"""Thin HTTP client for the hosted SciAtlas knowledge-graph API.

The request/response contracts mirror the live OpenAPI spec at
``http://sciatlas.openkg.cn/openapi.json``:

  * ``POST /v1/search``           — main graph retrieval (plan + options)
  * ``POST /v1/papers/search``    — vector paper search by title/abstract
  * ``POST /v1/keywords/search``  — keyword node lookup (exact/vector)
  * ``POST /v1/authors/related``  — related authors for a plan
  * ``GET  /v1/auth/token/status`` / ``/v1/auth/usage`` — quota helpers
  * ``GET  /healthz``             — health check

All retrieval routes are token-authenticated; we send both
``Authorization: Bearer`` and ``X-API-Key`` for compatibility, exactly as the
official CLI does.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from ..config import SciAtlasConfig
from ..utils import get_logger, read_json, retry, save_json, stable_hash

# A 4xx in the CLI's captured output is a caller error and must not be retried;
# everything else (5xx, the gateway's 502 on slow searches, 429, timeouts,
# transport HTTPError) is transient and worth retrying.
_HTTP_STATUS_RE = re.compile(r"HTTP status:\s*`?(\d{3})`?")

log = get_logger("sciatlas.client")

# Anchor levels accepted by the graph for bias options.
BIAS_LEVELS = ("low", "middle", "high")
RANKING_PROFILES = ("precision", "balanced", "discovery", "impact")
RETRIEVAL_MODES = ("keyword", "semantic", "title", "hybrid")


class SciAtlasError(RuntimeError):
    """Raised when the SciAtlas API returns a non-success response."""


class SciAtlasClient:
    def __init__(
        self,
        config: SciAtlasConfig | None = None,
        *,
        trace_path: Path | None = None,
    ) -> None:
        self.cfg = config or SciAtlasConfig()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "X-API-Key": self.cfg.api_key,
                "Content-Type": "application/json",
            }
        )
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = trace_path
        self._trace_context: str | None = None
        # Step 1's three sources (KG/S2/OpenAlex) trace from parallel threads, so
        # the read-modify-write of the trace file must be serialized.
        self._trace_lock = threading.Lock()

    def set_trace_context(self, context: str | None) -> None:
        self._trace_context = context

    def record_external_retrieval(
        self,
        *,
        source: str,
        query: str,
        papers: Iterable[Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record an OpenAlex/S2 retrieval into the run trace under the current step.

        SciAtlas KG calls trace themselves via :meth:`_append_trace`; OpenAlex and
        S2 run in separate code paths (step1/step2), so steps call this to keep the
        trace a complete picture of every source that contributed candidates.
        """
        records = [
            {
                "paper_id": getattr(paper, "paper_id", None),
                "title": getattr(paper, "title", "") or "",
                "abstract": getattr(paper, "abstract", "") or "",
                "year": getattr(paper, "year", None),
            }
            for paper in papers
        ]
        payload = {"source": source, "query": query}
        if extra:
            payload.update(extra)
        self._append_trace(
            command_name=source,
            payload=payload,
            result={"papers": records},
            cache_hit=False,
        )

    def _append_trace(
        self,
        *,
        command_name: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        cache_hit: bool,
    ) -> None:
        if self._trace_path is None:
            return
        papers: list[dict[str, Any]] = []
        for item in result.get("papers", []) or []:
            if not isinstance(item, dict):
                continue
            papers.append(
                {
                    "paper_id": item.get("paper_id") or item.get("id"),
                    "title": item.get("title") or item.get("name") or "",
                    "abstract": item.get("abstract") or item.get("summary") or "",
                    "year": item.get("year") or item.get("publication_year"),
                    "score": item.get("score") or item.get("rank_score") or item.get("relevance"),
                }
            )
        with self._trace_lock:
            trace = {"version": 1, "events": []}
            if self._trace_path.exists():
                trace = read_json(self._trace_path)
                if not isinstance(trace, dict):
                    trace = {"version": 1, "events": []}
            events = trace.setdefault("events", [])
            events.append(
                {
                    "index": len(events) + 1,
                    "step": self._trace_context,
                    "command_name": command_name,
                    "cache_hit": cache_hit,
                    "payload": payload,
                    "paper_count": len(papers),
                    "papers": papers,
                }
            )
            save_json(trace, self._trace_path)

    def _cache_path(self, namespace: str, payload: dict[str, Any]) -> Path:
        key = stable_hash(payload)
        return self.cfg.cache_dir / namespace / f"{key}.json"

    def _read_cache(self, namespace: str, payload: dict[str, Any]) -> dict | None:
        if not self.cfg.use_cache:
            return None
        path = self._cache_path(namespace, payload)
        if path.exists():
            log.info("SciAtlas cache hit: %s", path)
            return read_json(path)
        return None

    def _write_cache(self, namespace: str, payload: dict[str, Any], data: dict) -> None:
        if not self.cfg.use_cache:
            return
        save_json(data, self._cache_path(namespace, payload))

    # ------------------------------------------------------------------ #
    # low-level transport
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, *, json: dict | None = None) -> Any:
        url = f"{self.cfg.base_url.rstrip('/')}{path}"

        def _do() -> Any:
            resp = self._session.request(
                method, url, json=json, timeout=self.cfg.timeout
            )
            # 5xx (incl. the gateway's 502 on slow searches) and 429 are retryable.
            if resp.status_code >= 500 or resp.status_code == 429:
                raise SciAtlasError(
                    f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:200]}"
                )
            if resp.status_code >= 400:
                # 4xx are caller errors; do not retry.
                raise _NonRetryable(
                    f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()

        return retry(
            _do,
            max_retries=self.cfg.max_retries,
            backoff=self.cfg.retry_backoff,
            retry_on=(SciAtlasError, requests.RequestException),
            description=f"SciAtlas {method} {path}",
        )

    # ------------------------------------------------------------------ #
    # health / quota
    # ------------------------------------------------------------------ #
    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def token_status(self) -> dict:
        return self._request("GET", "/v1/auth/token/status")

    def usage(self, days: int = 7) -> dict:
        return self._request("GET", f"/v1/auth/usage?days={days}")

    # ------------------------------------------------------------------ #
    # main graph search  (POST /v1/search)
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_text: str,
        *,
        keywords: list[dict[str, Any]] | None = None,
        titles: list[dict[str, Any]] | None = None,
        reference_titles: list[str] | None = None,
        source_type: str = "idea_text",
        source_title: str | None = None,
        top_k: int = 20,
        retrieval_mode: str = "hybrid",
        after: str | None = None,
        before: str | None = None,
        target_field: str | None = None,
        retrieval_bias: dict[str, str] | None = None,
    ) -> dict:
        """Run KG search via the official SciAtlas CLI and return normalized results."""
        if self.cfg.use_local_kg:
            return self._run_local_kg_command(
                command_name="search-papers",
                payload={
                    "query_text": query_text,
                    "keywords": keywords or [{"text": query_text, "score": 8}],
                    "source_type": source_type,
                    "source_title": source_title,
                    "top_k": top_k,
                    "retrieval_mode": retrieval_mode,
                    "after": after,
                    "before": before,
                    "target_field": target_field,
                    "backend": "local_kg",
                },
                query_text=query_text,
                top_k=top_k,
                after=after,
                before=before,
                target_field=target_field,
                cache_namespace="local_kg_search",
            )
        extra_args: list[str] = []
        if after:
            extra_args.extend(["--after", after])
        if before:
            extra_args.extend(["--before", before])
        if target_field:
            extra_args.extend(["--target-field", target_field])
        if retrieval_bias:
            bias_map = {
                "keyword": "--bias-keyword",
                "citation": "--bias-citation",
                "paper_relatedness": "--bias-related",
                "authorship": "--bias-authorship",
                "coauthorship": "--bias-coauthorship",
                "cooccurrence": "--bias-cooccurrence",
                "exploration": "--bias-exploration",
            }
            for key, value in retrieval_bias.items():
                flag = bias_map.get(key)
                if flag and value:
                    extra_args.extend([flag, value])
        if titles:
            for item in titles:
                title = item.get("title")
                if title:
                    extra_args.extend(["--title", f"high:{title}"])
        if reference_titles:
            for title in reference_titles:
                extra_args.extend(["--reference", f"high:{title}"])
        return self._run_official_cli_command(
            command_name="search-papers",
            payload={
                "query_text": query_text,
                "keywords": keywords or [{"text": query_text, "score": 8}],
                "titles": titles or [],
                "reference_titles": reference_titles or [],
                "source_type": source_type,
                "source_title": source_title,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "after": after,
                "before": before,
                "target_field": target_field,
                "retrieval_bias": retrieval_bias or {},
            },
            cli_args=[
                "--query",
                query_text,
                "--top-k",
                str(top_k),
                "--top-keywords",
                "0",
                "--max-titles",
                "0",
                "--max-refs",
                "0",
                "--report-max-items",
                str(top_k),
                "--retrieval-mode",
                retrieval_mode,
                *extra_args,
            ],
            keyword_items=keywords,
            normalizer=_normalize_search_result,
            cache_namespace="official_cli_search",
        )

    # ------------------------------------------------------------------ #
    # lightweight vector helpers
    # ------------------------------------------------------------------ #
    def search_papers(
        self,
        query_text: str,
        *,
        field: str = "abstract",
        top_k: int = 10,
        after: str | None = None,
        before: str | None = None,
    ) -> dict:
        """Run low-level paper search via the official SciAtlas CLI."""
        if self.cfg.use_local_kg:
            return self._run_local_kg_command(
                command_name="paper-search",
                payload={
                    "query_text": query_text,
                    "field": field,
                    "top_k": top_k,
                    "after": after,
                    "before": before,
                    "backend": "local_kg",
                },
                query_text=query_text,
                top_k=top_k,
                after=after,
                before=before,
                target_field=None,
                cache_namespace="local_kg_paper_search",
            )
        extra_args: list[str] = []
        if after:
            extra_args.extend(["--after", after])
        if before:
            extra_args.extend(["--before", before])
        return self._run_official_cli_command(
            command_name="paper-search",
            payload={
                "query_text": query_text,
                "field": field,
                "top_k": top_k,
                "after": after,
                "before": before,
            },
            cli_args=[
                "--text",
                query_text,
                "--mode",
                "vector",
                "--field",
                field,
                "--top-k",
                str(top_k),
                "--report-max-items",
                str(top_k),
                *extra_args,
            ],
            normalizer=_normalize_search_result,
            cache_namespace="official_cli_paper_search",
        )

    def search_keywords(
        self, query_text: str, *, mode: str = "vector", top_k: int = 10
    ) -> dict:
        """Look up keyword nodes (``mode`` = ``exact`` or ``vector``)."""
        payload = {"query_text": query_text, "mode": mode, "options": {"top_k": top_k}}
        cached = self._read_cache("http_keywords_search", payload)
        if cached is not None:
            return _unwrap(cached)
        envelope = self._request("POST", "/v1/keywords/search", json=payload)
        self._write_cache("http_keywords_search", payload, envelope)
        return _unwrap(envelope)

    def related_authors(
        self, query_text: str, *, keywords: list[dict] | None = None, top_k: int = 10
    ) -> dict:
        return self._run_official_cli_command(
            command_name="related-authors",
            payload={
                "query_text": query_text,
                "keywords": keywords or [{"text": query_text, "score": 8}],
                "top_k": top_k,
            },
            cli_args=[
                "--query",
                query_text,
                "--top-k",
                str(top_k),
                "--report-max-items",
                str(top_k),
            ],
            keyword_items=keywords,
            cache_namespace="official_cli_related_authors",
        )

    def search_via_official_cli(
        self,
        query_text: str,
        *,
        keywords: list[dict[str, Any]] | None = None,
        top_k: int = 3,
        retrieval_mode: str = "hybrid",
    ) -> dict:
        """Backward-compatible wrapper around the CLI-backed search path."""
        return self.search(
            query_text,
            keywords=keywords,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )

    def _run_official_cli_command(
        self,
        *,
        command_name: str,
        payload: dict[str, Any],
        cli_args: list[str],
        keyword_items: list[dict[str, Any]] | None = None,
        normalizer: Callable[[dict], dict] | None = None,
        cache_namespace: str,
    ) -> dict:
        cached = self._read_cache(cache_namespace, payload)
        if cached is not None:
            result = _unwrap(cached)
            normalized = normalizer(result) if normalizer else result
            self._append_trace(
                command_name=command_name,
                payload=payload,
                result=normalized,
                cache_hit=True,
            )
            return normalized

        cli_root = self.cfg.official_cli_root.resolve()
        cli_script = (cli_root / "run_sciatlas.py").resolve()
        if not cli_script.exists():
            raise FileNotFoundError(f"Official SciAtlas CLI not found at {cli_script}")

        run_id = f"cache_{stable_hash({'command': command_name, 'payload': payload})[:16]}"
        runs_dir = (self.cfg.cache_dir / "official_cli_runs").resolve()
        runs_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        official_timeout = max(self.cfg.timeout, 900)
        env["SCIATLAS_API_BASE_URL"] = self.cfg.base_url
        env["SCIATLAS_API_KEY"] = self.cfg.api_key
        env["SCIATLAS_TIMEOUT"] = str(official_timeout)
        attempt = 0

        def _do() -> dict:
            nonlocal attempt
            attempt += 1
            attempt_run_id = f"{run_id}_try{attempt}"
            command = [
                sys.executable,
                str(cli_script),
                "--runs-dir",
                str(runs_dir),
                "--timeout",
                str(official_timeout),
                command_name,
                "--run-id",
                attempt_run_id,
                *cli_args,
            ]
            for item in keyword_items or []:
                text = item.get("text")
                if text:
                    command.extend(["--keyword", f"high:{text}"])

            log.info(
                "SciAtlas official CLI %s (attempt %d/%d): %s",
                command_name,
                attempt,
                self.cfg.max_retries,
                payload.get("query_text", ""),
            )
            proc = subprocess.run(
                command,
                cwd=cli_root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout).strip()
                status_code = _extract_cli_http_status(detail)
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise _NonRetryable(
                        f"Official CLI {command_name} failed ({proc.returncode}, HTTP {status_code}): {detail}"
                    )
                raise SciAtlasError(
                    f"Official CLI {command_name} failed ({proc.returncode}): {detail}"
                )

            response_path = runs_dir / attempt_run_id / "response.json"
            if not response_path.exists():
                raise SciAtlasError(f"Official CLI response artifact missing: {response_path}")
            response = read_json(response_path)
            data = response.get("data")
            if not isinstance(data, dict):
                raise SciAtlasError(f"Unexpected official CLI response shape: {response_path}")
            return data

        data = retry(
            _do,
            max_retries=self.cfg.max_retries,
            backoff=self.cfg.retry_backoff,
            retry_on=(SciAtlasError, OSError),
            description=f"Official CLI {command_name}",
        )
        self._write_cache(cache_namespace, payload, data)
        result = _unwrap(data)
        normalized = normalizer(result) if normalizer else result
        self._append_trace(
            command_name=command_name,
            payload=payload,
            result=normalized,
            cache_hit=False,
        )
        return normalized

    # ------------------------------------------------------------------ #
    # local KG engine (references/search :: innoeval_search)
    # ------------------------------------------------------------------ #
    def _run_local_kg_command(
        self,
        *,
        command_name: str,
        payload: dict[str, Any],
        query_text: str,
        top_k: int,
        after: str | None = None,
        before: str | None = None,
        target_field: str | None = None,
        cache_namespace: str,
    ) -> dict:
        """Run KG search against the bundled local Neo4j engine (`run_search.py`).

        Mirrors :meth:`_run_official_cli_command`'s cache/trace/normalize flow but
        shells out to ``references/search/run_search.py`` in KG-only mode instead
        of the hosted ``/v1/search`` gateway. The engine prints the merged result
        (with ``ranking``/``sources.kg`` papers) as JSON on stdout, which
        :func:`_normalize_search_result` flattens to ``result["papers"]``.
        """
        cached = self._read_cache(cache_namespace, payload)
        if cached is not None:
            normalized = _normalize_search_result(_unwrap(cached))
            self._append_trace(
                command_name=command_name,
                payload=payload,
                result=normalized,
                cache_hit=True,
            )
            return normalized

        search_root = self.cfg.local_search_root.resolve()
        search_script = (search_root / "run_search.py").resolve()
        if not search_script.exists():
            raise FileNotFoundError(
                f"Local KG search engine not found at {search_script}"
            )

        env = os.environ.copy()
        env_path = search_root / ".env"

        def _do() -> dict:
            command = [
                sys.executable,
                str(search_script),
                "--idea-text",
                query_text,
                "--disable-s2",
                "--disable-llm-ranking",
                "--disable-result-log",
                "--kg-top-k",
                str(top_k),
                "--s2-top-k",
                str(top_k),
                "--final-top-k",
                str(top_k),
            ]
            if env_path.exists():
                command.extend(["--env", str(env_path)])
            if after:
                command.extend(["--after", after])
            if before:
                command.extend(["--before", before])
            if target_field:
                command.extend(["--Target-Field", target_field])

            log.info("SciAtlas local KG search: %s", query_text)
            proc = subprocess.run(
                command,
                cwd=search_root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            stdout = proc.stdout or ""
            data = _parse_local_kg_stdout(stdout)
            if data is None:
                detail = (proc.stderr or stdout).strip()[-1200:]
                raise SciAtlasError(
                    f"Local KG search produced no parseable JSON "
                    f"(returncode={proc.returncode}): {detail}"
                )
            return data

        data = retry(
            _do,
            max_retries=self.cfg.max_retries,
            backoff=self.cfg.retry_backoff,
            retry_on=(SciAtlasError, OSError),
            description=f"Local KG {command_name}",
        )
        self._write_cache(cache_namespace, payload, data)
        normalized = _normalize_search_result(_unwrap(data))
        self._append_trace(
            command_name=command_name,
            payload=payload,
            result=normalized,
            cache_hit=False,
        )
        return normalized


class _NonRetryable(RuntimeError):
    """A 4xx error that should not be retried."""


def _extract_cli_http_status(text: str) -> int | None:
    match = _HTTP_STATUS_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_local_kg_stdout(stdout: str) -> dict | None:
    """Extract the merged-search JSON object printed by ``run_search.py``.

    The engine prints a single ``json.dumps(...)`` line on stdout, but model
    loaders may emit progress noise around it, so we fall back to scanning lines
    from the end for the last line that parses as a JSON object with ``status``.
    """
    import json

    text = stdout.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("status" in obj or "combined" in obj):
            return obj
    return None


def _unwrap(envelope: Any) -> dict:
    """Return the ``result`` payload from an API response envelope.

    The API wraps responses as ``{request_id, status, result, meta?}``. We pass
    the inner ``result`` to callers but tolerate already-unwrapped dicts.
    """
    if isinstance(envelope, dict) and "result" in envelope:
        status = envelope.get("status")
        if status not in (None, "ok", "success"):
            log.warning("SciAtlas response status=%s", status)
        return envelope["result"] or {}
    return envelope if isinstance(envelope, dict) else {"data": envelope}


def _normalize_search_result(result: dict) -> dict:
    """Normalize search outputs so downstream code can always read ``papers``."""
    if "papers" in result and isinstance(result["papers"], list):
        return result
    ranking = result.get("ranking")
    if isinstance(ranking, dict) and isinstance(ranking.get("papers"), list):
        normalized = dict(result)
        normalized["papers"] = ranking["papers"]
        return normalized
    sources = result.get("sources")
    if isinstance(sources, dict):
        kg = sources.get("kg")
        if isinstance(kg, dict) and isinstance(kg.get("papers"), list):
            normalized = dict(result)
            normalized["papers"] = kg["papers"]
            return normalized
    return result
