#!/usr/bin/env python3
"""Standalone literature-review search MVP.

The pipeline follows downstream/Literature Review Search MVP Implementation Plan.md:
topic profiling, probe search, time slicing, two search rounds, method clustering,
coverage diagnosis, and final search_result.json emission.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from openai import OpenAI


LR_SEARCH_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = LR_SEARCH_DIR.parent
INNOEVAL_DIR = DOWNSTREAM_DIR.parent
PROJECT_ROOT = INNOEVAL_DIR.parent
SEARCH_DIR = INNOEVAL_DIR / "search"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_DMX_API_URL = "https://www.dmxapi.cn/v1/chat/completions"
DEFAULT_DMX_MODEL = "deepseek-v3.2"
DEFAULT_CURRENT_YEAR = date.today().year
QUERY_STYLES = {"broad", "facet", "exact_title", "named_work", "frontier"}
EVIDENCE_ROLES = {
    "foundational",
    "original_method",
    "representative",
    "benchmark",
    "survey",
    "recent_frontier",
    "gap_refinement",
    "scope_probe",
}
ROUND1_INTENTS = {
    "core_topic_recall",
    "method_facet_recall",
    "foundational_recall",
    "recent_frontier_recall",
    "benchmark_or_evaluation_recall",
    "survey_or_taxonomy_recall",
}
ROUND2_INTENTS = {
    "missing_method_family_query",
    "cluster_deepening_query",
    "foundational_query",
    "recent_frontier_query",
    "transition_query",
    "representative_query",
    "disambiguation_query",
    "benchmark_or_evaluation_query",
    "survey_or_taxonomy_query",
    "canonical_anchor_query",
}

for extra_path in (INNOEVAL_DIR, SEARCH_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def progress_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[lr_search {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m{remainder:.1f}s"


@contextmanager
def progress_stage(enabled: bool, name: str):
    started_at = time.perf_counter()
    progress_log(enabled, f"START {name}")
    try:
        yield
    except Exception:
        progress_log(enabled, f"FAIL {name} after {format_elapsed(time.perf_counter() - started_at)}")
        raise
    progress_log(enabled, f"DONE {name} in {format_elapsed(time.perf_counter() - started_at)}")


def load_env_values(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        candidate = cleaned[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
            except ImportError:
                raise
            payload = repair_json(candidate, return_objects=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def truncate_text(value: Any, max_chars: int) -> str:
    text = normalize_whitespace(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def slugify(value: str, *, limit: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalize_whitespace(value).casefold()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:limit].rstrip("-") or "topic")


def normalize_query_style(value: Any, *, intent: str = "") -> str:
    text = normalize_whitespace(value).casefold()
    if text in QUERY_STYLES:
        return text
    if "frontier" in intent or "recent" in intent:
        return "frontier"
    if "foundational" in intent or "canonical" in intent:
        return "named_work"
    if "facet" in intent or "missing_method" in intent or "cluster" in intent:
        return "facet"
    return "broad"


def normalize_evidence_role(value: Any, *, intent: str = "") -> str:
    values = list_of_strings(value, limit=4) if isinstance(value, list) else [normalize_whitespace(value)]
    for item in values:
        lowered = item.casefold()
        if lowered in EVIDENCE_ROLES:
            return lowered
    if "foundational" in intent or "canonical" in intent:
        return "foundational"
    if "benchmark" in intent or "evaluation" in intent:
        return "benchmark"
    if "survey" in intent or "taxonomy" in intent:
        return "survey"
    if "recent" in intent or "frontier" in intent:
        return "recent_frontier"
    if "method" in intent:
        return "original_method"
    return "representative"


def sanitize_query_text(query: str, *, topic: str = "", intent: str = "") -> str:
    text = normalize_whitespace(query)
    if not text:
        return ""
    tokens = []
    for token in text.split():
        bare = token.strip("\"'`()[]{}.,;:")
        lowered = bare.casefold()
        if re.fullmatch(r"(19|20)\d{2}", bare):
            continue
        if lowered in {"seminal", "classic", "classical", "frontier", "foundational"}:
            continue
        tokens.append(token)
    cleaned = normalize_whitespace(" ".join(tokens))
    topic_text = normalize_whitespace(topic)
    if topic_text and topic_text.casefold() not in cleaned.casefold() and "exact_title" not in intent.casefold():
        intent_lower = intent.casefold()
        if intent_lower:
            cleaned = normalize_whitespace(f"{topic_text} {cleaned}")
    return cleaned


class DmxJsonClient:
    def __init__(
        self,
        *,
        env_path: Path,
        api_url: str,
        model: str,
        timeout: int,
        max_tokens: int,
        temperature: float,
        use_env_proxy: bool,
        api_key_override: str | None = None,
        wire_api: str = "chat_completions",
        mock: bool = False,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.wire_api = normalize_whitespace(wire_api) or "chat_completions"
        self.mock = mock
        self.call_count = 0
        env_values = load_env_values(env_path)
        self.api_key = (
            normalize_whitespace(api_key_override)
            or normalize_whitespace(os.environ.get("DMX-API-KEY"))
            or normalize_whitespace(os.environ.get("DMX_API_KEY"))
            or normalize_whitespace(os.environ.get("OPENAI_API_KEY"))
            or normalize_whitespace(env_values.get("DMX-API-KEY"))
            or normalize_whitespace(env_values.get("DMX_API_KEY"))
            or normalize_whitespace(env_values.get("OPENAI_API_KEY"))
        )
        if not self.api_key and not mock:
            raise RuntimeError(f"Missing DMX API key in environment or {env_path}")
        self.opener = urllib.request.build_opener() if use_env_proxy else urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )
        self.responses_client = (
            OpenAI(api_key=self.api_key, base_url=self.api_url)
            if (not mock and self.wire_api == "responses")
            else None
        )

    def chat_json(self, *, system_prompt: str, user_prompt: str, label: str) -> dict[str, Any]:
        content = self.chat_json_text(system_prompt=system_prompt, user_prompt=user_prompt, label=label)
        return parse_json_object(content)

    def chat_json_text(self, *, system_prompt: str, user_prompt: str, label: str) -> str:
        self.call_count += 1
        if self.mock:
            return json.dumps(mock_llm_response(label=label, user_prompt=user_prompt), ensure_ascii=False)
        if self.wire_api == "responses":
            return self._responses_json_text(system_prompt=system_prompt, user_prompt=user_prompt, label=label)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"] or "{}"
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"DMX HTTP {exc.code}: {detail}")
                if attempt < 3 and exc.code in {429, 500, 502, 503, 504}:
                    time.sleep(2**attempt)
                    continue
                raise last_error
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"DMX call failed for {label}: {last_error}") from exc
        raise RuntimeError(f"DMX call failed for {label}: {last_error}")

    def _responses_json_text(self, *, system_prompt: str, user_prompt: str, label: str) -> str:
        if self.responses_client is None:
            raise RuntimeError(f"Responses client is not initialized for {label}")
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.responses_client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    store=False,
                )
                output_text = normalize_whitespace(getattr(response, "output_text", ""))
                if output_text:
                    return output_text
                dumped = response.model_dump()
                chunks: list[str] = []
                for item in dumped.get("output", []):
                    for content in item.get("content", []):
                        text = normalize_whitespace(content.get("text"))
                        if text:
                            chunks.append(text)
                combined = "\n".join(chunks).strip()
                if combined:
                    return combined
                raise RuntimeError("Responses API returned no text output.")
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Responses API call failed for {label}: {last_error}") from exc
        raise RuntimeError(f"Responses API call failed for {label}: {last_error}")


class MergeSearchBackend:
    def __init__(
        self,
        *,
        env_path: Path,
        cache_dir: Path,
        use_env_proxy: bool,
        s2_mode: str | None,
        kg_embedding_device: str | None,
        kg_reranker_device: str | None,
        kg_policy_enabled: bool,
        mock: bool = False,
    ) -> None:
        self.env_path = env_path
        self.cache_dir = ensure_dir(cache_dir)
        self.use_env_proxy = use_env_proxy
        self.s2_mode = s2_mode
        self.kg_embedding_device = kg_embedding_device
        self.kg_reranker_device = kg_reranker_device
        self.kg_policy_enabled = kg_policy_enabled
        self.mock = mock

    def search(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.mock:
            return mock_search_payload(action)

        return self._run_sciatlas_backend(action)

    def _run_sciatlas_backend(self, action: dict[str, Any]) -> dict[str, Any]:
        top_k = self._action_top_k(action)
        run_id = f"lr_{slugify(str(action.get('action_id') or 'search'))}_{hashlib.sha1(normalize_whitespace(action.get('query')).encode('utf-8')).hexdigest()[:8]}"
        runs_dir = ensure_dir(self.cache_dir / "sciatlas_backend_runs")
        response_path = runs_dir / run_id / "response.json"
        cache_path = self.cache_dir / f"sciatlas_{slugify(str(action.get('action_id') or 'search'))}.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_sciatlas.py"),
            "--runs-dir",
            str(runs_dir),
            "--timeout",
            str(self._sciatlas_timeout()),
            "search-papers",
            "--run-id",
            run_id,
            "--query",
            normalize_whitespace(action.get("query")),
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
            "hybrid",
        ]
        time_window = action.get("time_window")
        if isinstance(time_window, dict):
            if time_window.get("start_year"):
                command.extend(["--after", f"{int(time_window['start_year'])}-01-01"])
            if time_window.get("end_year"):
                command.extend(["--before", f"{int(time_window['end_year'])}-12-31"])

        started_at = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=self._sciatlas_cli_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._sciatlas_timeout() + 30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            payload = self._error_payload(
                action,
                "TimeoutExpired",
                f"SciAtlas backend search timed out after {self._sciatlas_timeout()} seconds.",
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
            write_json(cache_path, payload)
            return payload

        response = self._read_sciatlas_response(response_path)
        if proc.returncode != 0 or not (isinstance(response, dict) and response.get("ok")):
            error_type = ""
            error = ""
            if isinstance(response, dict):
                error_type = normalize_whitespace(response.get("error_type"))
                error = normalize_whitespace(response.get("error"))
            payload = self._error_payload(
                action,
                error_type or f"CliExit{proc.returncode}",
                error or "SciAtlas backend search failed.",
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                stdout=proc.stdout,
                stderr=proc.stderr,
                response=response,
                response_path=response_path,
            )
            write_json(cache_path, payload)
            return payload

        payload = self._success_payload(
            action,
            response,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            response_path=response_path,
        )
        write_json(cache_path, payload)
        return payload

    def _sciatlas_cli_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key, value in load_env_values(self.env_path).items():
            env.setdefault(key, value)
        if not env.get("SCIATLAS_API_KEY") and env.get("SCISCHOLAR_API_KEY"):
            env["SCIATLAS_API_KEY"] = env["SCISCHOLAR_API_KEY"]
        if not env.get("SCIATLAS_API_BASE_URL") and env.get("SCISCHOLAR_API_BASE_URL"):
            env["SCIATLAS_API_BASE_URL"] = env["SCISCHOLAR_API_BASE_URL"]
        return env

    def _sciatlas_timeout(self) -> int:
        for key in ("SCIATLAS_SEARCH_TIMEOUT", "SCIATLAS_TIMEOUT"):
            value = os.environ.get(key)
            if value:
                try:
                    return max(30, int(value))
                except ValueError:
                    pass
        return 180

    def _action_top_k(self, action: dict[str, Any]) -> int:
        kg_top_k = parse_int(action.get("kg_top_k")) or 0
        s2_top_k = parse_int(action.get("s2_top_k")) or 0
        return max(1, min(50, kg_top_k + s2_top_k or kg_top_k or s2_top_k or 10))

    def _read_sciatlas_response(self, response_path: Path) -> dict[str, Any] | None:
        if not response_path.exists():
            return None
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _success_payload(
        self,
        action: dict[str, Any],
        response: dict[str, Any],
        *,
        elapsed_ms: float,
        response_path: Path,
    ) -> dict[str, Any]:
        result = self._unwrap_sciatlas_result(response)
        raw_papers = self._extract_sciatlas_papers(result)
        papers = self._dedupe_sciatlas_papers(raw_papers)
        kg_payload = {
            "source": "kg",
            "backend": "sciatlas",
            "status": "ok",
            "elapsed_ms": round(elapsed_ms, 1),
            "paper_count": len(papers),
            "papers": papers,
            "response_path": str(response_path),
        }
        return {
            "status": "ok",
            "successful_source_count": 1,
            "failed_source_count": 0,
            "query": action.get("query"),
            "backend": "sciatlas",
            "filter": {"unique_papers": papers, "unique_paper_count": len(papers)},
            "ranking": {"status": "sciatlas", "papers": papers},
            "combined": [
                {"source": "kg", "source_rank": index, "paper": paper}
                for index, paper in enumerate(papers, start=1)
            ],
            "sources": {
                "kg": kg_payload,
                "sciatlas": dict(kg_payload),
                "s2": {
                    "source": "s2",
                    "status": "skipped",
                    "reason": "Paper retrieval is restricted to the SciAtlas backend.",
                    "paper_count": 0,
                    "papers": [],
                },
            },
        }

    def _error_payload(
        self,
        action: dict[str, Any],
        error_type: str,
        error: str,
        *,
        elapsed_ms: float,
        stdout: str | bytes | None = None,
        stderr: str | bytes | None = None,
        response: dict[str, Any] | None = None,
        response_path: Path | None = None,
    ) -> dict[str, Any]:
        source_payload = {
            "source": "kg",
            "backend": "sciatlas",
            "status": "error",
            "error_type": normalize_whitespace(error_type) or "SciAtlasSearchError",
            "error": normalize_whitespace(error),
            "elapsed_ms": round(elapsed_ms, 1),
            "paper_count": 0,
            "papers": [],
            "stdout_tail": truncate_text(stdout, 1200),
            "stderr_tail": truncate_text(stderr, 1200),
        }
        if response_path is not None:
            source_payload["response_path"] = str(response_path)
        if response is not None:
            source_payload["response"] = response
        return {
            "status": "error",
            "successful_source_count": 0,
            "failed_source_count": 1,
            "query": action.get("query"),
            "backend": "sciatlas",
            "filter": {"unique_papers": [], "unique_paper_count": 0},
            "ranking": {"status": "skipped", "papers": []},
            "combined": [],
            "sources": {
                "kg": source_payload,
                "sciatlas": dict(source_payload),
                "s2": {
                    "source": "s2",
                    "status": "skipped",
                    "reason": "Paper retrieval is restricted to the SciAtlas backend.",
                    "paper_count": 0,
                    "papers": [],
                },
            },
        }

    def _unwrap_sciatlas_result(self, response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data")
        if isinstance(data, dict):
            result = data.get("result")
            if isinstance(result, dict):
                return result
            return data
        result = response.get("result")
        return result if isinstance(result, dict) else response

    def _extract_sciatlas_papers(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[Any] = []
        ranking = result.get("ranking") if isinstance(result, dict) else None
        if isinstance(ranking, dict):
            candidates.append(ranking.get("papers"))
        sources = result.get("sources") if isinstance(result, dict) else None
        if isinstance(sources, dict):
            kg = sources.get("kg")
            if isinstance(kg, dict):
                candidates.append(kg.get("papers"))
            sciatlas = sources.get("sciatlas")
            if isinstance(sciatlas, dict):
                candidates.append(sciatlas.get("papers"))
        candidates.append(result.get("papers") if isinstance(result, dict) else None)

        for value in candidates:
            if isinstance(value, list):
                papers = [
                    self._normalize_sciatlas_paper(item, rank)
                    for rank, item in enumerate(value, start=1)
                    if isinstance(item, dict)
                ]
                return [paper for paper in papers if paper is not None]
        return []

    def _normalize_sciatlas_paper(self, item: dict[str, Any], rank: int) -> dict[str, Any] | None:
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        title = normalize_whitespace(item.get("title") or paper.get("title"))
        if not title:
            return None
        normalized = dict(item)
        normalized["title"] = title
        normalized.setdefault("rank", rank)
        normalized.setdefault("source", "kg")
        source_set = normalized.get("source_set")
        if isinstance(source_set, list):
            source_values = {normalize_whitespace(value) for value in source_set}
            source_values.update({"kg", "sciatlas"})
            normalized["source_set"] = sorted(value for value in source_values if value)
        else:
            normalized["source_set"] = ["kg", "sciatlas"]
        normalized.setdefault("paper_url", normalize_whitespace(item.get("paper_url") or paper.get("id") or paper.get("url")))
        normalized.setdefault("abstract", normalize_whitespace(item.get("abstract") or paper.get("abstract")))
        normalized.setdefault("year", item.get("year") or paper.get("publication_year") or paper.get("year"))
        normalized.setdefault(
            "citation_count",
            item.get("citation_count") or item.get("cited_by_count") or paper.get("cited_by_count"),
        )
        identifiers = normalized.get("identifiers") if isinstance(normalized.get("identifiers"), list) else []
        if not identifiers:
            identifiers = self._identifiers_from_sciatlas_paper(normalized, paper)
            if identifiers:
                normalized["identifiers"] = identifiers
        return normalized

    def _identifiers_from_sciatlas_paper(self, item: dict[str, Any], paper: dict[str, Any]) -> list[str]:
        identifiers: list[str] = []
        doi = normalize_whitespace(item.get("doi") or paper.get("doi"))
        if doi:
            identifiers.append(f"doi:{doi}")
        openalex_id = normalize_whitespace(
            item.get("paper_id")
            or item.get("id")
            or item.get("paper_url")
            or paper.get("id")
            or paper.get("paper_url")
        )
        if openalex_id and "openalex.org" in openalex_id:
            identifiers.append(f"openalex:{openalex_id}")
        s2_id = normalize_whitespace(item.get("paperId") or paper.get("paperId"))
        if s2_id:
            identifiers.append(f"s2:{s2_id}")
        return identifiers

    def _dedupe_sciatlas_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for paper in papers:
            key = stable_key_from_candidate(paper)
            if key not in unique:
                unique[key] = paper
        return list(unique.values())


def build_kg_policy_args(action: dict[str, Any]) -> list[str]:
    """Map generic review-search intent to KG scoring knobs.

    These policies are topic-agnostic. They bias retrieval toward the evidence role
    requested by the action without hard-coding any method family or paper title.
    """
    query_style = normalize_query_style(action.get("query_style"), intent=normalize_whitespace(action.get("intent")))
    evidence_role = normalize_evidence_role(action.get("evidence_role"), intent=normalize_whitespace(action.get("intent")))
    intent = normalize_whitespace(action.get("intent"))

    args: list[str] = []
    if evidence_role in {"foundational", "original_method", "survey", "benchmark"} and intent != "canonical_anchor_query":
        args.extend(
            [
                "--kg-use-citation-importance",
                "--kg-final-weight-pre-graph",
                "0.45",
                "--kg-final-weight-graph",
                "0.35",
                "--kg-final-weight-importance",
                "0.20",
            ]
        )
    if query_style in {"exact_title", "named_work"} or evidence_role in {"foundational", "original_method"}:
        args.extend(
            [
                "--kg-weight-title-path",
                "1.35",
                "--kg-weight-embedding-path",
                "0.15",
                "--kg-title-exact-pre-graph-bonus",
                "0.80",
                "--kg-final-title-bonus",
                "0.35",
                "--kg-topk-title-rerank",
                "18",
                "--kg-topk-abstract-rerank",
                "10",
            ]
        )
    if query_style == "frontier" or evidence_role == "recent_frontier":
        args.extend(
            [
                "--kg-final-weight-pre-graph",
                "0.40",
                "--kg-final-weight-graph",
                "0.45",
                "--kg-final-weight-importance",
                "0.15",
            ]
        )
    return args


def annotate_kg_policy_metadata(actions: list[dict[str, Any]], *, enabled: bool) -> None:
    for action in actions:
        action["kg_policy_args"] = build_kg_policy_args(action) if enabled else []


def stable_title_hash(title: str) -> str:
    return hashlib.sha1(normalize_whitespace(title).casefold().encode("utf-8")).hexdigest()[:16]


def first_identifier(identifiers: list[Any], prefix: str) -> str:
    for value in identifiers:
        text = normalize_whitespace(value)
        if text.startswith(prefix):
            return text.split(":", 1)[1]
    return ""


def stable_key_from_candidate(candidate: dict[str, Any]) -> str:
    identifiers = candidate.get("identifiers") if isinstance(candidate.get("identifiers"), list) else []
    for prefix in ("doi:", "arxiv:", "s2:", "openalex:"):
        identifier = first_identifier(identifiers, prefix)
        if identifier:
            return f"{prefix}{identifier}".casefold()

    paper = candidate.get("paper") if isinstance(candidate.get("paper"), dict) else {}
    doi = normalize_whitespace(candidate.get("doi") or paper.get("doi"))
    if doi:
        return f"doi:{doi.casefold()}"
    s2_id = normalize_whitespace(candidate.get("paperId") or paper.get("paperId"))
    if s2_id:
        return f"s2:{s2_id}"
    openalex_id = normalize_whitespace(candidate.get("id") or paper.get("id"))
    if openalex_id:
        return f"openalex:{openalex_id}"
    title = normalize_whitespace(candidate.get("title") or paper.get("title"))
    return f"title:{stable_title_hash(title)}"


def normalize_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for item in value:
        if isinstance(item, str):
            name = normalize_whitespace(item)
        elif isinstance(item, dict):
            name = normalize_whitespace(item.get("name"))
        else:
            name = ""
        if name:
            authors.append(name)
    return authors[:20]


def normalize_paper_candidate(candidate: dict[str, Any], action_id: str, rank: int) -> dict[str, Any] | None:
    paper = candidate.get("paper") if isinstance(candidate.get("paper"), dict) else {}
    title = normalize_whitespace(candidate.get("title") or paper.get("title"))
    if not title:
        return None
    identifiers = candidate.get("identifiers") if isinstance(candidate.get("identifiers"), list) else []
    doi = first_identifier(identifiers, "doi:") or normalize_whitespace(candidate.get("doi") or paper.get("doi"))
    url = (
        normalize_whitespace(candidate.get("paper_url"))
        or normalize_whitespace(candidate.get("url"))
        or normalize_whitespace(paper.get("url"))
        or normalize_whitespace(candidate.get("pdf_url"))
    )
    source_set = candidate.get("source_set")
    if isinstance(source_set, list):
        source = sorted({normalize_whitespace(item) for item in source_set if normalize_whitespace(item)})
    else:
        source_value = normalize_whitespace(candidate.get("source") or paper.get("source"))
        source = [source_value] if source_value else []

    return {
        "stable_key": stable_key_from_candidate(candidate),
        "title": title,
        "abstract": normalize_whitespace(candidate.get("abstract") or paper.get("abstract")),
        "year": parse_year(candidate.get("year") or paper.get("year") or paper.get("publication_year")),
        "citation_count": parse_int(
            candidate.get("citation_count")
            or candidate.get("citationCount")
            or paper.get("citationCount")
            or paper.get("cited_by_count")
        ),
        "venue": normalize_whitespace(candidate.get("venue") or paper.get("venue") or paper.get("publicationVenue")),
        "authors": normalize_authors(candidate.get("authors") or paper.get("authors")),
        "doi": doi,
        "url": url,
        "source": source,
        "source_actions": [action_id],
        "retrieval_scores": {
            "source_rank_min": parse_int(candidate.get("rank") or candidate.get("source_rank") or rank),
        },
        "raw_group_id": normalize_whitespace(candidate.get("group_id")),
    }


def parse_year(value: Any) -> int | None:
    text = normalize_whitespace(value)
    if not text:
        return None
    match = re.match(r"^([12][0-9]{3})", text)
    if not match:
        return None
    year = int(match.group(1))
    if 1900 <= year <= DEFAULT_CURRENT_YEAR + 1:
        return year
    return None


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def candidates_from_search_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ranking_papers = payload.get("ranking", {}).get("papers")
    if isinstance(ranking_papers, list) and ranking_papers:
        return [item for item in ranking_papers if isinstance(item, dict)]
    unique_papers = payload.get("filter", {}).get("unique_papers")
    if isinstance(unique_papers, list):
        return [item for item in unique_papers if isinstance(item, dict)]
    return []


RELEVANCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "large",
    "language",
    "llm",
    "llms",
    "model",
    "models",
    "of",
    "on",
    "or",
    "paper",
    "the",
    "to",
    "using",
    "with",
}


def relevance_tokens(value: Any) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_-]*", normalize_whitespace(value).casefold()):
        token = token.replace("_", "-")
        if len(token) <= 2 or token in RELEVANCE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def is_candidate_relevant_to_action(card: dict[str, Any], action: dict[str, Any], *, current_year: int) -> bool:
    source = set(card.get("source", []))
    if "kg_embedding_gap" in source:
        return True
    if "s2" in source:
        return True
    if len(card.get("source_actions", [])) > 1:
        return True

    query_tokens = relevance_tokens(action.get("query"))
    if not query_tokens:
        return True
    title_tokens = relevance_tokens(card.get("title"))
    abstract_tokens = relevance_tokens(card.get("abstract"))
    text_tokens = title_tokens | set(list(abstract_tokens)[:80])

    shared = query_tokens & text_tokens
    title_shared = query_tokens & title_tokens
    has_topic_anchor = bool({"reasoning", "prompting", "thought", "cot", "chain"} & text_tokens)
    year = card.get("year")
    rank = (card.get("retrieval_scores") or {}).get("source_rank_min") or 9999
    citations = card.get("citation_count") or 0

    if isinstance(year, int) and year < 2020:
        return len(shared) >= 2 and bool(title_shared) and has_topic_anchor
    if isinstance(year, int) and year >= 2020:
        return bool(shared) or has_topic_anchor or citations >= 100 or rank <= 20
    if rank > 15 and len(shared) == 0:
        return False
    if citations >= 1000 and len(shared) == 0:
        return False
    return bool(shared) or has_topic_anchor or (isinstance(year, int) and year >= current_year - 2)


def merge_paper_cards(
    action_results: dict[str, dict[str, Any]],
    *,
    action_ids: list[str],
    actions_by_id: dict[str, dict[str, Any]] | None = None,
    abstract_char_limit: int,
    current_year: int = DEFAULT_CURRENT_YEAR,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    filtered_count = 0
    use_relevance_guard = actions_by_id is not None
    for action_id in action_ids:
        payload = action_results.get(action_id, {})
        action = (actions_by_id or {}).get(action_id, {"query": ""})
        for rank, candidate in enumerate(candidates_from_search_payload(payload), start=1):
            card = normalize_paper_candidate(candidate, action_id, rank)
            if card is None:
                continue
            if use_relevance_guard and not is_candidate_relevant_to_action(card, action, current_year=current_year):
                filtered_count += 1
                continue
            key = card["stable_key"]
            existing = merged.get(key)
            if existing is None:
                merged[key] = card
                continue
            for field in ("title", "abstract", "venue", "doi", "url"):
                if not existing.get(field) and card.get(field):
                    existing[field] = card[field]
            if card.get("abstract") and len(card["abstract"]) > len(existing.get("abstract", "")):
                existing["abstract"] = card["abstract"]
            if existing.get("year") is None and card.get("year") is not None:
                existing["year"] = card["year"]
            if card.get("citation_count") is not None:
                existing["citation_count"] = max(existing.get("citation_count") or 0, card["citation_count"])
            existing["source"] = sorted(set(existing.get("source", [])) | set(card.get("source", [])))
            existing["authors"] = existing.get("authors") or card.get("authors", [])
            existing["source_actions"] = sorted(set(existing["source_actions"]) | set(card["source_actions"]))
            existing["retrieval_scores"]["source_rank_min"] = min(
                existing["retrieval_scores"].get("source_rank_min") or 9999,
                card["retrieval_scores"].get("source_rank_min") or 9999,
            )

    cards = sorted(
        merged.values(),
        key=lambda item: (
            -(item.get("citation_count") or 0),
            item.get("year") is None,
            -(item.get("year") or 0),
            item.get("title", "").casefold(),
        ),
    )
    for index, card in enumerate(cards, start=1):
        card["paper_id"] = f"P{index:03d}"
        card["abstract"] = truncate_text(card.get("abstract"), abstract_char_limit)
        ordered = {
            "paper_id": card.pop("paper_id"),
            "stable_key": card.pop("stable_key"),
            **card,
        }
        cards[index - 1] = ordered
    if cards:
        cards[0].setdefault("retrieval_scores", {})["action_relevance_filtered_count_total"] = filtered_count
    return cards


def relevance_filtered_count(cards: list[dict[str, Any]]) -> int:
    if not cards:
        return 0
    return int((cards[0].get("retrieval_scores") or {}).get("action_relevance_filtered_count_total") or 0)


def paper_cards_for_llm(cards: list[dict[str, Any]], *, limit: int, abstract_char_limit: int) -> list[dict[str, Any]]:
    compact = []
    for card in cards[:limit]:
        compact.append(
            {
                "paper_id": card.get("paper_id"),
                "title": card.get("title"),
                "abstract": truncate_text(card.get("abstract"), abstract_char_limit),
                "year": card.get("year"),
                "citation_count": card.get("citation_count"),
                "source_actions": card.get("source_actions", []),
            }
        )
    return compact


def build_probe_action(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    normalized_topic = normalize_whitespace(profile.get("normalized_topic")) or args.topic
    keywords = [normalize_whitespace(item) for item in profile.get("likely_keywords", []) if normalize_whitespace(item)]
    foundations = [
        normalize_whitespace(item)
        for item in profile.get("upstream_foundations", [])
        if normalize_whitespace(item)
    ]
    query_parts = [normalized_topic] + keywords[:5] + foundations[:2]
    return {
        "action_id": "probe",
        "round": "probe",
        "query": " ".join(dict.fromkeys(query_parts)),
        "intent": "broad_probe",
        "query_style": "broad",
        "evidence_role": "scope_probe",
        "target_method_cluster": None,
        "target_role": ["scope_probe"],
        "kg_top_k": args.probe_kg_top_k,
        "s2_top_k": args.probe_s2_top_k,
        "rationale": "Broad probe to estimate terminology, paper-year distribution, and initial evidence pool.",
    }


SYSTEM_PROMPT = (
    "You are a rigorous academic literature-search planner. Return strict JSON only. "
    "Use only metadata and abstracts supplied in the prompt; do not invent paper-specific evidence."
)


def llm_topic_profile(client: DmxJsonClient, topic: str) -> dict[str, Any]:
    prompt = f"""Create a topic profile for literature-review search.

Return JSON with exactly these keys:
normalized_topic, scope, possible_method_families, upstream_foundations,
application_contexts, likely_keywords, ambiguous_terms, exclusion_rules.

Requirements:
- possible_method_families, upstream_foundations, application_contexts, likely_keywords,
  ambiguous_terms, and exclusion_rules must be arrays of strings.
- Keep the profile useful for academic paper retrieval.

Topic: {topic}
"""
    try:
        payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label="topic_profile")
        return {
            "normalized_topic": normalize_whitespace(payload.get("normalized_topic")) or topic,
            "scope": normalize_whitespace(payload.get("scope")),
            "possible_method_families": list_of_strings(payload.get("possible_method_families")),
            "upstream_foundations": list_of_strings(payload.get("upstream_foundations")),
            "application_contexts": list_of_strings(payload.get("application_contexts")),
            "likely_keywords": list_of_strings(payload.get("likely_keywords")),
            "ambiguous_terms": list_of_strings(payload.get("ambiguous_terms")),
            "exclusion_rules": list_of_strings(payload.get("exclusion_rules")),
        }
    except Exception as exc:
        return fallback_topic_profile(topic, error=str(exc))


def list_of_strings(value: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
    return result[:limit]


def fallback_topic_profile(topic: str, *, error: str | None = None) -> dict[str, Any]:
    profile = {
        "normalized_topic": topic,
        "scope": f"Academic literature related to {topic}.",
        "possible_method_families": [topic],
        "upstream_foundations": [],
        "application_contexts": [],
        "likely_keywords": [topic],
        "ambiguous_terms": [],
        "exclusion_rules": ["Exclude papers that only mention the topic tangentially."],
    }
    if error:
        profile["fallback_error"] = error
    return profile


def llm_time_windows(
    client: DmxJsonClient,
    *,
    profile: dict[str, Any],
    probe_cards: list[dict[str, Any]],
    current_year: int,
    min_year: int | None,
    max_year: int | None,
    llm_paper_limit: int,
) -> dict[str, Any]:
    prompt = f"""Generate 3-5 time windows for literature-review search.

Return JSON:
{{
  "time_window_source": "llm",
  "time_windows": [
    {{"label": "...", "start_year": 2000, "end_year": 2010, "rationale": "...", "search_focus": "..."}}
  ],
  "warnings": []
}}

Current year: {current_year}
User min year constraint: {min_year}
User max year constraint: {max_year}
Topic profile:
{json.dumps(profile, ensure_ascii=False)}

Probe papers:
{json.dumps(paper_cards_for_llm(probe_cards, limit=llm_paper_limit, abstract_char_limit=700), ensure_ascii=False)}
"""
    try:
        payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label="time_windows")
        windows = sanitize_time_windows(
            payload.get("time_windows"),
            current_year=current_year,
            min_year=min_year,
            max_year=max_year,
        )
        if len(windows) < 3:
            raise ValueError("LLM returned fewer than 3 valid windows.")
        return {
            "time_window_source": "llm",
            "time_windows": windows[:5],
            "warnings": list_of_strings(payload.get("warnings"), limit=10),
        }
    except Exception as exc:
        fallback = fallback_time_windows(
            probe_cards,
            current_year=current_year,
            min_year=min_year,
            max_year=max_year,
        )
        fallback["warnings"].append(f"LLM time slicing fallback: {exc}")
        return fallback


def sanitize_time_windows(
    value: Any,
    *,
    current_year: int,
    min_year: int | None,
    max_year: int | None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    windows: list[dict[str, Any]] = []
    lower_bound = min_year or 1900
    upper_bound = max_year or current_year
    for item in value:
        if not isinstance(item, dict):
            continue
        start = parse_year(item.get("start_year"))
        end = parse_year(item.get("end_year"))
        if start is None or end is None:
            continue
        start = max(start, lower_bound)
        end = min(end, upper_bound)
        if end < start:
            continue
        windows.append(
            {
                "label": slugify(normalize_whitespace(item.get("label")) or f"{start}_{end}", limit=40),
                "start_year": start,
                "end_year": end,
                "rationale": normalize_whitespace(item.get("rationale")),
                "search_focus": normalize_whitespace(item.get("search_focus")),
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for window in windows:
        key = (window["start_year"], window["end_year"], window["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(window)
    return deduped


def fallback_time_windows(
    cards: list[dict[str, Any]],
    *,
    current_year: int,
    min_year: int | None,
    max_year: int | None,
) -> dict[str, Any]:
    years = sorted({card["year"] for card in cards if isinstance(card.get("year"), int)})
    if years:
        start = min_year or min(years)
        end = max_year or max(max(years), current_year)
    else:
        start = min_year or max(2000, current_year - 20)
        end = max_year or current_year
    if end < start:
        start, end = end, start
    span = max(1, end - start + 1)
    if span <= 6:
        windows = [(start, end, "active_period")]
    else:
        first_end = start + span // 3 - 1
        second_end = start + (2 * span) // 3 - 1
        windows = [
            (start, first_end, "foundational"),
            (first_end + 1, second_end, "development"),
            (second_end + 1, end, "recent_frontier"),
        ]
    return {
        "time_window_source": "fallback",
        "time_windows": [
            {
                "label": label,
                "start_year": left,
                "end_year": right,
                "rationale": "Fallback split from available probe paper years.",
                "search_focus": label.replace("_", " "),
            }
            for left, right, label in windows
        ],
        "warnings": [],
    }


def llm_round1_actions(
    client: DmxJsonClient,
    *,
    profile: dict[str, Any],
    time_windows: list[dict[str, Any]],
    probe_cards: list[dict[str, Any]],
    action_limit: int,
    kg_top_k: int,
    s2_top_k: int,
    llm_paper_limit: int,
    clean_queries: bool,
) -> list[dict[str, Any]]:
    prompt = f"""Plan up to {action_limit} Round 1 literature-search actions.

Return JSON: {{"actions": [SEARCH_ACTION, ...]}}
Required SEARCH_ACTION fields: action_id, round, query, intent, query_style, evidence_role, kg_top_k, s2_top_k, rationale.
Optional fields: time_window, target_method_cluster, target_role.

Allowed intents: core_topic_recall, method_facet_recall, foundational_recall, recent_frontier_recall,
benchmark_or_evaluation_recall, survey_or_taxonomy_recall.
Allowed query_style values: broad, facet, named_work, exact_title, frontier.
Allowed evidence_role values: foundational, original_method, representative, benchmark, survey, recent_frontier.

Coverage quotas, subject to the action limit:
- Include one core_topic_recall action.
- Include one foundational_recall action aimed at original or highly influential work, without inventing paper titles.
- Include 2-4 method_facet_recall actions from topic_profile.possible_method_families.
- Include one benchmark_or_evaluation_recall action when benchmarks, datasets, or evaluation are relevant.
- Include one recent_frontier_recall action when the topic has recent activity.
- Include one survey_or_taxonomy_recall action when the field is broad or rapidly moving.

Do not create a full method-by-time Cartesian product. Select only high-value actions.
Keep each action focused: one primary intent and one primary method/evidence role per query.
Do not merge unrelated method families into one query.
Prefer method/task terms over broad historical or prestige terms.
Query construction rules:
- Every query must keep the normalized topic or a close topic anchor from the topic profile.
- Do not add standalone years as query terms; use time_window when time matters.
- Use at most one OR expression, and only between near-synonyms or the same method family.
- Do not use broad parent fields, historical AI subfields, or prestige words as substitutes for the topic.
- For foundational_recall, search for original or highly influential work inside this topic, not older upstream fields unless the topic profile explicitly identifies them as in-scope.
Use kg_top_k={kg_top_k} and s2_top_k={s2_top_k} unless there is a strong reason.
Set round to "round_1".

Topic profile:
{json.dumps(profile, ensure_ascii=False)}

Time windows:
{json.dumps(time_windows, ensure_ascii=False)}

Probe papers:
{json.dumps(paper_cards_for_llm(probe_cards, limit=llm_paper_limit, abstract_char_limit=600), ensure_ascii=False)}
"""
    try:
        payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label="round1_actions")
        actions = sanitize_actions(
            payload.get("actions"),
            round_name="round_1",
            limit=action_limit,
            kg_top_k=kg_top_k,
            s2_top_k=s2_top_k,
            topic=normalize_whitespace(profile.get("normalized_topic")),
            clean_queries=clean_queries,
            allowed_intents={
                "core_topic_recall",
                "method_facet_recall",
                "foundational_recall",
                "recent_frontier_recall",
                "benchmark_or_evaluation_recall",
                "survey_or_taxonomy_recall",
            },
        )
        if actions:
            return enforce_round1_action_coverage(
                actions,
                profile=profile,
                time_windows=time_windows,
                action_limit=action_limit,
                kg_top_k=kg_top_k,
                s2_top_k=s2_top_k,
            )
        raise ValueError("No valid Round 1 actions.")
    except Exception:
        return fallback_round1_actions(
            profile=profile,
            time_windows=time_windows,
            action_limit=action_limit,
            kg_top_k=kg_top_k,
            s2_top_k=s2_top_k,
        )


def sanitize_actions(
    value: Any,
    *,
    round_name: str,
    limit: int,
    kg_top_k: int,
    s2_top_k: int,
    allowed_intents: set[str],
    topic: str = "",
    clean_queries: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    actions: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        intent = normalize_whitespace(item.get("intent"))
        raw_query = normalize_whitespace(item.get("query"))
        query = sanitize_query_text(raw_query, topic=topic, intent=intent) if clean_queries else raw_query
        if not query or intent not in allowed_intents:
            continue
        query_key = query.casefold()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        raw_id = slugify(normalize_whitespace(item.get("action_id")) or f"{round_name}_{index}", limit=60)
        action_id = raw_id
        suffix = 2
        while action_id in seen_ids:
            action_id = f"{raw_id}_{suffix}"
            suffix += 1
        seen_ids.add(action_id)
        action = {
            "action_id": action_id,
            "round": round_name,
            "query": query,
            "intent": intent,
            "query_style": normalize_query_style(item.get("query_style"), intent=intent),
            "evidence_role": normalize_evidence_role(item.get("evidence_role"), intent=intent),
            "target_method_cluster": item.get("target_method_cluster"),
            "target_role": list_of_strings(item.get("target_role"), limit=8),
            "kg_top_k": parse_int(item.get("kg_top_k")) or kg_top_k,
            "s2_top_k": parse_int(item.get("s2_top_k")) or s2_top_k,
            "rationale": normalize_whitespace(item.get("rationale")) or "LLM-planned search action.",
        }
        time_window = item.get("time_window")
        if isinstance(time_window, dict) and time_window.get("start_year") and time_window.get("end_year"):
            action["time_window"] = {
                "label": normalize_whitespace(time_window.get("label")) or "window",
                "start_year": parse_year(time_window.get("start_year")),
                "end_year": parse_year(time_window.get("end_year")),
            }
        actions.append(action)
        if len(actions) >= limit:
            break
    return actions


def enforce_round1_action_coverage(
    actions: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    time_windows: list[dict[str, Any]],
    action_limit: int,
    kg_top_k: int,
    s2_top_k: int,
) -> list[dict[str, Any]]:
    if len(actions) >= action_limit:
        return actions[:action_limit]
    present = {action["intent"] for action in actions}
    supplements: list[dict[str, Any]] = []
    fallback_actions = fallback_round1_actions(
        profile=profile,
        time_windows=time_windows,
        action_limit=action_limit,
        kg_top_k=kg_top_k,
        s2_top_k=s2_top_k,
    )
    priority = {
        "foundational_recall": 0,
        "benchmark_or_evaluation_recall": 1,
        "recent_frontier_recall": 2,
        "survey_or_taxonomy_recall": 3,
        "core_topic_recall": 4,
        "method_facet_recall": 5,
    }
    fallback_actions.sort(key=lambda action: priority.get(action["intent"], 99))
    for fallback in fallback_actions:
        if fallback["intent"] in present:
            continue
        if normalize_whitespace(fallback["query"]).casefold() in {
            normalize_whitespace(action["query"]).casefold() for action in actions
        }:
            continue
        supplements.append(fallback)
        present.add(fallback["intent"])
        if len(actions) + len(supplements) >= action_limit:
            break
    return (actions + supplements)[:action_limit]


def fallback_round1_actions(
    *,
    profile: dict[str, Any],
    time_windows: list[dict[str, Any]],
    action_limit: int,
    kg_top_k: int,
    s2_top_k: int,
) -> list[dict[str, Any]]:
    topic = normalize_whitespace(profile.get("normalized_topic"))
    actions = [
        {
            "action_id": "round1_core",
            "round": "round_1",
            "query": topic,
            "intent": "core_topic_recall",
            "query_style": "broad",
            "evidence_role": "representative",
            "target_method_cluster": None,
            "target_role": ["representative"],
            "kg_top_k": kg_top_k,
            "s2_top_k": s2_top_k,
            "rationale": "Fallback core-topic recall.",
        }
    ]
    for family in list_of_strings(profile.get("possible_method_families"), limit=4):
        actions.append(
            {
                "action_id": f"round1_method_{slugify(family, limit=24)}",
                "round": "round_1",
                "query": f"{topic} {family}",
                "intent": "method_facet_recall",
                "query_style": "facet",
                "evidence_role": "original_method",
                "target_method_cluster": family,
                "target_role": ["method_facet"],
                "kg_top_k": kg_top_k,
                "s2_top_k": s2_top_k,
                "rationale": "Fallback method-facet recall.",
            }
        )
        if len(actions) >= action_limit:
            return actions
    if time_windows:
        foundational = time_windows[0]
        recent = time_windows[-1]
        for action_id, intent, window, role in (
            ("round1_foundational", "foundational_recall", foundational, "foundational"),
            ("round1_recent_frontier", "recent_frontier_recall", recent, "recent_frontier"),
        ):
            actions.append(
                {
                    "action_id": action_id,
                    "round": "round_1",
                    "query": topic,
                    "time_window": window,
                    "intent": intent,
                    "query_style": "named_work" if role == "foundational" else "frontier",
                    "evidence_role": "foundational" if role == "foundational" else "recent_frontier",
                    "target_method_cluster": None,
                    "target_role": [role],
                    "kg_top_k": kg_top_k,
                    "s2_top_k": s2_top_k,
                    "rationale": f"Fallback {role} recall.",
                }
            )
            if len(actions) >= action_limit:
                break
    for action_id, intent, suffix, role in (
        ("round1_benchmark_evaluation", "benchmark_or_evaluation_recall", "benchmark dataset evaluation", "benchmark"),
        ("round1_survey_taxonomy", "survey_or_taxonomy_recall", "survey taxonomy overview", "survey"),
    ):
        if len(actions) >= action_limit:
            break
        if intent in {action["intent"] for action in actions}:
            continue
        actions.append(
            {
                "action_id": action_id,
                "round": "round_1",
                "query": f"{topic} {suffix}",
                "intent": intent,
                "query_style": "facet",
                "evidence_role": role,
                "target_method_cluster": None,
                "target_role": [role],
                "kg_top_k": kg_top_k,
                "s2_top_k": s2_top_k,
                "rationale": f"Fallback {role} recall.",
            }
        )
    return actions[:action_limit]


def llm_clusters(
    client: DmxJsonClient,
    *,
    profile: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    label: str,
    llm_paper_limit: int,
) -> dict[str, Any]:
    prompt = f"""Cluster papers by topic-relevant methodological family or research thread.

Rules:
- Target 3-8 clusters.
- Do not cluster primarily by year, venue, or application domain.
- Each paper should have at most one primary cluster.
- The goal is not to preserve every retrievable technical direction. The goal is to identify the small set of method families that are genuinely central to this review topic.
- Before forming clusters, infer the scientific scope of the topic from the topic profile and paper set.
- Keep only clusters that are directly relevant to the topic's central scientific scope and could plausibly support a meaningful section in a domain review.
- If a set of papers forms a coherent technical line but is only loosely related to the topic, do not force it into a retained cluster. Mark those papers as outliers instead.
- Prefer a smaller number of topic-centered clusters over a larger number of noisy or weakly related clusters.
- Use `topic_relevance="core"` for clusters that should clearly remain in the main review narrative.
- Use `topic_relevance="peripheral"` for clusters that are only weakly related, specialized extensions, or likely better treated as excluded background rather than retained main families.
- If a cluster is peripheral, set `keep_in_review=false` unless there is a strong topic-specific reason to preserve it.
- Use `topic_relevance="off_topic"` for clusters that should be excluded from the review.
- Use outliers for off-topic papers, peripheral technical lines, or papers that do not belong in any retained cluster.
- Use uncertain_assignments for ambiguous method-family assignments.

Return JSON:
{{
  "clusters": [
    {{
      "cluster_id": "C1",
      "name": "...",
      "definition": "...",
      "topic_relevance": "core|peripheral|off_topic",
      "keep_in_review": true,
      "relevance_rationale": "...",
      "distinguishing_features": ["..."],
      "paper_ids": ["P001"],
      "representative_paper_ids": ["P001"],
      "missing_signals": ["..."]
    }}
  ],
  "outliers": [],
  "uncertain_assignments": []
}}

Topic profile:
{json.dumps(profile, ensure_ascii=False)}

Paper cards:
{json.dumps(paper_cards_for_llm(paper_cards, limit=llm_paper_limit, abstract_char_limit=750), ensure_ascii=False)}
"""
    try:
        payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label=label)
        return sanitize_clusters(payload, paper_cards)
    except Exception as exc:
        return fallback_clusters(paper_cards, error=str(exc))


def sanitize_clusters(payload: dict[str, Any], paper_cards: list[dict[str, Any]]) -> dict[str, Any]:
    valid_ids = {card["paper_id"] for card in paper_cards}
    clusters = []
    dropped_cluster_ids: list[str] = []
    dropped_paper_ids: list[str] = []
    for index, item in enumerate(payload.get("clusters", []), start=1):
        if not isinstance(item, dict):
            continue
        paper_ids = [pid for pid in list_of_strings(item.get("paper_ids"), limit=500) if pid in valid_ids]
        if not paper_ids:
            continue
        cluster_id = normalize_whitespace(item.get("cluster_id")) or f"C{index}"
        topic_relevance = normalize_whitespace(item.get("topic_relevance")).casefold() or "core"
        keep_in_review = item.get("keep_in_review")
        if keep_in_review is None:
            keep_in_review = topic_relevance == "core"
        keep_in_review = bool(keep_in_review)
        if not keep_in_review:
            dropped_cluster_ids.append(cluster_id)
            dropped_paper_ids.extend(paper_ids)
            continue
        clusters.append(
            {
                "cluster_id": cluster_id,
                "name": normalize_whitespace(item.get("name")) or cluster_id,
                "definition": normalize_whitespace(item.get("definition")),
                "topic_relevance": topic_relevance,
                "keep_in_review": keep_in_review,
                "relevance_rationale": normalize_whitespace(item.get("relevance_rationale")),
                "distinguishing_features": list_of_strings(item.get("distinguishing_features"), limit=12),
                "paper_ids": paper_ids,
                "representative_paper_ids": [
                    pid for pid in list_of_strings(item.get("representative_paper_ids"), limit=10) if pid in valid_ids
                ],
                "missing_signals": list_of_strings(item.get("missing_signals"), limit=10),
            }
        )
    assigned = {pid for cluster in clusters for pid in cluster["paper_ids"]}
    model_outliers = [pid for pid in list_of_strings(payload.get("outliers"), limit=500) if pid in valid_ids]
    residual_unassigned = sorted(valid_ids - assigned)
    hard_excluded_paper_ids = sorted(set(model_outliers) | set(dropped_paper_ids))
    outliers = sorted(set(model_outliers) | set(dropped_paper_ids) | set(residual_unassigned))
    uncertain = []
    for item in payload.get("uncertain_assignments", []):
        if not isinstance(item, dict):
            continue
        pid = normalize_whitespace(item.get("paper_id"))
        if pid in valid_ids:
            uncertain.append(
                {
                    "paper_id": pid,
                    "candidate_clusters": list_of_strings(item.get("candidate_clusters"), limit=8),
                    "reason": normalize_whitespace(item.get("reason")),
                }
            )
    if not clusters:
        return fallback_clusters(paper_cards)
    return {
        "clusters": clusters[:8],
        "outliers": outliers,
        "model_outliers": sorted(set(model_outliers)),
        "residual_unassigned_paper_ids": residual_unassigned,
        "hard_excluded_paper_ids": hard_excluded_paper_ids,
        "uncertain_assignments": uncertain,
        "dropped_cluster_ids": dropped_cluster_ids,
    }


def fallback_clusters(paper_cards: list[dict[str, Any]], *, error: str | None = None) -> dict[str, Any]:
    ids = [card["paper_id"] for card in paper_cards]
    representatives = ids[:3]
    payload = {
        "clusters": [
            {
                "cluster_id": "C1",
                "name": "Retrieved literature",
                "definition": "Fallback cluster containing retrieved papers pending method-family refinement.",
                "topic_relevance": "core",
                "keep_in_review": True,
                "relevance_rationale": "Fallback cluster retained because no valid topic-centered clustering result was available.",
                "distinguishing_features": [],
                "paper_ids": ids,
                "representative_paper_ids": representatives,
                "missing_signals": ["LLM clustering was unavailable or invalid."],
            }
        ]
        if ids
        else [],
        "outliers": [],
        "uncertain_assignments": [],
    }
    if error:
        payload["fallback_error"] = error
    return payload


def llm_coverage_diagnosis(
    client: DmxJsonClient,
    *,
    profile: dict[str, Any],
    time_windows: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    clusters: dict[str, Any],
    actions: list[dict[str, Any]],
    label: str,
    final: bool,
    llm_paper_limit: int,
) -> dict[str, Any]:
    prompt = f"""Diagnose literature-search coverage.

Check missing method families, under-populated clusters, weak foundational coverage,
weak recent frontier coverage, off-topic papers, weak representative evidence, and
missing benchmark/dataset/system/survey papers when relevant.
Use topic-agnostic criteria. Do not require any specific paper or method name unless
it is evident from the supplied topic profile, clusters, actions, or paper metadata.
If an important gap remains, include it in structured_gaps. Do not set
stop_recommendation=true when any high-severity structured gap has blocks_stop=true.
Write query_seed as a narrow, topic-grounded search seed:
- It must include the normalized topic or a close topic anchor.
- It should name one missing method, benchmark, dataset, system, or paper family.
- It must not be just a broad parent field or historical upstream area.
- Do not include standalone years; describe recency in target/severity instead.
- Use OR only for near-synonyms within the same gap.

Return JSON:
{{
  "coverage_status": "sufficient|needs_refinement|limited",
  "structured_gaps": [
    {{
      "gap_id": "G1",
      "axis": "method|time|representative|benchmark|canonical_anchor|off_topic",
      "target": "short description of the missing or weak evidence",
      "severity": "high|medium|low",
      "query_seed": "short topic-grounded search seed for a follow-up action",
      "suggested_intent": "missing_method_family_query|cluster_deepening_query|foundational_query|recent_frontier_query|representative_query|benchmark_or_evaluation_query|survey_or_taxonomy_query|canonical_anchor_query|disambiguation_query",
      "blocks_stop": true
    }}
  ],
  "method_gaps": [],
  "time_gaps": [],
  "representative_gaps": [],
  "off_topic_notes": [],
  "suggested_refine_actions": [],
  "stop_recommendation": false
}}

For final={final}, record remaining limitations only and do not imply another search round is required.

Topic profile:
{json.dumps(profile, ensure_ascii=False)}

Time windows:
{json.dumps(time_windows, ensure_ascii=False)}

Executed actions:
{json.dumps(actions, ensure_ascii=False)}

Clusters:
{json.dumps(clusters, ensure_ascii=False)}

Paper cards:
{json.dumps(paper_cards_for_llm(paper_cards, limit=llm_paper_limit, abstract_char_limit=600), ensure_ascii=False)}
"""
    try:
        payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label=label)
        structured_gaps = normalize_structured_gaps(payload.get("structured_gaps"))
        return {
            "coverage_status": normalize_whitespace(payload.get("coverage_status")) or "limited",
            "structured_gaps": structured_gaps,
            "method_gaps": normalize_gap_list(payload.get("method_gaps")),
            "time_gaps": normalize_gap_list(payload.get("time_gaps")),
            "representative_gaps": normalize_gap_list(payload.get("representative_gaps")),
            "off_topic_notes": normalize_gap_list(payload.get("off_topic_notes")),
            "suggested_refine_actions": normalize_gap_list(payload.get("suggested_refine_actions")),
            "stop_recommendation": (bool(payload.get("stop_recommendation")) and not has_blocking_gaps({"structured_gaps": structured_gaps})) if not final else True,
        }
    except Exception as exc:
        return fallback_coverage(paper_cards, clusters, final=final, error=str(exc))


def normalize_gap_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:20]


def normalize_structured_gaps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    gaps: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        target = normalize_whitespace(item.get("target") or item.get("description"))
        if not target:
            continue
        severity = normalize_whitespace(item.get("severity")).casefold()
        if severity not in {"high", "medium", "low"}:
            severity = "medium"
        gaps.append(
            {
                "gap_id": normalize_whitespace(item.get("gap_id")) or f"G{index}",
                "axis": normalize_whitespace(item.get("axis")) or "representative",
                "target": target,
                "severity": severity,
                "query_seed": normalize_whitespace(item.get("query_seed")) or target,
                "suggested_intent": normalize_whitespace(item.get("suggested_intent")) or "representative_query",
                "blocks_stop": bool(item.get("blocks_stop")) if "blocks_stop" in item else severity == "high",
            }
        )
    return gaps[:20]


def has_blocking_gaps(diagnosis: dict[str, Any]) -> bool:
    for gap in diagnosis.get("structured_gaps", []):
        if isinstance(gap, dict) and gap.get("blocks_stop") and gap.get("severity") == "high":
            return True
    return False


def fallback_coverage(
    paper_cards: list[dict[str, Any]],
    clusters: dict[str, Any],
    *,
    final: bool,
    error: str | None = None,
) -> dict[str, Any]:
    cluster_count = len(clusters.get("clusters", [])) if isinstance(clusters, dict) else 0
    status = "sufficient" if paper_cards and cluster_count >= 2 else "needs_refinement"
    payload = {
        "coverage_status": "limited" if final and status == "needs_refinement" else status,
        "structured_gaps": [],
        "method_gaps": [],
        "time_gaps": [],
        "representative_gaps": [] if paper_cards else ["No paper cards were retrieved."],
        "off_topic_notes": [],
        "suggested_refine_actions": [],
        "stop_recommendation": final or status == "sufficient",
    }
    if error:
        payload["fallback_error"] = error
    return payload


def llm_round2_actions(
    client: DmxJsonClient,
    *,
    profile: dict[str, Any],
    diagnosis: dict[str, Any],
    existing_actions: list[dict[str, Any]],
    action_limit: int,
    kg_top_k: int,
    s2_top_k: int,
    clean_queries: bool,
) -> list[dict[str, Any]]:
    if (
        diagnosis.get("stop_recommendation")
        and diagnosis.get("coverage_status") == "sufficient"
        and not has_blocking_gaps(diagnosis)
    ):
        return []
    prompt = f"""Plan up to {action_limit} Round 2 refinement search actions from the coverage diagnosis.

Allowed intents:
missing_method_family_query, cluster_deepening_query, foundational_query,
recent_frontier_query, transition_query, representative_query, disambiguation_query,
benchmark_or_evaluation_query, survey_or_taxonomy_query, canonical_anchor_query.

Requirements:
- Set round to "round_2".
- No duplicate queries.
- Do not repeat a Round 1 query without a narrower intent.
- Prefer structured_gaps with severity=high and blocks_stop=true.
- Use each structured gap's query_seed and suggested_intent when available.
- Prefer concrete method, benchmark, dataset, or system terms over broad historical or prestige terms.
- Round 2 is for precise gap repair, not broad exploration. Each action should target one diagnosed gap.
- Every query must keep the normalized topic or a close topic anchor from the topic profile.
- Avoid broad parent fields, historical upstream fields, and textbook-area terms unless the topic itself is that field.
- Do not combine a modern topic with an unrelated upstream field using OR.
- Use at most one OR expression, and only between near-synonyms or names in the same method family.
- Do not add standalone years as query terms; use recent_frontier_query or time_window semantics instead.
- Prefer missing_method_family_query, canonical_anchor_query, representative_query, benchmark_or_evaluation_query, or recent_frontier_query before cluster_deepening_query/foundational_query/survey_or_taxonomy_query when the gap can be repaired narrowly.
- Use kg_top_k={kg_top_k}, s2_top_k={s2_top_k}.

Return JSON: {{"actions": [SEARCH_ACTION, ...]}}
Each action must include query_style and evidence_role.

Topic profile:
{json.dumps(profile, ensure_ascii=False)}

Coverage diagnosis:
{json.dumps(diagnosis, ensure_ascii=False)}

Existing actions:
{json.dumps(existing_actions, ensure_ascii=False)}
"""
    allowed = {
        "missing_method_family_query",
        "cluster_deepening_query",
        "foundational_query",
        "recent_frontier_query",
        "transition_query",
        "representative_query",
        "disambiguation_query",
        "benchmark_or_evaluation_query",
        "survey_or_taxonomy_query",
        "canonical_anchor_query",
    }
    try:
        payload = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label="round2_actions")
        actions = sanitize_actions(
            payload.get("actions"),
            round_name="round_2",
            limit=action_limit,
            kg_top_k=kg_top_k,
            s2_top_k=s2_top_k,
            topic=normalize_whitespace(profile.get("normalized_topic")),
            clean_queries=clean_queries,
            allowed_intents=allowed,
        )
        existing_queries = {normalize_whitespace(action.get("query")).casefold() for action in existing_actions}
        actions = [action for action in actions if action["query"].casefold() not in existing_queries]
        if actions:
            return actions[:action_limit]
        raise ValueError("No valid Round 2 actions.")
    except Exception:
        return fallback_round2_actions(
            profile=profile,
            diagnosis=diagnosis,
            existing_actions=existing_actions,
            action_limit=action_limit,
            kg_top_k=kg_top_k,
            s2_top_k=s2_top_k,
            clean_queries=clean_queries,
        )


def fallback_round2_actions(
    *,
    profile: dict[str, Any],
    diagnosis: dict[str, Any],
    existing_actions: list[dict[str, Any]],
    action_limit: int,
    kg_top_k: int,
    s2_top_k: int,
    clean_queries: bool,
) -> list[dict[str, Any]]:
    topic = normalize_whitespace(profile.get("normalized_topic"))
    existing_queries = {normalize_whitespace(action.get("query")).casefold() for action in existing_actions}
    structured_gaps = [
        gap
        for gap in diagnosis.get("structured_gaps", [])
        if isinstance(gap, dict) and normalize_whitespace(gap.get("query_seed"))
    ]
    structured_gaps.sort(key=lambda gap: (0 if gap.get("severity") == "high" else 1, not bool(gap.get("blocks_stop"))))
    actions: list[dict[str, Any]] = []
    for index, gap in enumerate(structured_gaps, start=1):
        suggested_intent = normalize_whitespace(gap.get("suggested_intent")) or "representative_query"
        raw_seed_text = normalize_whitespace(gap.get("query_seed"))
        seed_text = sanitize_query_text(raw_seed_text, topic=topic, intent=suggested_intent) if clean_queries else raw_seed_text
        if not seed_text:
            continue
        query = seed_text if topic.casefold() in seed_text.casefold() else f"{topic} {seed_text}"
        if query.casefold() in existing_queries:
            continue
        actions.append(
            {
                "action_id": f"round2_refine_{index}",
                "round": "round_2",
                "query": query,
                "intent": suggested_intent if suggested_intent in ROUND2_INTENTS else "representative_query",
                "query_style": "named_work" if gap.get("axis") == "canonical_anchor" else "facet",
                "evidence_role": "foundational" if gap.get("axis") in {"canonical_anchor", "time"} else "gap_refinement",
                "target_method_cluster": None,
                "target_role": ["gap_refinement"],
                "kg_top_k": kg_top_k,
                "s2_top_k": s2_top_k,
                "rationale": f"Fallback refinement query from structured gap {gap.get('gap_id')}: {gap.get('target')}",
            }
        )
        if len(actions) >= action_limit:
            break
    seeds = diagnosis.get("method_gaps") or diagnosis.get("representative_gaps") or profile.get("possible_method_families")
    for index, seed in enumerate(seeds if isinstance(seeds, list) else [], start=len(actions) + 1):
        if len(actions) >= action_limit:
            break
        seed_text = normalize_whitespace(seed if isinstance(seed, str) else json.dumps(seed, ensure_ascii=False))
        if not seed_text:
            continue
        query = f"{topic} {seed_text} representative methods"
        if query.casefold() in existing_queries:
            continue
        actions.append(
            {
                "action_id": f"round2_refine_{index}",
                "round": "round_2",
                "query": query,
                "intent": "representative_query",
                "query_style": "facet",
                "evidence_role": "gap_refinement",
                "target_method_cluster": None,
                "target_role": ["gap_refinement"],
                "kg_top_k": kg_top_k,
                "s2_top_k": s2_top_k,
                "rationale": "Fallback refinement query from diagnosed coverage gaps.",
            }
        )
    if not actions and topic.casefold() not in existing_queries:
        actions.append(
            {
                "action_id": "round2_representative",
                "round": "round_2",
                "query": f"{topic} survey benchmark dataset representative methods",
                "intent": "representative_query",
                "query_style": "facet",
                "evidence_role": "representative",
                "target_method_cluster": None,
                "target_role": ["representative"],
                "kg_top_k": kg_top_k,
                "s2_top_k": s2_top_k,
                "rationale": "Fallback representative-evidence refinement.",
            }
        )
    return actions


def run_action_searches(
    backend: MergeSearchBackend,
    actions: list[dict[str, Any]],
    *,
    output_dir: Path,
    progress: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    action_result_dir = ensure_dir(output_dir / "action_results")
    total = len(actions)
    for index, action in enumerate(actions, start=1):
        if "kg_policy_args" not in action:
            action["kg_policy_args"] = build_kg_policy_args(action) if backend.kg_policy_enabled else []
        policy_state = f"policy_args={len(action.get('kg_policy_args') or [])}" if backend.kg_policy_enabled else "policy=off"
        label = (
            f"search action {index}/{total} {action['action_id']} "
            f"intent={action.get('intent')} kg_top_k={action.get('kg_top_k')} "
            f"s2_top_k={action.get('s2_top_k')} {policy_state}"
        )
        with progress_stage(progress, label):
            payload = backend.search(action)
        results[action["action_id"]] = payload
        write_json(action_result_dir / f"{action['action_id']}.json", payload)
        unique_count = len((payload.get("filter") or {}).get("unique_papers") or [])
        kg_payload = (payload.get("sources") or {}).get("kg") or {}
        s2_payload = (payload.get("sources") or {}).get("s2") or {}
        progress_log(
            progress,
            (
                f"ACTION_RESULT {action['action_id']} status={payload.get('status')} "
                f"unique={unique_count} kg_status={kg_payload.get('status')} "
                f"kg_papers={kg_payload.get('paper_count')} s2_status={s2_payload.get('status')} "
                f"s2_papers={s2_payload.get('paper_count')}"
            ),
        )
    return results


def build_embedding_expansion_action(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action_id": "embedding_expand_round_2",
        "round": "round_2",
        "query": f"{args.topic} embedding gap expansion",
        "intent": "embedding_gap_expansion",
        "query_style": "embedding_region",
        "evidence_role": "gap_refinement",
        "target_method_cluster": None,
        "target_role": ["gap_refinement"],
        "kg_top_k": 0,
        "s2_top_k": 0,
        "rationale": "KG embedding-space expansion from Round 1 paper clusters.",
        "kg_policy_args": [],
    }


def expansion_result_to_search_payload(expansion_result: dict[str, Any]) -> dict[str, Any]:
    papers = []
    for rank, item in enumerate(expansion_result.get("recall_papers", []), start=1):
        if not isinstance(item, dict):
            continue
        title = normalize_whitespace(item.get("title"))
        if not title:
            continue
        kg_paper_id = normalize_whitespace(item.get("kg_paper_id"))
        identifiers = [f"openalex:{kg_paper_id}"] if kg_paper_id else []
        papers.append(
            {
                "title": title,
                "abstract": normalize_whitespace(item.get("abstract")),
                "year": item.get("year"),
                "citation_count": item.get("citation_count"),
                "source": "kg_embedding_gap",
                "source_set": ["kg", "kg_embedding_gap"],
                "source_rank": rank,
                "rank": rank,
                "paper_url": kg_paper_id,
                "identifiers": identifiers,
                "group_id": f"embedding-gap-{normalize_whitespace(item.get('region_id')) or 'unclustered'}-{rank:03d}",
                "paper": {
                    "id": kg_paper_id,
                    "title": title,
                    "abstract": normalize_whitespace(item.get("abstract")),
                    "publication_year": item.get("year"),
                    "cited_by_count": item.get("citation_count"),
                },
                "embedding_gap": {
                    "region_id": item.get("region_id"),
                    "gap_score": item.get("gap_score"),
                    "novelty": item.get("novelty"),
                    "density": item.get("density"),
                    "method_relevance": item.get("method_relevance"),
                    "nearest_known_cluster_id": item.get("nearest_known_cluster_id"),
                    "nearest_known_similarity": item.get("nearest_known_similarity"),
                },
            }
        )
    return {
        "status": "ok",
        "successful_source_count": 1,
        "failed_source_count": 0,
        "filter": {"unique_papers": papers, "unique_paper_count": len(papers)},
        "ranking": {"status": "embedding_gap", "papers": papers},
        "sources": {
            "kg_embedding_gap": {
                "status": "ok",
                "paper_count": len(papers),
                "region_count": len(expansion_result.get("gap_regions", [])),
                "diagnostics": (expansion_result.get("diagnostics") or {}).get("gap_search", {}),
            }
        },
    }


def run_embedding_expansion_round2(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    topic_profile: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    clusters: dict[str, Any],
    progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from explore_embedding_gaps import build_parser as build_expansion_parser
    from explore_embedding_gaps import render_markdown as render_expansion_markdown
    from explore_embedding_gaps import run as run_expansion

    action = build_embedding_expansion_action(args)
    expansion_input = {
        "topic": args.topic,
        "topic_profile": topic_profile,
        "paper_cards": paper_cards,
        "method_clusters": clusters.get("clusters", []),
    }
    input_path = output_dir / "embedding_expansion_round_2.input.json"
    write_json(input_path, expansion_input)
    expansion_args = build_expansion_parser().parse_args(
        [
            "--search-result",
            str(input_path),
            "--topic",
            normalize_whitespace(topic_profile.get("normalized_topic")) or args.topic,
            "--embedding-field",
            args.embedding_expansion_field,
            "--topic-title-top-k",
            str(args.embedding_expansion_topic_title_top_k),
            "--topic-abstract-top-k",
            str(args.embedding_expansion_topic_abstract_top_k),
            "--top-seed-pool",
            str(args.embedding_expansion_top_seed_pool),
            "--max-regions",
            str(args.embedding_expansion_max_regions),
            "--min-region-size",
            str(args.embedding_expansion_min_region_size),
            "--recall-paper-count",
            str(args.embedding_expansion_paper_count),
            "--cluster-algorithm",
            args.embedding_expansion_cluster_algorithm,
            "--hdbscan-min-cluster-size",
            str(args.embedding_expansion_hdbscan_min_cluster_size),
            "--hdbscan-min-samples",
            str(args.embedding_expansion_hdbscan_min_samples),
            "--mmr-lambda",
            str(args.embedding_expansion_mmr_lambda),
            "--max-fuzzy-fallbacks",
            str(args.embedding_expansion_max_fuzzy_fallbacks),
            "--quiet",
        ]
    )
    if args.kg_embedding_device:
        expansion_args.embedding_device = args.kg_embedding_device
    with progress_stage(progress, "Round 2 embedding gap expansion"):
        expansion_result = run_expansion(expansion_args)
    write_json(output_dir / "embedding_gap_candidates_round_2.json", expansion_result)
    (output_dir / "embedding_gap_candidates_round_2.md").write_text(
        render_expansion_markdown(expansion_result),
        encoding="utf-8",
    )
    payload = expansion_result_to_search_payload(expansion_result)
    write_json(output_dir / "action_results" / f"{action['action_id']}.json", payload)
    progress_log(
        progress,
        (
            f"EMBEDDING_EXPANSION regions={len(expansion_result.get('gap_regions', []))} "
            f"papers={len(expansion_result.get('recall_papers', []))}"
        ),
    )
    return action, payload


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    progress = not args.quiet
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    cache_dir = ensure_dir(
        Path(args.search_cache_dir).expanduser().resolve()
        if args.search_cache_dir
        else output_dir / "cache"
    )
    current_year = args.current_year
    pipeline_started_at = time.perf_counter()
    progress_log(progress, f"OUTPUT {output_dir}")
    progress_log(progress, f"CACHE {cache_dir}")
    progress_log(progress, f"CONFIG topic={args.topic!r} model={args.llm_model} mock={args.mock_backend}")
    client = DmxJsonClient(
        env_path=Path(args.env).expanduser().resolve(),
        api_url=args.llm_api_url,
        model=args.llm_model,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        use_env_proxy=args.use_env_proxy,
        mock=args.mock_backend,
    )
    backend = MergeSearchBackend(
        env_path=Path(args.env).expanduser().resolve(),
        cache_dir=cache_dir,
        use_env_proxy=args.use_env_proxy,
        s2_mode=args.s2_mode,
        kg_embedding_device=args.kg_embedding_device,
        kg_reranker_device=args.kg_reranker_device,
        kg_policy_enabled=args.enable_kg_policy,
        mock=args.mock_backend,
    )

    with progress_stage(progress, "LLM topic profile"):
        topic_profile = llm_topic_profile(client, args.topic)
    write_json(output_dir / "topic_profile.json", topic_profile)
    progress_log(progress, f"TOPIC normalized={topic_profile.get('normalized_topic')!r}")

    probe_action = build_probe_action(topic_profile, args)
    annotate_kg_policy_metadata([probe_action], enabled=args.enable_kg_policy)
    write_json(output_dir / "actions_probe.json", {"action": probe_action})
    probe_results = run_action_searches(backend, [probe_action], output_dir=output_dir, progress=progress)
    write_json(output_dir / "probe_search_result.json", probe_results[probe_action["action_id"]])

    all_action_results = dict(probe_results)
    all_actions_by_id = {probe_action["action_id"]: probe_action}
    probe_cards = merge_paper_cards(
        all_action_results,
        action_ids=[probe_action["action_id"]],
        actions_by_id=all_actions_by_id if args.enable_relevance_guard else None,
        abstract_char_limit=args.abstract_char_limit,
        current_year=current_year,
    )
    write_json(output_dir / "paper_cards_probe.json", probe_cards)
    progress_log(progress, f"PROBE_PAPERS merged={len(probe_cards)} filtered={relevance_filtered_count(probe_cards)}")

    with progress_stage(progress, "LLM time windows"):
        time_window_payload = llm_time_windows(
            client,
            profile=topic_profile,
            probe_cards=probe_cards,
            current_year=current_year,
            min_year=args.min_year,
            max_year=args.max_year,
            llm_paper_limit=args.llm_paper_limit,
        )
    write_json(output_dir / "time_windows.json", time_window_payload)
    time_windows = time_window_payload["time_windows"]
    progress_log(progress, f"TIME_WINDOWS count={len(time_windows)} source={time_window_payload.get('time_window_source')}")

    with progress_stage(progress, "LLM Round 1 action planning"):
        round1_actions = llm_round1_actions(
            client,
            profile=topic_profile,
            time_windows=time_windows,
            probe_cards=probe_cards,
            action_limit=args.round1_action_limit,
            kg_top_k=args.round_kg_top_k,
            s2_top_k=args.round_s2_top_k,
            llm_paper_limit=args.llm_paper_limit,
            clean_queries=args.enable_query_cleaning,
        )
    annotate_kg_policy_metadata(round1_actions, enabled=args.enable_kg_policy)
    write_json(output_dir / "actions_round_1.json", {"actions": round1_actions})
    progress_log(progress, f"ROUND1_ACTIONS count={len(round1_actions)}")
    all_actions_by_id.update({action["action_id"]: action for action in round1_actions})
    round1_results = run_action_searches(backend, round1_actions, output_dir=output_dir, progress=progress)
    all_action_results.update(round1_results)

    round1_action_ids = [probe_action["action_id"]] + [action["action_id"] for action in round1_actions]
    paper_cards_round1 = merge_paper_cards(
        all_action_results,
        action_ids=round1_action_ids,
        actions_by_id=all_actions_by_id if args.enable_relevance_guard else None,
        abstract_char_limit=args.abstract_char_limit,
        current_year=current_year,
    )
    write_json(output_dir / "paper_cards_round_1.json", paper_cards_round1)
    progress_log(progress, f"ROUND1_PAPERS merged={len(paper_cards_round1)} filtered={relevance_filtered_count(paper_cards_round1)}")

    with progress_stage(progress, "LLM Round 1 clustering"):
        clusters_round1 = llm_clusters(
            client,
            profile=topic_profile,
            paper_cards=paper_cards_round1,
            label="clusters_round_1",
            llm_paper_limit=args.llm_paper_limit,
        )
    write_json(output_dir / "clusters_round_1.json", clusters_round1)
    progress_log(progress, f"ROUND1_CLUSTERS count={len(clusters_round1.get('clusters', []))}")

    executed_round1_actions = [probe_action] + round1_actions
    with progress_stage(progress, "LLM Round 1 coverage diagnosis"):
        diagnosis_round1 = llm_coverage_diagnosis(
            client,
            profile=topic_profile,
            time_windows=time_windows,
            paper_cards=paper_cards_round1,
            clusters=clusters_round1,
            actions=executed_round1_actions,
            label="coverage_diagnosis_round_1",
            final=False,
            llm_paper_limit=args.llm_paper_limit,
        )
    write_json(output_dir / "coverage_diagnosis_round_1.json", diagnosis_round1)
    progress_log(
        progress,
        (
            f"ROUND1_DIAG status={diagnosis_round1.get('coverage_status')} "
            f"stop={diagnosis_round1.get('stop_recommendation')} "
            f"structured_gaps={len(diagnosis_round1.get('structured_gaps', []))}"
        ),
    )

    if args.enable_round2 and args.round2_action_limit > 0:
        with progress_stage(progress, "LLM Round 2 action planning"):
            round2_actions = llm_round2_actions(
                client,
                profile=topic_profile,
                diagnosis=diagnosis_round1,
                existing_actions=executed_round1_actions,
                action_limit=args.round2_action_limit,
                kg_top_k=args.round_kg_top_k,
                s2_top_k=args.round_s2_top_k,
                clean_queries=args.enable_query_cleaning,
            )
    else:
        round2_actions = []
        progress_log(progress, "ROUND2_SKIPPED enable_round2=false")
    annotate_kg_policy_metadata(round2_actions, enabled=args.enable_kg_policy)
    write_json(output_dir / "actions_round_2.json", {"actions": round2_actions})
    progress_log(progress, f"ROUND2_ACTIONS count={len(round2_actions)}")
    all_actions_by_id.update({action["action_id"]: action for action in round2_actions})
    round2_results = run_action_searches(backend, round2_actions, output_dir=output_dir, progress=progress)
    all_action_results.update(round2_results)

    embedding_expansion_action: dict[str, Any] | None = None
    embedding_expansion_result: dict[str, Any] | None = None
    if args.enable_embedding_expansion:
        if args.mock_backend:
            progress_log(progress, "EMBEDDING_EXPANSION_SKIPPED mock_backend=true")
        else:
            embedding_expansion_action, embedding_expansion_payload = run_embedding_expansion_round2(
                args=args,
                output_dir=output_dir,
                topic_profile=topic_profile,
                paper_cards=paper_cards_round1,
                clusters=clusters_round1,
                progress=progress,
            )
            embedding_expansion_result = {
                "action": embedding_expansion_action,
                "paper_count": (embedding_expansion_payload.get("filter") or {}).get("unique_paper_count", 0),
                "region_count": ((embedding_expansion_payload.get("sources") or {}).get("kg_embedding_gap") or {}).get(
                    "region_count",
                    0,
                ),
            }
            all_actions_by_id[embedding_expansion_action["action_id"]] = embedding_expansion_action
            all_action_results[embedding_expansion_action["action_id"]] = embedding_expansion_payload
    else:
        progress_log(progress, "EMBEDDING_EXPANSION_SKIPPED enable_embedding_expansion=false")

    final_action_ids = round1_action_ids + [action["action_id"] for action in round2_actions]
    if embedding_expansion_action:
        final_action_ids.append(embedding_expansion_action["action_id"])
    paper_cards_final = merge_paper_cards(
        all_action_results,
        action_ids=final_action_ids,
        actions_by_id=all_actions_by_id if args.enable_relevance_guard else None,
        abstract_char_limit=args.abstract_char_limit,
        current_year=current_year,
    )
    write_json(output_dir / "paper_cards_final.json", paper_cards_final)
    progress_log(progress, f"FINAL_PAPERS merged={len(paper_cards_final)} filtered={relevance_filtered_count(paper_cards_final)}")

    with progress_stage(progress, "LLM final clustering"):
        clusters_final = llm_clusters(
            client,
            profile=topic_profile,
            paper_cards=paper_cards_final,
            label="clusters_final",
            llm_paper_limit=args.llm_paper_limit,
        )
    write_json(output_dir / "clusters_final.json", clusters_final)
    progress_log(progress, f"FINAL_CLUSTERS count={len(clusters_final.get('clusters', []))}")

    final_executed_actions = [probe_action] + round1_actions + round2_actions
    if embedding_expansion_action:
        final_executed_actions.append(embedding_expansion_action)

    with progress_stage(progress, "LLM final coverage diagnosis"):
        final_diagnosis = llm_coverage_diagnosis(
            client,
            profile=topic_profile,
            time_windows=time_windows,
            paper_cards=paper_cards_final,
            clusters=clusters_final,
            actions=final_executed_actions,
            label="coverage_diagnosis_final",
            final=True,
            llm_paper_limit=args.llm_paper_limit,
        )
    write_json(output_dir / "coverage_diagnosis_final.json", final_diagnosis)

    search_result = {
        "topic": args.topic,
        "topic_profile": topic_profile,
        "time_window_source": time_window_payload["time_window_source"],
        "time_windows": time_windows,
        "paper_cards": paper_cards_final,
        "method_clusters": clusters_final.get("clusters", []),
        "coverage_report": final_diagnosis,
        "search_actions": {
            "probe": probe_action,
            "round_1": round1_actions,
            "round_2": round2_actions,
            "embedding_expansion_round_2": embedding_expansion_action,
        },
        "embedding_expansion": embedding_expansion_result,
        "diagnostics": {
            "output_dir": str(output_dir),
            "paper_count": len(paper_cards_final),
            "round1_paper_count": len(paper_cards_round1),
            "llm_model": args.llm_model,
            "llm_call_count": client.call_count,
            "mock_backend": args.mock_backend,
            "kg_policy_enabled": args.enable_kg_policy,
            "round2_enabled": args.enable_round2,
            "query_cleaning_enabled": args.enable_query_cleaning,
            "relevance_guard_enabled": args.enable_relevance_guard,
            "embedding_expansion_enabled": args.enable_embedding_expansion,
            "search_cache_dir": str(cache_dir),
            "elapsed_seconds": round(time.perf_counter() - pipeline_started_at, 3),
            "probe_relevance_filtered_count": relevance_filtered_count(probe_cards),
            "round1_relevance_filtered_count": relevance_filtered_count(paper_cards_round1),
            "final_relevance_filtered_count": relevance_filtered_count(paper_cards_final),
        },
    }
    write_json(output_dir / "search_result.json", search_result)
    progress_log(
        progress,
        (
            f"COMPLETE papers={len(paper_cards_final)} round1_actions={len(round1_actions)} "
            f"round2_actions={len(round2_actions)} elapsed={format_elapsed(time.perf_counter() - pipeline_started_at)}"
        ),
    )
    return search_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run literature-review search MVP.")
    parser.add_argument("--topic", required=True, help="Literature-review topic.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for lr_search artifacts, e.g. downstream/runs/example/artifacts/lr_search.",
    )
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with DMX and S2 keys.")
    parser.add_argument("--llm-api-url", default=DEFAULT_DMX_API_URL, help="DMX chat completions endpoint.")
    parser.add_argument("--llm-model", default=DEFAULT_DMX_MODEL, help="DMX model name.")
    parser.add_argument("--llm-timeout", type=int, default=120, help="LLM timeout in seconds.")
    parser.add_argument("--llm-max-tokens", type=int, default=4000, help="LLM max tokens per JSON call.")
    parser.add_argument("--llm-temperature", type=float, default=0.1, help="LLM temperature.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use HTTP(S)_PROXY for DMX and S2.")
    parser.add_argument("--s2-mode", choices=("search", "recommend", "hybrid"), default=None, help="S2 mode.")
    parser.add_argument("--kg-embedding-device", default=None, help="Torch device for KG embedding model.")
    parser.add_argument("--kg-reranker-device", default=None, help="Torch device for KG reranker model.")
    parser.add_argument(
        "--enable-kg-policy",
        action="store_true",
        help="Map action roles/styles to experimental KG citation, title, and scoring knobs.",
    )
    parser.add_argument(
        "--enable-round2",
        action="store_true",
        help="Enable experimental Round 2 refinement actions from structured coverage gaps.",
    )
    parser.add_argument(
        "--enable-query-cleaning",
        action="store_true",
        help="Enable experimental query cleaning that removes standalone years and broad prestige terms.",
    )
    parser.add_argument(
        "--enable-relevance-guard",
        action="store_true",
        help="Enable experimental lightweight filtering of weak KG-only action candidates.",
    )
    parser.add_argument(
        "--enable-embedding-expansion",
        action="store_true",
        help="Enable Round 2 KG embedding-space gap expansion from Round 1 clusters.",
    )
    parser.add_argument(
        "--embedding-expansion-field",
        choices=("abstract_embedding", "title_embedding"),
        default="abstract_embedding",
        help="Paper embedding field used for embedding expansion.",
    )
    parser.add_argument("--embedding-expansion-paper-count", type=int, default=50, help="Expansion recall paper cap.")
    parser.add_argument(
        "--embedding-expansion-topic-title-top-k",
        type=int,
        default=500,
        help="Expansion topic universe title-vector top-k.",
    )
    parser.add_argument(
        "--embedding-expansion-topic-abstract-top-k",
        type=int,
        default=1000,
        help="Expansion topic universe abstract-vector top-k.",
    )
    parser.add_argument("--embedding-expansion-top-seed-pool", type=int, default=500, help="Expansion residual cap.")
    parser.add_argument("--embedding-expansion-max-regions", type=int, default=10, help="Expansion region cap.")
    parser.add_argument("--embedding-expansion-min-region-size", type=int, default=4, help="Expansion min region size.")
    parser.add_argument(
        "--embedding-expansion-cluster-algorithm",
        choices=("auto", "hdbscan", "greedy"),
        default="auto",
        help="Expansion residual clustering algorithm.",
    )
    parser.add_argument(
        "--embedding-expansion-hdbscan-min-cluster-size",
        type=int,
        default=5,
        help="Expansion HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--embedding-expansion-hdbscan-min-samples",
        type=int,
        default=2,
        help="Expansion HDBSCAN min_samples.",
    )
    parser.add_argument("--embedding-expansion-mmr-lambda", type=float, default=0.75, help="Expansion MMR lambda.")
    parser.add_argument(
        "--embedding-expansion-max-fuzzy-fallbacks",
        type=int,
        default=0,
        help="Expansion title fuzzy fallback count for mapping current papers to KG.",
    )
    parser.add_argument(
        "--search-cache-dir",
        default=None,
        help="Optional shared cache directory for per-action SciAtlas backend cache files.",
    )
    parser.add_argument("--probe-kg-top-k", type=int, default=30, help="Probe SciAtlas budget component.")
    parser.add_argument("--probe-s2-top-k", type=int, default=30, help="Probe SciAtlas budget component.")
    parser.add_argument("--round-kg-top-k", type=int, default=20, help="Round 1/2 SciAtlas budget component.")
    parser.add_argument("--round-s2-top-k", type=int, default=20, help="Round 1/2 SciAtlas budget component.")
    parser.add_argument("--round1-action-limit", type=int, default=7, help="Maximum Round 1 actions.")
    parser.add_argument("--round2-action-limit", type=int, default=4, help="Maximum Round 2 actions.")
    parser.add_argument("--min-year", type=int, default=None, help="Optional lower year constraint.")
    parser.add_argument("--max-year", type=int, default=None, help="Optional upper year constraint.")
    parser.add_argument("--current-year", type=int, default=DEFAULT_CURRENT_YEAR, help="Current year for slicing.")
    parser.add_argument("--abstract-char-limit", type=int, default=1800, help="Stored abstract truncation length.")
    parser.add_argument("--llm-paper-limit", type=int, default=80, help="Max paper cards sent to LLM stages.")
    parser.add_argument("--mock-backend", action="store_true", help="Use deterministic mock LLM/search for smoke tests.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs on stderr.")
    parser.add_argument("--pretty", action="store_true", help="Print pretty JSON summary.")
    return parser


def mock_llm_response(*, label: str, user_prompt: str) -> dict[str, Any]:
    if label == "topic_profile":
        return {
            "normalized_topic": "LLM reasoning",
            "scope": "Methods and evaluations for reasoning in large language models.",
            "possible_method_families": ["chain-of-thought prompting", "tree search reasoning", "self-refinement"],
            "upstream_foundations": ["transformers", "neural symbolic reasoning"],
            "application_contexts": ["mathematical reasoning", "agent planning"],
            "likely_keywords": ["large language models", "reasoning", "chain-of-thought", "planning"],
            "ambiguous_terms": ["reasoning"],
            "exclusion_rules": ["Exclude generic NLP papers without reasoning evaluation."],
        }
    if label == "time_windows":
        return {
            "time_window_source": "llm",
            "time_windows": [
                {
                    "label": "foundational",
                    "start_year": 2018,
                    "end_year": 2021,
                    "rationale": "Early prompting and transformer foundations.",
                    "search_focus": "foundational reasoning and prompting",
                },
                {
                    "label": "cot_expansion",
                    "start_year": 2022,
                    "end_year": 2023,
                    "rationale": "Chain-of-thought and tool reasoning expanded.",
                    "search_focus": "chain-of-thought and decomposition",
                },
                {
                    "label": "recent_frontier",
                    "start_year": 2024,
                    "end_year": 2026,
                    "rationale": "Recent agentic and search-based reasoning.",
                    "search_focus": "test-time search and agent reasoning",
                },
            ],
            "warnings": [],
        }
    if label == "round1_actions":
        return {
            "actions": [
                {
                    "action_id": "round1_core",
                    "round": "round_1",
                    "query": "LLM reasoning chain-of-thought planning self-reflection",
                    "intent": "core_topic_recall",
                    "query_style": "broad",
                    "evidence_role": "representative",
                    "kg_top_k": 3,
                    "s2_top_k": 3,
                    "rationale": "Core recall.",
                },
                {
                    "action_id": "round1_recent",
                    "round": "round_1",
                    "query": "LLM reasoning test-time search agent planning 2024",
                    "intent": "recent_frontier_recall",
                    "query_style": "frontier",
                    "evidence_role": "recent_frontier",
                    "time_window": {"label": "recent_frontier", "start_year": 2024, "end_year": 2026},
                    "kg_top_k": 3,
                    "s2_top_k": 3,
                    "rationale": "Recent frontier recall.",
                },
            ]
        }
    if label == "round2_actions":
        return {
            "actions": [
                {
                    "action_id": "round2_benchmarks",
                    "round": "round_2",
                    "query": "LLM reasoning benchmarks mathematical reasoning planning evaluation",
                    "intent": "representative_query",
                    "query_style": "facet",
                    "evidence_role": "gap_refinement",
                    "kg_top_k": 3,
                    "s2_top_k": 3,
                    "rationale": "Deepen benchmark coverage.",
                }
            ]
        }
    if label.startswith("clusters"):
        ids = sorted(set(re.findall(r'"paper_id":\s*"(P[0-9]{3})"', user_prompt)))
        return {
            "clusters": [
                {
                    "cluster_id": "C1",
                    "name": "Prompted reasoning",
                    "definition": "Prompting methods that elicit intermediate reasoning traces.",
                    "topic_relevance": "core",
                    "keep_in_review": True,
                    "relevance_rationale": "Directly relevant to the topic's core reasoning-method focus.",
                    "distinguishing_features": ["chain-of-thought", "decomposition"],
                    "paper_ids": ids[: max(1, len(ids) // 2)],
                    "representative_paper_ids": ids[:1],
                    "missing_signals": [],
                },
                {
                    "cluster_id": "C2",
                    "name": "Search and agentic reasoning",
                    "definition": "Methods that use search, planning, or self-refinement around LLMs.",
                    "topic_relevance": "core",
                    "keep_in_review": True,
                    "relevance_rationale": "Directly relevant to the topic's core reasoning-method focus.",
                    "distinguishing_features": ["tree search", "planning"],
                    "paper_ids": ids[max(1, len(ids) // 2) :],
                    "representative_paper_ids": ids[max(1, len(ids) // 2) : max(2, len(ids) // 2 + 1)],
                    "missing_signals": [],
                },
            ],
            "outliers": [],
            "uncertain_assignments": [],
        }
    if label.startswith("coverage"):
        return {
            "coverage_status": "needs_refinement" if "round_1" in label else "limited",
            "structured_gaps": [
                {
                    "gap_id": "G1",
                    "axis": "benchmark",
                    "target": "Need more benchmark and evaluation papers.",
                    "severity": "high" if "round_1" in label else "low",
                    "query_seed": "benchmarks mathematical reasoning planning evaluation",
                    "suggested_intent": "benchmark_or_evaluation_query",
                    "blocks_stop": "round_1" in label,
                }
            ],
            "method_gaps": ["Need more benchmark and evaluation papers."],
            "time_gaps": [],
            "representative_gaps": [],
            "off_topic_notes": [],
            "suggested_refine_actions": [
                {
                    "query": "LLM reasoning benchmarks mathematical reasoning planning evaluation",
                    "intent": "representative_query",
                }
            ],
            "stop_recommendation": False,
        }
    return {}


def mock_search_payload(action: dict[str, Any]) -> dict[str, Any]:
    base = [
        {
            "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            "abstract": "Chain-of-thought prompting improves multi-step reasoning in large language models.",
            "year": 2022,
            "citation_count": 5000,
            "paper_url": "https://example.org/cot",
            "identifiers": ["s2:cot"],
            "source_set": ["s2"],
            "source_rank": 1,
            "paper": {"authors": [{"name": "Jason Wei"}], "venue": "NeurIPS"},
        },
        {
            "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
            "abstract": "Tree of Thoughts uses search over intermediate thoughts for deliberate reasoning.",
            "year": 2023,
            "citation_count": 1200,
            "paper_url": "https://example.org/tot",
            "identifiers": ["s2:tot"],
            "source_set": ["kg", "s2"],
            "source_rank": 2,
            "paper": {"authors": [{"name": "Shunyu Yao"}], "venue": "NeurIPS"},
        },
        {
            "title": "Self-Refine: Iterative Refinement with Self-Feedback",
            "abstract": "Self-Refine lets language models improve outputs through iterative feedback.",
            "year": 2023,
            "citation_count": 900,
            "paper_url": "https://example.org/self-refine",
            "identifiers": ["s2:selfrefine"],
            "source_set": ["s2"],
            "source_rank": 3,
            "paper": {"authors": [{"name": "Madaan"}], "venue": "ACL"},
        },
    ]
    if action["round"] == "round_2":
        base.append(
            {
                "title": "Measuring Mathematical Problem Solving With Language Models",
                "abstract": "Benchmarks evaluate mathematical reasoning and problem solving in language models.",
                "year": 2024,
                "citation_count": 300,
                "paper_url": "https://example.org/math-bench",
                "identifiers": ["s2:mathbench"],
                "source_set": ["kg"],
                "source_rank": 1,
                "paper": {"authors": [{"name": "Benchmark Author"}], "venue": "ICLR"},
            }
        )
    return {
        "status": "ok",
        "successful_source_count": 2,
        "failed_source_count": 0,
        "filter": {"unique_papers": base, "unique_paper_count": len(base)},
        "ranking": {"status": "skipped", "papers": base},
        "sources": {"kg": {"status": "ok"}, "s2": {"status": "ok"}},
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline(args)
    summary = {
        "search_result": str(Path(args.output_dir).expanduser().resolve() / "search_result.json"),
        "paper_count": result["diagnostics"]["paper_count"],
        "round1_actions": len(result["search_actions"]["round_1"]),
        "round2_actions": len(result["search_actions"]["round_2"]),
        "llm_model": result["diagnostics"]["llm_model"],
        "mock_backend": result["diagnostics"]["mock_backend"],
    }
    print(json.dumps(summary if not args.pretty else result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
