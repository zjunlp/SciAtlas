#!/usr/bin/env python3
"""Resolve YYYY-MM timestamps for method-timeline anchors.

Input:
  - current method timeline spec YAML/JSON
  - optional search_result.json to reuse known identifiers from paper_cards

Output:
  - enriched spec with per-anchor month fields
  - resolution report with coverage statistics

This script aims only to add a reliable month-level timestamp when possible.
It does not modify rendering logic by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LR_SEARCH_DIR = Path(__file__).resolve().parent
if str(LR_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(LR_SEARCH_DIR))

ARS_SCRIPTS_DIR = LR_SEARCH_DIR.parent / "repos" / "academic-research-skills" / "scripts"
if str(ARS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ARS_SCRIPTS_DIR))

from literature_review_search import normalize_whitespace  # noqa: E402
from migrate_method_spec import extract_s2_id_from_card  # noqa: E402
from openalex_client import OpenAlexClient, OpenAlexUnavailable  # noqa: E402
from crossref_client import CrossrefClient, CrossrefUnavailable  # noqa: E402

try:
    from semantic_scholar_client import SemanticScholarClient, SemanticScholarUnavailable  # noqa: E402
except ImportError:  # pragma: no cover
    SemanticScholarClient = None  # type: ignore
    SemanticScholarUnavailable = Exception  # type: ignore


ARXIV_API_BASE = "https://export.arxiv.org/api/query"
S2_GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_ID_RE = re.compile(r"(?:arxiv:|abs/)?([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})", re.I)


def load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected top-level mapping")
    return payload


def dump_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10_000),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_title(text: Any) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = cleaned.casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return normalize_whitespace(cleaned)


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return cleaned[:180].strip("._-") or "item"


def parse_yyyy_mm(text: str | None) -> str | None:
    raw = normalize_whitespace(text or "")
    if not raw:
        return None
    m = re.match(r"^(\d{4})-(\d{2})", raw)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}"


def extract_arxiv_id_from_values(*values: Any) -> str | None:
    for value in values:
        text = normalize_whitespace(value)
        if not text:
            continue
        if "10." in text and "arxiv" not in text.casefold():
            continue
        match = ARXIV_ID_RE.search(text)
        if match:
            return match.group(1)
    return None


def first_date_parts(date_parts_blob: Any) -> str | None:
    if not isinstance(date_parts_blob, list) or not date_parts_blob:
        return None
    first = date_parts_blob[0]
    if not isinstance(first, list) or not first:
        return None
    try:
        year = int(first[0])
    except (TypeError, ValueError):
        return None
    month = 1
    if len(first) >= 2:
        try:
            month = int(first[1])
        except (TypeError, ValueError):
            month = 1
    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}"


def crossref_yyyy_mm(item: dict[str, Any]) -> str | None:
    for key in ("published-online", "published-print", "issued", "published", "created"):
        val = item.get(key)
        if isinstance(val, dict):
            parsed = first_date_parts(val.get("date-parts"))
            if parsed:
                return parsed
    return None


def title_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def openalex_year(work: dict[str, Any]) -> int | None:
    try:
        return int(work.get("publication_year"))
    except (TypeError, ValueError):
        return None


def crossref_year(item: dict[str, Any]) -> int | None:
    for key in ("issued", "published-print", "published-online", "published", "created"):
        val = item.get(key)
        if not isinstance(val, dict):
            continue
        parts = val.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                return None
    return None


def candidate_is_plausible(title: str, candidate_title: str, target_year: int | None, candidate_year: int | None) -> bool:
    if title_similarity(title, candidate_title) < 0.82:
        return False
    if target_year is not None and candidate_year is not None and abs(candidate_year - target_year) > 2:
        return False
    return True


@dataclass
class AnchorContext:
    lane_id: str
    lane_name: str
    anchor: dict[str, Any]
    title: str
    label: str
    year: int | None
    doi: str = ""
    arxiv_id: str = ""
    s2_id: str = ""
    openalex_id: str = ""
    corpus_paper_id: str = ""
    matched_card_title: str = ""


class ArxivMonthClient:
    def __init__(self, *, use_env_proxy: bool = False, min_interval: float = 0.5):
        self._last_request_at: float | None = None
        self._min_interval = min_interval
        self._opener = urllib.request.build_opener() if use_env_proxy else urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    def lookup_month(self, arxiv_id: str) -> str | None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()
        params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
        url = f"{ARXIV_API_BASE}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "innoeval-timeline/1.0"})
        try:
            with self._opener.open(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            return None
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        published = entry.findtext("a:published", default="", namespaces=ns)
        return parse_yyyy_mm(published)


class S2TitleMatchClient:
    def __init__(self, *, min_interval: float = 1.2):
        self._last_request_at: float | None = None
        self._min_interval = min_interval

    def match(self, title: str) -> dict[str, Any] | None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()
        params = urllib.parse.urlencode(
            {"query": title, "fields": "title,year,publicationDate,externalIds,venue"}
        )
        url = f"{S2_GRAPH_BASE}/paper/search/match?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "innoeval-month/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
        data = payload.get("data")
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(payload, dict) and payload.get("title"):
            return payload
        return None

    def paper_month(self, paper_id: str) -> str | None:
        if not paper_id:
            return None
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()
        url = f"{S2_GRAPH_BASE}/paper/{urllib.parse.quote(paper_id)}?fields=publicationDate"
        req = urllib.request.Request(url, headers={"User-Agent": "innoeval-month/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
        return parse_yyyy_mm(payload.get("publicationDate") if isinstance(payload, dict) else None)


class JsonFileCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, namespace: str, key: str) -> Any | None:
        path = self.root / namespace / f"{safe_filename(key)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, namespace: str, key: str, payload: Any) -> None:
        path = self.root / namespace / f"{safe_filename(key)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_paper_card_index(search_result_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_title: dict[str, dict[str, Any]] = {}
    by_pid: dict[str, dict[str, Any]] = {}
    if not search_result_path or not search_result_path.exists():
        return by_title, by_pid
    payload = read_json(search_result_path)
    paper_cards = payload.get("paper_cards") if isinstance(payload, dict) else None
    if not isinstance(paper_cards, list):
        return by_title, by_pid
    for card in paper_cards:
        if not isinstance(card, dict):
            continue
        paper_id = normalize_whitespace(card.get("paper_id"))
        title_key = norm_title(card.get("title"))
        if paper_id:
            by_pid[paper_id] = card
        if title_key and title_key not in by_title:
            by_title[title_key] = card
    return by_title, by_pid


def enrich_anchor_contexts(spec: dict[str, Any], *, by_title: dict[str, dict[str, Any]], by_pid: dict[str, dict[str, Any]]) -> list[AnchorContext]:
    contexts: list[AnchorContext] = []
    for cluster in spec.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        lane_id = str(cluster.get("id") or "")
        lane_name = str(cluster.get("name") or lane_id)
        for anchor in cluster.get("anchors") or []:
            if not isinstance(anchor, dict):
                continue
            title = normalize_whitespace(anchor.get("full_title") or anchor.get("title") or anchor.get("label"))
            label = normalize_whitespace(anchor.get("label") or title)
            year = anchor.get("year")
            try:
                year_val = int(year) if year is not None else None
            except (TypeError, ValueError):
                year_val = None
            card = None
            corpus_paper_id = normalize_whitespace(anchor.get("corpus_paper_id"))
            if corpus_paper_id and corpus_paper_id in by_pid:
                maybe_card = by_pid[corpus_paper_id]
                if candidate_is_plausible(title, str(maybe_card.get("title") or ""), year_val, maybe_card.get("year")):
                    card = maybe_card
            if card is None:
                card = by_title.get(norm_title(title))
            doi = normalize_whitespace(anchor.get("doi") or "")
            arxiv_id = normalize_whitespace(anchor.get("arxiv_id") or "")
            s2_id = normalize_whitespace(anchor.get("s2_paper_id") or "")
            openalex_id = normalize_whitespace(anchor.get("openalex_id") or "")
            matched_title = ""
            if card:
                matched_title = normalize_whitespace(card.get("title") or "")
                doi = doi or normalize_whitespace(card.get("doi") or "")
                s2_ext = extract_s2_id_from_card(card)
                s2_id = s2_id or normalize_whitespace(s2_ext or "")
                stable_key = normalize_whitespace(card.get("stable_key") or "")
                if stable_key.startswith("openalex:"):
                    openalex_id = openalex_id or stable_key.split(":", 1)[1]
                arxiv_id = arxiv_id or extract_arxiv_id_from_values(
                    stable_key,
                    card.get("doi"),
                    card.get("url"),
                    card.get("title"),
                ) or ""
            arxiv_id = arxiv_id or extract_arxiv_id_from_values(
                title,
                label,
                anchor.get("notes"),
                anchor.get("doi"),
            ) or ""
            contexts.append(
                AnchorContext(
                    lane_id=lane_id,
                    lane_name=lane_name,
                    anchor=anchor,
                    title=title,
                    label=label,
                    year=year_val,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    s2_id=s2_id,
                    openalex_id=openalex_id,
                    corpus_paper_id=corpus_paper_id,
                    matched_card_title=matched_title,
                )
            )
    return contexts


def fill_identifiers_from_crossref_item(ctx: AnchorContext, item: dict[str, Any]) -> None:
    doi = normalize_whitespace(item.get("DOI") or item.get("doi") or "")
    if doi and not ctx.doi:
        ctx.doi = doi.casefold()


def fill_identifiers_from_openalex_work(ctx: AnchorContext, work: dict[str, Any]) -> None:
    doi = normalize_whitespace(work.get("doi") or "")
    if doi and not ctx.doi:
        ctx.doi = doi.replace("https://doi.org/", "").casefold()
    openalex_id = normalize_whitespace(work.get("id") or "")
    if openalex_id and not ctx.openalex_id:
        ctx.openalex_id = openalex_id


def fill_identifiers_from_s2_match(ctx: AnchorContext, item: dict[str, Any]) -> None:
    ext = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    doi = normalize_whitespace(ext.get("DOI") or "")
    if doi and not ctx.doi:
        ctx.doi = doi.casefold().replace("https://doi.org/", "")
    arxiv_id = normalize_whitespace(ext.get("ArXiv") or "")
    if arxiv_id and not ctx.arxiv_id:
        ctx.arxiv_id = arxiv_id
    paper_id = normalize_whitespace(item.get("paperId") or "")
    if paper_id and not ctx.s2_id:
        ctx.s2_id = paper_id


def title_search_openalex_raw(client: OpenAlexClient, title: str, year: int | None = None) -> dict[str, Any] | None:
    try:
        return client.title_search(title, year=year)
    except OpenAlexUnavailable:
        return None


def title_search_crossref_raw(client: CrossrefClient, title: str, year: int | None = None) -> dict[str, Any] | None:
    try:
        return client.title_search(title, year=year)
    except CrossrefUnavailable:
        return None


def maybe_backfill_identifiers(
    ctx: AnchorContext,
    *,
    openalex_client: OpenAlexClient,
    crossref_client: CrossrefClient | None,
    s2_title_client: S2TitleMatchClient | None,
    cache: JsonFileCache,
    allow_crossref_title: bool = False,
) -> None:
    if ctx.doi or ctx.arxiv_id or ctx.openalex_id:
        return
    title_key = f"{ctx.title}|{ctx.year or ''}"
    cached = cache.get("title_resolution", title_key)
    if isinstance(cached, dict):
        ctx.doi = normalize_whitespace(cached.get("doi") or ctx.doi)
        ctx.arxiv_id = normalize_whitespace(cached.get("arxiv_id") or ctx.arxiv_id)
        ctx.openalex_id = normalize_whitespace(cached.get("openalex_id") or ctx.openalex_id)
        return

    resolved: dict[str, str] = {"doi": "", "arxiv_id": "", "openalex_id": "", "s2_id": ""}

    work = title_search_openalex_raw(openalex_client, ctx.title, year=ctx.year)
    if isinstance(work, dict) and candidate_is_plausible(
        ctx.title,
        str(work.get("title") or ""),
        ctx.year,
        openalex_year(work),
    ):
        fill_identifiers_from_openalex_work(ctx, work)
        resolved["doi"] = ctx.doi
        resolved["openalex_id"] = ctx.openalex_id

    if s2_title_client is not None and not ctx.doi and not ctx.arxiv_id:
        matched = s2_title_client.match(ctx.title)
        if isinstance(matched, dict) and candidate_is_plausible(
            ctx.title,
            str(matched.get("title") or ""),
            ctx.year,
            matched.get("year"),
        ):
            fill_identifiers_from_s2_match(ctx, matched)
            resolved["doi"] = ctx.doi
            resolved["arxiv_id"] = ctx.arxiv_id
            resolved["s2_id"] = ctx.s2_id

    if allow_crossref_title and crossref_client is not None and not ctx.doi:
        item = title_search_crossref_raw(crossref_client, ctx.title, year=ctx.year)
        if isinstance(item, dict) and candidate_is_plausible(
            ctx.title,
            " ".join(item.get("title") or []) if isinstance(item.get("title"), list) else str(item.get("title") or ""),
            ctx.year,
            crossref_year(item),
        ):
            fill_identifiers_from_crossref_item(ctx, item)
            resolved["doi"] = ctx.doi

    if ctx.doi:
        ctx.arxiv_id = ctx.arxiv_id or extract_arxiv_id_from_values(ctx.doi) or ""

    cache.put(
        "title_resolution",
        title_key,
        {
            "doi": ctx.doi,
            "arxiv_id": ctx.arxiv_id,
            "openalex_id": ctx.openalex_id,
            "s2_id": ctx.s2_id,
        },
    )


def best_openalex_month(client: OpenAlexClient, ctx: AnchorContext, *, allow_title_search: bool) -> str | None:
    try:
        if ctx.doi:
            work = client.doi_lookup_with_title_check(ctx.doi, ctx.title)
            if work and isinstance(work, dict):
                return parse_yyyy_mm(work.get("publication_date"))
        if allow_title_search:
            work = client.title_search(ctx.title, year=ctx.year)
            if work and isinstance(work, dict):
                return parse_yyyy_mm(work.get("publication_date"))
    except OpenAlexUnavailable:
        return None
    return None


def best_crossref_month(client: CrossrefClient, ctx: AnchorContext, *, allow_title_search: bool) -> str | None:
    try:
        if ctx.doi:
            work = client.doi_lookup_with_title_check(ctx.doi, ctx.title)
            if work and isinstance(work, dict):
                return crossref_yyyy_mm(work)
    except CrossrefUnavailable:
        return None
    return None


def best_s2_month(client: Any, ctx: AnchorContext, *, allow_title_search: bool) -> str | None:
    if client is None:
        return None
    try:
        if ctx.doi:
            result = client.lookup({"doi": ctx.doi, "title": ctx.title, "year": ctx.year})
            if result.get("matched") and result.get("paperId"):
                paper_id = result["paperId"]
                data = client._request(f"/paper/{urllib.parse.quote(paper_id)}?fields=publicationDate")  # type: ignore[attr-defined]
                return parse_yyyy_mm(data.get("publicationDate") if isinstance(data, dict) else None)
        if allow_title_search and ctx.title:
            result = client.lookup({"title": ctx.title, "year": ctx.year})
            if result.get("matched") and result.get("paperId"):
                paper_id = result["paperId"]
                data = client._request(f"/paper/{urllib.parse.quote(paper_id)}?fields=publicationDate")  # type: ignore[attr-defined]
                return parse_yyyy_mm(data.get("publicationDate") if isinstance(data, dict) else None)
    except SemanticScholarUnavailable:
        return None
    except Exception:
        return None
    return None


def resolve_month(
    ctx: AnchorContext,
    *,
    arxiv_client: ArxivMonthClient,
    openalex_client: OpenAlexClient,
    crossref_client: CrossrefClient | None,
    s2_client: Any,
    s2_title_client: S2TitleMatchClient | None,
    allow_title_search: bool,
    enable_crossref: bool,
    enable_s2: bool,
    cache: JsonFileCache,
    allow_crossref_title: bool = False,
) -> tuple[str | None, str]:
    if ctx.arxiv_id:
        month = arxiv_client.lookup_month(ctx.arxiv_id)
        if month:
            return month, "arxiv"
    if ctx.s2_id and s2_title_client is not None:
        month = s2_title_client.paper_month(ctx.s2_id)
        if month:
            return month, "s2"
    if allow_title_search:
        maybe_backfill_identifiers(
            ctx,
            openalex_client=openalex_client,
            crossref_client=crossref_client if enable_crossref else None,
            s2_title_client=s2_title_client,
            cache=cache,
            allow_crossref_title=allow_crossref_title,
        )
        if ctx.arxiv_id:
            month = arxiv_client.lookup_month(ctx.arxiv_id)
            if month:
                return month, "arxiv"
        if ctx.s2_id and s2_title_client is not None:
            month = s2_title_client.paper_month(ctx.s2_id)
            if month:
                return month, "s2"
        if s2_title_client is not None:
            matched = s2_title_client.match(ctx.title)
            if isinstance(matched, dict) and candidate_is_plausible(
                ctx.title,
                str(matched.get("title") or ""),
                ctx.year,
                matched.get("year"),
            ):
                month = parse_yyyy_mm(matched.get("publicationDate"))
                if month:
                    fill_identifiers_from_s2_match(ctx, matched)
                    return month, "s2"
    month = best_openalex_month(openalex_client, ctx, allow_title_search=allow_title_search)
    if month:
        return month, "openalex"
    if enable_crossref and crossref_client is not None:
        month = best_crossref_month(crossref_client, ctx, allow_title_search=allow_title_search)
        if month:
            return month, "crossref"
    if enable_s2:
        month = best_s2_month(s2_client, ctx, allow_title_search=allow_title_search)
        if month:
            return month, "s2"
    return None, ""


def update_spec_with_months(spec: dict[str, Any], contexts: list[AnchorContext], resolved: dict[int, tuple[str | None, str]]) -> dict[str, Any]:
    idx = 0
    for cluster in spec.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        for anchor in cluster.get("anchors") or []:
            if not isinstance(anchor, dict):
                continue
            month, source = resolved.get(idx, (None, ""))
            if month:
                anchor["year_month"] = month
                anchor["timeline_month_source"] = source
            idx += 1
    return spec


def build_report(contexts: list[AnchorContext], resolved: dict[int, tuple[str | None, str]]) -> dict[str, Any]:
    total = len(contexts)
    success = 0
    by_source: dict[str, int] = {}
    unresolved: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    for idx, ctx in enumerate(contexts):
        month, source = resolved.get(idx, (None, ""))
        if month:
            success += 1
            by_source[source] = by_source.get(source, 0) + 1
            resolved_rows.append(
                {
                    "lane_id": ctx.lane_id,
                    "lane_name": ctx.lane_name,
                    "label": ctx.label,
                    "title": ctx.title,
                    "year": ctx.year,
                    "year_month": month,
                    "source": source,
                    "doi": ctx.doi,
                    "arxiv_id": ctx.arxiv_id,
                }
            )
        else:
            unresolved.append(
                {
                    "lane_id": ctx.lane_id,
                    "lane_name": ctx.lane_name,
                    "label": ctx.label,
                    "title": ctx.title,
                    "year": ctx.year,
                    "doi": ctx.doi,
                    "arxiv_id": ctx.arxiv_id,
                    "s2_id": ctx.s2_id,
                    "openalex_id": ctx.openalex_id,
                }
            )
    return {
        "anchor_count": total,
        "resolved_count": success,
        "resolved_pct": round((success / total) * 100.0, 2) if total else 0.0,
        "by_source": by_source,
        "resolved": resolved_rows,
        "unresolved": unresolved,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve YYYY-MM timestamps for method-timeline anchors.")
    parser.add_argument("--spec", required=True, help="Input spec YAML/JSON.")
    parser.add_argument("--search-result", default=None, help="Optional search_result.json for paper_cards identifiers.")
    parser.add_argument("--out-spec", required=True, help="Output spec path with year_month fields.")
    parser.add_argument("--report", required=True, help="Output JSON report path.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use HTTP(S)_PROXY from environment.")
    parser.add_argument("--allow-title-search", action="store_true", help="Allow title-based fallback on external APIs.")
    parser.add_argument("--enable-crossref", action="store_true", help="Enable Crossref fallback.")
    parser.add_argument("--enable-s2", action="store_true", help="Enable Semantic Scholar fallback.")
    parser.add_argument("--cache-dir", default=None, help="Optional cache dir for title/date lookups.")
    parser.add_argument("--allow-crossref-title", action="store_true", help="Allow slow Crossref title-based identifier backfill.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    out_spec_path = Path(args.out_spec).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    search_result_path = Path(args.search_result).expanduser().resolve() if args.search_result else None
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else out_spec_path.parent / ".timeline_month_cache"
    )

    spec = load_spec(spec_path)
    by_title, by_pid = build_paper_card_index(search_result_path)
    contexts = enrich_anchor_contexts(spec, by_title=by_title, by_pid=by_pid)

    arxiv_client = ArxivMonthClient(use_env_proxy=args.use_env_proxy)
    openalex_client = OpenAlexClient()
    crossref_client = CrossrefClient() if args.enable_crossref else None
    s2_client = SemanticScholarClient() if (args.enable_s2 and SemanticScholarClient is not None) else None
    s2_title_client = S2TitleMatchClient()
    cache = JsonFileCache(cache_dir)

    resolved: dict[int, tuple[str | None, str]] = {}
    for idx, ctx in enumerate(contexts):
        month, source = resolve_month(
            ctx,
            arxiv_client=arxiv_client,
            openalex_client=openalex_client,
            crossref_client=crossref_client,
            s2_client=s2_client,
            s2_title_client=s2_title_client,
            allow_title_search=args.allow_title_search,
            enable_crossref=args.enable_crossref,
            enable_s2=args.enable_s2,
            cache=cache,
            allow_crossref_title=args.allow_crossref_title,
        )
        resolved[idx] = (month, source)

    enriched = update_spec_with_months(spec, contexts, resolved)
    report = build_report(contexts, resolved)

    dump_spec(out_spec_path, enriched)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_spec": str(out_spec_path), "report": str(report_path), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
