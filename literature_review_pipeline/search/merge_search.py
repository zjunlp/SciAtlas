#!/usr/bin/env python3
"""Combined KG + S2 paper search with deduplication, filtering, and LLM reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent
INNOEVAL_DIR = BASE_DIR.parent
PROJECT_ROOT = INNOEVAL_DIR.parent
S2API_DIR = INNOEVAL_DIR / "s2api"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CACHE_PATH = BASE_DIR / "target"
RESULT_DIR = BASE_DIR / "result"
RESULT_MARKDOWN = RESULT_DIR / "merge_search_log.md"
DMX_API_URL = "https://www.dmxapi.cn/v1/chat/completions"
DEFAULT_DMX_MODEL = "deepseek-v3.2"
DEFAULT_LLM_BATCH_SIZE = 4
DEFAULT_LLM_PAPER_COVERAGE = 2
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "the",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "using",
    "via",
    "with",
}
LLM_SYSTEM_PROMPT = (
    "You are a rigorous academic literature relevance judge. "
    "Score each candidate paper independently and return strict JSON only."
)


for extra_path in (INNOEVAL_DIR, S2API_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

import search_s2 as s2_search  # noqa: E402


def _build_source_error_payload(
    source_name: str,
    *,
    error_type: str,
    error: str,
    elapsed_ms: float = 0.0,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source_name,
        "status": "error",
        "error_type": error_type,
        "error": error,
        "elapsed_ms": round(elapsed_ms, 1),
        "paper_count": 0,
        "papers": [],
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    if source_name == "kg":
        payload["author_count"] = 0
        payload["authors"] = []
    return payload


def _load_kg_runner() -> Callable[[argparse.Namespace], Any]:
    from kg_search.service import run_search_with_authors  # noqa: WPS433

    return run_search_with_authors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run KG + Semantic Scholar search, deduplicate, filter, and LLM-rerank papers."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-text", help="Idea text used for retrieval.")
    input_group.add_argument("--pdf-path", help="PDF path used for retrieval.")

    parser.add_argument("--kg-top-k", type=int, default=20, help="Top-k paper count returned from KG search.")
    parser.add_argument("--s2-top-k", type=int, default=20, help="Top-k paper count returned from S2 search.")
    parser.add_argument(
        "--s2-mode",
        choices=("search", "recommend", "hybrid"),
        default=None,
        help="S2 retrieval mode. Default inside merge_search is `search`.",
    )
    parser.add_argument(
        "--s2-search-top-k",
        type=int,
        default=None,
        help="Optional cap on S2 search candidates before final output.",
    )
    parser.add_argument(
        "--s2-recommend-top-k",
        type=int,
        default=None,
        help="Optional cap on S2 recommendation candidates before final output.",
    )
    parser.add_argument(
        "--s2-per-keyword-limit",
        type=int,
        default=10,
        help="Number of papers requested from Semantic Scholar for each extracted keyword.",
    )
    parser.add_argument(
        "--include-s2-per-keyword-results",
        action="store_true",
        help="Include raw per-keyword search results in the S2 payload.",
    )

    parser.add_argument("--Target-Field", dest="target_field", default=None, help="KG field filter.")
    parser.add_argument("--after", default=None, help="KG lower date bound in YYYY-MM-DD format.")
    parser.add_argument("--before", default=None, help="KG upper date bound in YYYY-MM-DD format.")
    parser.add_argument("--kg-embedding-device", default=None, help="Torch device for KG embedding model.")
    parser.add_argument("--kg-reranker-device", default=None, help="Torch device for KG reranker model.")
    parser.add_argument(
        "--unable-title-ft",
        action="store_true",
        help="Disable KG title fulltext fallback and use exact title matching only.",
    )
    parser.add_argument(
        "--max-titles-from-pdf-references",
        type=int,
        default=None,
        help="Override KG max reference titles selected from PDF references.",
    )
    parser.add_argument(
        "--pdf-reference-selection-mode",
        choices=("llm", "all"),
        default=None,
        help="Select PDF references with the LLM or use the first max reference titles directly.",
    )
    parser.add_argument(
        "--max-seed-papers",
        type=int,
        default=None,
        help="Override KG graph max seed papers.",
    )
    parser.add_argument(
        "--kg-graph-method",
        dest="kg_graph_method",
        choices=("none", "A", "B"),
        default=None,
        help="Override KG graph reranking method. Use `none` for no graphwalk.",
    )
    parser.add_argument(
        "--disable-kg-keywords",
        action="store_true",
        help="Disable KG keyword extraction, keyword recall, and keyword graph seeds.",
    )
    parser.add_argument(
        "--kg-pdf-debug-dir",
        default=None,
        help="Optional directory where KG PDF-search debug artifacts are written.",
    )
    parser.add_argument(
        "--kg-use-citation-importance",
        action="store_true",
        help="Use KG citation-derived importance instead of uniform importance.",
    )
    parser.add_argument("--kg-final-weight-pre-graph", type=float, default=None, help="KG final score pre-graph weight.")
    parser.add_argument("--kg-final-weight-graph", type=float, default=None, help="KG final score graph weight.")
    parser.add_argument("--kg-final-weight-importance", type=float, default=None, help="KG final score citation/importance weight.")
    parser.add_argument("--kg-weight-title-path", type=float, default=None, help="KG pre-graph title-path weight.")
    parser.add_argument("--kg-weight-embedding-path", type=float, default=None, help="KG pre-graph embedding-path weight.")
    parser.add_argument("--kg-topk-title-rerank", type=int, default=None, help="KG title-vector rerank keep top-k.")
    parser.add_argument("--kg-topk-abstract-rerank", type=int, default=None, help="KG abstract-vector rerank keep top-k.")
    parser.add_argument("--kg-topk-title-vector", type=int, default=None, help="KG title-vector recall top-k.")
    parser.add_argument("--kg-topk-abstract-vector", type=int, default=None, help="KG abstract-vector recall top-k.")
    parser.add_argument("--kg-title-exact-pre-graph-bonus", type=float, default=None, help="KG exact-title pre-graph bonus.")
    parser.add_argument("--kg-title-fuzzy-pre-graph-bonus", type=float, default=None, help="KG fuzzy-title pre-graph bonus.")
    parser.add_argument("--kg-final-title-bonus", type=float, default=None, help="KG exact-title final score bonus.")

    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Loose JSON cache file.")
    parser.add_argument(
        "--disable-cache-reuse",
        action="store_true",
        help="Ignore cache-path and always recompute KG and PDF extraction.",
    )
    parser.add_argument(
        "--reuse-cached-s2",
        action="store_true",
        help="Reuse cached S2 papers from cache-path when available and input matches.",
    )

    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env file for S2 and DMX.")
    parser.add_argument("--keyword-model", default=s2_search.DEFAULT_KEYWORD_MODEL, help="Keyword extractor model.")
    parser.add_argument("--keyword-api-url", default=s2_search.KEYWORD_API_URL, help="Keyword extractor API URL.")
    parser.add_argument("--keyword-timeout", type=int, default=60, help="Keyword extractor timeout in seconds.")
    parser.add_argument("--search-timeout", type=int, default=s2_search.DEFAULT_TIMEOUT, help="S2 timeout in seconds.")
    parser.add_argument("--search-retries", type=int, default=3, help="S2 retry count.")
    parser.add_argument(
        "--recommendation-limit",
        type=int,
        default=20,
        help="Number of S2 recommendation papers requested for PDF title flow.",
    )
    parser.add_argument("--grobid-base-url", default=s2_search.DEFAULT_GROBID_BASE_URL, help="GROBID base URL.")
    parser.add_argument("--grobid-start-page", type=int, default=None, help="First PDF page sent to GROBID.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the environment for S2 and DMX requests.",
    )

    parser.add_argument(
        "--disable-llm-ranking",
        action="store_true",
        help="Skip DMX batch scoring and only return deduplicated papers.",
    )
    parser.add_argument("--llm-model", default=DEFAULT_DMX_MODEL, help="DMX model used for reranking.")
    parser.add_argument("--llm-timeout", type=int, default=60, help="DMX timeout in seconds.")
    parser.add_argument("--llm-max-tokens", type=int, default=900, help="Max tokens for each DMX batch call.")
    parser.add_argument("--llm-temperature", type=float, default=0.1, help="DMX temperature.")
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=DEFAULT_LLM_BATCH_SIZE,
        help="Paper count in each LLM scoring batch.",
    )
    parser.add_argument(
        "--llm-paper-coverage",
        type=int,
        default=DEFAULT_LLM_PAPER_COVERAGE,
        help="How many different batches each paper should appear in.",
    )
    parser.add_argument(
        "--llm-max-parallel",
        type=int,
        default=20,
        help="Maximum parallel DMX batch calls.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help="Optional top-k slice on the final LLM-ranked sequence.",
    )

    parser.add_argument("--result-tag", default="merge_search", help="Tag used for result artifacts.")
    parser.add_argument(
        "--disable-result-log",
        action="store_true",
        help="Do not write JSON and markdown artifacts into search/result/.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def build_kg_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        idea_text=args.idea_text,
        pdf_path=args.pdf_path,
        top_k=args.kg_top_k,
        target_field=args.target_field,
        after=args.after,
        before=args.before,
        embedding_device=getattr(args, "kg_embedding_device", None),
        reranker_device=getattr(args, "kg_reranker_device", None),
        unable_title_ft=args.unable_title_ft,
        max_titles_from_pdf_references=getattr(args, "max_titles_from_pdf_references", None),
        pdf_reference_selection_mode=getattr(args, "pdf_reference_selection_mode", None),
        max_seed_papers=getattr(args, "max_seed_papers", None),
        graph_method=getattr(args, "kg_graph_method", None),
        disable_keywords=getattr(args, "disable_kg_keywords", False),
        pdf_debug_dir=getattr(args, "kg_pdf_debug_dir", None),
        use_citation_importance=getattr(args, "kg_use_citation_importance", False),
        final_weight_pre_graph=getattr(args, "kg_final_weight_pre_graph", None),
        final_weight_graph=getattr(args, "kg_final_weight_graph", None),
        final_weight_importance=getattr(args, "kg_final_weight_importance", None),
        weight_embedding_path=getattr(args, "kg_weight_embedding_path", None),
        weight_title_path=getattr(args, "kg_weight_title_path", None),
        topk_title_rerank=getattr(args, "kg_topk_title_rerank", None),
        topk_abstract_rerank=getattr(args, "kg_topk_abstract_rerank", None),
        topk_title_vector=getattr(args, "kg_topk_title_vector", None),
        topk_abstract_vector=getattr(args, "kg_topk_abstract_vector", None),
        title_exact_pre_graph_bonus=getattr(args, "kg_title_exact_pre_graph_bonus", None),
        title_fuzzy_pre_graph_bonus=getattr(args, "kg_title_fuzzy_pre_graph_bonus", None),
        final_title_bonus=getattr(args, "kg_final_title_bonus", None),
        pretty=False,
    )


def build_s2_args(args: argparse.Namespace, *, pre_extracted_pdf: dict[str, str] | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        idea_text=args.idea_text,
        pdf_path=args.pdf_path,
        env=args.env,
        mode=args.s2_mode or "search",
        top_k=args.s2_top_k,
        search_top_k=args.s2_search_top_k,
        recommend_top_k=args.s2_recommend_top_k,
        per_keyword_limit=args.s2_per_keyword_limit,
        include_per_keyword_results=args.include_s2_per_keyword_results,
        keyword_model=args.keyword_model,
        keyword_api_url=args.keyword_api_url,
        keyword_timeout=args.keyword_timeout,
        search_timeout=args.search_timeout,
        search_retries=args.search_retries,
        recommendation_limit=args.recommendation_limit,
        grobid_base_url=args.grobid_base_url,
        grobid_start_page=args.grobid_start_page,
        use_env_proxy=args.use_env_proxy,
        disable_keywords=getattr(args, "disable_kg_keywords", False),
        pre_extracted_pdf=pre_extracted_pdf,
        pretty=False,
    )


def _capture_source(
    *,
    source_name: str,
    runner: Callable[[argparse.Namespace], Any],
    runner_args: argparse.Namespace,
    payload_builder: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        payload = payload_builder(runner(runner_args))
    except Exception as exc:
        return _build_source_error_payload(
            source_name,
            error_type=exc.__class__.__name__,
            error=str(exc),
            traceback_text=traceback.format_exc(),
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
        )

    payload["source"] = source_name
    payload["status"] = "ok"
    payload["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    return payload


def _build_kg_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        papers = payload.get("papers")
        authors = payload.get("authors")
    else:
        papers = payload
        authors = []

    if not isinstance(papers, list):
        raise TypeError("KG payload missing papers list")
    if not isinstance(authors, list):
        authors = []

    return {
        "paper_count": len(papers),
        "papers": papers,
        "author_count": len(authors),
        "authors": authors,
    }


def _build_s2_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    papers = result.get("papers")
    result["paper_count"] = len(papers) if isinstance(papers, list) else 0
    return result


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def normalize_title(text: str | None) -> str:
    cleaned = normalize_whitespace(text).casefold()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def tokenize_title(text: str | None) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_whitespace(text).casefold())
        if token and token not in TITLE_STOPWORDS
    }
    return tokens


def canonicalize_doi(value: Any) -> str:
    text = normalize_whitespace(str(value) if value is not None else "")
    if not text:
        return ""
    text = text.casefold()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip("/")


def extract_arxiv_id(value: Any) -> str:
    text = normalize_whitespace(str(value) if value is not None else "")
    if not text:
        return ""
    doi_match = re.search(r"arxiv[./:]([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text, flags=re.IGNORECASE)
    if doi_match:
        return doi_match.group(1).casefold()
    match = re.search(r"([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).casefold()
    return ""


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    text = normalize_whitespace(str(value))
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.match(r"^([0-9]{4})", text)
    if match:
        return int(match.group(1))
    return None


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def parse_loose_json(path: Path) -> dict[str, Any]:
    text = strip_ansi(path.read_text(encoding="utf-8"))
    stripped = text.lstrip()
    if stripped.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])
    if stripped and not stripped.startswith('"'):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    return json.loads("{" + text + "}")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Cannot parse JSON object from: {text!r}") from None
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object, got: {payload!r}")
    return payload


def load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = "" if value is None else str(value).strip()
        if text:
            return text
    return ""


def apply_env_values(env_path: Path) -> dict[str, str]:
    values = load_env_values(env_path)
    for key, value in values.items():
        if key and value:
            os.environ[key] = value

    api_key = first_non_empty(
        values.get("SEARCH_LLM_API_KEY"),
        values.get("DMX-API-KEY"),
        values.get("DMX_API_KEY"),
        values.get("OPENAI_API_KEY"),
        values.get("LLM_API_KEY"),
        os.environ.get("SEARCH_LLM_API_KEY"),
        os.environ.get("DMX-API-KEY"),
        os.environ.get("DMX_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("LLM_API_KEY"),
    )
    if api_key:
        os.environ.setdefault("SEARCH_LLM_API_KEY", api_key)
        os.environ.setdefault("DMX-API-KEY", api_key)

    api_url = first_non_empty(
        values.get("SEARCH_LLM_API_URL"),
        values.get("LLM_API_URL"),
        os.environ.get("SEARCH_LLM_API_URL"),
        os.environ.get("LLM_API_URL"),
    )
    if api_url:
        os.environ.setdefault("SEARCH_LLM_API_URL", api_url)

    model = first_non_empty(values.get("SEARCH_LLM_MODEL"), values.get("LLM_MODEL"), os.environ.get("SEARCH_LLM_MODEL"), os.environ.get("LLM_MODEL"))
    if model:
        os.environ.setdefault("SEARCH_LLM_MODEL", model)
    return values


def build_opener(env_values: dict[str, str], *, use_env_proxy: bool = False) -> urllib.request.OpenerDirector:
    if not use_env_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    proxy_map: dict[str, str] = {}
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        value = env_values.get(name) or os.environ.get(name)
        if not value:
            continue
        scheme = name.split("_", 1)[0].lower()
        proxy_map[scheme] = value

    if proxy_map:
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxy_map))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_cache_payload(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return parse_loose_json(path)


def cache_matches_args(cache_payload: dict[str, Any], args: argparse.Namespace) -> bool:
    if getattr(args, "idea_text", None):
        cached_text = normalize_whitespace(cache_payload.get("idea_text"))
        return bool(cached_text) and cached_text == normalize_whitespace(args.idea_text)

    if getattr(args, "pdf_path", None):
        cached_pdf_path = normalize_whitespace(cache_payload.get("pdf_path"))
        if not cached_pdf_path:
            return False
        return Path(cached_pdf_path).resolve() == Path(args.pdf_path).resolve()

    return False


def extract_cached_pdf(cache_payload: dict[str, Any]) -> dict[str, str] | None:
    raw_pdf = cache_payload.get("pdf")
    if not isinstance(raw_pdf, dict):
        return None
    return {
        "title": normalize_whitespace(raw_pdf.get("title")),
        "abstract": normalize_whitespace(raw_pdf.get("abstract")),
        "body": normalize_whitespace(raw_pdf.get("body")),
    }


def build_source_payload_from_cache(source_name: str, cache_payload: dict[str, Any]) -> dict[str, Any] | None:
    cached: Any = None
    sources = cache_payload.get("sources")
    if isinstance(sources, dict):
        cached = sources.get(source_name)
    if cached is None:
        cached = cache_payload.get(source_name)
    if cached is None:
        return None

    payload = dict(cached) if isinstance(cached, dict) else {"papers": cached}
    papers = payload.get("papers")
    if not isinstance(papers, list):
        return None
    if payload.get("status") not in {None, "ok"}:
        return None

    payload["source"] = source_name
    payload["status"] = "ok"
    payload["paper_count"] = len(papers)
    if source_name == "kg":
        authors = payload.get("authors")
        if not isinstance(authors, list):
            authors = []
        payload["authors"] = authors
        payload["author_count"] = len(authors)
    payload["elapsed_ms"] = round(float(payload.get("elapsed_ms") or 0.0), 1)
    payload["from_cache"] = True
    return payload


def _combine_source_papers(source_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for source_name in ("kg", "s2"):
        payload = source_payloads[source_name]
        if payload.get("status") != "ok":
            continue
        papers = payload.get("papers")
        if not isinstance(papers, list):
            continue
        for rank, paper in enumerate(papers, start=1):
            combined.append(
                {
                    "source": source_name,
                    "source_rank": rank,
                    "paper": paper,
                }
            )
    return combined


def extract_pdf_url(source: str, paper: dict[str, Any]) -> str:
    if source == "kg":
        pdf_url = normalize_whitespace(paper.get("pdf_url"))
        if pdf_url:
            return pdf_url

    open_access_pdf = paper.get("openAccessPdf")
    if isinstance(open_access_pdf, dict):
        url = normalize_whitespace(open_access_pdf.get("url"))
        if url:
            return url

    for key in ("pdf_url", "pdfUrl"):
        url = normalize_whitespace(paper.get(key))
        if url:
            return url

    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict):
        arxiv_id = extract_arxiv_id(external_ids.get("ArXiv")) or extract_arxiv_id(external_ids.get("DOI"))
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}"

    arxiv_id = extract_arxiv_id(paper.get("url")) or extract_arxiv_id(paper.get("id")) or extract_arxiv_id(paper.get("doi"))
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"
    return ""


def infer_paper_url(source: str, paper: dict[str, Any], pdf_url: str) -> str:
    for value in (paper.get("url"), paper.get("paper_url"), paper.get("id")):
        url = normalize_whitespace(value)
        if url:
            return url

    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict):
        arxiv_id = extract_arxiv_id(external_ids.get("ArXiv"))
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
        doi = canonicalize_doi(external_ids.get("DOI"))
        arxiv_id = extract_arxiv_id(doi)
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
        if doi:
            return f"https://doi.org/{doi}"

    doi = canonicalize_doi(paper.get("doi"))
    arxiv_id = extract_arxiv_id(doi)
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    if doi:
        return f"https://doi.org/{doi}"

    arxiv_id = extract_arxiv_id(pdf_url)
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"

    if source == "s2":
        paper_id = normalize_whitespace(paper.get("paperId"))
        if paper_id:
            return f"https://www.semanticscholar.org/paper/{paper_id}"

    return ""


def build_identifier_set(source: str, paper: dict[str, Any], pdf_url: str) -> list[str]:
    identifiers: set[str] = set()

    doi = canonicalize_doi(paper.get("doi"))
    if doi:
        identifiers.add(f"doi:{doi}")

    paper_url = normalize_whitespace(paper.get("url"))
    arxiv_from_url = extract_arxiv_id(paper_url) or extract_arxiv_id(pdf_url)
    if arxiv_from_url:
        identifiers.add(f"arxiv:{arxiv_from_url}")

    if source == "kg":
        openalex_id = normalize_whitespace(paper.get("id"))
        if openalex_id:
            identifiers.add(f"openalex:{openalex_id}")
    else:
        paper_id = normalize_whitespace(paper.get("paperId"))
        if paper_id:
            identifiers.add(f"s2:{paper_id}")
        corpus_id = normalize_whitespace(paper.get("corpusId"))
        if corpus_id:
            identifiers.add(f"corpus:{corpus_id}")

        external_ids = paper.get("externalIds")
        if isinstance(external_ids, dict):
            doi = canonicalize_doi(external_ids.get("DOI"))
            if doi:
                identifiers.add(f"doi:{doi}")
            arxiv_id = extract_arxiv_id(external_ids.get("ArXiv"))
            if arxiv_id:
                identifiers.add(f"arxiv:{arxiv_id}")

    return sorted(identifiers)


def build_candidate_record(
    item: dict[str, Any],
    *,
    candidate_index: int,
) -> dict[str, Any] | None:
    source = item["source"]
    paper = item["paper"]
    if not isinstance(paper, dict):
        return None

    title = normalize_whitespace(paper.get("title"))
    if not title:
        return None

    source_rank = int(item.get("source_rank") or candidate_index)
    abstract = normalize_whitespace(paper.get("abstract"))
    pdf_url = extract_pdf_url(source, paper)
    paper_url = infer_paper_url(source, paper, pdf_url)
    identifiers = build_identifier_set(source, paper, pdf_url)
    title_text = normalize_whitespace(title).casefold()
    citation_count = int(paper.get("citationCount") or paper.get("cited_by_count") or 0)

    return {
        "candidate_id": f"{source}-{source_rank:03d}-{candidate_index:03d}",
        "source": source,
        "source_rank": source_rank,
        "title": title,
        "title_text": title_text,
        "title_key": normalize_title(title),
        "title_tokens": tokenize_title(title),
        "abstract": abstract,
        "year": parse_year(paper.get("year") or paper.get("publication_year") or paper.get("publication_date")),
        "citation_count": citation_count,
        "pdf_url": pdf_url,
        "paper_url": paper_url,
        "identifiers": identifiers,
        "paper": paper,
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    source_priority = 1 if candidate["source"] == "s2" else 0
    has_doi = 1 if any(identifier.startswith("doi:") for identifier in candidate["identifiers"]) else 0
    return (
        1 if candidate["abstract"] else 0,
        len(candidate["abstract"]),
        source_priority,
        has_doi,
        candidate["citation_count"],
        -candidate["source_rank"],
        candidate["title_text"],
    )


def prefix_before_colon(text: str) -> str:
    if ":" not in text:
        return ""
    return normalize_title(text.split(":", 1)[0])


def detect_duplicate_evidence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    left_ids = set(left["identifiers"])
    right_ids = set(right["identifiers"])
    shared_ids = sorted(left_ids & right_ids)
    if shared_ids:
        strong_shared_ids = [
            identifier for identifier in shared_ids if identifier.startswith("doi:") or identifier.startswith("arxiv:")
        ]
        if strong_shared_ids:
            return {
                "kind": "shared_identifier",
                "shared_identifiers": strong_shared_ids,
                "sequence_ratio": 1.0,
                "token_jaccard": 1.0,
                "token_overlap": 1.0,
            }

    if left["title_key"] and left["title_key"] == right["title_key"]:
        return {
            "kind": "exact_title",
            "shared_identifiers": shared_ids,
            "sequence_ratio": 1.0,
            "token_jaccard": 1.0,
            "token_overlap": 1.0,
        }

    left_tokens = left["title_tokens"]
    right_tokens = right["title_tokens"]
    if not left_tokens or not right_tokens:
        return None

    shared_tokens = sorted(left_tokens & right_tokens)
    union_tokens = left_tokens | right_tokens
    min_token_count = min(len(left_tokens), len(right_tokens))
    token_jaccard = len(shared_tokens) / len(union_tokens)
    token_overlap = len(shared_tokens) / min_token_count
    sequence_ratio = SequenceMatcher(None, left["title_text"], right["title_text"]).ratio()
    year_gap = (
        abs(left["year"] - right["year"])
        if left.get("year") is not None and right.get("year") is not None
        else 0
    )
    prefix_match = prefix_before_colon(left["title"]) and prefix_before_colon(left["title"]) == prefix_before_colon(
        right["title"]
    )

    if sequence_ratio >= 0.97 and token_overlap >= 0.85:
        return {
            "kind": "near_exact_title",
            "shared_identifiers": shared_ids,
            "sequence_ratio": round(sequence_ratio, 4),
            "token_jaccard": round(token_jaccard, 4),
            "token_overlap": round(token_overlap, 4),
        }

    if (
        len(shared_tokens) >= 5
        and token_overlap >= 0.88
        and sequence_ratio >= 0.86
        and year_gap <= 1
    ):
        return {
            "kind": "high_token_overlap",
            "shared_identifiers": shared_ids,
            "shared_tokens": shared_tokens,
            "sequence_ratio": round(sequence_ratio, 4),
            "token_jaccard": round(token_jaccard, 4),
            "token_overlap": round(token_overlap, 4),
        }

    if prefix_match and token_overlap >= 0.8 and sequence_ratio >= 0.82 and year_gap <= 1:
        return {
            "kind": "matching_title_prefix",
            "shared_identifiers": shared_ids,
            "shared_tokens": shared_tokens,
            "sequence_ratio": round(sequence_ratio, 4),
            "token_jaccard": round(token_jaccard, 4),
            "token_overlap": round(token_overlap, 4),
        }

    return None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
            return
        if self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
            return
        self.parent[right_root] = left_root
        self.rank[left_root] += 1


def serialize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "source": candidate["source"],
        "source_rank": candidate["source_rank"],
        "title": candidate["title"],
        "abstract": candidate["abstract"],
        "year": candidate["year"],
        "citation_count": candidate["citation_count"],
        "pdf_url": candidate["pdf_url"],
        "paper_url": candidate["paper_url"],
        "identifiers": candidate["identifiers"],
        "paper": candidate["paper"],
    }


def build_unique_group_entry(group: dict[str, Any]) -> dict[str, Any]:
    representative = group["representative"]
    return {
        "group_id": group["group_id"],
        "source_set": sorted({member["source"] for member in group["all_members"]}),
        "variant_count": len(group["all_members"]),
        "candidate_count_with_pdf": len(group["members_with_pdf"]),
        "candidate_count_without_pdf": len(group["members_without_pdf"]),
        "has_pdf_url": bool(representative["pdf_url"]),
        "title": representative["title"],
        "abstract": representative["abstract"],
        "year": representative["year"],
        "citation_count": representative["citation_count"],
        "source": representative["source"],
        "source_rank": representative["source_rank"],
        "pdf_url": representative["pdf_url"],
        "paper_url": representative["paper_url"],
        "identifiers": representative["identifiers"],
        "paper": representative["paper"],
        "variants": [serialize_candidate(member) for member in group["all_members"]],
        "match_reasons": group["match_edges"],
    }


def group_sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    representative = group["representative"] or max(group["all_members"], key=candidate_sort_key)
    return (
        representative["source_rank"],
        -representative["citation_count"],
        representative["title_text"],
    )


def filter_and_group_papers(combined_papers: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    dropped_missing_title_count = 0

    for candidate_index, item in enumerate(combined_papers, start=1):
        candidate = build_candidate_record(item, candidate_index=candidate_index)
        if candidate is None:
            dropped_missing_title_count += 1
            continue
        candidates.append(candidate)

    if not candidates:
        return {
            "input_candidate_count": 0,
            "dropped_missing_title_count": dropped_missing_title_count,
            "duplicate_group_count": 0,
            "dropped_group_count_without_pdf": 0,
            "dropped_member_count_without_pdf": 0,
            "group_count_without_pdf": 0,
            "member_count_without_pdf": 0,
            "unique_paper_count_without_pdf": 0,
            "unique_paper_count": 0,
            "duplicate_groups": [],
            "unique_papers": [],
        }

    union_find = UnionFind(len(candidates))
    match_edges: list[dict[str, Any]] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            evidence = detect_duplicate_evidence(candidates[left_index], candidates[right_index])
            if evidence is None:
                continue
            union_find.union(left_index, right_index)
            match_edges.append(
                {
                    "left_candidate_id": candidates[left_index]["candidate_id"],
                    "right_candidate_id": candidates[right_index]["candidate_id"],
                    **evidence,
                }
            )

    members_by_root: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        members_by_root.setdefault(union_find.find(index), []).append(candidate)

    groups: list[dict[str, Any]] = []
    candidate_root_by_id = {
        candidate["candidate_id"]: union_find.find(index)
        for index, candidate in enumerate(candidates)
    }
    edges_by_root: dict[int, list[dict[str, Any]]] = {}
    for edge in match_edges:
        root = candidate_root_by_id[edge["left_candidate_id"]]
        edges_by_root.setdefault(root, []).append(edge)

    for root, members in members_by_root.items():
        sorted_members = sorted(members, key=candidate_sort_key, reverse=True)
        members_with_pdf = [member for member in sorted_members if member["pdf_url"]]
        members_without_pdf = [member for member in sorted_members if not member["pdf_url"]]
        representative = max(sorted_members, key=candidate_sort_key)
        groups.append(
            {
                "root": root,
                "all_members": sorted_members,
                "members_with_pdf": members_with_pdf,
                "members_without_pdf": members_without_pdf,
                "representative": representative,
                "match_edges": edges_by_root.get(root, []),
            }
        )

    groups.sort(key=group_sort_key)
    for group_index, group in enumerate(groups, start=1):
        group["group_id"] = f"group-{group_index:03d}"

    unique_papers = [build_unique_group_entry(group) for group in groups if group["representative"] is not None]
    duplicate_groups = [
        {
            "group_id": group["group_id"],
            "candidate_count": len(group["all_members"]),
            "kept_candidate_count": len(group["members_with_pdf"]),
            "candidate_count_without_pdf": len(group["members_without_pdf"]),
            "source_set": sorted({member["source"] for member in group["all_members"]}),
            "representative": serialize_candidate(group["representative"]) if group["representative"] else None,
            "members": [serialize_candidate(member) for member in group["all_members"]],
            "members_without_pdf": [serialize_candidate(member) for member in group["members_without_pdf"]],
            "match_reasons": group["match_edges"],
        }
        for group in groups
    ]

    return {
        "input_candidate_count": len(candidates),
        "dropped_missing_title_count": dropped_missing_title_count,
        "duplicate_group_count": len(groups),
        "group_count_without_pdf": sum(1 for group in groups if not group["members_with_pdf"]),
        "member_count_without_pdf": sum(len(group["members_without_pdf"]) for group in groups),
        "dropped_group_count_without_pdf": 0,
        "dropped_member_count_without_pdf": 0,
        "unique_paper_count_without_pdf": sum(1 for paper in unique_papers if not paper["has_pdf_url"]),
        "unique_paper_count": len(unique_papers),
        "duplicate_groups": duplicate_groups,
        "unique_papers": unique_papers,
    }


def build_query_context(
    args: argparse.Namespace,
    source_payloads: dict[str, dict[str, Any]],
    cached_pdf: dict[str, str] | None,
) -> dict[str, Any]:
    if getattr(args, "idea_text", None):
        idea_text = normalize_whitespace(args.idea_text)
        return {
            "input_type": "idea_text",
            "title": "",
            "abstract": idea_text,
            "text": idea_text,
        }

    pdf_payload: dict[str, Any] | None = None
    s2_payload = source_payloads.get("s2", {})
    if isinstance(s2_payload, dict):
        maybe_pdf = s2_payload.get("pdf")
        if isinstance(maybe_pdf, dict):
            pdf_payload = maybe_pdf
    if pdf_payload is None:
        pdf_payload = cached_pdf

    if not isinstance(pdf_payload, dict):
        raise ValueError("PDF input requires extracted title/abstract, but none was found in cache or S2 payload")

    title = normalize_whitespace(pdf_payload.get("title"))
    abstract = normalize_whitespace(pdf_payload.get("abstract"))
    body = normalize_whitespace(pdf_payload.get("body"))
    return {
        "input_type": "pdf",
        "title": title,
        "abstract": abstract,
        "text": abstract or body or title,
    }


def build_scoring_batches(
    *,
    paper_count: int,
    batch_size: int,
    paper_coverage: int,
    seed: int,
) -> list[dict[str, Any]]:
    if paper_count <= 0:
        return []

    if paper_count <= batch_size:
        return [
            {
                "batch_id": "batch-001",
                "round": 1,
                "paper_indices": list(range(paper_count)),
            }
        ]

    effective_batch_size = max(2, batch_size)
    effective_coverage = max(1, paper_coverage)
    batch_count_per_round = math.ceil(paper_count / effective_batch_size)
    rng = random.Random(seed)
    batches: list[dict[str, Any]] = []

    for round_index in range(effective_coverage):
        shuffled = list(range(paper_count))
        rng.shuffle(shuffled)
        round_batches: list[list[int]] = [[] for _ in range(batch_count_per_round)]
        for position, paper_index in enumerate(shuffled):
            round_batches[position % batch_count_per_round].append(paper_index)
        for paper_indices in round_batches:
            batches.append(
                {
                    "batch_id": f"batch-{len(batches) + 1:03d}",
                    "round": round_index + 1,
                    "paper_indices": paper_indices,
                }
            )

    return batches


def build_relevance_prompt(query_context: dict[str, Any], batch_papers: list[dict[str, Any]]) -> str:
    query_lines = []
    if query_context["input_type"] == "idea_text":
        query_lines.append("Query type: idea_text")
        query_lines.append(f"Query text:\n{query_context['text']}")
    else:
        query_lines.append("Query type: paper_abstract")
        if query_context.get("title"):
            query_lines.append(f"Query paper title: {query_context['title']}")
        query_lines.append(f"Query paper abstract:\n{query_context['text']}")

    paper_blocks = []
    for slot, paper in enumerate(batch_papers, start=1):
        abstract = paper["abstract"] or "[missing abstract]"
        paper_blocks.append(f"[{slot}] Title: {paper['title']}\nAbstract: {abstract}")

    return "\n\n".join(
        [
            "Evaluate how substantively relevant each candidate paper is to the query.",
            "",
            "Scoring target:",
            "- Judge each paper independently, not comparatively.",
            "- Focus on overlap in the core research problem, evaluation target, inputs/outputs, and methodological purpose.",
            "- A paper can score high even if the method differs, as long as it directly addresses the same problem.",
            "- Penalize papers that only share broad themes such as LLMs, scientific discovery, benchmarks, or evaluation.",
            "- Ignore citation counts, venue prestige, popularity, and writing polish.",
            "",
            "Rubric:",
            "- 9-10: Extremely close neighbor; essentially the same problem or an immediately adjacent paper you would definitely cite.",
            "- 7-8: Directly relevant; strong task or evaluation overlap.",
            "- 5-6: Meaningful partial overlap, but not the same core problem.",
            "- 3-4: Weak topical relation only.",
            "- 0-2: Effectively irrelevant.",
            "",
            "Return ONLY JSON in this format:",
            '{"papers":[{"paper_index":1,"score":8.7,"reason":"short reason"}]}',
            "",
            "Requirements for output:",
            "- Include every paper exactly once.",
            "- score must be a number in [0, 10].",
            "- reason must be concise and mention the main match or mismatch.",
            "",
            "\n".join(query_lines),
            "Candidate papers:\n" + "\n\n".join(paper_blocks),
        ]
    )


class DmxPaperBatchScorer:
    def __init__(
        self,
        *,
        env_values: dict[str, str],
        model: str,
        timeout: int,
        max_tokens: int,
        temperature: float,
        use_env_proxy: bool,
    ) -> None:
        api_key = first_non_empty(
            env_values.get("SEARCH_LLM_API_KEY"),
            env_values.get("DMX-API-KEY"),
            env_values.get("DMX_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
            env_values.get("LLM_API_KEY"),
            os.environ.get("SEARCH_LLM_API_KEY"),
            os.environ.get("DMX-API-KEY"),
            os.environ.get("DMX_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("LLM_API_KEY"),
        )
        if not api_key:
            raise ValueError("LLM API key not found in env file or environment")
        self.api_key = api_key
        self.api_url = first_non_empty(
            env_values.get("SEARCH_LLM_API_URL"),
            env_values.get("LLM_API_URL"),
            os.environ.get("SEARCH_LLM_API_URL"),
            os.environ.get("LLM_API_URL"),
            DMX_API_URL,
        )
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.opener = build_opener(env_values, use_env_proxy=use_env_proxy)

    def score_batch(self, *, query_context: dict[str, Any], batch_papers: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = build_relevance_prompt(query_context, batch_papers)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        started_at = time.perf_counter()
        for attempt in range(4):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                response_payload = json.loads(body)
                content = response_payload["choices"][0]["message"]["content"]
                parsed_scores = parse_batch_score_response(content, expected_size=len(batch_papers))
                return {
                    "prompt": prompt,
                    "content": content,
                    "scores": parsed_scores,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"DMX HTTP {exc.code}: {detail}")
                if attempt < 3 and exc.code in {429, 500, 502, 503, 504}:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"DMX batch scoring failed: {last_error}")


def parse_batch_score_response(content: str, *, expected_size: int) -> list[dict[str, Any]]:
    payload = parse_json_object(content)
    values = payload.get("papers") or payload.get("scores")
    if not isinstance(values, list):
        raise ValueError(f"Missing paper list in batch response: {payload!r}")

    parsed: dict[int, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        paper_index = item.get("paper_index")
        if paper_index is None:
            paper_index = item.get("index")
        try:
            slot = int(paper_index)
        except (TypeError, ValueError):
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if slot < 1 or slot > expected_size:
            continue
        score = max(0.0, min(10.0, score))
        parsed[slot] = {
            "paper_index": slot,
            "score": score,
            "reason": normalize_whitespace(item.get("reason")),
        }

    missing_slots = [slot for slot in range(1, expected_size + 1) if slot not in parsed]
    if missing_slots:
        raise ValueError(f"Batch response missing paper indices: {missing_slots}")

    return [parsed[slot] for slot in range(1, expected_size + 1)]


def run_scoring_batch(
    *,
    scorer: DmxPaperBatchScorer,
    query_context: dict[str, Any],
    batch: dict[str, Any],
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_papers = [papers[index] for index in batch["paper_indices"]]
    request_papers = [
        {
            "paper_index": slot,
            "group_id": paper["group_id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
        }
        for slot, paper in enumerate(batch_papers, start=1)
    ]
    scored = scorer.score_batch(query_context=query_context, batch_papers=request_papers)
    return {
        "batch_id": batch["batch_id"],
        "round": batch["round"],
        "papers": request_papers,
        **scored,
    }


def compute_score_std(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return variance ** 0.5


def ranking_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -item["llm_score"],
        -item["score_count"],
        -item["citation_count"],
        item["title"].casefold(),
    )


def score_unique_papers_with_llm(
    *,
    args: argparse.Namespace,
    query_context: dict[str, Any],
    unique_papers: list[dict[str, Any]],
) -> dict[str, Any]:
    if not unique_papers:
        return {
            "status": "skipped",
            "reason": "no_unique_papers_after_filter",
            "papers": [],
        }

    seed_input = f"{query_context['input_type']}::{query_context['title']}::{query_context['text']}"
    seed = int(hashlib.sha1(seed_input.encode("utf-8")).hexdigest()[:8], 16)
    batches = build_scoring_batches(
        paper_count=len(unique_papers),
        batch_size=max(2, int(getattr(args, "llm_batch_size", DEFAULT_LLM_BATCH_SIZE))),
        paper_coverage=max(1, int(getattr(args, "llm_paper_coverage", DEFAULT_LLM_PAPER_COVERAGE))),
        seed=seed,
    )

    env_values = load_env_values(Path(args.env))
    scorer = DmxPaperBatchScorer(
        env_values=env_values,
        model=getattr(args, "llm_model", DEFAULT_DMX_MODEL),
        timeout=int(getattr(args, "llm_timeout", 60)),
        max_tokens=int(getattr(args, "llm_max_tokens", 900)),
        temperature=float(getattr(args, "llm_temperature", 0.1)),
        use_env_proxy=bool(getattr(args, "use_env_proxy", False)),
    )

    batch_results: list[dict[str, Any]] = []
    batch_errors: list[dict[str, Any]] = []
    max_parallel = max(1, min(int(getattr(args, "llm_max_parallel", 20)), len(batches)))

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {
            executor.submit(
                run_scoring_batch,
                scorer=scorer,
                query_context=query_context,
                batch=batch,
                papers=unique_papers,
            ): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            batch = future_map[future]
            try:
                batch_results.append(future.result())
            except Exception as exc:
                batch_errors.append(
                    {
                        "batch_id": batch["batch_id"],
                        "round": batch["round"],
                        "paper_indices": list(batch["paper_indices"]),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                )

    batch_results.sort(key=lambda item: item["batch_id"])
    score_details_by_group: dict[str, list[dict[str, Any]]] = {paper["group_id"]: [] for paper in unique_papers}
    for batch_result in batch_results:
        papers_by_slot = {paper["paper_index"]: paper for paper in batch_result["papers"]}
        for score_item in batch_result["scores"]:
            paper = papers_by_slot[score_item["paper_index"]]
            score_details_by_group[paper["group_id"]].append(
                {
                    "batch_id": batch_result["batch_id"],
                    "round": batch_result["round"],
                    "score": round(float(score_item["score"]), 4),
                    "reason": score_item["reason"],
                }
            )

    ranked_papers: list[dict[str, Any]] = []
    unscored_group_ids: list[str] = []
    for paper in unique_papers:
        group_id = paper["group_id"]
        details = score_details_by_group.get(group_id, [])
        if not details:
            unscored_group_ids.append(group_id)
            continue
        scores = [detail["score"] for detail in details]
        ranked_paper = dict(paper)
        ranked_paper["llm_score"] = round(sum(scores) / len(scores), 4)
        ranked_paper["score_count"] = len(scores)
        ranked_paper["score_std"] = round(compute_score_std(scores), 4)
        ranked_paper["reasons"] = [detail["reason"] for detail in details]
        ranked_paper["score_details"] = details
        ranked_papers.append(ranked_paper)

    ranked_papers.sort(key=ranking_sort_key)
    for rank, paper in enumerate(ranked_papers, start=1):
        paper["rank"] = rank

    final_top_k = getattr(args, "final_top_k", None)
    returned_papers = ranked_papers[:final_top_k] if final_top_k is not None else ranked_papers

    if ranked_papers and not batch_errors and not unscored_group_ids:
        status = "ok"
    elif ranked_papers:
        status = "partial_error"
    else:
        status = "error"

    return {
        "status": status,
        "strategy": "dmx_batch_mean",
        "model": getattr(args, "llm_model", DEFAULT_DMX_MODEL),
        "batch_size": max(2, int(getattr(args, "llm_batch_size", DEFAULT_LLM_BATCH_SIZE))),
        "paper_coverage": 1
        if len(unique_papers) <= max(2, int(getattr(args, "llm_batch_size", DEFAULT_LLM_BATCH_SIZE)))
        else max(1, int(getattr(args, "llm_paper_coverage", DEFAULT_LLM_PAPER_COVERAGE))),
        "batch_count": len(batches),
        "full_paper_count": len(ranked_papers),
        "returned_paper_count": len(returned_papers),
        "prompt_system": LLM_SYSTEM_PROMPT,
        "papers": returned_papers,
        "all_papers": ranked_papers,
        "batches": batch_results,
        "batch_errors": batch_errors,
        "unscored_group_ids": unscored_group_ids,
    }


def merge_status(source_status: str, ranking_status: str | None) -> str:
    if source_status == "error":
        return "error"
    if ranking_status in {"error", "partial_error"}:
        return "partial_error"
    return source_status


def persist_result_payload(tag: str, payload: dict[str, Any]) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULT_DIR / f"{stamp}_{tag}.json"

    output = dict(payload)
    output["artifacts"] = {
        "json": str(summary_path),
        "markdown": str(RESULT_MARKDOWN),
    }

    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    ranking = output.get("ranking", {})
    top_titles = [paper.get("title") for paper in ranking.get("papers", [])[:10]]
    cache_info = output.get("cache", {})
    lines = [
        f"## {stamp} `{tag}`",
        "",
        f"- status: `{output.get('status')}`",
        f"- input_type: `{output.get('input_type')}`",
        f"- cache_path: `{cache_info.get('path')}`",
        f"- cache_reused_sources: {json.dumps(cache_info.get('reused_sources', []), ensure_ascii=False)}",
        f"- cache_reused_pdf: `{cache_info.get('reused_pdf')}`",
        f"- source_successful_count: `{output.get('successful_source_count')}`",
        f"- unique_paper_count: `{output.get('filter', {}).get('unique_paper_count')}`",
        f"- ranking_status: `{ranking.get('status')}`",
        f"- ranking_batch_count: `{ranking.get('batch_count')}`",
        f"- llm_model: `{ranking.get('model')}`",
        f"- top_titles: {json.dumps(top_titles, ensure_ascii=False)}",
        "",
    ]
    with RESULT_MARKDOWN.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return output


def run_combined_search(args: argparse.Namespace) -> dict[str, Any]:
    apply_env_values(Path(args.env).expanduser().resolve())
    cache_info: dict[str, Any] = {
        "path": str(getattr(args, "cache_path", "")),
        "matched_input": False,
        "reused_sources": [],
        "reused_pdf": False,
    }
    cached_payload: dict[str, Any] | None = None
    cached_pdf: dict[str, str] | None = None

    if not bool(getattr(args, "disable_cache_reuse", False)):
        try:
            cached_payload = load_cache_payload(getattr(args, "cache_path", None))
        except Exception as exc:
            cache_info["error"] = str(exc)
            cached_payload = None
        if cached_payload is not None:
            cache_info["matched_input"] = cache_matches_args(cached_payload, args)
            if cache_info["matched_input"]:
                cached_pdf = extract_cached_pdf(cached_payload)
                cache_info["reused_pdf"] = cached_pdf is not None

    tasks: dict[str, tuple[Callable[[argparse.Namespace], Any], argparse.Namespace, Callable[[Any], dict[str, Any]]]] = {}
    source_payloads: dict[str, dict[str, Any]] = {}

    if cached_payload is not None and cache_info["matched_input"]:
        cached_kg = build_source_payload_from_cache("kg", cached_payload)
        if cached_kg is not None:
            source_payloads["kg"] = cached_kg
            cache_info["reused_sources"].append("kg")

        if bool(getattr(args, "reuse_cached_s2", False)):
            cached_s2 = build_source_payload_from_cache("s2", cached_payload)
            if cached_s2 is not None:
                source_payloads["s2"] = cached_s2
                cache_info["reused_sources"].append("s2")

    if "kg" not in source_payloads:
        kg_top_k = int(getattr(args, "kg_top_k", 0) or 0)
        if kg_top_k <= 0:
            source_payloads["kg"] = _build_source_error_payload(
                "kg",
                error_type="SourceDisabled",
                error="KG search skipped because kg_top_k <= 0.",
            )
        else:
            try:
                run_kg_search = _load_kg_runner()
            except Exception as exc:
                source_payloads["kg"] = _build_source_error_payload(
                    "kg",
                    error_type="DependencyUnavailable",
                    error=f"KG search unavailable: {exc}",
                    traceback_text=traceback.format_exc(),
                )
            else:
                tasks["kg"] = (
                    run_kg_search,
                    build_kg_args(args),
                    _build_kg_payload,
                )

    if "s2" not in source_payloads:
        tasks["s2"] = (
            s2_search.run_pipeline,
            build_s2_args(args, pre_extracted_pdf=cached_pdf),
            _build_s2_payload,
        )

    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_map = {
                executor.submit(
                    _capture_source,
                    source_name=source_name,
                    runner=runner,
                    runner_args=runner_args,
                    payload_builder=payload_builder,
                ): source_name
                for source_name, (runner, runner_args, payload_builder) in tasks.items()
            }
            for future in as_completed(future_map):
                source_name = future_map[future]
                source_payloads[source_name] = future.result()

    for source_name in ("kg", "s2"):
        if source_name not in source_payloads:
            source_payloads[source_name] = _build_source_error_payload(
                source_name,
                error_type="MissingSourcePayload",
                error=f"{source_name} payload missing",
            )

    combined_papers = _combine_source_papers(source_payloads)
    filter_payload = filter_and_group_papers(combined_papers)

    ranking_payload: dict[str, Any]
    if bool(getattr(args, "disable_llm_ranking", False)):
        ranking_payload = {
            "status": "skipped",
            "reason": "llm_disabled",
            "papers": filter_payload["unique_papers"],
        }
    else:
        try:
            query_context = build_query_context(args, source_payloads, cached_pdf)
            ranking_payload = score_unique_papers_with_llm(
                args=args,
                query_context=query_context,
                unique_papers=filter_payload["unique_papers"],
            )
        except Exception as exc:
            ranking_payload = {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "papers": [],
            }

    success_count = sum(1 for payload in source_payloads.values() if payload.get("status") == "ok")
    failed_count = 2 - success_count
    if success_count == 2:
        source_status = "ok"
    elif success_count == 0:
        source_status = "error"
    else:
        source_status = "partial_error"

    result: dict[str, Any] = {
        "status": merge_status(source_status, ranking_payload.get("status")),
        "input_type": "pdf" if getattr(args, "pdf_path", None) else "idea_text",
        "successful_source_count": success_count,
        "failed_source_count": failed_count,
        "cache": cache_info,
        "sources": {
            "kg": source_payloads["kg"],
            "s2": source_payloads["s2"],
        },
        "combined": {
            "paper_count": len(combined_papers),
            "papers": combined_papers,
        },
        "filter": filter_payload,
        "ranking": ranking_payload,
    }
    if getattr(args, "idea_text", None):
        result["idea_text"] = args.idea_text
    if getattr(args, "pdf_path", None):
        result["pdf_path"] = str(Path(args.pdf_path).resolve())

    if not bool(getattr(args, "disable_result_log", False)):
        result = persist_result_payload(getattr(args, "result_tag", "merge_search"), result)

    return result


def main() -> int:
    args = build_parser().parse_args()
    payload = run_combined_search(args)
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["successful_source_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
