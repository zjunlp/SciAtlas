#!/usr/bin/env python3
"""Standalone Semantic Scholar search pipeline for idea text or PDF input."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
INNOEVAL_DIR = MODULE_DIR.parent
PROJECT_ROOT = INNOEVAL_DIR.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
RECO_BASE = "https://api.semanticscholar.org/recommendations/v1"
DEFAULT_GROBID_BASE_URL = "http://127.0.0.1:8070"
DEFAULT_TIMEOUT = 30

RICH_PAPER_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "publicationVenue",
        "year",
        "referenceCount",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
        "publicationDate",
        "journal",
        "citationStyles",
        "authors",
    ]
)


for extra_path in (INNOEVAL_DIR, MODULE_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from kw_extractor import (  # noqa: E402
    API_URL as KEYWORD_API_URL,
    DEFAULT_MODEL as DEFAULT_KEYWORD_MODEL,
    KeywordExtractor,
    KeywordExtractorConfig,
)
from kg_search.pdf_extraction import extract_pdf  # noqa: E402


class SemanticScholarSearchError(RuntimeError):
    """Raised when Semantic Scholar search fails."""


@dataclass(frozen=True)
class SearchConfig:
    env_path: Path = DEFAULT_ENV_PATH
    timeout: int = DEFAULT_TIMEOUT
    retries: int = 3
    use_env_proxy: bool = False


def load_s2_api_keys(env_path: Path) -> list[str]:
    candidates = ("S2-API-KEY", "S2_API_KEY", "S2-API-KEY1", "S2-API-KEY2")
    values: dict[str, str] = {key: os.environ.get(key, "") for key in candidates}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    api_keys: list[str] = []
    for key_name in candidates:
        value = values.get(key_name)
        if value and value not in api_keys:
            api_keys.append(value)
    if api_keys:
        return api_keys

    raise ValueError(
        f"Missing Semantic Scholar API key. Set one of {', '.join(candidates)} "
        f"in the environment or {env_path}."
    )


def load_s2_api_key(env_path: Path) -> str:
    return load_s2_api_keys(env_path)[0]


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def flatten_pdf_body(sections: list[Any]) -> str:
    blocks: list[str] = []

    def visit(items: list[Any]) -> None:
        for section in items:
            heading = normalize_whitespace(getattr(section, "heading", ""))
            if heading:
                blocks.append(heading)

            for paragraph in getattr(section, "paragraphs", []):
                text = normalize_whitespace(paragraph)
                if text:
                    blocks.append(text)

            visit(list(getattr(section, "subsections", [])))

    visit(sections)
    return "\n\n".join(blocks)


def build_keyword_input(idea_text: str | None, pdf_title: str, pdf_abstract: str) -> str:
    if idea_text is not None:
        text = normalize_whitespace(idea_text)
        if not text:
            raise ValueError("idea_text must not be empty")
        return text

    title = normalize_whitespace(pdf_title)
    abstract = normalize_whitespace(pdf_abstract)
    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    if abstract:
        return abstract
    if title:
        return title
    raise ValueError("PDF title and abstract are empty; cannot extract search seed phrases")


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(needle.casefold() in lowered for needle in needles)


def extract_title_task_phrase(title: str) -> str | None:
    normalized_title = normalize_whitespace(title)
    if not normalized_title:
        return None

    candidate_source = normalized_title
    if ":" in normalized_title:
        prefix, suffix = normalized_title.split(":", 1)
        if len(prefix.split()) <= 4 and normalize_whitespace(suffix):
            candidate_source = normalize_whitespace(suffix)

    for pattern in (
        r"(?i)\b(?:on|towards?|toward)\s+(.+?)\s+as\b",
        r"(?i)\b(?:on|towards?|toward)\s+(.+?)\s+for\b",
    ):
        match = re.search(pattern, candidate_source)
        if match:
            candidate_source = normalize_whitespace(match.group(1))
            break

    lowered = candidate_source.casefold()
    for pattern in (
        r"\bresearch idea evaluation\b",
        r"\bscientific idea evaluation\b",
        r"\bidea evaluation\b",
        r"\bscientific idea generation\b",
        r"\bresearch idea generation\b",
        r"\bscientific idea judgments?\b",
        r"\bnovelty assessment\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            return match.group(0)

    words = re.findall(r"[A-Za-z]+", candidate_source)
    if 2 <= len(words) <= 5:
        return " ".join(word.casefold() for word in words)
    return None


def is_viable_search_seed(phrase: str) -> bool:
    normalized = normalize_whitespace(phrase).casefold()
    if not normalized:
        return False
    if normalized in {
        "large language models",
        "knowledge grounding",
        "scientific evaluation",
        "multi-criteria decision making",
        "innovation assessment",
        "heterogeneous knowledge retrieval",
        "peer review benchmarking",
    }:
        return False
    if any(
        blocked in normalized
        for blocked in (
            "framework",
            "multi-perspective",
            "review board",
            "consensus",
            "knowledge grounded reasoning",
        )
    ):
        return False
    return any(token in normalized for token in ("idea", "scientific", "research", "novelty", "judgment", "benchmark"))


def compose_search_queries(
    seed_phrases: list[str],
    *,
    title: str,
    abstract: str,
) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    context = f"{normalize_whitespace(title)}\n{normalize_whitespace(abstract)}".casefold()

    def maybe_add(query: str | None) -> None:
        if not query:
            return
        normalized = normalize_whitespace(query).casefold()
        if not normalized or normalized in seen:
            return
        if normalized == "idea evaluation" and any(
            existing in {"research idea evaluation", "scientific idea evaluation"} for existing in queries
        ):
            return
        seen.add(normalized)
        queries.append(normalized)

    maybe_add(extract_title_task_phrase(title))
    for phrase in seed_phrases:
        if is_viable_search_seed(phrase):
            maybe_add(phrase)

    if contains_any(context, ("research idea",)) and any("idea evaluation" in query for query in queries):
        maybe_add("research idea evaluation")

    idea_eval_in_context = contains_any(context, ("research idea", "scientific idea", "scientific ideas")) or any(
        "idea evaluation" in query for query in queries
    )

    if idea_eval_in_context and contains_any(context, ("large language model", "large language models", "llm", "llms")):
        maybe_add("idea evaluation large language models")

    if idea_eval_in_context and contains_any(
        context,
        ("benchmark", "point-wise", "pair-wise", "group-wise", "judgment patterns", "human experts"),
    ):
        maybe_add("scientific idea evaluation benchmark")
    elif idea_eval_in_context and contains_any(context, ("judgment", "judge", "judgments")):
        maybe_add("scientific idea judgment")

    if idea_eval_in_context and contains_any(
        context,
        ("knowledge-grounded", "knowledge grounded", "knowledgeable grounding", "literature", "peer-reviewed", "papers"),
    ):
        maybe_add("literature grounded scientific ideas")

    if not queries:
        for phrase in seed_phrases:
            maybe_add(phrase)

    return queries[:4]


class SemanticScholarSearchClient:
    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.api_keys = load_s2_api_keys(self.config.env_path)
        self.api_key_index = 0
        self.opener = self._build_opener()

    def _build_opener(self) -> urllib.request.OpenerDirector:
        if self.config.use_env_proxy:
            return urllib.request.build_opener()
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def search_papers(self, query: str, limit: int) -> dict[str, Any]:
        normalized_query = normalize_whitespace(query)
        if not normalized_query:
            raise ValueError("search query must not be empty")

        params = urllib.parse.urlencode(
            {
                "query": normalized_query,
                "limit": limit,
                "fields": RICH_PAPER_FIELDS,
            }
        )
        url = f"{GRAPH_BASE}/paper/search?{params}"
        payload = self._request_json(url)

        papers = payload.get("data")
        if papers is None and int(payload.get("total") or 0) == 0:
            payload["data"] = []
            return payload
        if not isinstance(papers, list):
            raise SemanticScholarSearchError(f"Unexpected search payload for query={normalized_query!r}: {payload!r}")
        return payload

    def paper_match_by_title(self, title: str) -> dict[str, Any]:
        normalized_title = normalize_whitespace(title)
        if not normalized_title:
            raise ValueError("title must not be empty")

        params = urllib.parse.urlencode(
            {
                "query": normalized_title,
                "fields": RICH_PAPER_FIELDS,
            }
        )
        url = f"{GRAPH_BASE}/paper/search/match?{params}"
        payload = self._request_json(url)

        paper_id = normalize_whitespace(payload.get("paperId"))
        if paper_id:
            return payload

        data = payload.get("data")
        if isinstance(data, list) and data:
            candidate = data[0]
            if isinstance(candidate, dict):
                return candidate

        raise SemanticScholarSearchError(f"No matched paper found for title={normalized_title!r}")

    def recommend_papers(self, seed_paper_id: str, limit: int, source_pool: str = "all-cs") -> dict[str, Any]:
        normalized_paper_id = normalize_whitespace(seed_paper_id)
        if not normalized_paper_id:
            raise ValueError("seed_paper_id must not be empty")

        params = urllib.parse.urlencode(
            {
                "from": source_pool,
                "limit": limit,
                "fields": RICH_PAPER_FIELDS,
            }
        )
        url = f"{RECO_BASE}/papers/forpaper/{urllib.parse.quote(normalized_paper_id)}?{params}"
        payload = self._request_json(url)
        recommended_papers = payload.get("recommendedPapers")
        if not isinstance(recommended_papers, list):
            raise SemanticScholarSearchError(
                f"Unexpected recommendation payload for paper_id={normalized_paper_id!r}: {payload!r}"
            )
        return payload

    def _request_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        api_keys = self.api_keys
        key_count = len(api_keys)
        for attempt in range(self.config.retries + 1):
            for offset in range(key_count):
                api_key = api_keys[(self.api_key_index + offset) % key_count]
                request = urllib.request.Request(
                    url,
                    headers={
                        "accept": "application/json",
                        "user-agent": "search-s2/1.0",
                        "x-api-key": api_key,
                    },
                    method="GET",
                )
                try:
                    with self.opener.open(request, timeout=self.config.timeout) as response:
                        body = response.read().decode("utf-8")
                        self.api_key_index = (self.api_key_index + offset) % key_count
                        return json.loads(body)
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    last_error = SemanticScholarSearchError(f"HTTP {exc.code} for {url}: {detail}")
                    if exc.code == 429:
                        retry_after = exc.headers.get("Retry-After")
                        sleep_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                        time.sleep(sleep_seconds)
                        continue
                    raise last_error
                except urllib.error.URLError as exc:
                    last_error = SemanticScholarSearchError(f"Request failed for {url}: {exc}")
                    if attempt < self.config.retries:
                        time.sleep(2 ** attempt)
                        continue
                    raise last_error
                except json.JSONDecodeError as exc:
                    raise SemanticScholarSearchError(f"Invalid JSON response from {url}") from exc

        raise SemanticScholarSearchError(f"Request failed for {url}: {last_error}")


def extract_pdf_payload(
    pdf_path: Path,
    *,
    grobid_base_url: str,
    grobid_start_page: int | None,
) -> dict[str, str]:
    document = extract_pdf(
        pdf_path,
        base_url=grobid_base_url,
        start_page=grobid_start_page,
    )
    return {
        "title": normalize_whitespace(document.title),
        "abstract": normalize_whitespace(document.abstract),
        "body": flatten_pdf_body(document.body),
    }


def aggregate_keyword_searches(
    client: SemanticScholarSearchClient,
    keywords: list[str],
    per_keyword_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keyword_results: list[dict[str, Any]] = []
    merged: dict[str, dict[str, Any]] = {}
    anonymous_counter = 0

    for keyword in keywords:
        response = client.search_papers(keyword, per_keyword_limit)
        papers = response.get("data", [])
        keyword_results.append(
            {
                "keyword": keyword,
                "total": response.get("total"),
                "count": len(papers),
                "papers": papers,
            }
        )

        for rank, paper in enumerate(papers, start=1):
            anonymous_counter = merge_paper(
                merged,
                paper,
                source="keyword_search",
                rank=rank,
                keyword=keyword,
                anonymous_counter=anonymous_counter,
            )

    papers = sort_merged_papers(merged)
    return papers, keyword_results


def merge_paper(
    merged: dict[str, dict[str, Any]],
    paper: Any,
    *,
    source: str,
    rank: int | None,
    anonymous_counter: int,
    keyword: str | None = None,
) -> int:
    if not isinstance(paper, dict):
        return anonymous_counter

    paper_id = normalize_whitespace(paper.get("paperId"))
    if not paper_id:
        title = normalize_whitespace(paper.get("title"))
        paper_id = f"__anon__::{title or 'untitled'}::{anonymous_counter}"
        anonymous_counter += 1

    current = merged.get(paper_id)
    if current is None:
        paper_copy = dict(paper)
        paper_copy["_retrieval"] = {
            "sources": [source],
            "matched_keywords": [keyword] if keyword else [],
            "keyword_hit_count": 1 if keyword else 0,
            "best_keyword_rank": rank if keyword and rank is not None else None,
            "recommendation_rank": rank if source == "title_recommendation" else None,
        }
        merged[paper_id] = paper_copy
        return anonymous_counter

    retrieval = current.setdefault("_retrieval", {})
    sources = retrieval.setdefault("sources", [])
    if source not in sources:
        sources.append(source)

    matched_keywords = retrieval.setdefault("matched_keywords", [])
    if keyword and keyword not in matched_keywords:
        matched_keywords.append(keyword)
    retrieval["keyword_hit_count"] = len(matched_keywords)

    if keyword and rank is not None:
        best_keyword_rank = retrieval.get("best_keyword_rank")
        retrieval["best_keyword_rank"] = rank if best_keyword_rank is None else min(int(best_keyword_rank), rank)

    if source == "title_recommendation" and rank is not None:
        recommendation_rank = retrieval.get("recommendation_rank")
        retrieval["recommendation_rank"] = rank if recommendation_rank is None else min(int(recommendation_rank), rank)

    return anonymous_counter


def sort_merged_papers(merged: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    papers = list(merged.values())
    papers.sort(
        key=lambda item: (
            -int(item.get("_retrieval", {}).get("keyword_hit_count", 0)),
            -len(item.get("_retrieval", {}).get("sources", [])),
            _sortable_rank(item.get("_retrieval", {}).get("best_keyword_rank")),
            _sortable_rank(item.get("_retrieval", {}).get("recommendation_rank")),
            -int(item.get("citationCount") or 0),
            normalize_whitespace(item.get("title")).casefold(),
        )
    )
    return papers


def _sortable_rank(value: Any) -> int:
    if value is None:
        return 10**9
    return int(value)


def recommendation_search_by_pdf_title(
    client: SemanticScholarSearchClient,
    pdf_title: str,
    recommendation_limit: int,
) -> dict[str, Any]:
    normalized_title = normalize_whitespace(pdf_title)
    if not normalized_title:
        raise ValueError("PDF title is empty; cannot run title recommendation flow")

    seed_paper = client.paper_match_by_title(normalized_title)
    seed_paper_id = normalize_whitespace(seed_paper.get("paperId"))
    if not seed_paper_id:
        raise SemanticScholarSearchError(f"Matched paper for title={normalized_title!r} has no paperId")

    recommendation_payload = client.recommend_papers(seed_paper_id, recommendation_limit)
    recommended_papers = recommendation_payload.get("recommendedPapers", [])
    return {
        "seed_title": normalized_title,
        "seed_paper": seed_paper,
        "recommended_papers": recommended_papers,
    }


def build_search_payload(
    *,
    keyword_source: str,
    keyword_input_text: str,
    keywords: list[str],
    search_queries: list[str],
    papers: list[dict[str, Any]],
    idea_text: str | None,
    extracted_pdf: dict[str, str] | None,
    pdf_path: str | None,
    per_keyword_results: list[dict[str, Any]] | None,
    include_per_keyword_results: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "search",
        "input_type": "pdf" if extracted_pdf is not None else "idea_text",
        "keywords_source": keyword_source,
        "keyword_input_text": keyword_input_text,
        "keywords": keywords,
        "search_queries": search_queries,
        "paper_count": len(papers),
        "papers": papers,
    }
    if idea_text:
        payload["idea_text"] = normalize_whitespace(idea_text)
    if extracted_pdf is not None and pdf_path is not None:
        payload["pdf_path"] = str(Path(pdf_path).resolve())
        payload["pdf"] = extracted_pdf
    if include_per_keyword_results and per_keyword_results is not None:
        payload["per_keyword_results"] = per_keyword_results
    return payload


def build_recommendation_payload(
    *,
    extracted_pdf: dict[str, str],
    pdf_path: str,
    recommendation_result: dict[str, Any],
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "recommend",
        "input_type": "pdf",
        "pdf_path": str(Path(pdf_path).resolve()),
        "pdf": extracted_pdf,
        "recommendation": {
            "seed_title": recommendation_result["seed_title"],
            "seed_paper": recommendation_result["seed_paper"],
            "count": len(recommendation_result["recommended_papers"]),
        },
        "paper_count": len(papers),
        "papers": papers,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    mode = normalize_whitespace(getattr(args, "mode", "")) or ("hybrid" if args.pdf_path else "search")
    if mode not in {"search", "recommend", "hybrid"}:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if mode in {"recommend", "hybrid"} and not args.pdf_path:
        raise ValueError(f"mode={mode!r} requires --pdf-path so recommendation can resolve a paper title")

    extracted_pdf: dict[str, str] | None = None
    keyword_source = "idea_text"
    keyword_input_text = normalize_whitespace(args.idea_text) if args.idea_text else ""
    pre_extracted_pdf = getattr(args, "pre_extracted_pdf", None)

    if args.pdf_path:
        if isinstance(pre_extracted_pdf, dict):
            extracted_pdf = {
                "title": normalize_whitespace(pre_extracted_pdf.get("title")),
                "abstract": normalize_whitespace(pre_extracted_pdf.get("abstract")),
                "body": normalize_whitespace(pre_extracted_pdf.get("body")),
            }
        else:
            extracted_pdf = extract_pdf_payload(
                Path(args.pdf_path),
                grobid_base_url=args.grobid_base_url,
                grobid_start_page=args.grobid_start_page,
            )
        keyword_source = "pdf_title_abstract"
        keyword_input_text = build_keyword_input(None, extracted_pdf["title"], extracted_pdf["abstract"])
    else:
        keyword_input_text = build_keyword_input(args.idea_text, "", "")

    keywords: list[str] = []
    search_queries: list[str] = []
    merged_papers: list[dict[str, Any]] = []
    per_keyword_results: list[dict[str, Any]] = []

    if mode in {"search", "hybrid"}:
        if bool(getattr(args, "disable_keywords", False)):
            keyword_source = f"{keyword_source}_raw_query"
            search_queries = [keyword_input_text]
        else:
            keyword_extractor = KeywordExtractor(
                KeywordExtractorConfig(
                    api_url=args.keyword_api_url,
                    model=args.keyword_model,
                    env_path=Path(args.env),
                    timeout=args.keyword_timeout,
                    use_env_proxy=args.use_env_proxy,
                )
            )
            keywords = keyword_extractor.extract_keywords(keyword_input_text)
            search_queries = compose_search_queries(
                keywords,
                title=extracted_pdf["title"] if extracted_pdf is not None else "",
                abstract=extracted_pdf["abstract"] if extracted_pdf is not None else keyword_input_text,
            )
            if not search_queries:
                search_queries = keywords

    search_client = SemanticScholarSearchClient(
        SearchConfig(
            env_path=Path(args.env),
            timeout=args.search_timeout,
            retries=args.search_retries,
            use_env_proxy=args.use_env_proxy,
        )
    )

    if mode in {"search", "hybrid"}:
        merged_papers, per_keyword_results = aggregate_keyword_searches(
            search_client,
            search_queries,
            args.per_keyword_limit,
        )

    recommendation_result: dict[str, Any] | None = None
    recommendation_papers: list[dict[str, Any]] = []

    if mode in {"recommend", "hybrid"} and extracted_pdf is not None:
        recommendation_result = recommendation_search_by_pdf_title(
            search_client,
            extracted_pdf["title"],
            args.recommendation_limit,
        )
        recommendation_papers = list(recommendation_result["recommended_papers"])

    search_top_k = getattr(args, "search_top_k", None)
    recommend_top_k = getattr(args, "recommend_top_k", None)

    if mode == "search":
        papers = merged_papers[: search_top_k] if search_top_k is not None else merged_papers
        papers = papers[: args.top_k] if args.top_k is not None else papers
        return build_search_payload(
            keyword_source=keyword_source,
            keyword_input_text=keyword_input_text,
            keywords=keywords,
            search_queries=search_queries,
            papers=papers,
            idea_text=args.idea_text,
            extracted_pdf=extracted_pdf,
            pdf_path=args.pdf_path,
            per_keyword_results=per_keyword_results,
            include_per_keyword_results=args.include_per_keyword_results,
        )

    if mode == "recommend":
        papers = recommendation_papers[: recommend_top_k] if recommend_top_k is not None else recommendation_papers
        papers = papers[: args.top_k] if args.top_k is not None else papers
        return build_recommendation_payload(
            extracted_pdf=extracted_pdf,
            pdf_path=args.pdf_path,
            recommendation_result=recommendation_result,
            papers=papers,
        )

    search_candidates = merged_papers[: search_top_k] if search_top_k is not None else merged_papers
    recommend_candidates = (
        recommendation_papers[: recommend_top_k] if recommend_top_k is not None else recommendation_papers
    )

    merged_index: dict[str, dict[str, Any]] = {}
    anonymous_counter = 0
    for paper in search_candidates:
        anonymous_counter = merge_paper(
            merged_index,
            paper,
            source="keyword_search",
            rank=paper.get("_retrieval", {}).get("best_keyword_rank"),
            anonymous_counter=anonymous_counter,
        )
        retrieval = merged_index[next(reversed(merged_index))].get("_retrieval", {})
        retrieval["matched_keywords"] = list(paper.get("_retrieval", {}).get("matched_keywords", []))
        retrieval["keyword_hit_count"] = int(paper.get("_retrieval", {}).get("keyword_hit_count", 0))

    for rank, paper in enumerate(recommend_candidates, start=1):
        anonymous_counter = merge_paper(
            merged_index,
            paper,
            source="title_recommendation",
            rank=rank,
            anonymous_counter=anonymous_counter,
        )

    papers = sort_merged_papers(merged_index)
    if args.top_k is not None:
        papers = papers[: args.top_k]

    payload = build_search_payload(
        keyword_source=keyword_source,
        keyword_input_text=keyword_input_text,
        keywords=keywords,
        search_queries=search_queries,
        papers=papers,
        idea_text=args.idea_text,
        extracted_pdf=extracted_pdf,
        pdf_path=args.pdf_path,
        per_keyword_results=per_keyword_results,
        include_per_keyword_results=args.include_per_keyword_results,
    )
    payload["mode"] = "hybrid"
    payload["search"] = {
        "paper_count": len(search_candidates),
        "papers": search_candidates,
    }
    if args.include_per_keyword_results:
        payload["search"]["per_keyword_results"] = per_keyword_results
    payload["recommendation"] = {
        "seed_title": recommendation_result["seed_title"],
        "seed_paper": recommendation_result["seed_paper"],
        "count": len(recommend_candidates),
        "papers": recommend_candidates,
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Semantic Scholar papers from idea text or PDF-derived abstract keywords."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-text", help="Idea text used for keyword extraction.")
    input_group.add_argument("--pdf-path", help="PDF path. Title/abstract/body are extracted via GROBID.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env file.")
    parser.add_argument(
        "--mode",
        choices=("search", "recommend", "hybrid"),
        default=None,
        help="S2 retrieval mode. Default: search for idea-text, hybrid for pdf-path.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Final number of deduplicated papers to return.")
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=None,
        help="Optional cap on search candidates before final output/fusion.",
    )
    parser.add_argument(
        "--recommend-top-k",
        type=int,
        default=None,
        help="Optional cap on recommendation candidates before final output/fusion.",
    )
    parser.add_argument(
        "--per-keyword-limit",
        type=int,
        default=10,
        help="Number of papers requested from Semantic Scholar for each extracted keyword.",
    )
    parser.add_argument(
        "--include-per-keyword-results",
        action="store_true",
        help="Include raw per-keyword search results in output JSON.",
    )
    parser.add_argument("--keyword-model", default=DEFAULT_KEYWORD_MODEL, help="Keyword extractor model.")
    parser.add_argument("--keyword-api-url", default=KEYWORD_API_URL, help="Keyword extractor API URL.")
    parser.add_argument("--keyword-timeout", type=int, default=60, help="Keyword extractor timeout in seconds.")
    parser.add_argument(
        "--disable-keywords",
        action="store_true",
        help="Skip keyword extraction and search Semantic Scholar with the raw idea/PDF query.",
    )
    parser.add_argument("--search-timeout", type=int, default=DEFAULT_TIMEOUT, help="Semantic Scholar timeout in seconds.")
    parser.add_argument("--search-retries", type=int, default=3, help="Semantic Scholar retry count.")
    parser.add_argument(
        "--recommendation-limit",
        type=int,
        default=20,
        help="Number of recommendation papers requested for PDF title -> paperId -> recommendation flow.",
    )
    parser.add_argument("--grobid-base-url", default=DEFAULT_GROBID_BASE_URL, help="GROBID base URL.")
    parser.add_argument("--grobid-start-page", type=int, default=None, help="First PDF page sent to GROBID, 1-based.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the environment for both keyword extraction and search requests.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_pipeline(args)
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
