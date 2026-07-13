"""Step 1 — query refinement + multi-query seed-paper retrieval.

First refine the raw query with an LLM into ``{research_problem, keywords,
refined_research_question}``. Then diversify the refined topic into
``seed_num_queries`` (k) complementary search queries and retrieve for each in
parallel using the unchanged KG + S2 + OpenAlex per-query logic, keeping the top
``seed_per_query`` (m) candidates of each. The pooled ``k*m`` candidates are
deduplicated and the final ``k_step1`` seeds are chosen by a diversity-aware LLM
selection (the merge_search final-rerank prompt augmented to emphasize diversity).
Produces a :class:`SeedSet`, replacing the former Target Paper Pool.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import requests

from ..clients.llm_client import LLMClient
from ..clients.sciatlas_client import SciAtlasClient
from ..config import PipelineConfig
from ..models import Paper, SeedSet
from ..prompts import (
    MERGE_RERANK_RELEVANCE_DIVERSE,
    QUERY_REFINEMENT,
    SEED_QUERY_DIVERSIFICATION,
)
from ..utils import get_logger, read_json, save_json, stable_hash

log = get_logger("sciatlas.step1")

_S2_MODULE: Any = None


def _load_s2_module(innoeval_root: Path) -> Any:
    global _S2_MODULE
    if _S2_MODULE is not None:
        return _S2_MODULE
    # If merge_search_client already installed a stub, we need to replace it with the real module.
    s2api_dir = str((innoeval_root / "s2api").resolve())
    innoeval_dir = str(innoeval_root.resolve())
    for path in (s2api_dir, innoeval_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    # Remove any previously-installed stub so we get the real module.
    sys.modules.pop("search_s2", None)
    try:
        _S2_MODULE = importlib.import_module("search_s2")
    except Exception as exc:
        log.warning("  search_s2 unavailable: %s", exc)
        return None
    return _S2_MODULE


def _paper_from_s2(record: dict[str, Any]) -> Paper | None:
    title = str(record.get("title") or "").strip()
    if not title:
        return None
    paper_id = str(record.get("paperId") or "").strip()
    if not paper_id:
        return None
    abstract = str(record.get("abstract") or "").strip()
    year = record.get("year")
    citation_count = int(record.get("citationCount") or 0)
    external_ids = record.get("externalIds") or {}
    doi = str(external_ids.get("DOI") or "").strip().lower()
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year if isinstance(year, int) else None,
        references=[],
        citations=[],
        is_influential=citation_count > 100,
        score=0.0,
        raw={
            **record,
            "cited_by_count": citation_count,
            "citation_count": citation_count,
            "doi": doi or None,
            "s2_paper_id": paper_id,
        },
    )


def _s2_seed_search(
    innoeval_root: Path,
    refined_query: str,
    request_top_k: int,
) -> list[Paper]:
    """Retrieve seed candidates from Semantic Scholar via keyword search."""
    s2 = _load_s2_module(innoeval_root)
    if s2 is None:
        return []
    env_path = innoeval_root.parent / ".env"
    args = argparse.Namespace(
        idea_text=refined_query,
        pdf_path=None,
        env=str(env_path),
        mode="search",
        top_k=request_top_k,
        search_top_k=None,
        recommend_top_k=None,
        per_keyword_limit=10,
        include_per_keyword_results=False,
        keyword_model=s2.DEFAULT_KEYWORD_MODEL,
        keyword_api_url=s2.KEYWORD_API_URL,
        keyword_timeout=60,
        search_timeout=s2.DEFAULT_TIMEOUT,
        search_retries=3,
        recommendation_limit=20,
        grobid_base_url=s2.DEFAULT_GROBID_BASE_URL,
        grobid_start_page=None,
        use_env_proxy=False,
        disable_keywords=False,
        pre_extracted_pdf=None,
        pretty=False,
    )
    try:
        result = s2.run_pipeline(args)
    except Exception as exc:
        log.warning("  S2 seed search failed: %s", exc)
        return []
    papers: list[Paper] = []
    for record in result.get("papers") or []:
        if not isinstance(record, dict):
            continue
        paper = _paper_from_s2(record)
        if paper is not None:
            papers.append(paper)
    log.info("  S2 seed search returned %d candidates", len(papers))
    return papers


def _s2_pdf_extract_and_search(
    cfg: PipelineConfig,
    pdf_path: str,
    request_top_k: int,
) -> tuple[Paper | None, list[Paper]]:
    """Extract a PDF (title/abstract via GROBID) and run a PDF-seeded S2 search.

    Returns ``(pdf_paper, related)``:
      * ``pdf_paper`` — the paper the PDF itself represents, built from its
        GROBID-extracted title/abstract and flagged ``is_pdf_input`` /
        ``force_include`` so the caller can guarantee it anchors the graph.
      * ``related`` — S2 hits for the PDF's title/abstract keywords (recall).

    Returns ``(None, [])`` on any failure (missing S2 module, GROBID down, no
    extractable title) so a PDF that cannot be read never aborts the run.
    """
    s2 = _load_s2_module(cfg.innoeval_root)
    if s2 is None:
        return None, []
    grobid_url = getattr(cfg, "grobid_base_url", s2.DEFAULT_GROBID_BASE_URL)
    env_path = cfg.innoeval_root.parent / ".env"

    # Step A (ESSENTIAL): extract the PDF's title/abstract via GROBID. This is the
    # only part required to force-include the PDF as a seed, so it is isolated from
    # the (optional, network-heavy) S2 search below — an S2 failure must never drop
    # the seed.
    try:
        extracted = s2.extract_pdf_payload(
            Path(pdf_path), grobid_base_url=grobid_url, grobid_start_page=None
        )
    except Exception as exc:
        log.warning("  PDF GROBID extraction failed for %s: %s", pdf_path, exc)
        return None, []
    title = str((extracted or {}).get("title") or "").strip()
    abstract = str((extracted or {}).get("abstract") or "").strip()
    if not title:
        return None, []
    pdf_paper = Paper(
        paper_id="pdf:" + stable_hash({"title": title.lower()})[:16],
        title=title,
        abstract=abstract,
        raw={
            "title": title,
            "abstract": abstract,
            "is_pdf_input": True,
            "force_include": True,
            "pdf_path": str(pdf_path),
            "retrieval_sources": ["pdf"],
        },
    )

    # Step B (OPTIONAL): reuse the already-extracted payload to run a PDF-seeded S2
    # search for extra recall. Any failure here (no proxy to S2, keyword API down)
    # is swallowed — we still return the PDF seed from Step A.
    related: list[Paper] = []
    args = argparse.Namespace(
        idea_text=None,
        pdf_path=str(pdf_path),
        env=str(env_path),
        mode="search",
        top_k=request_top_k,
        search_top_k=None,
        recommend_top_k=None,
        per_keyword_limit=10,
        include_per_keyword_results=False,
        keyword_model=s2.DEFAULT_KEYWORD_MODEL,
        keyword_api_url=s2.KEYWORD_API_URL,
        keyword_timeout=60,
        search_timeout=s2.DEFAULT_TIMEOUT,
        search_retries=3,
        recommendation_limit=20,
        grobid_base_url=grobid_url,
        grobid_start_page=None,
        use_env_proxy=False,
        disable_keywords=False,
        pre_extracted_pdf=extracted,  # reuse Step A; skip a second GROBID call
        pretty=False,
    )
    try:
        result = s2.run_pipeline(args)
        for record in (result.get("papers") if isinstance(result, dict) else None) or []:
            if not isinstance(record, dict):
                continue
            paper = _paper_from_s2(record)
            if paper is not None:
                related.append(paper)
    except Exception as exc:
        log.warning("  PDF S2 recall search failed for %s (seed kept): %s", pdf_path, exc)
    log.info(
        "  PDF %s -> title=%r, %d related candidate(s)",
        pdf_path,
        title[:80],
        len(related),
    )
    return pdf_paper, related


def _s2_reference_count(innoeval_root: Path, title: str) -> int:
    """Best-effort Semantic Scholar reference count for a paper title.

    OpenAlex frequently returns an empty ``referenced_works`` list for arXiv/LLM
    preprints, which floors Step 2's reference-count-driven graph budget at the
    minimum. S2 covers CS preprints far better, so we match the seed by title and
    read its ``referenceCount`` to recover a realistic budget. Returns 0 on any
    failure (no match, network error) so the caller falls back to the minimum.
    """
    s2 = _load_s2_module(innoeval_root)
    if s2 is None or not (title or "").strip():
        return 0
    try:
        client = s2.SemanticScholarSearchClient(
            s2.SearchConfig(env_path=innoeval_root.parent / ".env")
        )
        payload = client.paper_match_by_title(title)
    except Exception as exc:
        log.warning("  S2 reference-count lookup failed for %r: %s", title[:60], exc)
        return 0
    count = payload.get("referenceCount") if isinstance(payload, dict) else None
    return int(count) if isinstance(count, (int, float)) else 0


def _s2_paper_references(innoeval_root: Path, node: "Paper", limit: int) -> list["Paper"]:
    """Fetch a paper's references (the works it cites) from Semantic Scholar.

    OpenAlex often has an empty ``referenced_works`` list for arXiv/LLM preprints,
    which starves Step 2's *backward* (foundational-paper) expansion — the only
    direction that can reach seminal older work. S2 covers CS preprints far better,
    so we resolve the node's S2 paperId (its ``s2_paper_id`` if it came from S2, else
    a title match) and pull its reference list, returning the cited papers
    high-citation-first so foundational work surfaces. Returns [] on any failure.
    """
    s2 = _load_s2_module(innoeval_root)
    if s2 is None:
        return []
    raw = node.raw if isinstance(node.raw, dict) else {}
    paper_id = raw.get("s2_paper_id")
    title = (node.title or "").strip()
    try:
        client = s2.SemanticScholarSearchClient(
            s2.SearchConfig(env_path=innoeval_root.parent / ".env")
        )
        if not paper_id and title:
            matched = client.paper_match_by_title(title)
            paper_id = matched.get("paperId") if isinstance(matched, dict) else None
        if not paper_id:
            return []
        url = (
            f"{s2.GRAPH_BASE}/paper/{quote(str(paper_id))}/references"
            f"?fields={s2.RICH_PAPER_FIELDS}&limit={max(limit * 3, 30)}"
        )
        payload = client._request_json(url)
    except Exception as exc:
        log.warning("  S2 references lookup failed for %r: %s", title[:60], exc)
        return []
    refs: list[Paper] = []
    for item in (payload.get("data") if isinstance(payload, dict) else None) or []:
        cited = item.get("citedPaper") if isinstance(item, dict) else None
        if not isinstance(cited, dict):
            continue
        paper = _paper_from_s2(cited)
        if paper is not None and paper.title:
            refs.append(paper)
    refs.sort(
        key=lambda p: (p.raw or {}).get("cited_by_count", 0) if isinstance(p.raw, dict) else 0,
        reverse=True,
    )
    return refs


def _extract_papers(result: dict) -> list[Paper]:
    """Pull a paper list out of the free-form ``result`` dict."""
    for key in ("papers", "results", "items", "hits", "data", "nodes"):
        val = result.get(key)
        if isinstance(val, list) and val:
            return [Paper.from_api(r) for r in val if isinstance(r, dict)]
    return []


def _normalize_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalize_title(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_whitespace(text).casefold())


def _paper_type(paper: Paper) -> str:
    sources: list[dict[str, Any]] = []
    if isinstance(paper.raw, dict):
        sources.append(paper.raw)
        nested = paper.raw.get("paper")
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        value = source.get("type")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _paper_cite_count(paper: Paper) -> int:
    sources: list[dict[str, Any]] = []
    if isinstance(paper.raw, dict):
        sources.append(paper.raw)
        nested = paper.raw.get("paper")
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        value = source.get("cited_by_count") or source.get("citation_count")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return len(paper.citations or [])


def _is_review_like(paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract}".lower()
    markers = (
        "survey",
        "review",
        "overview",
        "literature review",
        "systematic review",
        "meta-analysis",
        "tutorial",
        "primer",
        "handbook",
    )
    return any(marker in text for marker in markers)


def _is_primary_research_like(paper: Paper) -> bool:
    paper_type = _paper_type(paper)
    if paper_type in {"book", "book-series", "book chapter", "dataset", "editorial", "reference-entry"}:
        return False
    return not _is_review_like(paper)


def _query_overlap(query: str, paper: Paper) -> int:
    query_terms = {token for token in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(token) > 2}
    paper_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9]+", f"{paper.title} {paper.abstract}".lower())
        if len(token) > 2
    }
    return len(query_terms & paper_terms)


def _anchor_priority(query: str, paper: Paper) -> tuple[int, int, int, int, float, int, int]:
    has_graph_metadata = int(bool(paper.references or paper.citations))
    has_abstract = int(bool((paper.abstract or "").strip()))
    primary_research = int(_is_primary_research_like(paper))
    overlap = _query_overlap(query, paper)
    score = float(paper.score) if isinstance(paper.score, (int, float)) else 0.0
    citations = _paper_cite_count(paper)
    year = paper.year or 0
    return (primary_research, has_graph_metadata, has_abstract, overlap, score, citations, year)


def _paper_identifiers(paper: Paper) -> set[str]:
    identifiers: set[str] = set()
    if paper.paper_id:
        identifiers.add(f"paper:{paper.paper_id}")
    sources: list[dict[str, Any]] = []
    if isinstance(paper.raw, dict):
        sources.append(paper.raw)
        nested = paper.raw.get("paper")
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        for key in ("id", "paper_id", "paper_url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                identifiers.add(f"paper:{value.strip()}")
        doi = source.get("doi")
        if isinstance(doi, str) and doi.strip():
            identifiers.add(f"doi:{doi.strip().lower()}")
    return identifiers


def _paper_quality_key(query: str, paper: Paper, *, source_rank: int) -> tuple[int, int, int, int, float, int, int]:
    return (
        int(bool(paper.abstract)),
        len(paper.abstract or ""),
        *_anchor_priority(query, paper)[:5],
        -source_rank,
    )


def _merge_duplicate_papers(query: str, candidates: list[tuple[str, int, Paper]]) -> list[Paper]:
    groups: list[dict[str, Any]] = []
    for source, source_rank, paper in candidates:
        identifiers = _paper_identifiers(paper)
        title_key = _normalize_title(paper.title)
        matched_group: dict[str, Any] | None = None
        for group in groups:
            if identifiers & group["identifiers"]:
                matched_group = group
                break
            if title_key and title_key == group["title_key"]:
                matched_group = group
                break
        if matched_group is None:
            matched_group = {
                "identifiers": set(identifiers),
                "title_key": title_key,
                "members": [],
            }
            groups.append(matched_group)
        matched_group["identifiers"].update(identifiers)
        matched_group["members"].append((source, source_rank, paper))

    merged: list[Paper] = []
    for group in groups:
        ranked_members = sorted(
            group["members"],
            key=lambda item: _paper_quality_key(query, item[2], source_rank=item[1]),
            reverse=True,
        )
        representative = ranked_members[0][2]
        for _, _, variant in ranked_members[1:]:
            if not representative.abstract and variant.abstract:
                representative.abstract = variant.abstract
            if not representative.references and variant.references:
                representative.references = list(variant.references)
            if not representative.citations and variant.citations:
                representative.citations = list(variant.citations)
            if not representative.is_influential and variant.is_influential:
                representative.is_influential = True
            if (
                not isinstance(representative.score, (int, float))
                and isinstance(variant.score, (int, float))
            ):
                representative.score = variant.score
            if isinstance(representative.raw, dict):
                representative.raw.setdefault("retrieval_sources", [])
                representative.raw["retrieval_sources"].extend(
                    source_name for source_name, _, _ in ranked_members
                )
        if isinstance(representative.raw, dict):
            representative.raw["retrieval_sources"] = sorted(set(representative.raw.get("retrieval_sources", [])))
        merged.append(representative)
    return merged


def _abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, offsets in index.items():
        if not isinstance(token, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int):
                positions.append((offset, token))
    positions.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positions)


def _extract_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            candidate = str(
                item.get("paper_id")
                or item.get("id")
                or item.get("openalex_id")
                or item.get("work_id")
                or ""
            ).strip()
        else:
            candidate = ""
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _existing_graph_ids(paper: Paper, *keys: str) -> list[str]:
    sources: list[dict[str, Any]] = []
    if isinstance(paper.raw, dict):
        sources.append(paper.raw)
        nested = paper.raw.get("paper")
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        for key in keys:
            ids = _extract_id_list(source.get(key))
            if ids:
                return ids
    return []


def _openalex_work_id(paper: Paper) -> str | None:
    candidates = [
        paper.paper_id,
        paper.raw.get("paper_id") if isinstance(paper.raw, dict) else None,
        paper.raw.get("id") if isinstance(paper.raw, dict) else None,
        paper.raw.get("paper_url") if isinstance(paper.raw, dict) else None,
    ]
    nested = paper.raw.get("paper") if isinstance(paper.raw, dict) else None
    if isinstance(nested, dict):
        candidates.extend(
            [
                nested.get("id"),
                nested.get("paper_id"),
                nested.get("openalex_id"),
            ]
        )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        match = re.search(r"(https?://openalex\.org/)?(W\d+)", candidate)
        if match:
            return f"https://openalex.org/{match.group(2)}"
    return None


def _openalex_cache_path(client: SciAtlasClient, namespace: str, key: str) -> Path:
    path = client.cfg.cache_dir / namespace / f"{stable_hash({'key': key})}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _openalex_get_json(client: SciAtlasClient, *, url: str, cache_key: str) -> dict[str, Any]:
    cache_path = _openalex_cache_path(client, "openalex", cache_key)
    if client.cfg.use_cache:
        try:
            return read_json(cache_path)
        except Exception:
            pass
    response = requests.get(
        url,
        headers={"User-Agent": "sciatlas-idea-gen/1.0"},
        timeout=min(client.cfg.timeout, 60),
    )
    response.raise_for_status()
    data = response.json()
    if client.cfg.use_cache:
        save_json(data, cache_path)
    return data if isinstance(data, dict) else {}


def _enrich_graph_fields(client: SciAtlasClient, paper: Paper) -> Paper:
    references = paper.references or _existing_graph_ids(
        paper,
        "references",
        "reference_ids",
        "referenced_works",
        "cites",
        "outgoing_citations",
    )
    citations = paper.citations or _existing_graph_ids(
        paper,
        "citations",
        "cited_by",
        "incoming_citations",
    )
    if references and citations:
        paper.references = references
        paper.citations = citations
        return paper

    work_id = _openalex_work_id(paper)
    if not work_id:
        paper.references = references
        paper.citations = citations
        return paper

    short_id = work_id.rsplit("/", 1)[-1]
    try:
        work = _openalex_get_json(
            client,
            url=(
                "https://api.openalex.org/works/"
                f"{short_id}?select=id,referenced_works,abstract_inverted_index,cited_by_count,type"
            ),
            cache_key=f"work:{short_id}",
        )
        if not references:
            references = _extract_id_list(work.get("referenced_works"))
        if not paper.abstract:
            paper.abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
        if isinstance(paper.raw, dict):
            if "cited_by_count" in work:
                paper.raw["cited_by_count"] = work.get("cited_by_count")
            if "type" in work:
                paper.raw["type"] = work.get("type")
    except Exception as exc:
        log.warning("  openalex reference enrichment failed for %s: %s", paper.title, exc)

    try:
        if not citations:
            citing = _openalex_get_json(
                client,
                url=f"https://api.openalex.org/works?filter=cites:{short_id}&per-page=200&select=id",
                cache_key=f"cites:{short_id}",
            )
            citations = _extract_id_list(
                [item.get("id") for item in citing.get("results", []) if isinstance(item, dict)]
            )
    except Exception as exc:
        log.warning("  openalex citation enrichment failed for %s: %s", paper.title, exc)

    paper.references = references
    paper.citations = citations
    if isinstance(paper.raw, dict):
        paper.raw["references"] = references
        paper.raw["citations"] = citations
        if paper.abstract:
            paper.raw["abstract"] = paper.abstract
        nested = paper.raw.get("paper")
        if isinstance(nested, dict):
            nested["referenced_works"] = references
            if paper.abstract:
                nested["abstract"] = paper.abstract
    return paper


def _fallback_keywords(query: str) -> list[dict[str, int]]:
    """Generate a few English keyword hints when the caller only provides a topic.

    The SciAtlas API requires at least one keyword or title in ``plan``. We keep
    this fallback simple and deterministic so a plain English research topic can
    still bootstrap the pipeline without an extra LLM call.
    """
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
        "help", "how", "in", "into", "is", "of", "on", "or", "that", "the",
        "their", "this", "to", "using", "with",
    }
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    terms = [w for w in words if len(w) > 2 and w not in stopwords]

    phrases: list[str] = []
    for size in (2, 1):
        for i in range(len(terms) - size + 1):
            phrase = " ".join(terms[i : i + size]).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)

    return [{"text": phrase, "score": max(10 - i, 6)} for i, phrase in enumerate(phrases[:5])]


def _normalize_openalex_id(value: str | None) -> str | None:
    match = re.search(r"(W\d+)", value or "")
    return f"https://openalex.org/{match.group(1)}" if match else None


def _paper_from_openalex_work(work: dict[str, Any]) -> Paper:
    paper_id = _normalize_openalex_id(str(work.get("id", ""))) or str(work.get("id", "")).strip()
    title = str(work.get("title", "") or "").strip()
    abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
    year = work.get("publication_year")
    references = [
        normalized
        for normalized in (_normalize_openalex_id(ref) for ref in work.get("referenced_works", []) or [])
        if normalized
    ]
    cited_by_count = work.get("cited_by_count")
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year if isinstance(year, int) else None,
        references=references,
        citations=[],
        is_influential=isinstance(cited_by_count, int) and cited_by_count > 0,
        raw={
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "publication_year": year,
            "references": references,
            "cited_by_count": cited_by_count if isinstance(cited_by_count, int) else 0,
        },
    )


def _openalex_search_once(client: SciAtlasClient, search_text: str, per_page: int) -> list[Paper]:
    """Run a single OpenAlex relevance search and parse it into Papers."""
    # OpenAlex treats ? and * as wildcards (which require exact search), so strip them.
    safe = re.sub(r"\s+", " ", re.sub(r"[?*]", " ", search_text)).strip()
    if not safe:
        return []
    url = (
        "https://api.openalex.org/works?search="
        f"{quote(safe)}&per-page={max(per_page, 1)}"
        "&select=id,title,publication_year,referenced_works,cited_by_count,abstract_inverted_index"
    )
    try:
        data = _openalex_get_json(client, url=url, cache_key=f"seed_search:{safe}:{per_page}")
    except Exception as exc:
        log.warning("  OpenAlex seed search failed for %r: %s", safe, exc)
        return []
    papers: list[Paper] = []
    for item in data.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        paper = _paper_from_openalex_work(item)
        if paper.paper_id and paper.title:
            papers.append(paper)
    return papers


def _openalex_seed_search(
    client: SciAtlasClient,
    query: str,
    top_k: int,
    *,
    keywords: list[str] | None = None,
) -> list[Paper]:
    """Retrieve seed papers from OpenAlex (scinet-independent fallback).

    The refined keywords (e.g. "chain-of-thought prompting", "self-consistency")
    are far more discriminative than the long natural-language question, so query
    OpenAlex once per keyword and merge the results, keeping the full query only as
    a low-priority safety net. Without this, BM25 on the long question surfaces
    generic surveys/off-topic papers and seminal method papers never appear.
    """
    seen: set[str] = set()
    papers: list[Paper] = []
    # Keyword searches first so they dominate the candidate ordering; a few hits
    # per keyword keeps the pool focused and bounds the number of HTTP calls.
    per_keyword = max(top_k // 3, 4)
    for keyword in (keywords or [])[:6]:
        for paper in _openalex_search_once(client, keyword, per_keyword):
            if paper.paper_id not in seen:
                seen.add(paper.paper_id)
                papers.append(paper)
    # Full query as a fallback signal (lowest priority).
    for paper in _openalex_search_once(client, query, max(top_k, 1)):
        if paper.paper_id not in seen:
            seen.add(paper.paper_id)
            papers.append(paper)
    return papers


def _generate_diverse_queries(
    llm: LLMClient,
    research_problem: str,
    refined_query: str,
    keywords: list[str],
    k: int,
) -> list[str]:
    """Ask the LLM for ``k`` complementary search queries spanning the topic.

    The refined query is always kept as the first query (so the original intent is
    never lost), and the LLM-generated queries are appended and de-duplicated. On
    any failure we fall back to just ``[refined_query]`` so retrieval still runs.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        text = " ".join(str(candidate or "").split()).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            queries.append(text)

    _add(refined_query)
    if k <= 1:
        return queries[:1] or [refined_query]
    try:
        data = llm.chat_json(
            SEED_QUERY_DIVERSIFICATION.format(
                research_problem=research_problem or refined_query,
                refined_query=refined_query,
                keywords=", ".join(keywords) if keywords else "(none)",
                k=k,
            ),
            temperature=0.7,
        )
        for item in (data.get("queries", []) if isinstance(data, dict) else []):
            _add(item)
    except Exception as exc:
        log.warning("  diverse query generation failed, using refined query only: %s", exc)
    return queries[:k] or [refined_query]


def _select_diverse_seeds(
    llm: LLMClient,
    seed_topic: str,
    candidates: list[Paper],
    k: int,
) -> list[Paper]:
    """Final seed selection from the pooled multi-query candidates.

    Reuses the merge_search final-rerank scoring prompt (survey/review papers score
    0) augmented with an explicit diversity emphasis, scoring all pooled candidates
    in one pass and keeping the top ``k``. ``candidates`` is pre-ranked heuristically
    so any failure (or empty result) falls back to the heuristic top-``k``.
    """
    if k <= 0 or not candidates:
        return candidates[: max(k, 0)]
    if len(candidates) <= k:
        return candidates[:k]
    blocks = "\n\n".join(
        f"[{slot}] Title: {paper.title}\nAbstract: {(paper.abstract or '[missing abstract]')[:500]}"
        for slot, paper in enumerate(candidates, start=1)
    )
    prompt = MERGE_RERANK_RELEVANCE_DIVERSE.format(
        query_type="research_topic",
        query_title="",
        query_text=seed_topic,
        candidate_blocks=blocks,
    )
    scored: list[tuple[float, Paper]] = []
    try:
        data = llm.chat_json(prompt, temperature=0.2)
        for item in (data.get("papers", []) if isinstance(data, dict) else []):
            if not isinstance(item, dict):
                continue
            try:
                slot = int(item.get("paper_index"))
                score = max(0.0, min(10.0, float(item.get("score"))))
            except (TypeError, ValueError):
                continue
            if 1 <= slot <= len(candidates):
                scored.append((score, candidates[slot - 1]))
    except Exception as exc:
        log.warning("  diverse seed selection failed, using heuristic top-%d: %s", k, exc)
    if not scored:
        return candidates[:k]
    # Stable sort by score desc; ties keep the incoming heuristic order.
    order = {id(paper): idx for idx, paper in enumerate(candidates)}
    scored.sort(key=lambda pair: (-pair[0], order.get(id(pair[1]), 0)))
    selected: list[Paper] = []
    for _, paper in scored:
        if paper not in selected:
            selected.append(paper)
        if len(selected) >= k:
            break
    return selected[:k] or candidates[:k]


def _refine_query(llm: LLMClient, raw_query: str) -> tuple[str, list[str], str]:
    """LLM-refine the raw query into (research_problem, keywords, refined_question)."""
    try:
        data = llm.chat_json(QUERY_REFINEMENT.format(raw_query=raw_query), temperature=0.2)
    except Exception as exc:
        log.warning("  query refinement failed, using raw query: %s", exc)
        data = {}
    research_problem = str((data or {}).get("research_problem", "") or "").strip()
    refined = str((data or {}).get("refined_research_question", "") or "").strip() or raw_query
    keywords = [
        str(item).strip()
        for item in ((data or {}).get("keywords", []) or [])
        if str(item).strip()
    ]
    return research_problem, keywords, refined


def _retrieve_seed_candidates(
    client: SciAtlasClient,
    cfg: PipelineConfig,
    *,
    query_text: str,
    keyword_items: list[dict],
    openalex_keywords: list[str],
    pdf_title_items: list[dict],
    domain: str | None,
    request_top_k: int,
) -> list[Paper]:
    """Run the (unchanged) KG + S2 + OpenAlex retrieval for a single query and
    return a heuristically-ranked, recency/citation-filtered pool of expandable
    primary-research papers. This is the per-query retrieval reused by the
    multi-query seed search; the caller keeps the top ``seed_per_query`` of it.
    """

    def _run_sciatlas() -> list[tuple[str, int, Paper]]:
        try:
            result = client.search(
                query_text=query_text,
                keywords=keyword_items,
                titles=pdf_title_items or None,
                source_title=pdf_title_items[0]["title"] if pdf_title_items else None,
                top_k=request_top_k,
                retrieval_mode="hybrid",
                target_field=domain,
            )
            primary = _extract_papers(result)
            papers = [("kg", rank, p) for rank, p in enumerate(primary, start=1)]
            if len(primary) < request_top_k:
                supplement = client.search_papers(query_text=query_text, field="abstract", top_k=request_top_k)
                papers.extend(
                    ("paper_search_abstract", rank, p)
                    for rank, p in enumerate(_extract_papers(supplement), start=1)
                )
            return papers
        except Exception as exc:
            log.warning("  SciAtlas seed retrieval failed for %r: %s", query_text[:60], exc)
            return []

    def _run_s2() -> list[tuple[str, int, Paper]]:
        papers = _s2_seed_search(cfg.innoeval_root, query_text, request_top_k)
        return [("s2", rank, p) for rank, p in enumerate(papers, start=1)]

    def _run_openalex() -> list[tuple[str, int, Paper]]:
        papers = _openalex_seed_search(client, query_text, request_top_k, keywords=openalex_keywords)
        return [("openalex", rank, p) for rank, p in enumerate(papers, start=1)]

    source_candidates: list[tuple[str, int, Paper]] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_sciatlas): "kg",
            executor.submit(_run_s2): "s2",
            executor.submit(_run_openalex): "openalex",
        }
        for future, source in futures.items():
            try:
                source_candidates.extend(future.result())
            except Exception as exc:
                log.warning("  %s parallel retrieval failed for %r: %s", source, query_text[:60], exc)

    # Record S2/OpenAlex into the run trace (KG calls self-trace).
    for traced_source in ("s2", "openalex"):
        traced_papers = [p for src, _, p in source_candidates if src == traced_source]
        if traced_papers:
            client.record_external_retrieval(source=traced_source, query=query_text, papers=traced_papers)

    # Interleave the sources round-robin BEFORE capping so the cap never drops a
    # whole source (OpenAlex carries the foundational/older papers + their refs).
    by_source: dict[str, list[tuple[str, int, Paper]]] = {}
    for item in source_candidates:
        by_source.setdefault(item[0], []).append(item)
    interleaved = [
        item
        for tier in zip_longest(*by_source.values())
        for item in tier
        if item is not None
    ]
    enriched = [
        (source, source_rank, paper if (paper.references and paper.abstract) else _enrich_graph_fields(client, paper))
        for source, source_rank, paper in interleaved[: request_top_k * 2]
    ]
    merged = _merge_duplicate_papers(query_text, enriched)
    primary = [paper for paper in merged if _is_primary_research_like(paper)] or merged
    ranked = sorted(primary, key=lambda paper: _anchor_priority(query_text, paper), reverse=True)
    # Seeds must be recent OR foundational (highly-cited older papers are kept).
    recent_min_year = datetime.date.today().year - max(1, cfg.seed_recent_years) + 1
    recent = [
        paper
        for paper in ranked
        if (isinstance(paper.year, int) and paper.year >= recent_min_year)
        or _paper_cite_count(paper) >= cfg.seed_min_citations
    ]
    pool = recent or ranked
    with_graph = [paper for paper in pool if paper.references or paper.citations]
    return with_graph or pool


def run(
    client: SciAtlasClient,
    llm: LLMClient,
    query: str,
    cfg: PipelineConfig,
    *,
    seed_keywords: list[dict] | None = None,
    domain: str | None = None,
    pdf_paths: list[str] | None = None,
) -> SeedSet:
    log.info("Step 1: query refinement + seed retrieval for query=%r", query)
    seed_request_multiplier = max(1, int(getattr(cfg, "seed_request_multiplier", 6)))
    seed_request_floor = max(1, int(getattr(cfg, "seed_request_floor", 12)))
    anchor_top_k = max(1, int(getattr(cfg, "anchor_top_k", 5)))
    request_top_k = max(cfg.k_step1 * seed_request_multiplier, seed_request_floor, anchor_top_k)

    # --- PDF inputs ---------------------------------------------------------
    # Each uploaded PDF is (a) extracted into a forced seed paper that MUST end
    # up anchoring the research graph and (b) used to seed an S2 search for extra
    # recall. GROBID extraction + S2 search are independent per PDF.
    pdf_seed_papers: list[Paper] = []
    pdf_related: list[tuple[str, int, Paper]] = []
    for path in pdf_paths or []:
        pdf_paper, related = _s2_pdf_extract_and_search(cfg, path, request_top_k)
        if pdf_paper is not None:
            pdf_seed_papers.append(pdf_paper)
        else:
            log.warning(
                "  could not extract a usable title from PDF %s; it cannot be force-included",
                path,
            )
        pdf_related.extend(("s2_pdf", rank, p) for rank, p in enumerate(related, start=1))

    # Fold the PDF title/abstract into the refinement input so the refined query
    # and keywords reflect the uploaded paper(s), not just the topic string.
    refine_input = query or ""
    if pdf_seed_papers:
        pdf_context = "\n\n".join(
            f"Attached paper: {paper.title}\nAbstract: {(paper.abstract or '')[:1500]}"
            for paper in pdf_seed_papers
        )
        refine_input = f"{refine_input}\n\n{pdf_context}".strip() if refine_input.strip() else pdf_context
    research_problem, refined_keywords, refined_query = _refine_query(llm, refine_input)
    log.info("  refined query: %r (%d keywords)", refined_query, len(refined_keywords))
    log.info("  refined keywords: %s", ", ".join(refined_keywords) if refined_keywords else "(none)")

    keyword_items = seed_keywords or (
        [{"text": kw, "score": max(10 - i, 6)} for i, kw in enumerate(refined_keywords[:5])]
        if refined_keywords
        else _fallback_keywords(refined_query)
    )
    # Hand the PDF titles to the SciAtlas KG search as high-weight title anchors so
    # the graph retrieval is biased toward the uploaded paper(s) and their citation
    # neighbourhood, not only the refined keyword query.
    pdf_title_items = [{"title": paper.title} for paper in pdf_seed_papers if paper.title]

    # --- Multi-query seed retrieval ----------------------------------------
    # Diversify the refined topic into `seed_num_queries` complementary queries,
    # retrieve for each independently/in parallel (same per-query retrieval logic),
    # keep the top `seed_per_query` of each, then pool + dedup the survivors. This
    # widens recall across sub-directions before the final diversity-aware select.
    k_queries = max(1, cfg.seed_num_queries)
    m_per_query = max(1, cfg.seed_per_query)
    per_query_top_k = max(m_per_query * seed_request_multiplier, seed_request_floor, anchor_top_k)
    queries = _generate_diverse_queries(
        llm, research_problem, refined_query, refined_keywords, k_queries
    )
    log.info("  generated %d diverse seed query(ies):", len(queries))
    for idx, q in enumerate(queries, start=1):
        log.info("    q%d: %s", idx, q)

    def _kw_items_for(q: str) -> list[dict]:
        # Reuse the refined keyword anchors for the refined query; derive simple
        # keyword hints per diverse query so KG/OpenAlex stay keyword-driven.
        if q == refined_query and keyword_items:
            return keyword_items
        return _fallback_keywords(q)

    def _retrieve_for_query(q: str) -> tuple[str, list[Paper]]:
        kw_items = _kw_items_for(q)
        oa_keywords = [
            str(item.get("text", "")).strip()
            for item in kw_items
            if str(item.get("text", "")).strip()
        ]
        candidates = _retrieve_seed_candidates(
            client,
            cfg,
            query_text=q,
            keyword_items=kw_items,
            openalex_keywords=oa_keywords,
            pdf_title_items=pdf_title_items,
            domain=domain,
            request_top_k=per_query_top_k,
        )
        return q, candidates[:m_per_query]

    pooled: list[tuple[str, int, Paper]] = []
    with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as executor:
        for q, kept in executor.map(_retrieve_for_query, queries):
            log.info("  query %r -> kept %d seed candidate(s)", q[:60], len(kept))
            for rank, paper in enumerate(kept, start=1):
                pooled.append((f"query::{q}", rank, paper))

    # Fold PDF-seeded S2 hits into the pooled candidates (recall around the upload).
    pooled.extend(pdf_related)
    if pdf_related:
        client.record_external_retrieval(
            source="s2_pdf",
            query=refined_query,
            papers=[p for _, _, p in pdf_related],
        )

    # Dedup the k*m pooled candidates, then heuristically rank them so the final
    # LLM selection (and its fallback top-k) sees an on-topic, expandable ordering.
    merged = _merge_duplicate_papers(refined_query, pooled)
    primary = [paper for paper in merged if _is_primary_research_like(paper)] or merged
    ranked = sorted(primary, key=lambda paper: _anchor_priority(refined_query, paper), reverse=True)
    with_graph = [paper for paper in ranked if paper.references or paper.citations]
    expandable = with_graph or ranked
    # Final selection is LLM-driven over the pooled candidates: same merge_search
    # final-rerank prompt (survey papers score 0) augmented to emphasize DIVERSITY,
    # so the seed set spans different sub-directions instead of near-duplicates.
    seed_topic = research_problem or refined_query or query
    # Principle: every uploaded PDF MUST anchor the research graph. PDF papers are
    # force-included as seeds (step 2 always materializes every seed paper), so the
    # LLM only fills the remaining k_step1 slots. Enrich the PDF seeds with
    # references/abstract first so step 2 can expand their citation neighbourhood.
    forced_seeds = [_enrich_graph_fields(client, paper) for paper in pdf_seed_papers]
    forced_ids: set[str] = set()
    for paper in forced_seeds:
        forced_ids |= _paper_identifiers(paper)
    llm_budget = max(0, cfg.k_step1 - len(forced_seeds)) if forced_seeds else max(1, cfg.k_step1)
    llm_seeds = (
        _select_diverse_seeds(llm, seed_topic, expandable, llm_budget) if llm_budget > 0 else []
    )
    # Drop any LLM pick that duplicates a forced PDF seed, then prepend the forced seeds.
    llm_seeds = [paper for paper in llm_seeds if not (_paper_identifiers(paper) & forced_ids)]
    seed_papers = forced_seeds + llm_seeds
    log.info(
        "  selected %d seed paper(s): %d forced from PDF + %d via diversity-aware LLM "
        "select (%d quer(ies) x %d kept -> pool=%d)",
        len(seed_papers),
        len(forced_seeds),
        len(llm_seeds),
        len(queries),
        m_per_query,
        len(expandable),
    )
    for idx, paper in enumerate(seed_papers, start=1):
        log.info(
            "    seed %d: (%s, cites=%d) %s",
            idx,
            paper.year if isinstance(paper.year, int) else "n.d.",
            _paper_cite_count(paper),
            paper.title,
        )

    return SeedSet(
        seed_papers=seed_papers,
        research_problem=research_problem,
        keywords=refined_keywords,
        refined_query=refined_query,
    )
