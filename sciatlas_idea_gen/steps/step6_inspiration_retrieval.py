"""Step 6 — multi-radius inspiration retrieval."""
from __future__ import annotations

from typing import Any

from ..clients.llm_client import LLMClient
from ..clients.sciatlas_client import SciAtlasClient
from ..config import PipelineConfig
from ..models import Inspiration, Paper, SeedSet
from ..prompts import DOMAIN_QUERY_GEN
from ..utils import get_logger
from .step1_anchor_retrieval import _openalex_seed_search

log = get_logger("sciatlas.step6")


def _extract_papers(result: dict[str, Any]) -> list[Paper]:
    for key in ("papers", "results", "items", "hits", "data", "nodes"):
        value = result.get(key)
        if isinstance(value, list):
            return [Paper.from_api(item) for item in value if isinstance(item, dict)]
    return []


def _same_field_candidates(
    sciatlas: SciAtlasClient,
    seeds: SeedSet,
    rss: dict[str, Any],
    top_k: int,
) -> list[tuple[str, Paper]]:
    if top_k <= 0:
        return []
    query = rss.get("core_gap") or seeds.refined_query
    try:
        result = sciatlas.search_papers(query_text=str(query), field="abstract", top_k=top_k)
        papers = _extract_papers(result)
    except Exception as exc:
        log.warning("  same-field SciAtlas search failed, falling back to OpenAlex: %s", exc)
        papers = _openalex_seed_search(sciatlas, str(query), top_k)
    out: list[tuple[str, Paper]] = []
    seen_titles = {p.title.lower() for p in seeds.seed_papers if p.title}
    for paper in papers:
        if not paper.title or paper.title.lower() in seen_titles:
            continue
        out.append(("same_field", paper))
    return out


def _cross_domain_candidates(
    sciatlas: SciAtlasClient,
    llm: LLMClient,
    problem_text: str,
    *,
    num_domains: int,
    top_k_per_domain: int,
) -> list[tuple[str, Paper]]:
    if num_domains <= 0 or top_k_per_domain <= 0:
        return []
    prompt = DOMAIN_QUERY_GEN.format(
        problem_text=problem_text,
        num_domains=num_domains,
    )
    data = llm.chat_json(prompt, temperature=0.4)
    domain_queries = data.get("domain_queries", []) if isinstance(data, dict) else []

    out: list[tuple[str, Paper]] = []
    for item in domain_queries:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", "cross_domain"))
        query_text = str(item.get("query_text", "")).strip()
        keywords = item.get("keywords", [])
        if not query_text:
            continue
        try:
            result = sciatlas.search(
                query_text=query_text,
                keywords=keywords if isinstance(keywords, list) else None,
                top_k=top_k_per_domain,
                retrieval_mode="semantic",
            )
            papers = _extract_papers(result)
        except Exception as exc:
            log.warning("  cross-domain SciAtlas search failed, falling back to OpenAlex: %s", exc)
            papers = _openalex_seed_search(sciatlas, query_text, top_k_per_domain)
        for paper in papers:
            if paper.title:
                out.append((domain, paper))
    return out


def run(
    sciatlas: SciAtlasClient,
    llm: LLMClient,
    seeds: SeedSet,
    rss: dict[str, Any],
    cfg: PipelineConfig,
    *,
    plan: dict[str, Any] | None = None,
) -> list[Inspiration]:
    log.info("Step 6: multi-radius inspiration retrieval")
    plan = plan or {}
    problem_text = str(rss.get("core_gap") or seeds.refined_query)
    same_field_top_k = int(plan.get("same_field_top_k", cfg.inspiration_top_k_same_field) or 0)
    num_cross_domains = int(plan.get("num_cross_domains", cfg.num_cross_domains) or 0)
    cross_domain_top_k_per_domain = int(
        plan.get("cross_domain_top_k_per_domain", cfg.inspiration_top_k_per_domain) or 0
    )
    use_same_field = bool(plan.get("use_same_field", same_field_top_k > 0))
    use_cross_domain = bool(plan.get("use_cross_domain", num_cross_domains > 0))

    candidates: list[tuple[str, str, Paper]] = []
    if use_same_field:
        for domain, paper in _same_field_candidates(sciatlas, seeds, rss, same_field_top_k):
            candidates.append((domain, "R1", paper))
    if use_cross_domain:
        for domain, paper in _cross_domain_candidates(
            sciatlas,
            llm,
            problem_text,
            num_domains=num_cross_domains,
            top_k_per_domain=cross_domain_top_k_per_domain,
        ):
            candidates.append((domain, "R2", paper))

    deduped: dict[str, tuple[str, str, Paper]] = {}
    for domain, radius, paper in candidates:
        key = (paper.paper_id or paper.title or "").strip().lower()
        if key and key not in deduped:
            deduped[key] = (domain, radius, paper)

    results: list[Inspiration] = []
    for index, (domain, radius, paper) in enumerate(deduped.values(), start=1):
        results.append(
            Inspiration(
                domain=domain,
                paper_title=paper.title,
                paper_abstract=paper.abstract,
                radius=radius,
                source_paper=paper,
                candidate_id=index,
            )
        )
    log.info("  retrieved %d inspiration candidates", len(results))
    return results
