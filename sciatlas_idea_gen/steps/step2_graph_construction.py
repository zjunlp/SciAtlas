"""Step 2 — citation idea-graph construction (merge_search-style expansion).

For each of the ``k_step1`` seed papers, repeatedly retrieve the node's citation
neighbourhood (OpenAlex references = backward / citing papers = forward), dedup +
LLM-rerank it with merge_search's reused machinery (survey papers scored 0), let
an LLM pick ``num_candidate_step2`` predecessors, randomly expand
``num_expansion_step2`` of them while interleaving forward/backward, until the
node budget is filled. The collected papers are then deduped by citation
relations into one :class:`ResearchGraph`.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from collections import Counter, defaultdict, deque
from statistics import median
from typing import Any
from urllib.parse import quote

import requests

from ..clients.llm_client import LLMClient
from ..clients.merge_search_client import merge_rerank
from ..config import PipelineConfig
from ..models import GraphEdge, Paper, ResearchGraph, SeedSet
from ..prompts import IDEA_FIELDS_EXTRACTION, PHASE_ASSIGNMENT, PREDECESSOR_SELECTION
from ..utils import get_logger, read_json, save_json, stable_hash
# Reuse the S2 module loader and paper converter from step1 (lazy, cached).
from .step1_anchor_retrieval import (
    _load_s2_module,
    _paper_from_s2,
    _s2_paper_references,
    _s2_reference_count,
)

log = get_logger("sciatlas.step2")

# Max papers fetched from a node's citation neighbourhood before dedup + rerank.
CANDIDATE_FETCH_LIMIT = 40

IDEA_FIELD_ORDER = (
    "Problem",
    "Existing Methods",
    "Motivation",
    "Proposed Method",
    "Experiment Plan",
)


def _budget(num_refs: int, cfg: PipelineConfig) -> int:
    """Return the target total number of papers in the final graph."""
    return min(
        cfg.graph_budget_max,
        max(cfg.graph_budget_min, math.ceil(cfg.graph_budget_ratio * max(num_refs, 1))),
    )


def _progress_bar(current: int, total: int, *, width: int = 20) -> str:
    if total <= 0:
        return "[--------------------] 0/0"
    filled = min(width, math.floor(width * current / total))
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total}"


def _paper_cite_count(paper: Paper) -> int:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    value = raw.get("cited_by_count")
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
    )
    return any(marker in text for marker in markers)


def _paper_direction(paper: Paper) -> str:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    return str(raw.get("expansion_direction", "unknown") or "unknown")


def _mark_expansion_direction(paper: Paper, direction: str) -> Paper:
    if isinstance(paper.raw, dict):
        paper.raw["expansion_direction"] = direction
    return paper


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")
_ARXIV_RE = re.compile(r"(?:arxiv[:/]|abs/|pdf/)\s*([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)


def _doi_from(*texts: str | None) -> str | None:
    for text in texts:
        match = _DOI_RE.search(text or "")
        if match:
            return match.group(0).rstrip(".").lower()
    return None


def _resolve_seed_to_openalex(sciatlas, seed: Paper) -> Paper | None:
    """Resolve a non-OpenAlex seed (e.g. a local-KG arXiv preprint) to its OpenAlex
    work so it gains a ``W…`` id + references and becomes expandable.

    Tries the DOI (from the id or the raw ``paper_url``/``doi``), then a title
    search. Returns ``None`` if nothing resolves.
    """
    raw = seed.raw if isinstance(seed.raw, dict) else {}
    select = "id,title,publication_year,referenced_works,cited_by_count,abstract_inverted_index"
    doi = _doi_from(seed.paper_id, raw.get("paper_url"), raw.get("doi"), raw.get("pdf_url"))
    if doi:
        try:
            work = _openalex_get_json(
                sciatlas,
                url=f"https://api.openalex.org/works/doi:{quote(doi)}?select={select}",
                cache_key=f"seed_doi:{doi}",
            )
            paper = _paper_from_openalex(work)
            if paper.paper_id and paper.title:
                return paper
        except Exception as exc:
            log.warning("  seed DOI->OpenAlex resolve failed (%s): %s", doi, exc)
    title = (seed.title or "").strip()
    if title:
        # OpenAlex treats ? and * as wildcard operators and 400s ("Wildcards
        # require exact (no-stem) search") on a plain ``search=`` containing them;
        # strip them so titles like "...Formal Language?" resolve normally.
        search_title = re.sub(r"[?*]", " ", title).strip()
        try:
            data = _openalex_get_json(
                sciatlas,
                url=f"https://api.openalex.org/works?search={quote(search_title)}&per-page=1&select={select}",
                cache_key=f"seed_title:{title[:120]}",
            )
            for item in data.get("results", []) or []:
                paper = _paper_from_openalex(item)
                if paper.paper_id and paper.title:
                    return paper
        except Exception as exc:
            log.warning("  seed title->OpenAlex resolve failed (%s): %s", title[:60], exc)
    return None


def _ensure_seed_metadata(sciatlas, seed: Paper) -> Paper:
    has_wid = _normalize_openalex_id(seed.paper_id) is not None
    # Fast path: already OpenAlex-addressable, just backfill refs/abstract if missing.
    if has_wid and (not seed.references or not seed.abstract):
        fetched = _fetch_openalex_paper(sciatlas, seed.paper_id)
        if fetched is not None:
            seed = _merge_openalex_fields(seed, fetched)
    # Local-KG / DOI seeds with NO W-id: resolve to OpenAlex so the citation-
    # neighbourhood expansion has a W-id to address and refs to walk. We deliberately
    # do NOT re-resolve when a W-id already exists: title/DOI search only serves to
    # FIND a W-id, and re-searching a ref-less work just re-finds the same record
    # (no new refs) while triggering needless 400s on wildcard-containing titles.
    if not _normalize_openalex_id(seed.paper_id):
        resolved = _resolve_seed_to_openalex(sciatlas, seed)
        if resolved is not None:
            if _normalize_openalex_id(resolved.paper_id):
                if isinstance(seed.raw, dict):
                    seed.raw.setdefault("original_paper_id", seed.paper_id)
                seed.paper_id = resolved.paper_id
            seed = _merge_openalex_fields(seed, resolved)
    return seed


def _normalize_openalex_id(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(W\d+)", value)
    if not match:
        return None
    return f"https://openalex.org/{match.group(1)}"


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


def _openalex_cache_path(sciatlas, cache_key: str):
    path = sciatlas.cfg.cache_dir / "openalex" / f"{stable_hash({'key': cache_key})}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _openalex_get_json(sciatlas, *, url: str, cache_key: str) -> dict[str, Any]:
    cache_path = _openalex_cache_path(sciatlas, cache_key)
    if sciatlas.cfg.use_cache:
        try:
            return read_json(cache_path)
        except Exception:
            pass
    response = requests.get(
        url,
        headers={"User-Agent": "sciatlas-idea-gen/1.0"},
        timeout=min(max(int(sciatlas.cfg.timeout), 30), 60),
    )
    response.raise_for_status()
    data = response.json()
    if sciatlas.cfg.use_cache:
        save_json(data, cache_path)
    return data if isinstance(data, dict) else {}


def _paper_from_openalex(work: dict[str, Any]) -> Paper:
    paper_id = _normalize_openalex_id(str(work.get("id", "")).strip()) or str(work.get("id", "")).strip()
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


def _fetch_openalex_paper(sciatlas, work_id: str) -> Paper | None:
    normalized_id = _normalize_openalex_id(work_id)
    if not normalized_id:
        return None
    short_id = normalized_id.rsplit("/", 1)[-1]
    try:
        work = _openalex_get_json(
            sciatlas,
            url=(
                "https://api.openalex.org/works/"
                f"{short_id}?select=id,title,publication_year,referenced_works,cited_by_count,abstract_inverted_index"
            ),
            cache_key=f"work_meta:{short_id}",
        )
    except Exception as exc:
        log.warning("  openalex expansion failed for %s: %s", normalized_id, exc)
        return None
    if not work:
        return None
    paper = _paper_from_openalex(work)
    return paper if paper.paper_id and paper.title else None


def _fetch_openalex_papers_batch(
    sciatlas, work_ids: list[str], *, batch_size: int = 50
) -> dict[str, Paper]:
    """Fetch many works in one OpenAlex multi-id request per ``batch_size`` ids.

    Returns a ``{normalized_id: Paper}`` map. Merged/deleted/unindexed ids simply do
    not appear in the response, so this avoids the per-id 404 that the single-GET
    path (`_fetch_openalex_paper`) logged for every unresolvable reference.
    """
    short_ids: list[str] = []
    for work_id in work_ids:
        normalized = _normalize_openalex_id(work_id)
        if normalized:
            short_ids.append(normalized.rsplit("/", 1)[-1])
    results: dict[str, Paper] = {}
    select = "id,title,publication_year,referenced_works,cited_by_count,abstract_inverted_index"
    for start in range(0, len(short_ids), batch_size):
        chunk = short_ids[start : start + batch_size]
        try:
            data = _openalex_get_json(
                sciatlas,
                url=(
                    "https://api.openalex.org/works"
                    f"?filter=openalex_id:{'|'.join(chunk)}&per-page={len(chunk)}&select={select}"
                ),
                cache_key=f"works_batch:{stable_hash({'ids': chunk})}",
            )
        except Exception as exc:
            log.warning("  openalex batch expansion failed (%d ids): %s", len(chunk), exc)
            continue
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            paper = _paper_from_openalex(item)
            if paper.paper_id and paper.title:
                results[paper.paper_id] = paper
    return results


def _fetch_openalex_citing_papers(
    sciatlas,
    work_id: str,
    *,
    per_page: int = 25,
) -> list[Paper]:
    normalized_id = _normalize_openalex_id(work_id)
    if not normalized_id:
        return []
    short_id = normalized_id.rsplit("/", 1)[-1]
    try:
        data = _openalex_get_json(
            sciatlas,
            url=(
                "https://api.openalex.org/works"
                f"?filter=cites:{short_id}&per-page={per_page}"
                "&select=id,title,publication_year,referenced_works,cited_by_count,abstract_inverted_index"
            ),
            cache_key=f"citing_meta:{short_id}:{per_page}",
        )
    except Exception as exc:
        log.warning("  openalex forward expansion failed for %s: %s", normalized_id, exc)
        return []
    papers: list[Paper] = []
    for item in data.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        paper = _paper_from_openalex(item)
        if paper.paper_id and paper.title:
            papers.append(paper)
    return papers


def _merge_openalex_fields(base: Paper, fetched: Paper | None) -> Paper:
    if fetched is None:
        return base
    if not base.abstract and fetched.abstract:
        base.abstract = fetched.abstract
    if not base.references and fetched.references:
        base.references = fetched.references
    if not base.citations and fetched.citations:
        base.citations = fetched.citations
    if not base.is_influential and fetched.is_influential:
        base.is_influential = True
    if isinstance(base.raw, dict) and isinstance(fetched.raw, dict):
        merged_raw = dict(fetched.raw)
        merged_raw.update(base.raw)
        if base.references:
            merged_raw["references"] = base.references
        if base.abstract:
            merged_raw["abstract"] = base.abstract
        base.raw = merged_raw
    return base


def _query_overlap(query: str, paper: Paper) -> int:
    query_terms = {token for token in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(token) > 2}
    paper_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9]+", f"{paper.title} {paper.abstract}".lower())
        if len(token) > 2
    }
    return len(query_terms & paper_terms)


def _normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").casefold())


def _paper_identifiers(paper: Paper) -> set[str]:
    identifiers: set[str] = set()
    if paper.paper_id:
        identifiers.add(f"paper:{paper.paper_id}")
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    nested = raw.get("paper") if isinstance(raw.get("paper"), dict) else {}
    for source in (raw, nested):
        for key in ("id", "paper_id", "paper_url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                identifiers.add(f"paper:{value.strip()}")
        doi = source.get("doi")
        if isinstance(doi, str) and doi.strip():
            identifiers.add(f"doi:{doi.strip().lower()}")
    return identifiers


def _candidate_quality_key(query_text: str, paper: Paper, *, source_rank: int) -> tuple[int, int, int, int, float, int, int]:
    return (
        int(bool(paper.abstract)),
        len(paper.abstract or ""),
        int(not _is_review_like(paper)),
        int(bool(paper.references or paper.citations)),
        float(paper.score or 0.0),
        _paper_cite_count(paper),
        -source_rank + _query_overlap(query_text, paper),
    )


def _dedupe_candidate_papers(query_text: str, candidates: list[tuple[str, int, Paper]]) -> list[Paper]:
    groups: list[dict[str, Any]] = []
    for _, source_rank, paper in candidates:
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
            matched_group = {"identifiers": set(identifiers), "title_key": title_key, "members": []}
            groups.append(matched_group)
        matched_group["identifiers"].update(identifiers)
        matched_group["members"].append((source_rank, paper))

    deduped: list[Paper] = []
    for group in groups:
        ranked_members = sorted(
            group["members"],
            key=lambda item: _candidate_quality_key(query_text, item[1], source_rank=item[0]),
            reverse=True,
        )
        representative = ranked_members[0][1]
        for _, variant in ranked_members[1:]:
            representative = _merge_openalex_fields(representative, variant)
        deduped.append(representative)
    return deduped


def _is_within_backward_time_cone(candidate: Paper, seed: Paper) -> bool:
    if candidate.paper_id == seed.paper_id:
        return False
    if seed.year is None or candidate.year is None:
        return True
    return candidate.year <= seed.year


def _is_within_forward_time_cone(candidate: Paper, source: Paper) -> bool:
    if candidate.paper_id == source.paper_id:
        return False
    if source.year is None or candidate.year is None:
        return True
    return candidate.year >= source.year


def _default_idea_fields(paper: Paper) -> dict[str, str]:
    abstract = (paper.abstract or "").strip()
    brief = abstract[:300] if abstract else paper.title
    return {
        "Problem": brief,
        "Existing Methods": "Unavailable from local metadata.",
        "Motivation": brief,
        "Proposed Method": brief,
        "Experiment Plan": "Unavailable from local metadata.",
    }


def _extract_idea_fields(llm: LLMClient, paper: Paper) -> dict[str, str]:
    prompt = IDEA_FIELDS_EXTRACTION.format(
        paper_title=paper.title,
        paper_abstract=paper.abstract[:2500],
    )
    try:
        data = llm.chat_json(prompt, temperature=0.2)
    except Exception as exc:
        log.warning("  idea extraction failed for %s: %s", paper.title, exc)
        return _default_idea_fields(paper)
    result = _default_idea_fields(paper)
    if isinstance(data, dict):
        for key in IDEA_FIELD_ORDER:
            value = str(data.get(key, "") or "").strip()
            if value:
                result[key] = value
    return result


def _safe_median(values: list[int]) -> float:
    clean = [value for value in values if isinstance(value, (int, float))]
    return float(median(clean)) if clean else 0.0


def _phase_layers(papers: dict[str, Paper], edges: list[GraphEdge]) -> list[list[str]]:
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.relation not in {"explicit_pred", "parallel_pred", "direct_to_seed"}:
            continue
        if edge.source in papers and edge.target in papers:
            incoming[edge.target].add(edge.source)
    remaining = set(papers)
    layers: list[list[str]] = []
    while remaining:
        layer = [pid for pid in remaining if not (incoming[pid] & remaining)]
        if not layer:
            layer = sorted(remaining, key=lambda p: (papers[p].year or 0, papers[p].title))
        else:
            layer = sorted(layer, key=lambda p: (papers[p].year or 0, papers[p].title))
        layers.append(layer)
        remaining -= set(layer)
    return layers


def _enforce_phase_precedence(
    phases: list[list[str]],
    papers: dict[str, Paper],
    edges: list[GraphEdge],
) -> list[list[str]]:
    phase_lookup: dict[str, int] = {}
    for phase_index, phase in enumerate(phases):
        for paper_id in phase:
            phase_lookup[paper_id] = phase_index
    if not phase_lookup:
        return phases
    for _ in range(len(phase_lookup)):
        changed = False
        for edge in edges:
            if edge.relation not in {"explicit_pred", "parallel_pred", "direct_to_seed"}:
                continue
            if edge.source not in phase_lookup or edge.target not in phase_lookup:
                continue
            if phase_lookup[edge.source] > phase_lookup[edge.target]:
                phase_lookup[edge.target] = phase_lookup[edge.source]
                changed = True
        if not changed:
            break
    grouped: dict[int, list[str]] = defaultdict(list)
    for paper_id, phase_index in phase_lookup.items():
        grouped[phase_index].append(paper_id)
    return [
        sorted(grouped[phase_index], key=lambda pid: (papers[pid].year or 0, papers[pid].title))
        for phase_index in sorted(grouped)
    ]


def _assign_phases_with_llm(
    llm: LLMClient,
    graph: ResearchGraph,
    seeds: list[Paper],
) -> list[list[str]]:
    seed_id_set = {seed.paper_id for seed in seeds}
    candidate_ids = [paper_id for paper_id in graph.papers if paper_id not in seed_id_set]
    if not candidate_ids:
        return []
    if len(candidate_ids) == 1:
        return [candidate_ids]

    candidate_id_set = set(candidate_ids)
    paper_lines = []
    for paper_id in sorted(candidate_ids, key=lambda pid: (graph.papers[pid].year or 0, graph.papers[pid].title)):
        paper = graph.papers[paper_id]
        paper_lines.append(
            "\n".join(
                [
                    f"- paper_id: {paper_id}",
                    f"  title: {paper.title}",
                    f"  year: {paper.year or 'n.d.'}",
                    f"  direction: {_paper_direction(paper)}",
                    f"  evidence_score: {graph.evidence_scores.get(paper_id, 'n/a')}",
                    f"  abstract: {(paper.abstract or '')[:500]}",
                ]
            )
        )

    edge_lines = []
    for edge in graph.edges:
        if edge.source not in candidate_id_set or edge.target not in candidate_id_set:
            continue
        if edge.relation not in {"explicit_pred", "parallel_pred", "direct_to_seed"}:
            continue
        edge_lines.append(f"- {edge.source} -> {edge.target} ({edge.relation})")
    if not edge_lines:
        edge_lines.append("- none")

    seed_title = " | ".join(seed.title for seed in seeds if seed.title) or seeds[0].title
    per_seed_budget = max(1200 // max(len(seeds), 1), 300)
    seed_abstract = "\n\n".join(
        f"[{seed.title}] {(seed.abstract or '')[:per_seed_budget]}" for seed in seeds
    )
    prompt = PHASE_ASSIGNMENT.format(
        seed_paper_title=seed_title,
        seed_paper_abstract=seed_abstract,
        paper_list="\n".join(paper_lines),
        edge_list="\n".join(edge_lines),
    )
    fallback = _phase_layers(
        {paper_id: graph.papers[paper_id] for paper_id in candidate_ids},
        graph.edges,
    )
    try:
        data = llm.chat_json(prompt, temperature=0.0)
    except Exception as exc:
        log.warning("  phase assignment failed, fallback to structural layers: %s", exc)
        return fallback

    phases: list[list[str]] = []
    seen: set[str] = set()
    for phase in data.get("phases", []) if isinstance(data, dict) else []:
        if not isinstance(phase, dict):
            continue
        ids: list[str] = []
        for paper_id in phase.get("paper_ids", []) or []:
            if paper_id in candidate_id_set and paper_id not in seen:
                ids.append(paper_id)
                seen.add(paper_id)
        if ids:
            phases.append(ids)

    remaining = [
        paper_id
        for paper_id in sorted(candidate_ids, key=lambda pid: (graph.papers[pid].year or 0, graph.papers[pid].title))
        if paper_id not in seen
    ]
    if remaining:
        phases.append(remaining)
    return _enforce_phase_precedence(phases or fallback, graph.papers, graph.edges)


def _connected_component_count(graph: ResearchGraph) -> int:
    node_ids = set(graph.papers)
    if not node_ids:
        return 0
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.source in node_ids and edge.target in node_ids:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    seen: set[str] = set()
    components = 0
    for paper_id in node_ids:
        if paper_id in seen:
            continue
        components += 1
        queue: deque[str] = deque([paper_id])
        seen.add(paper_id)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
    return components


def _build_predecessor_edges(
    graph: ResearchGraph,
    seeds: list[Paper],
    s_cite_lookup: dict[str, float],
    cfg: PipelineConfig,
) -> list[GraphEdge]:
    # All seeds are first-class anchors: layer-depth counts a target as depth-1 if
    # it is referenced by ANY seed, "direct_to_seed" applies to edges into ANY seed,
    # and the (low-confidence) delta_year feature is taken w.r.t. the primary seed.
    primary_seed = seeds[0]
    seed_id_set = {seed.paper_id for seed in seeds}
    raw_predecessors: dict[str, list[str]] = defaultdict(list)
    cited_by_subgraph: dict[str, int] = defaultdict(int)

    for target in graph.papers.values():
        for ref in target.references or []:
            if ref not in graph.papers or ref == target.paper_id:
                continue
            predecessor = graph.papers[ref]
            if predecessor.year is not None and target.year is not None and predecessor.year > target.year:
                continue
            raw_predecessors[target.paper_id].append(ref)
            cited_by_subgraph[ref] += 1

    citation_median = _safe_median([_paper_cite_count(paper) for paper in graph.papers.values()])
    seed_refs: set[str] = set()
    for seed in seeds:
        seed_refs.update(seed.references or [])
    candidate_edges: dict[tuple[str, str], GraphEdge] = {}

    for target_id, predecessor_ids in raw_predecessors.items():
        target = graph.papers[target_id]
        ranked_pred_ids = sorted(
            set(predecessor_ids),
            key=lambda pid: (
                s_cite_lookup.get(pid, 0.0),
                cited_by_subgraph.get(pid, 0),
                _paper_cite_count(graph.papers[pid]),
                -(
                    (target.year - graph.papers[pid].year)
                    if target.year is not None and graph.papers[pid].year is not None
                    else 999
                ),
            ),
            reverse=True,
        )[: cfg.graph_max_predecessors_per_paper]
        for pred_id in ranked_pred_ids:
            predecessor = graph.papers[pred_id]
            relation = "parallel_pred" if predecessor.year is not None and predecessor.year == target.year else "explicit_pred"
            features = {
                "layer_depth": 1 if target.paper_id in seed_refs else 2,
                "cited_in_sections": [],
                "cite_count": _paper_cite_count(target),
                "section_weight": 0.0,
                "delta_year": (
                    (primary_seed.year - target.year)
                    if primary_seed.year is not None and target.year is not None
                    else None
                ),
                "is_influential_raw": target.is_influential or _paper_cite_count(target) >= citation_median,
                "low_confidence": True,
                "cited_by_subgraph": cited_by_subgraph.get(target.paper_id, 0),
            }
            candidate_edges[(pred_id, target_id)] = GraphEdge(
                source=pred_id,
                target=target_id,
                relation=relation,
                edge_type=relation,
                weight=graph.evidence_scores.get(pred_id, 1.0),
                features=features,
            )

    removed_pairs: set[tuple[str, str]] = set()
    for source_id, target_id in list(candidate_edges):
        reverse_key = (target_id, source_id)
        if reverse_key not in candidate_edges or reverse_key in removed_pairs:
            continue
        source_score = s_cite_lookup.get(source_id, 0.0)
        reverse_score = s_cite_lookup.get(target_id, 0.0)
        keep_forward = (
            (source_score > reverse_score)
            or (source_score == reverse_score and source_id < target_id)
        )
        if keep_forward:
            removed_pairs.add(reverse_key)
        else:
            removed_pairs.add((source_id, target_id))

    edges = [edge for key, edge in candidate_edges.items() if key not in removed_pairs]
    predecessor_incoming: Counter[str] = Counter(edge.target for edge in edges)

    for edge in edges:
        if edge.target in seed_id_set and predecessor_incoming.get(edge.source, 0) == 0:
            edge.relation = "direct_to_seed"
            edge.edge_type = "direct_to_seed"

    return edges




def _query_context(paper: Paper) -> dict[str, str]:
    return {
        "query_type": "paper_abstract",
        "query_title": paper.title or "",
        "query_text": (paper.abstract or paper.title or "").strip(),
    }


def _backward_candidates(sciatlas, node: Paper, *, seen_ids: set[str], limit: int) -> list[Paper]:
    ref_ids: list[str] = []
    seen_local: set[str] = set()
    for ref_id in node.references or []:
        normalized = _normalize_openalex_id(ref_id) or ref_id
        if not normalized or normalized in seen_ids or normalized in seen_local:
            continue
        seen_local.add(normalized)
        ref_ids.append(normalized)
    ref_ids = ref_ids[: max(limit * 3, 30)]
    if not ref_ids:
        return []
    fetched_map = _fetch_openalex_papers_batch(sciatlas, ref_ids)
    cone: list[Paper] = []
    for ref_id in ref_ids:
        paper = fetched_map.get(ref_id)
        if not paper or not _is_within_backward_time_cone(paper, node):
            continue
        _mark_expansion_direction(paper, "backward")
        cone.append(paper)
    # Surface foundational references first: order backward candidates by citation
    # count (descending) rather than the paper's raw reference order, so seminal
    # older work (e.g. the papers a seed is built on) enters the graph preferentially.
    cone.sort(
        key=lambda p: (p.raw or {}).get("cited_by_count", 0) if isinstance(p.raw, dict) else 0,
        reverse=True,
    )
    return cone[:limit]


def _forward_candidates(sciatlas, node: Paper, *, seen_ids: set[str], limit: int) -> list[Paper]:
    results: list[Paper] = []
    for paper in _fetch_openalex_citing_papers(sciatlas, node.paper_id, per_page=max(limit * 2, 25)):
        if len(results) >= limit:
            break
        if paper.paper_id in seen_ids or not _is_within_forward_time_cone(paper, node):
            continue
        _mark_expansion_direction(paper, "forward")
        results.append(paper)
    return results


def _s2_related_candidates(
    cfg: "PipelineConfig",
    node: Paper,
    *,
    seen_ids: set[str],
    limit: int,
) -> list[Paper]:
    """Retrieve papers related to ``node`` via S2 text search on its title/abstract."""
    s2 = _load_s2_module(cfg.innoeval_root)
    if s2 is None:
        return []
    query = f"{node.title}. {(node.abstract or '')[:400]}".strip(". ")
    if not query:
        return []
    env_path = cfg.innoeval_root.parent / ".env"
    args = argparse.Namespace(
        idea_text=query,
        pdf_path=None,
        env=str(env_path),
        mode="search",
        top_k=limit,
        search_top_k=None,
        recommend_top_k=None,
        per_keyword_limit=8,
        include_per_keyword_results=False,
        keyword_model=s2.DEFAULT_KEYWORD_MODEL,
        keyword_api_url=s2.KEYWORD_API_URL,
        keyword_timeout=60,
        search_timeout=s2.DEFAULT_TIMEOUT,
        search_retries=2,
        recommendation_limit=10,
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
        log.warning("  S2 node search failed for %r: %s", node.title[:60], exc)
        return []
    papers: list[Paper] = []
    for record in result.get("papers") or []:
        if not isinstance(record, dict):
            continue
        paper = _paper_from_s2(record)
        if paper is None or paper.paper_id in seen_ids:
            continue
        _mark_expansion_direction(paper, "s2_related")
        papers.append(paper)
    return papers[:limit]


def _node_candidate_universe(
    sciatlas, node: Paper, direction: str, *, seen_ids: set[str], limit: int, cfg: "PipelineConfig | None" = None
) -> list[Paper]:
    """Citation neighbourhood of ``node`` in ``direction``.

    Forward (citing papers) is augmented by S2 *text* search for lateral recall.
    Backward (references) stays genealogical: it uses real reference edges only —
    OpenAlex ``referenced_works`` topped up with S2's reference list when OpenAlex is
    thin — and deliberately does NOT fall back to S2 text search, whose relevance
    ranking drifts toward recent on-topic papers and would mask the absence of true
    foundational references (the graph's only source of historical depth).
    """
    node = _ensure_seed_metadata(sciatlas, node)

    if direction == "forward":
        openalex = _forward_candidates(sciatlas, node, seen_ids=seen_ids, limit=limit)
        if openalex:
            sciatlas.record_external_retrieval(
                source="openalex", query=node.title, papers=openalex,
                extra={"direction": "forward", "node_id": node.paper_id},
            )
        if cfg is None:
            return openalex
        existing_ids = seen_ids | {p.paper_id for p in openalex}
        s2_papers = _s2_related_candidates(cfg, node, seen_ids=existing_ids, limit=max(limit // 2, 10))
        if s2_papers:
            sciatlas.record_external_retrieval(
                source="s2", query=node.title, papers=s2_papers,
                extra={"direction": "s2_related", "node_id": node.paper_id},
            )
        return openalex + s2_papers

    # backward
    openalex = _backward_candidates(sciatlas, node, seen_ids=seen_ids, limit=limit)
    if openalex:
        sciatlas.record_external_retrieval(
            source="openalex", query=node.title, papers=openalex,
            extra={"direction": "backward", "node_id": node.paper_id},
        )
    if cfg is None:
        return openalex

    backward = list(openalex)
    if len(backward) < limit:
        existing_ids = seen_ids | {p.paper_id for p in backward}
        s2_refs = _s2_paper_references(cfg.innoeval_root, node, limit=limit)
        added_refs: list[Paper] = []
        for paper in s2_refs:
            if paper.paper_id in existing_ids:
                continue
            existing_ids.add(paper.paper_id)
            _mark_expansion_direction(paper, "backward")
            added_refs.append(paper)
            if len(backward) + len(added_refs) >= limit:
                break
        if added_refs:
            sciatlas.record_external_retrieval(
                source="s2_refs", query=node.title, papers=added_refs,
                extra={"direction": "backward", "node_id": node.paper_id},
            )
        backward.extend(added_refs)

    if not backward:
        log.warning(
            "  node %s has no backward references (OpenAlex + S2 both empty)",
            node.paper_id[-12:] if node.paper_id else "?",
        )
    return backward


def _select_predecessors_with_llm(
    llm: LLMClient,
    node: Paper,
    ranked: list[dict[str, Any]],
    num_candidate: int,
    *,
    research_topic: str = "",
) -> list[Paper]:
    """LLM picks up to ``num_candidate`` predecessors weighing score + reason (surveys excluded).

    ``research_topic`` biases selection toward candidates relevant to BOTH the node
    and the overall topic, so repeated expansion keeps the graph centered on the
    topic instead of drifting into whatever adjacent area a node happened to cite.
    """
    usable = [item for item in ranked if item["llm_score"] > 0 and not _is_review_like(item["paper"])]
    if not usable:
        return []
    # Topic-aware ordering also makes the fallback (top-N) topic-centered, not just
    # the prompt: break ties on node-relevance by overlap with the research topic.
    if research_topic:
        usable.sort(
            key=lambda item: (item["llm_score"], _query_overlap(research_topic, item["paper"])),
            reverse=True,
        )
    lines = [
        "\n".join(
            [
                f"- paper_id: {item['paper_id']}",
                f"  title: {item['paper'].title}",
                f"  relevance_score: {item['llm_score']:.2f}",
                f"  reason: {item.get('reason', '')}",
                f"  abstract: {(item['paper'].abstract or '')[:400]}",
            ]
        )
        for item in usable
    ]
    prompt = PREDECESSOR_SELECTION.format(
        research_topic=research_topic or "(not specified)",
        node_title=node.title,
        node_abstract=(node.abstract or "")[:1200],
        candidate_list="\n".join(lines),
        num_candidate=num_candidate,
    )
    by_id = {item["paper_id"]: item["paper"] for item in usable}
    selected_ids: list[str] = []
    try:
        data = llm.chat_json(prompt, temperature=0.0)
        for paper_id in (data.get("selected_paper_ids", []) if isinstance(data, dict) else []):
            if paper_id in by_id and paper_id not in selected_ids:
                selected_ids.append(paper_id)
    except Exception as exc:
        log.warning("  predecessor selection failed, using top-ranked: %s", exc)
    if not selected_ids:
        selected_ids = [item["paper_id"] for item in usable[:num_candidate]]
    return [by_id[paper_id] for paper_id in selected_ids[:num_candidate]]


def _seed_int(text: str) -> int:
    return int(hashlib.sha1((text or "seed").encode("utf-8")).hexdigest()[:8], 16)


def run(sciatlas, llm: LLMClient, seeds: SeedSet, cfg: PipelineConfig) -> ResearchGraph:
    log.info("Step 2: citation idea-graph construction")
    seed_papers = list(seeds.seed_papers) or [Paper("seed", seeds.refined_query or "Seed paper")]
    seed_papers = [_ensure_seed_metadata(sciatlas, paper) for paper in seed_papers]
    seed_ids = {paper.paper_id for paper in seed_papers}
    # Keep expansion centered on the original topic, not just on each node paper.
    research_topic = (seeds.research_problem or seeds.refined_query or "").strip()

    # Budget is driven by the size of the seeds' citation neighbourhood — the
    # richer of its backward (references) and forward (citing papers) sides, so a
    # reference-poor but highly-cited seed still yields a full graph. OpenAlex
    # often has no referenced_works for arXiv/LLM seeds, so fall back to S2's
    # referenceCount to avoid flooring every graph at graph_budget_min.
    def _seed_neighbourhood_size(paper: Paper) -> int:
        refs = len(paper.references)
        raw = paper.raw if isinstance(paper.raw, dict) else {}
        cites = raw.get("cited_by_count") or raw.get("citation_count") or 0
        cites = int(cites) if isinstance(cites, (int, float)) else 0
        if refs == 0 and cites == 0:
            refs = _s2_reference_count(cfg.innoeval_root, paper.title)
        return max(refs, cites)

    max_seed_neigh = max((_seed_neighbourhood_size(paper) for paper in seed_papers), default=0)
    total_budget = _budget(max_seed_neigh or cfg.graph_budget_min, cfg)
    log.info("  graph budget: %d papers (max seed neighbourhood: %d)", total_budget, max_seed_neigh)
    rng = random.Random(_seed_int(seeds.refined_query))

    nodes: dict[str, Paper] = {}
    for paper in seed_papers:
        _mark_expansion_direction(paper, "anchor")
        nodes[paper.paper_id] = paper

    rerank_score: dict[str, float] = {paper.paper_id: 1.0 for paper in seed_papers}
    # Seeds expand in both directions so a reference-light seed still grows forward.
    frontier: deque[tuple[Paper, str]] = deque()
    for paper in seed_papers:
        frontier.append((paper, "backward"))
        frontier.append((paper, "forward"))

    iterations = 0
    max_iterations = total_budget * 4
    while frontier and len(nodes) < total_budget and iterations < max_iterations:
        iterations += 1
        node, direction = frontier.popleft()
        candidate_fetch_limit = max(
            1,
            int(getattr(cfg, "graph_candidate_fetch_limit", CANDIDATE_FETCH_LIMIT)),
        )
        candidates = _node_candidate_universe(
            sciatlas, node, direction, seen_ids=set(nodes), limit=candidate_fetch_limit, cfg=cfg
        )
        if not candidates:
            continue
        log.info(
            "  node %s expand %s -> %d candidates %s",
            node.paper_id,
            direction,
            len(candidates),
            _progress_bar(len(nodes), total_budget),
        )
        ranked = merge_rerank(
            llm,
            query_context=_query_context(node),
            candidates=candidates,
            innoeval_root=cfg.innoeval_root,
        )
        for item in ranked:
            rerank_score[item["paper_id"]] = max(
                rerank_score.get(item["paper_id"], 0.0), item["llm_score"] / 10.0
            )
        predecessors = _select_predecessors_with_llm(
            llm, node, ranked, cfg.num_candidate_step2, research_topic=research_topic
        )
        added: list[Paper] = []
        for paper in predecessors:
            if len(nodes) >= total_budget:
                break
            if paper.paper_id not in nodes:
                nodes[paper.paper_id] = paper
                added.append(paper)
        if not added:
            continue
        # Forward and backward are independent lineages: a paper selected while
        # expanding in `direction` continues in that SAME direction — forward keeps
        # walking citations from forward-selected papers, backward keeps walking
        # references from backward-selected papers. The two interleave because the
        # frontier is seeded with both directions per seed (above) and drained FIFO,
        # never crossing a backward-selected paper over into forward exploration.
        for paper in rng.sample(added, min(cfg.num_expansion_step2, len(added))):
            frontier.append((paper, direction))

    log.info("  collected %d papers (budget %d, %d iterations)", len(nodes), total_budget, iterations)

    # Requirement 7: dedup by identifiers/title, then build the citation graph.
    deduped = _dedupe_candidate_papers(
        seeds.refined_query,
        [("node", rank, paper) for rank, paper in enumerate(nodes.values(), start=1)],
    )

    graph = ResearchGraph(
        seed_paper_id=seed_papers[0].paper_id,
        seed_paper_ids=[paper.paper_id for paper in seed_papers],
    )
    for paper in deduped:
        graph.papers[paper.paper_id] = paper
        graph.evidence_scores[paper.paper_id] = (
            1.0 if paper.paper_id in seed_ids else round(rerank_score.get(paper.paper_id, 0.0), 4)
        )
    for paper in seed_papers:
        if paper.paper_id not in graph.papers:
            graph.papers[paper.paper_id] = paper
            graph.evidence_scores[paper.paper_id] = 1.0

    # Principle: every uploaded PDF (force-included as a seed in Step 1) MUST be
    # present in the research graph. Seeds are re-added above, so this is a guard
    # against a future regression (e.g. dedup dropping a seed id) rather than a
    # normal code path — log loudly if a PDF seed somehow went missing.
    for paper in seed_papers:
        is_pdf = isinstance(paper.raw, dict) and paper.raw.get("is_pdf_input")
        if is_pdf and paper.paper_id not in graph.papers:
            log.warning("  re-inserting missing PDF-input seed into graph: %s", paper.title)
            graph.papers[paper.paper_id] = paper
            graph.evidence_scores[paper.paper_id] = 1.0

    for paper_id, paper in graph.papers.items():
        if paper_id in seed_ids:
            paper.idea_fields = paper.idea_fields or _default_idea_fields(paper)
        else:
            paper.idea_fields = _extract_idea_fields(llm, paper)

    s_cite_lookup = {paper_id: graph.evidence_scores.get(paper_id, 0.0) for paper_id in graph.papers}
    for paper_id in seed_ids:
        s_cite_lookup[paper_id] = 1.0
    graph.edges.extend(_build_predecessor_edges(graph, seed_papers, s_cite_lookup, cfg))

    for paper_id, paper in graph.papers.items():
        for kw in paper.keywords or []:
            keyword = kw.lower().strip()
            if keyword:
                graph.edges.append(GraphEdge(paper_id, f"keyword:{keyword}", "has_keyword"))

    graph.phases = _assign_phases_with_llm(llm, graph, seed_papers)
    log.info(
        "  built graph: %d papers, %d edges, %d phases, %d connected components",
        len(graph.papers),
        len(graph.edges),
        len(graph.phases),
        _connected_component_count(graph),
    )
    return graph
