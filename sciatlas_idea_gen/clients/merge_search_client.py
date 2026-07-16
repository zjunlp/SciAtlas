"""Reuse innoeval's merge_search dedup + rerank machinery for Step 2.

This is a thin wrapper around ``/data1/zbs/innoeval/search/merge_search.py``. The
candidate *universe* in this pipeline is the seed/node citation neighbourhood
(references / citing papers), so we do not call merge_search's KG/S2 *search*
runners — we reuse only its source-agnostic, pure helpers:

  * ``filter_and_group_papers`` — the exact union-find dedup ("去重").
  * ``build_scoring_batches`` / ``compute_score_std`` / ``ranking_sort_key`` —
    the batch-rerank scaffolding (coverage, mean-of-batches, final ordering).

The LLM relevance call itself uses this repo's :class:`LLMClient` with the
repo-local ``MERGE_RERANK_RELEVANCE`` prompt, whose only difference from
merge_search's ``build_relevance_prompt`` is that survey/review papers are
scored 0.

``merge_search`` imports ``kg_search.service`` + ``search_s2`` at module top
(torch/neo4j), so the import here is **lazy** and only happens when Step 2 runs.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..models import Paper
from ..prompts import MERGE_RERANK_RELEVANCE
from ..utils import get_logger

log = get_logger("sciatlas.merge_search")

_MERGE_SEARCH_MODULE: Any = None


def _install_runner_stubs() -> None:
    """Stub merge_search's KG/S2 *retrieval runners* so its import succeeds here.

    ``merge_search`` imports ``kg_search.service`` and ``search_s2`` at module top
    purely for the semantic-search runners. Those modules load GPU/Neo4j config
    from a hardcoded, inaccessible ``.env`` at import time in this environment,
    and this pipeline never calls them (the candidate universe is the OpenAlex
    citation neighbourhood). We only need merge_search's pure dedup + rerank
    helpers, so we satisfy the import with no-op stubs.
    """
    if "kg_search.service" not in sys.modules:
        kg_pkg = sys.modules.get("kg_search") or types.ModuleType("kg_search")
        service = types.ModuleType("kg_search.service")
        service.run_search_with_authors = lambda *args, **kwargs: {"papers": [], "authors": []}
        kg_pkg.service = service
        sys.modules.setdefault("kg_search", kg_pkg)
        sys.modules["kg_search.service"] = service
    if "search_s2" not in sys.modules:
        s2 = types.ModuleType("search_s2")
        s2.run_pipeline = lambda *args, **kwargs: {"papers": []}
        sys.modules["search_s2"] = s2


def _load_merge_search(innoeval_root: Path) -> Any:
    """Import and cache the innoeval ``merge_search`` module (lazy)."""
    global _MERGE_SEARCH_MODULE
    if _MERGE_SEARCH_MODULE is not None:
        return _MERGE_SEARCH_MODULE

    root = innoeval_root.resolve()
    package_merge = root / "combined" / "merge_search.py"
    source_merge = root / "src" / "innoeval_search" / "combined" / "merge_search.py"
    old_search_merge = root / "search" / "merge_search.py"
    flat_merge = root / "merge_search.py"

    if package_merge.exists():
        src_dir = str(root.parent.resolve())
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        _MERGE_SEARCH_MODULE = importlib.import_module(f"{root.name}.combined.merge_search")
        return _MERGE_SEARCH_MODULE

    if source_merge.exists():
        src_dir = str((root / "src").resolve())
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        _MERGE_SEARCH_MODULE = importlib.import_module("innoeval_search.combined.merge_search")
        return _MERGE_SEARCH_MODULE

    if old_search_merge.exists():
        search_dir = str((root / "search").resolve())
        if search_dir not in sys.path:
            sys.path.insert(0, search_dir)
        _install_runner_stubs()
        _MERGE_SEARCH_MODULE = importlib.import_module("merge_search")
        return _MERGE_SEARCH_MODULE

    if flat_merge.exists():
        flat_dir = str(root)
        if flat_dir not in sys.path:
            sys.path.insert(0, flat_dir)
        _install_runner_stubs()
        _MERGE_SEARCH_MODULE = importlib.import_module("merge_search")
        return _MERGE_SEARCH_MODULE

    raise ModuleNotFoundError(f"merge_search.py not found under INNOEVAL_ROOT={root}")
    return _MERGE_SEARCH_MODULE


def _cite_count(paper: Paper) -> int:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    value = raw.get("cited_by_count") or raw.get("citation_count")
    if isinstance(value, (int, float)):
        return int(value)
    return len(paper.citations or [])


def _paper_to_record(paper: Paper) -> dict[str, Any]:
    """Build a merge_search-style paper dict from a :class:`Paper`."""
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    record: dict[str, Any] = {
        "title": paper.title,
        "abstract": paper.abstract,
        "year": paper.year,
        "cited_by_count": _cite_count(paper),
        # source "kg" makes build_identifier_set read the OpenAlex id from "id".
        "id": paper.paper_id,
        "url": paper.paper_id if str(paper.paper_id).startswith("http") else raw.get("url"),
        # merge_search drops groups without a pdf_url (it is built for PDF fetching).
        # We only reuse its dedup, so stand in the OpenAlex id to keep the group.
        "pdf_url": raw.get("pdf_url") or paper.paper_id,
        "doi": raw.get("doi"),
        "_seed_ref": paper.paper_id,
    }
    return record


def merge_rerank(
    llm,
    *,
    query_context: dict[str, str],
    candidates: list[Paper],
    innoeval_root: Path,
    batch_size: int = 4,
    paper_coverage: int = 2,
) -> list[dict[str, Any]]:
    """Dedup + LLM-rerank ``candidates`` against ``query_context``.

    Returns a relevance-sorted list of ``{"paper", "paper_id", "llm_score",
    "score_count", "reason"}`` for the deduplicated survivors. Survey/review
    papers are scored 0 by the prompt.
    """
    candidates = [paper for paper in candidates if paper.paper_id and paper.title]
    if not candidates:
        return []

    merge_search = _load_merge_search(innoeval_root)
    papers_by_ref: dict[str, Paper] = {paper.paper_id: paper for paper in candidates}

    combined = [
        {"source": "kg", "source_rank": rank, "paper": _paper_to_record(paper)}
        for rank, paper in enumerate(candidates, start=1)
    ]
    filter_payload = merge_search.filter_and_group_papers(combined)
    unique = filter_payload.get("unique_papers", []) or []
    if not unique:
        return []

    # Map each deduped group back to one of our Paper objects.
    group_to_paper: dict[str, Paper] = {}
    rerank_items: list[dict[str, Any]] = []
    for entry in unique:
        ref = ((entry.get("paper") or {}).get("_seed_ref")) or entry.get("paper_url") or ""
        paper = papers_by_ref.get(ref)
        if paper is None:
            # Fall back to title match against our candidates.
            title_key = merge_search.normalize_title(entry.get("title"))
            paper = next(
                (cand for cand in candidates if merge_search.normalize_title(cand.title) == title_key),
                None,
            )
        if paper is None:
            continue
        group_id = entry.get("group_id") or paper.paper_id
        group_to_paper[group_id] = paper
        rerank_items.append(
            {
                "group_id": group_id,
                "title": entry.get("title") or paper.title,
                "abstract": entry.get("abstract") or paper.abstract,
            }
        )

    if not rerank_items:
        return []

    seed_input = f"{query_context.get('query_type')}::{query_context.get('query_title')}::{query_context.get('query_text')}"
    seed = int(hashlib.sha1(seed_input.encode("utf-8")).hexdigest()[:8], 16)
    batches = merge_search.build_scoring_batches(
        paper_count=len(rerank_items),
        batch_size=max(2, batch_size),
        paper_coverage=max(1, paper_coverage),
        seed=seed,
    )

    scores_by_group: dict[str, list[float]] = {item["group_id"]: [] for item in rerank_items}
    reasons_by_group: dict[str, list[str]] = {item["group_id"]: [] for item in rerank_items}

    def _score_batch(batch: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
        batch_items = [rerank_items[index] for index in batch["paper_indices"]]
        blocks = "\n\n".join(
            f"[{slot}] Title: {item['title']}\nAbstract: {item['abstract'] or '[missing abstract]'}"
            for slot, item in enumerate(batch_items, start=1)
        )
        prompt = MERGE_RERANK_RELEVANCE.format(
            query_type=query_context.get("query_type", "paper_abstract"),
            query_title=query_context.get("query_title", ""),
            query_text=query_context.get("query_text", ""),
            candidate_blocks=blocks,
        )
        try:
            return batch_items, llm.chat_json(prompt, temperature=0.1)
        except Exception as exc:  # noqa: BLE001 - keep reranking the other batches
            log.warning("  merge rerank batch failed: %s", exc)
            return batch_items, None

    with ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        batch_results = list(executor.map(_score_batch, batches))

    for batch_items, data in batch_results:
        for item in (data.get("papers", []) if isinstance(data, dict) else []):
            if not isinstance(item, dict):
                continue
            try:
                slot = int(item.get("paper_index"))
                score = max(0.0, min(10.0, float(item.get("score"))))
            except (TypeError, ValueError):
                continue
            if slot < 1 or slot > len(batch_items):
                continue
            group_id = batch_items[slot - 1]["group_id"]
            scores_by_group[group_id].append(score)
            reason = str(item.get("reason") or "").strip()
            if reason:
                reasons_by_group[group_id].append(reason)

    ranked: list[dict[str, Any]] = []
    for item in rerank_items:
        group_id = item["group_id"]
        scores = scores_by_group.get(group_id, [])
        if not scores:
            continue
        paper = group_to_paper[group_id]
        reasons = reasons_by_group.get(group_id, [])
        ranked.append(
            {
                "paper": paper,
                "paper_id": paper.paper_id,
                "llm_score": round(sum(scores) / len(scores), 4),
                "score_count": len(scores),
                "score_std": round(merge_search.compute_score_std(scores), 4),
                "citation_count": _cite_count(paper),
                "title": paper.title,
                "reason": reasons[0] if reasons else "",
                "reasons": reasons,
            }
        )

    ranked.sort(key=merge_search.ranking_sort_key)
    return ranked
