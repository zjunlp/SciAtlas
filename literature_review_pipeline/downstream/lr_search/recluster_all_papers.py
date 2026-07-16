#!/usr/bin/env python3
"""Full-corpus re-clustering for lr_search artifacts.

The original literature_review_search.py truncates the cluster prompt at
``--llm-paper-limit`` (default 80), which silently dumps every other paper into
``residual_unassigned`` -> ``outliers``. That makes 80-90% of retrieved papers
invisible to the clustering LLM. This script re-clusters from scratch using all
paper cards via a Preview + Map-Reduce + Stitch + optional Lexical-Rescue
pipeline.

Usage:
    python3 recluster_all_papers.py \
        --search-result <path/to/search_result.json> \
        --output        <path/to/clusters_recovered.json>

The output schema matches downstream/lr_search/clusters_final.json so it can be
fed directly to:
    organize_search_result.py --clusters clusters_recovered.json ...
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LR_SEARCH_DIR = Path(__file__).resolve().parent
if str(LR_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(LR_SEARCH_DIR))

from literature_review_search import (  # noqa: E402
    DEFAULT_DMX_API_URL,
    DEFAULT_DMX_MODEL,
    DEFAULT_ENV_PATH,
    DmxJsonClient,
    SYSTEM_PROMPT,
    format_elapsed,
    list_of_strings,
    normalize_whitespace,
    parse_json_object,
    paper_cards_for_llm,
    progress_log,
    progress_stage,
    truncate_text,
    write_json,
)
from organize_search_result import (  # noqa: E402
    build_cluster_profiles,
    paper_text,
    score_cluster,
    tokenize,
)


# ----------------------------- IO helpers -----------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def stable_sort_paper_ids(paper_ids: list[str]) -> list[str]:
    return sorted(set(paper_ids))


# ----------------------------- Preview step -----------------------------


def build_preview_prompt(
    *,
    topic_profile: dict[str, Any],
    paper_titles: list[dict[str, Any]],
    target_min: int,
    target_max: int,
) -> str:
    return (
        "You are scoping the method-family taxonomy of a literature corpus before any clustering.\n"
        "Read the topic profile and the title-only list, then propose candidate method families / research threads.\n"
        "These candidates are a HINT for later batched clustering; they do not yet bind any paper.\n"
        f"Target {target_min}-{target_max} candidates that collectively cover the visible breadth of the corpus.\n"
        "Each candidate must be a coherent technical line, not a venue, year, or application domain.\n"
        "Return strict JSON only.\n\n"
        "Schema:\n"
        "{\n"
        '  "preview_clusters": [\n'
        '    {"tentative_name": "...", "one_line_definition": "...", "representative_title_examples": ["title", "title"]}\n'
        "  ],\n"
        '  "global_observations": "1-2 lines about visible eras, paradigms, or near-duplicate families"\n'
        "}\n\n"
        f"Topic profile:\n{json.dumps(topic_profile, ensure_ascii=False)}\n\n"
        f"Paper titles ({len(paper_titles)} items):\n{json.dumps(paper_titles, ensure_ascii=False)}\n"
    )


def run_preview(
    *,
    client: DmxJsonClient,
    boosted_client: DmxJsonClient,
    topic_profile: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    preview_title_limit: int,
    target_min: int,
    target_max: int,
    out_path: Path,
    resume: bool,
    progress: bool,
) -> dict[str, Any]:
    if resume and file_exists_nonempty(out_path):
        progress_log(progress, f"PREVIEW resume from {out_path}")
        return read_json(out_path)

    titles = [
        {"paper_id": card.get("paper_id"), "title": truncate_text(card.get("title"), 220)}
        for card in paper_cards[:preview_title_limit]
        if card.get("paper_id")
    ]
    prompt = build_preview_prompt(
        topic_profile=topic_profile,
        paper_titles=titles,
        target_min=target_min,
        target_max=target_max,
    )
    with progress_stage(progress, "preview"):
        payload = call_llm_with_retry(client, boosted_client, prompt=prompt, label="recluster_preview")
    write_json(out_path, payload)
    return payload


# ----------------------------- Map step -----------------------------


def build_batch_prompt(
    *,
    topic_profile: dict[str, Any],
    preview_clusters: list[dict[str, Any]],
    batch_id: str,
    batch_cards: list[dict[str, Any]],
    abstract_char_limit: int,
    min_local_cluster_size: int,
) -> str:
    compact_cards = paper_cards_for_llm(
        batch_cards,
        limit=len(batch_cards),
        abstract_char_limit=abstract_char_limit,
    )
    return (
        "You cluster a SINGLE BATCH of papers into local method-family clusters.\n"
        "Use the preview hints as suggestions; you may adopt them, refine them, or invent new ones if the batch warrants it.\n"
        "Rules:\n"
        "- Assign each paper to EXACTLY ONE local cluster, or mark it OUT for this batch.\n"
        f"- Each local cluster must contain >= {min_local_cluster_size} papers; otherwise mark its papers OUT.\n"
        "- Cluster names should describe a method family / research line, NOT a venue, year, or application domain.\n"
        "- When possible reuse a preview tentative_name verbatim so downstream merging is easier.\n"
        "- Do not invent paper_ids outside the batch input.\n"
        "Return strict JSON only.\n\n"
        "Schema:\n"
        "{\n"
        f'  "batch_id": "{batch_id}",\n'
        '  "local_clusters": [\n'
        f'    {{"local_cluster_id": "{batch_id}_L1", "name": "...", "definition": "...", '
        '"paper_ids": ["..."], "representative_paper_ids": ["..."]}\n'
        "  ],\n"
        '  "out_paper_ids": ["..."]\n'
        "}\n\n"
        f"Topic profile:\n{json.dumps(topic_profile, ensure_ascii=False)}\n\n"
        f"Preview cluster hints:\n{json.dumps(preview_clusters, ensure_ascii=False)}\n\n"
        f"Batch {batch_id} papers ({len(batch_cards)} items):\n"
        f"{json.dumps(compact_cards, ensure_ascii=False)}\n"
    )


def sanitize_batch_payload(payload: dict[str, Any], *, batch_id: str, batch_ids_set: set[str]) -> dict[str, Any]:
    local_clusters_in = payload.get("local_clusters") if isinstance(payload, dict) else None
    local_clusters_in = local_clusters_in if isinstance(local_clusters_in, list) else []

    local_clusters_out: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()
    for idx, item in enumerate(local_clusters_in, start=1):
        if not isinstance(item, dict):
            continue
        local_id = normalize_whitespace(item.get("local_cluster_id")) or f"{batch_id}_L{idx}"
        # Force batch prefix to make local ids globally unique
        if not local_id.startswith(batch_id + "_"):
            local_id = f"{batch_id}_L{idx}"
        paper_ids = [
            pid for pid in list_of_strings(item.get("paper_ids"), limit=2000)
            if pid in batch_ids_set and pid not in assigned_ids
        ]
        if not paper_ids:
            continue
        reps = [pid for pid in list_of_strings(item.get("representative_paper_ids"), limit=10) if pid in set(paper_ids)]
        local_clusters_out.append(
            {
                "local_cluster_id": local_id,
                "name": normalize_whitespace(item.get("name")) or local_id,
                "definition": normalize_whitespace(item.get("definition")),
                "paper_ids": sorted(paper_ids),
                "representative_paper_ids": reps[:5],
                "paper_count": len(paper_ids),
            }
        )
        assigned_ids.update(paper_ids)

    declared_out = [pid for pid in list_of_strings((payload or {}).get("out_paper_ids"), limit=2000) if pid in batch_ids_set]
    out_set = (set(declared_out) | (batch_ids_set - assigned_ids)) - assigned_ids
    return {
        "batch_id": batch_id,
        "local_clusters": local_clusters_out,
        "out_paper_ids": sorted(out_set),
        "input_paper_count": len(batch_ids_set),
        "assigned_paper_count": len(assigned_ids),
    }


def run_map(
    *,
    client: DmxJsonClient,
    boosted_client: DmxJsonClient,
    topic_profile: dict[str, Any],
    preview_clusters: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    batch_size: int,
    abstract_char_limit: int,
    min_local_cluster_size: int,
    shuffle_seed: int,
    max_parallel: int,
    batches_dir: Path,
    resume: bool,
    progress: bool,
) -> list[dict[str, Any]]:
    cards_by_id = {card["paper_id"]: card for card in paper_cards if card.get("paper_id")}
    paper_ids_sorted = sorted(cards_by_id.keys())
    rng = random.Random(shuffle_seed)
    rng.shuffle(paper_ids_sorted)

    batches: list[tuple[str, list[str]]] = []
    for i in range(0, len(paper_ids_sorted), batch_size):
        chunk = paper_ids_sorted[i : i + batch_size]
        batch_id = f"B{(i // batch_size) + 1:02d}"
        batches.append((batch_id, chunk))

    batches_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    def run_one(batch_id: str, batch_paper_ids: list[str]) -> tuple[str, dict[str, Any]]:
        out_path = batches_dir / f"{batch_id}.json"
        if resume and file_exists_nonempty(out_path):
            progress_log(progress, f"MAP {batch_id} resume")
            return batch_id, read_json(out_path)
        batch_cards = [cards_by_id[pid] for pid in batch_paper_ids]
        batch_ids_set = set(batch_paper_ids)
        prompt = build_batch_prompt(
            topic_profile=topic_profile,
            preview_clusters=preview_clusters,
            batch_id=batch_id,
            batch_cards=batch_cards,
            abstract_char_limit=abstract_char_limit,
            min_local_cluster_size=min_local_cluster_size,
        )
        started = time.perf_counter()
        try:
            payload = call_llm_with_retry(
                client,
                boosted_client,
                prompt=prompt,
                label=f"recluster_map_{batch_id}",
            )
        except Exception as exc:  # pragma: no cover - degraded mode
            progress_log(progress, f"MAP {batch_id} FAILED after retries: {exc}")
            payload = {"local_clusters": [], "out_paper_ids": sorted(batch_ids_set)}
        sanitized = sanitize_batch_payload(payload, batch_id=batch_id, batch_ids_set=batch_ids_set)
        sanitized["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        write_json(out_path, sanitized)
        progress_log(
            progress,
            f"MAP {batch_id} clusters={len(sanitized['local_clusters'])} "
            f"assigned={sanitized['assigned_paper_count']}/{sanitized['input_paper_count']} "
            f"in {format_elapsed(sanitized['elapsed_seconds'])}",
        )
        return batch_id, sanitized

    max_workers = max(1, min(max_parallel, len(batches)))
    with progress_stage(progress, f"map ({len(batches)} batches, parallel={max_workers})"):
        if max_workers == 1:
            for batch_id, ids in batches:
                bid, payload = run_one(batch_id, ids)
                results[bid] = payload
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_one, bid, ids): bid for bid, ids in batches}
                for fut in concurrent.futures.as_completed(futures):
                    bid, payload = fut.result()
                    results[bid] = payload

    ordered = [results[bid] for bid, _ in batches if bid in results]
    return ordered


# ----------------------------- Reduce step -----------------------------


def build_reduce_prompt(
    *,
    topic_profile: dict[str, Any],
    local_cluster_descriptors: list[dict[str, Any]],
    target_min: int,
    target_max: int,
    min_global_cluster_size: int,
) -> str:
    return (
        "You merge per-batch local clusters into a final global taxonomy.\n"
        "Local clusters were produced by independent batches over the same corpus, so many will overlap or duplicate.\n"
        "You do NOT see paper abstracts here; reason only over names, definitions, representative_titles, and paper_count.\n"
        "Rules:\n"
        f"- Output {target_min}-{target_max} global clusters, each describing a method family or research line.\n"
        f"- Each global cluster must end up with >= {min_global_cluster_size} papers across the local clusters it merges.\n"
        "- Every local_cluster_id must either appear in merged_from_local_cluster_ids of exactly one global cluster, "
        "or appear in discarded_local_cluster_ids (if its content is not a coherent method family that belongs in the review).\n"
        "- Do not invent global clusters that have no local cluster mapped into them.\n"
        "- Prefer splitting an over-broad local cluster across multiple global clusters when its description spans multiple method families.\n"
        "- Mark global clusters as core (central narrative) or peripheral (kept but tangential).\n"
        "Return strict JSON only.\n\n"
        "Schema:\n"
        "{\n"
        '  "global_clusters": [\n'
        '    {"cluster_id": "C1", "name": "...", "definition": "...", '
        '"topic_relevance": "core", "merged_from_local_cluster_ids": ["B01_L2"], '
        '"distinguishing_features": ["..."]}\n'
        "  ],\n"
        '  "discarded_local_cluster_ids": ["..."],\n'
        '  "merge_rationale": "..."\n'
        "}\n\n"
        f"Topic profile:\n{json.dumps(topic_profile, ensure_ascii=False)}\n\n"
        f"Local cluster descriptors ({len(local_cluster_descriptors)} items):\n"
        f"{json.dumps(local_cluster_descriptors, ensure_ascii=False)}\n"
    )


def collect_local_cluster_descriptors(
    batch_payloads: list[dict[str, Any]],
    paper_cards_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    descriptors = []
    for batch in batch_payloads:
        for cluster in batch.get("local_clusters", []):
            rep_titles = []
            for pid in cluster.get("representative_paper_ids", []) or cluster.get("paper_ids", [])[:3]:
                card = paper_cards_by_id.get(pid)
                if card and card.get("title"):
                    rep_titles.append(truncate_text(card.get("title"), 180))
                if len(rep_titles) >= 3:
                    break
            descriptors.append(
                {
                    "local_cluster_id": cluster["local_cluster_id"],
                    "batch_id": batch["batch_id"],
                    "name": cluster["name"],
                    "definition": cluster["definition"],
                    "paper_count": cluster["paper_count"],
                    "representative_titles": rep_titles,
                }
            )
    return descriptors


def run_reduce(
    *,
    client: DmxJsonClient,
    boosted_client: DmxJsonClient,
    topic_profile: dict[str, Any],
    batch_payloads: list[dict[str, Any]],
    paper_cards_by_id: dict[str, dict[str, Any]],
    target_min: int,
    target_max: int,
    min_global_cluster_size: int,
    out_path: Path,
    resume: bool,
    progress: bool,
) -> dict[str, Any]:
    if resume and file_exists_nonempty(out_path):
        progress_log(progress, f"REDUCE resume from {out_path}")
        return read_json(out_path)
    descriptors = collect_local_cluster_descriptors(batch_payloads, paper_cards_by_id)
    if not descriptors:
        payload = {"global_clusters": [], "discarded_local_cluster_ids": [], "merge_rationale": "no local clusters produced"}
        write_json(out_path, payload)
        return payload

    prompt = build_reduce_prompt(
        topic_profile=topic_profile,
        local_cluster_descriptors=descriptors,
        target_min=target_min,
        target_max=target_max,
        min_global_cluster_size=min_global_cluster_size,
    )
    with progress_stage(progress, "reduce"):
        payload = call_llm_with_retry(client, boosted_client, prompt=prompt, label="recluster_reduce")
    payload["_local_cluster_descriptors"] = descriptors  # for debugging
    write_json(out_path, payload)
    return payload


# ----------------------------- Stitch step -----------------------------


def stitch_global_assignments(
    *,
    reduce_payload: dict[str, Any],
    batch_payloads: list[dict[str, Any]],
    paper_cards_by_id: dict[str, dict[str, Any]],
    min_global_cluster_size: int,
) -> dict[str, Any]:
    # local_cluster_id -> global_cluster_id ("DISCARDED" if discarded)
    mapping: dict[str, str] = {}
    global_meta: dict[str, dict[str, Any]] = {}

    for index, item in enumerate(reduce_payload.get("global_clusters", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        cluster_id = normalize_whitespace(item.get("cluster_id")) or f"C{index}"
        topic_relevance = normalize_whitespace(item.get("topic_relevance")).casefold() or "core"
        keep_in_review = topic_relevance != "off_topic"
        global_meta[cluster_id] = {
            "cluster_id": cluster_id,
            "name": normalize_whitespace(item.get("name")) or cluster_id,
            "definition": normalize_whitespace(item.get("definition")),
            "topic_relevance": topic_relevance,
            "keep_in_review": keep_in_review,
            "distinguishing_features": list_of_strings(item.get("distinguishing_features"), limit=12),
        }
        for local_id in list_of_strings(item.get("merged_from_local_cluster_ids"), limit=2000):
            mapping[local_id] = cluster_id

    for local_id in list_of_strings(reduce_payload.get("discarded_local_cluster_ids"), limit=2000):
        mapping.setdefault(local_id, "DISCARDED")

    # Collect papers per global cluster.
    cluster_paper_ids: dict[str, list[str]] = defaultdict(list)
    paper_assignment_log: dict[str, dict[str, Any]] = {}
    discarded_paper_ids: set[str] = set()
    batch_out_paper_ids: set[str] = set()
    unmapped_local_clusters: set[str] = set()

    for batch in batch_payloads:
        for local_cluster in batch.get("local_clusters", []):
            local_id = local_cluster["local_cluster_id"]
            target = mapping.get(local_id)
            if target is None:
                unmapped_local_clusters.add(local_id)
                continue
            if target == "DISCARDED":
                discarded_paper_ids.update(local_cluster["paper_ids"])
                continue
            for pid in local_cluster["paper_ids"]:
                if pid in paper_assignment_log:
                    continue
                cluster_paper_ids[target].append(pid)
                paper_assignment_log[pid] = {
                    "cluster_id": target,
                    "via_local_cluster_id": local_id,
                    "source": "llm_map_reduce",
                }
        for pid in batch.get("out_paper_ids", []):
            if pid not in paper_assignment_log:
                batch_out_paper_ids.add(pid)

    # Build final cluster records, dropping clusters that lost too many papers.
    final_clusters: list[dict[str, Any]] = []
    dropped_cluster_ids: list[str] = []
    for cluster_id, meta in global_meta.items():
        ids = sorted(set(cluster_paper_ids.get(cluster_id, [])))
        if len(ids) < min_global_cluster_size or not meta["keep_in_review"]:
            dropped_cluster_ids.append(cluster_id)
            # Move papers from a dropped cluster to outliers (do NOT silently re-route).
            for pid in ids:
                paper_assignment_log[pid] = {
                    "cluster_id": None,
                    "via_local_cluster_id": paper_assignment_log[pid]["via_local_cluster_id"],
                    "source": "dropped_undersized_cluster",
                }
                discarded_paper_ids.add(pid)
            continue
        reps_sorted = sorted(
            ids,
            key=lambda pid: -(paper_cards_by_id.get(pid, {}).get("citation_count") or 0),
        )
        final_clusters.append(
            {
                "cluster_id": cluster_id,
                "name": meta["name"],
                "definition": meta["definition"],
                "topic_relevance": meta["topic_relevance"],
                "keep_in_review": True,
                "relevance_rationale": "",
                "distinguishing_features": meta["distinguishing_features"],
                "paper_ids": ids,
                "representative_paper_ids": reps_sorted[:5],
                "missing_signals": [],
            }
        )

    assigned_ids = {pid for cluster in final_clusters for pid in cluster["paper_ids"]}
    outlier_ids = sorted(
        set(paper_cards_by_id.keys()) - assigned_ids
    )
    return {
        "clusters": final_clusters,
        "outliers": outlier_ids,
        "discarded_paper_ids": sorted(discarded_paper_ids),
        "unmapped_local_cluster_ids": sorted(unmapped_local_clusters),
        "batch_out_paper_ids": sorted(batch_out_paper_ids),
        "paper_assignment_log": paper_assignment_log,
        "dropped_cluster_ids": dropped_cluster_ids,
    }


# ----------------------------- Lexical rescue -----------------------------


def lexical_rescue(
    *,
    stitched: dict[str, Any],
    paper_cards_by_id: dict[str, dict[str, Any]],
    threshold: float,
    min_token_overlap: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not stitched["clusters"] or not stitched["outliers"]:
        return stitched, {"rescued": 0, "still_outlier": len(stitched["outliers"])}

    # Build cluster-profile structures compatible with organize_search_result.build_cluster_profiles.
    clusters_for_profile = []
    for cluster in stitched["clusters"]:
        clusters_for_profile.append(
            {
                "cluster_id": cluster["cluster_id"],
                "name": cluster["name"],
                "definition": cluster["definition"],
                "distinguishing_features": cluster["distinguishing_features"],
                "seed_paper_ids": cluster["paper_ids"],
                "representative_seed_paper_ids": cluster["representative_paper_ids"],
                "missing_signals": [],
            }
        )
    profiles = build_cluster_profiles(clusters_for_profile, paper_cards_by_id)
    clusters_by_id = {cluster["cluster_id"]: cluster for cluster in stitched["clusters"]}

    rescued = 0
    new_outliers: list[str] = []
    for paper_id in stitched["outliers"]:
        paper = paper_cards_by_id.get(paper_id)
        if not paper:
            continue
        tokens = tokenize(paper_text(paper))
        if not tokens:
            new_outliers.append(paper_id)
            continue
        scored = [(cid, score_cluster(paper, cid, profile, tokens)) for cid, profile in profiles.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_cid, best_score = scored[0] if scored else ("", 0.0)
        best_profile_tokens = profiles.get(best_cid, {}).get("tokens", set())
        overlap = len(tokens & best_profile_tokens) if best_profile_tokens else 0
        if best_score >= threshold and overlap >= min_token_overlap:
            cluster = clusters_by_id[best_cid]
            cluster["paper_ids"] = sorted(set(cluster["paper_ids"]) | {paper_id})
            stitched["paper_assignment_log"][paper_id] = {
                "cluster_id": best_cid,
                "via_local_cluster_id": None,
                "source": "lexical_rescue",
                "score": best_score,
                "token_overlap": overlap,
            }
            rescued += 1
        else:
            new_outliers.append(paper_id)

    stitched["outliers"] = sorted(new_outliers)
    stats = {"rescued": rescued, "still_outlier": len(stitched["outliers"])}
    # Refresh representative_paper_ids for clusters that gained members.
    for cluster in stitched["clusters"]:
        ids_sorted = sorted(
            cluster["paper_ids"],
            key=lambda pid: -(paper_cards_by_id.get(pid, {}).get("citation_count") or 0),
        )
        cluster["representative_paper_ids"] = ids_sorted[:5]
    return stitched, stats


# ----------------------------- LLM retry shim -----------------------------


def call_llm_with_retry(
    client: DmxJsonClient,
    boosted_client: DmxJsonClient,
    *,
    prompt: str,
    label: str,
) -> dict[str, Any]:
    """Call the LLM, falling back to a higher-max-tokens client on parse failure."""
    try:
        return client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label=label)
    except Exception as exc:  # noqa: BLE001
        last_error: Exception = exc
    try:
        return boosted_client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, label=label + "_retry")
    except Exception as exc2:  # noqa: BLE001
        raise RuntimeError(f"LLM call failed twice for label={label}: {last_error}; retry={exc2}") from exc2


# ----------------------------- Assembly -----------------------------


def assemble_final_payload(
    *,
    rescued: dict[str, Any],
    paper_cards_by_id: dict[str, dict[str, Any]],
    reduce_payload: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    clusters_out: list[dict[str, Any]] = []
    for cluster in rescued["clusters"]:
        ids_sorted = sorted(cluster["paper_ids"])
        reps_sorted = sorted(
            ids_sorted,
            key=lambda pid: -(paper_cards_by_id.get(pid, {}).get("citation_count") or 0),
        )[:5]
        clusters_out.append(
            {
                "cluster_id": cluster["cluster_id"],
                "name": cluster["name"],
                "definition": cluster["definition"],
                "topic_relevance": cluster["topic_relevance"],
                "keep_in_review": True,
                "relevance_rationale": "",
                "distinguishing_features": cluster["distinguishing_features"],
                "paper_ids": ids_sorted,
                "representative_paper_ids": reps_sorted,
                "missing_signals": [],
            }
        )

    return {
        "clusters": clusters_out,
        "outliers": rescued["outliers"],
        "model_outliers": rescued["batch_out_paper_ids"],
        "residual_unassigned_paper_ids": rescued["outliers"],
        "hard_excluded_paper_ids": rescued["discarded_paper_ids"],
        "uncertain_assignments": [],
        "dropped_cluster_ids": rescued["dropped_cluster_ids"],
        "recluster_metadata": {
            **metadata,
            "cluster_count": len(clusters_out),
            "papers_assigned_by_llm": sum(
                1 for v in rescued["paper_assignment_log"].values() if v.get("source") == "llm_map_reduce"
            ),
            "papers_assigned_by_lexical_rescue": sum(
                1 for v in rescued["paper_assignment_log"].values() if v.get("source") == "lexical_rescue"
            ),
            "papers_outlier": len(rescued["outliers"]),
            "coverage": (
                round(1 - len(rescued["outliers"]) / max(1, len(paper_cards_by_id)), 4)
            ),
            "merge_rationale": normalize_whitespace(reduce_payload.get("merge_rationale")),
            "unmapped_local_cluster_ids": rescued["unmapped_local_cluster_ids"],
            "paper_assignment_log": rescued["paper_assignment_log"],
        },
    }


# ----------------------------- Driver -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-corpus re-clustering for lr_search artifacts.")
    parser.add_argument("--search-result", required=True, help="Path to lr_search search_result.json.")
    parser.add_argument("--output", default=None, help="Output clusters_recovered.json path.")
    parser.add_argument("--work-dir", default=None, help="Intermediate artifact directory (preview/batches/reduce).")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env.")
    parser.add_argument("--llm-api-url", default=DEFAULT_DMX_API_URL)
    parser.add_argument("--llm-model", default=DEFAULT_DMX_MODEL)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-max-tokens", type=int, default=6000)
    parser.add_argument("--llm-retry-max-tokens", type=int, default=10000)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--abstract-char-limit", type=int, default=700)
    parser.add_argument("--preview-title-limit", type=int, default=600)
    parser.add_argument("--target-cluster-count-min", type=int, default=5)
    parser.add_argument("--target-cluster-count-max", type=int, default=15)
    parser.add_argument("--min-local-cluster-size", type=int, default=2)
    parser.add_argument("--min-global-cluster-size", type=int, default=3)
    parser.add_argument("--max-parallel-batches", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--enable-lexical-rescue", action="store_true")
    parser.add_argument("--lexical-rescue-threshold", type=float, default=0.05)
    parser.add_argument("--lexical-rescue-min-overlap", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse preview / batches / reduce files if present.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    progress = not args.quiet

    search_result_path = Path(args.search_result).expanduser().resolve()
    search_result = read_json(search_result_path)
    if not isinstance(search_result, dict):
        raise SystemExit(f"search_result.json must be a JSON object: {search_result_path}")
    paper_cards = [card for card in search_result.get("paper_cards", []) if isinstance(card, dict) and card.get("paper_id")]
    if not paper_cards:
        raise SystemExit(f"No paper_cards found in {search_result_path}")

    topic_profile = search_result.get("topic_profile") or {}
    paper_cards_by_id = {card["paper_id"]: card for card in paper_cards}

    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else search_result_path.parent / "recluster"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else search_result_path.parent / "clusters_recovered.json"
    )

    progress_log(
        progress,
        f"START papers={len(paper_cards)} batch_size={args.batch_size} "
        f"parallel={args.max_parallel_batches} model={args.llm_model}",
    )

    client = DmxJsonClient(
        env_path=Path(args.env).expanduser().resolve(),
        api_url=args.llm_api_url,
        model=args.llm_model,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        use_env_proxy=args.use_env_proxy,
    )
    boosted_client = DmxJsonClient(
        env_path=Path(args.env).expanduser().resolve(),
        api_url=args.llm_api_url,
        model=args.llm_model,
        timeout=args.llm_timeout * 2,
        max_tokens=args.llm_retry_max_tokens,
        temperature=args.llm_temperature,
        use_env_proxy=args.use_env_proxy,
    )

    started_at = time.perf_counter()

    preview_payload = run_preview(
        client=client,
        boosted_client=boosted_client,
        topic_profile=topic_profile,
        paper_cards=sorted(paper_cards, key=lambda c: -(c.get("citation_count") or 0)),
        preview_title_limit=args.preview_title_limit,
        target_min=args.target_cluster_count_min,
        target_max=args.target_cluster_count_max,
        out_path=work_dir / "preview.json",
        resume=args.resume,
        progress=progress,
    )
    preview_clusters = preview_payload.get("preview_clusters") or []

    batch_payloads = run_map(
        client=client,
        boosted_client=boosted_client,
        topic_profile=topic_profile,
        preview_clusters=preview_clusters,
        paper_cards=paper_cards,
        batch_size=args.batch_size,
        abstract_char_limit=args.abstract_char_limit,
        min_local_cluster_size=args.min_local_cluster_size,
        shuffle_seed=args.shuffle_seed,
        max_parallel=args.max_parallel_batches,
        batches_dir=work_dir / "batches",
        resume=args.resume,
        progress=progress,
    )

    reduce_payload = run_reduce(
        client=client,
        boosted_client=boosted_client,
        topic_profile=topic_profile,
        batch_payloads=batch_payloads,
        paper_cards_by_id=paper_cards_by_id,
        target_min=args.target_cluster_count_min,
        target_max=args.target_cluster_count_max,
        min_global_cluster_size=args.min_global_cluster_size,
        out_path=work_dir / "reduce.json",
        resume=args.resume,
        progress=progress,
    )

    stitched = stitch_global_assignments(
        reduce_payload=reduce_payload,
        batch_payloads=batch_payloads,
        paper_cards_by_id=paper_cards_by_id,
        min_global_cluster_size=args.min_global_cluster_size,
    )
    write_json(work_dir / "stitched.json", {**stitched, "paper_assignment_log_size": len(stitched["paper_assignment_log"])})

    rescue_stats = {"rescued": 0, "still_outlier": len(stitched["outliers"])}
    if args.enable_lexical_rescue:
        with progress_stage(progress, "lexical_rescue"):
            stitched, rescue_stats = lexical_rescue(
                stitched=stitched,
                paper_cards_by_id=paper_cards_by_id,
                threshold=args.lexical_rescue_threshold,
                min_token_overlap=args.lexical_rescue_min_overlap,
            )

    metadata = {
        "method": "preview+map_reduce+stitch" + ("+lexical_rescue" if args.enable_lexical_rescue else ""),
        "search_result_path": str(search_result_path),
        "work_dir": str(work_dir),
        "preview_path": str((work_dir / "preview.json").resolve()),
        "reduce_path": str((work_dir / "reduce.json").resolve()),
        "stitched_path": str((work_dir / "stitched.json").resolve()),
        "batch_count": len(batch_payloads),
        "batch_size": args.batch_size,
        "max_parallel_batches": args.max_parallel_batches,
        "shuffle_seed": args.shuffle_seed,
        "llm_model": args.llm_model,
        "llm_temperature": args.llm_temperature,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_retry_max_tokens": args.llm_retry_max_tokens,
        "lexical_rescue_enabled": args.enable_lexical_rescue,
        "lexical_rescue_threshold": args.lexical_rescue_threshold,
        "lexical_rescue_min_overlap": args.lexical_rescue_min_overlap,
        "lexical_rescue_rescued": rescue_stats["rescued"],
        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
        "total_llm_calls": client.call_count + boosted_client.call_count,
        "preview_cluster_count": len(preview_clusters),
        "input_paper_count": len(paper_cards),
    }
    final_payload = assemble_final_payload(
        rescued=stitched,
        paper_cards_by_id=paper_cards_by_id,
        reduce_payload=reduce_payload,
        metadata=metadata,
    )
    write_json(output_path, final_payload)

    summary = {
        "output": str(output_path),
        "cluster_count": len(final_payload["clusters"]),
        "papers_total": len(paper_cards),
        "papers_assigned": len(paper_cards) - len(final_payload["outliers"]),
        "papers_outlier": len(final_payload["outliers"]),
        "coverage": final_payload["recluster_metadata"]["coverage"],
        "elapsed": format_elapsed(metadata["elapsed_seconds"]),
        "total_llm_calls": metadata["total_llm_calls"],
        "cluster_sizes": Counter(
            cluster["cluster_id"] + ":" + str(len(cluster["paper_ids"]))
            for cluster in final_payload["clusters"]
        ),
    }
    if args.pretty:
        print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
