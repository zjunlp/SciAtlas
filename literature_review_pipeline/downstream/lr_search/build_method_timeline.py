#!/usr/bin/env python3
"""Build per-method timeline data from a recluster_all_papers.py output.

Produces RAW, ANNOTATED per-paper records. Non-landmark papers are NOT dropped;
landmark status is exposed as flags / scores so the downstream consumer (human
or LLM) can re-filter without losing information.

Inputs:
    - search_result.json   (lr_search artifact; supplies paper cards + time windows)
    - clusters_recovered.json  (recluster_all_papers.py output; supplies clusters
                                and paper_assignment_log)

Outputs (under --output-dir):
    - papers_assigned.csv        : every assigned paper, with annotation columns
    - landmarks.csv              : convenience subset of papers flagged as landmarks
    - method_timeline.md         : human-readable per-cluster timeline tables
    - method_timeline.json       : structured per-axis data for visualization
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LR_SEARCH_DIR = Path(__file__).resolve().parent
if str(LR_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(LR_SEARCH_DIR))

from literature_review_search import normalize_whitespace, write_json  # noqa: E402
from organize_search_result import (  # noqa: E402
    build_cluster_profiles,
    paper_text,
    score_cluster,
    tokenize,
)


# ----------------------------- IO helpers -----------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ----------------------------- Formatting helpers -----------------------------


def first_author_last_name(authors: Any) -> str:
    if not isinstance(authors, list) or not authors:
        return ""
    first = authors[0]
    name = normalize_whitespace(first.get("name") if isinstance(first, dict) else first)
    if not name:
        return ""
    parts = name.split()
    return parts[-1] if parts else name


def first_author_label(authors: Any, year: Any) -> str:
    last = first_author_last_name(authors)
    if not last:
        return ""
    if year is None or year == "":
        return last
    return f"{last} {year}"


def authors_short(authors: Any, *, max_n: int = 4) -> str:
    if not isinstance(authors, list):
        return ""
    names = []
    for entry in authors[:max_n]:
        name = normalize_whitespace(entry.get("name") if isinstance(entry, dict) else entry)
        if name:
            names.append(name)
    rendered = ", ".join(names)
    if len(authors) > max_n:
        rendered += " et al."
    return rendered


def authors_full(authors: Any) -> str:
    if not isinstance(authors, list):
        return ""
    names = []
    for entry in authors:
        name = normalize_whitespace(entry.get("name") if isinstance(entry, dict) else entry)
        if name:
            names.append(name)
    return "; ".join(names)


def short_title(title: Any, max_chars: int = 90) -> str:
    text = normalize_whitespace(title)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# ----------------------------- Time buckets -----------------------------


def build_time_buckets(time_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = []
    for window in time_windows:
        if not isinstance(window, dict):
            continue
        label = normalize_whitespace(window.get("label"))
        start = window.get("start_year")
        end = window.get("end_year")
        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            continue
        if not label:
            continue
        buckets.append({"label": label, "start": start_i, "end": end_i})
    buckets.sort(key=lambda b: b["start"])
    return buckets


def find_bucket_label(year: Any, buckets: list[dict[str, Any]]) -> str:
    if year is None or year == "":
        return "unknown"
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "unknown"
    for bucket in buckets:
        if bucket["start"] <= y <= bucket["end"]:
            return bucket["label"]
    if buckets and y < buckets[0]["start"]:
        return f"before_{buckets[0]['start']}"
    if buckets and y > buckets[-1]["end"]:
        return f"after_{buckets[-1]['end']}"
    return "unknown"


# ----------------------------- Scoring helpers -----------------------------


def percentile_rank(sorted_values: list[float], value: float) -> float:
    """Return percentile rank in [0, 1]. Max value -> 1.0, min value -> 0.0."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    # binary search: count values <= value (right insertion index)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    # lo is now count of values <= value, so the 0-indexed rank is lo - 1
    rank = max(0, lo - 1)
    return rank / (n - 1)


def zscore(values: list[float], value: float) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return 0.0
    return (value - mean) / stdev


def citation_per_year(citation_count: Any, year: Any, current_year: int) -> float:
    try:
        cite = float(citation_count or 0)
    except (TypeError, ValueError):
        cite = 0.0
    try:
        y = int(year)
    except (TypeError, ValueError):
        return 0.0
    span = max(1, current_year - y + 1)
    return cite / span


# ----------------------------- Core build -----------------------------


def build_paper_records(
    *,
    paper_cards_by_id: dict[str, dict[str, Any]],
    clusters: list[dict[str, Any]],
    paper_assignment_log: dict[str, Any],
    time_buckets: list[dict[str, Any]],
    focus_cluster_ids: set[str],
    current_year: int,
    cluster_profiles: dict[str, dict[str, Any]],
    cluster_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Pre-compute per-cluster sorted distributions for percentile / zscore.
    per_cluster_cpy: dict[str, list[float]] = {}
    per_cluster_similarity: dict[str, list[float]] = {}
    per_cluster_citation: dict[str, list[float]] = {}
    cluster_member_payload: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        cpy_values: list[float] = []
        sim_values: list[float] = []
        cite_values: list[float] = []
        for pid in cluster["paper_ids"]:
            card = paper_cards_by_id.get(pid)
            if not card:
                continue
            cpy = citation_per_year(card.get("citation_count"), card.get("year"), current_year)
            try:
                cite = float(card.get("citation_count") or 0)
            except (TypeError, ValueError):
                cite = 0.0
            tokens = tokenize(paper_text(card))
            profile = cluster_profiles.get(cluster_id, {"tokens": set(), "seed_paper_ids": set()})
            sim = score_cluster(card, cluster_id, profile, tokens)
            cpy_values.append(cpy)
            sim_values.append(sim)
            cite_values.append(cite)
            cluster_member_payload[cluster_id].append(
                {
                    "paper_id": pid,
                    "card": card,
                    "citation_per_year": cpy,
                    "similarity": sim,
                    "citation_count": cite,
                }
            )
        per_cluster_cpy[cluster_id] = sorted(cpy_values)
        per_cluster_similarity[cluster_id] = sorted(sim_values)
        per_cluster_citation[cluster_id] = sorted(cite_values)

    # Global distribution for retrieval-hit normalization.
    all_hit_counts: list[float] = []
    for cluster in clusters:
        for pid in cluster["paper_ids"]:
            card = paper_cards_by_id.get(pid)
            if not card:
                continue
            actions = card.get("source_actions") or []
            all_hit_counts.append(float(len(actions)))
    max_hits = max(all_hit_counts) if all_hit_counts else 1.0

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        cluster_name = cluster["name"]
        is_focus = cluster_id in focus_cluster_ids if focus_cluster_ids else True
        cpy_dist = per_cluster_cpy[cluster_id]
        sim_dist = per_cluster_similarity[cluster_id]
        cite_dist = per_cluster_citation[cluster_id]
        for member in cluster_member_payload[cluster_id]:
            card = member["card"]
            pid = member["paper_id"]
            year = card.get("year")
            cpy = member["citation_per_year"]
            sim = member["similarity"]
            cite = member["citation_count"]
            actions = card.get("source_actions") or []
            hit_count = float(len(actions))
            cpy_pct = percentile_rank(cpy_dist, cpy)
            sim_pct = percentile_rank(sim_dist, sim)
            cite_pct = percentile_rank(cite_dist, cite)
            cite_z = zscore(cite_dist, cite)
            cpy_z = zscore(cpy_dist, cpy)
            hit_pct = hit_count / max_hits if max_hits else 0.0
            log_entry = paper_assignment_log.get(pid, {})
            source = normalize_whitespace(log_entry.get("source")) or "unknown"
            confidence = 1.0 if source == "llm_map_reduce" else 0.5 if source == "lexical_rescue" else 0.5
            rescue_score = log_entry.get("score") if source == "lexical_rescue" else None
            rescue_overlap = log_entry.get("token_overlap") if source == "lexical_rescue" else None
            records.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "is_focus_cluster": is_focus,
                    "paper_id": pid,
                    "year": year,
                    "time_bucket": find_bucket_label(year, time_buckets),
                    "title": normalize_whitespace(card.get("title")),
                    "short_title": short_title(card.get("title"), 90),
                    "first_author": first_author_last_name(card.get("authors")),
                    "first_author_year_label": first_author_label(card.get("authors"), year),
                    "authors_short": authors_short(card.get("authors")),
                    "authors_full": authors_full(card.get("authors")),
                    "venue": normalize_whitespace(card.get("venue")),
                    "doi": normalize_whitespace(card.get("doi")),
                    "url": normalize_whitespace(card.get("url")),
                    "citation_count": int(cite),
                    "citation_per_year": round(cpy, 3),
                    "citation_pct_in_cluster": round(cite_pct, 4),
                    "citation_per_year_pct_in_cluster": round(cpy_pct, 4),
                    "citation_zscore_in_cluster": round(cite_z, 3),
                    "citation_per_year_zscore_in_cluster": round(cpy_z, 3),
                    "cluster_similarity": round(sim, 4),
                    "cluster_similarity_pct_in_cluster": round(sim_pct, 4),
                    "retrieval_hit_count": int(hit_count),
                    "retrieval_hit_pct_corpus": round(hit_pct, 4),
                    "source_actions": ";".join(actions),
                    "assignment_source": source,
                    "assignment_confidence": confidence,
                    "lexical_rescue_score": rescue_score if rescue_score is not None else "",
                    "lexical_rescue_token_overlap": rescue_overlap if rescue_overlap is not None else "",
                }
            )
    return records


def compute_landmark_scores(
    records: list[dict[str, Any]],
    *,
    w_cpy: float,
    w_sim: float,
    w_hits: float,
    w_conf: float,
) -> None:
    for record in records:
        score = (
            w_cpy * record["citation_per_year_pct_in_cluster"]
            + w_sim * record["cluster_similarity_pct_in_cluster"]
            + w_hits * record["retrieval_hit_pct_corpus"]
            + w_conf * record["assignment_confidence"]
        )
        record["landmark_score"] = round(score, 4)


def flag_landmarks(
    records: list[dict[str, Any]],
    *,
    top_k_per_cluster: int,
    top_n_per_bucket: int,
) -> None:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record["is_landmark_topk_in_cluster"] = False
        record["is_landmark_topn_in_bucket"] = False
        by_cluster[record["cluster_id"]].append(record)

    for cluster_records in by_cluster.values():
        cluster_records_sorted = sorted(cluster_records, key=lambda r: -r["landmark_score"])
        for record in cluster_records_sorted[:top_k_per_cluster]:
            record["is_landmark_topk_in_cluster"] = True

    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_bucket[(record["cluster_id"], record["time_bucket"])].append(record)
    for bucket_records in by_bucket.values():
        bucket_records_sorted = sorted(bucket_records, key=lambda r: -r["landmark_score"])
        for record in bucket_records_sorted[:top_n_per_bucket]:
            record["is_landmark_topn_in_bucket"] = True

    for record in records:
        record["is_landmark_any"] = (
            record["is_landmark_topk_in_cluster"] or record["is_landmark_topn_in_bucket"]
        )


# ----------------------------- Rendering -----------------------------


CSV_FIELDS = [
    "cluster_id",
    "cluster_name",
    "is_focus_cluster",
    "paper_id",
    "year",
    "time_bucket",
    "title",
    "short_title",
    "first_author",
    "first_author_year_label",
    "authors_short",
    "venue",
    "doi",
    "url",
    "citation_count",
    "citation_per_year",
    "citation_pct_in_cluster",
    "citation_per_year_pct_in_cluster",
    "citation_zscore_in_cluster",
    "citation_per_year_zscore_in_cluster",
    "cluster_similarity",
    "cluster_similarity_pct_in_cluster",
    "retrieval_hit_count",
    "retrieval_hit_pct_corpus",
    "assignment_source",
    "assignment_confidence",
    "lexical_rescue_score",
    "lexical_rescue_token_overlap",
    "source_actions",
    "landmark_score",
    "is_landmark_topk_in_cluster",
    "is_landmark_topn_in_bucket",
    "is_landmark_any",
    "authors_full",
]


def render_markdown(
    *,
    records: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    time_buckets: list[dict[str, Any]],
    focus_cluster_ids: set[str],
    topic_text: str,
    weights: dict[str, float],
    top_k_per_cluster: int,
    top_n_per_bucket: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Method × Time Timeline\n")
    if topic_text:
        lines.append(f"Topic: {topic_text}\n")
    lines.append(
        f"Scoring weights: cpy={weights['cpy']}, similarity={weights['sim']}, "
        f"retrieval_hits={weights['hits']}, confidence={weights['conf']}.  "
        f"Landmark flags: top {top_k_per_cluster} per cluster + top {top_n_per_bucket} per (cluster, bucket).\n"
    )
    lines.append("## Time buckets\n")
    for bucket in time_buckets:
        lines.append(f"- `{bucket['label']}`: {bucket['start']}–{bucket['end']}")
    lines.append("")

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cluster[record["cluster_id"]].append(record)

    focus_clusters = [c for c in clusters if c["cluster_id"] in focus_cluster_ids] if focus_cluster_ids else clusters
    other_clusters = [c for c in clusters if c["cluster_id"] not in focus_cluster_ids] if focus_cluster_ids else []

    def render_cluster_section(cluster: dict[str, Any], is_focus: bool) -> None:
        cluster_id = cluster["cluster_id"]
        cluster_records = sorted(by_cluster.get(cluster_id, []), key=lambda r: (r.get("year") or 0, -r["landmark_score"]))
        if not cluster_records:
            return
        source_counter = Counter(r["assignment_source"] for r in cluster_records)
        landmark_count = sum(1 for r in cluster_records if r["is_landmark_any"])
        years = [r["year"] for r in cluster_records if isinstance(r.get("year"), int)]
        year_span = f"{min(years)}–{max(years)}" if years else "unknown"
        focus_tag = "FOCUS" if is_focus else "non-focus"
        lines.append(f"## {cluster_id} — {cluster['name']}  [{focus_tag}]\n")
        lines.append(
            f"- n={len(cluster_records)}  "
            f"landmarks={landmark_count}  "
            f"year_span={year_span}  "
            f"source={dict(source_counter)}\n"
        )
        bucket_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in cluster_records:
            bucket_to_records[record["time_bucket"]].append(record)
        ordered_bucket_labels: list[str] = []
        seen_labels: set[str] = set()
        for bucket in time_buckets:
            if bucket["label"] in bucket_to_records:
                ordered_bucket_labels.append(bucket["label"])
                seen_labels.add(bucket["label"])
        for label in bucket_to_records:
            if label not in seen_labels:
                ordered_bucket_labels.append(label)
        for label in ordered_bucket_labels:
            bucket_records_sorted = sorted(
                bucket_to_records[label],
                key=lambda r: (r.get("year") or 0, -r["landmark_score"]),
            )
            lines.append(f"### {label}\n")
            lines.append(
                "| Year | ★ | Title | Citations | Cites/yr | sim | source | First author | Venue |"
            )
            lines.append("| ---: | :-: | --- | ---: | ---: | ---: | :-- | --- | --- |")
            for record in bucket_records_sorted:
                star = "★" if record["is_landmark_any"] else ""
                source_short = {"llm_map_reduce": "llm", "lexical_rescue": "rsc"}.get(
                    record["assignment_source"], record["assignment_source"][:3]
                )
                title_short = short_title(record["title"], 80).replace("|", "\\|")
                first_author = record["first_author"].replace("|", "\\|")
                venue = short_title(record["venue"], 40).replace("|", "\\|")
                lines.append(
                    f"| {record.get('year') or '?'} | {star} | {title_short} "
                    f"| {record['citation_count']:,} | {record['citation_per_year']:.1f} "
                    f"| {record['cluster_similarity']:.3f} | {source_short} | {first_author} | {venue} |"
                )
            lines.append("")

    for cluster in focus_clusters:
        render_cluster_section(cluster, is_focus=True)
    if other_clusters:
        lines.append("---\n")
        lines.append("# Non-focus clusters\n")
        for cluster in other_clusters:
            render_cluster_section(cluster, is_focus=False)
    return "\n".join(lines).rstrip() + "\n"


def assemble_json(
    *,
    records: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    time_buckets: list[dict[str, Any]],
    focus_cluster_ids: set[str],
    topic_text: str,
    weights: dict[str, float],
    top_k_per_cluster: int,
    top_n_per_bucket: int,
    current_year: int,
) -> dict[str, Any]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cluster[record["cluster_id"]].append(record)
    axes = []
    for cluster in clusters:
        cluster_records = sorted(by_cluster.get(cluster["cluster_id"], []), key=lambda r: (r.get("year") or 0, -r["landmark_score"]))
        axes.append(
            {
                "cluster_id": cluster["cluster_id"],
                "name": cluster["name"],
                "definition": cluster.get("definition", ""),
                "is_focus": cluster["cluster_id"] in focus_cluster_ids if focus_cluster_ids else True,
                "paper_count": len(cluster_records),
                "landmark_count": sum(1 for r in cluster_records if r["is_landmark_any"]),
                "papers": [
                    {
                        "paper_id": r["paper_id"],
                        "year": r["year"],
                        "time_bucket": r["time_bucket"],
                        "title": r["title"],
                        "first_author": r["first_author"],
                        "first_author_year_label": r["first_author_year_label"],
                        "authors": r["authors_full"],
                        "venue": r["venue"],
                        "doi": r["doi"],
                        "url": r["url"],
                        "citation_count": r["citation_count"],
                        "citation_per_year": r["citation_per_year"],
                        "citation_per_year_pct_in_cluster": r["citation_per_year_pct_in_cluster"],
                        "cluster_similarity": r["cluster_similarity"],
                        "cluster_similarity_pct_in_cluster": r["cluster_similarity_pct_in_cluster"],
                        "retrieval_hit_count": r["retrieval_hit_count"],
                        "assignment_source": r["assignment_source"],
                        "landmark_score": r["landmark_score"],
                        "is_landmark_topk_in_cluster": r["is_landmark_topk_in_cluster"],
                        "is_landmark_topn_in_bucket": r["is_landmark_topn_in_bucket"],
                        "is_landmark": r["is_landmark_any"],
                    }
                    for r in cluster_records
                ],
            }
        )
    return {
        "topic": topic_text,
        "current_year": current_year,
        "time_buckets": time_buckets,
        "scoring": {
            "weights": weights,
            "top_k_per_cluster": top_k_per_cluster,
            "top_n_per_bucket": top_n_per_bucket,
        },
        "focus_cluster_ids": sorted(focus_cluster_ids),
        "axes": axes,
    }


# ----------------------------- Driver -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-method timeline data (CSV / MD / JSON) from recluster_all_papers.py output."
    )
    parser.add_argument("--search-result", required=True, help="Path to lr_search search_result.json.")
    parser.add_argument("--clusters", required=True, help="Path to clusters_recovered.json (or clusters_final.json).")
    parser.add_argument("--output-dir", required=True, help="Directory to write timeline artifacts to.")
    parser.add_argument(
        "--focus-clusters",
        default="",
        help="Comma-separated cluster_ids marked as is_focus. Empty -> all clusters treated as focus.",
    )
    parser.add_argument("--current-year", type=int, default=2026, help="Used for citation-per-year normalization.")
    parser.add_argument("--top-k-per-cluster", type=int, default=15, help="Flag top-K landmarks per cluster.")
    parser.add_argument("--top-n-per-bucket", type=int, default=3, help="Flag top-N landmarks per (cluster, bucket).")
    parser.add_argument("--w-cpy", type=float, default=1.0, help="Weight: citation-per-year percentile in cluster.")
    parser.add_argument("--w-similarity", type=float, default=0.3, help="Weight: cluster similarity percentile in cluster.")
    parser.add_argument("--w-hits", type=float, default=0.2, help="Weight: retrieval hit count percentile across corpus.")
    parser.add_argument("--w-confidence", type=float, default=0.2, help="Weight: assignment confidence (1.0 LLM / 0.5 rescue).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    search_result_path = Path(args.search_result).expanduser().resolve()
    clusters_path = Path(args.clusters).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    search_result = read_json(search_result_path)
    clusters_payload = read_json(clusters_path)
    if not isinstance(search_result, dict):
        raise SystemExit(f"{search_result_path} must contain a JSON object")
    if not isinstance(clusters_payload, dict):
        raise SystemExit(f"{clusters_path} must contain a JSON object")

    paper_cards_by_id: dict[str, dict[str, Any]] = {}
    for card in search_result.get("paper_cards", []):
        if isinstance(card, dict) and card.get("paper_id"):
            paper_cards_by_id[card["paper_id"]] = card
    if not paper_cards_by_id:
        raise SystemExit(f"No paper_cards found in {search_result_path}")

    clusters_raw = clusters_payload.get("clusters") or []
    clusters: list[dict[str, Any]] = []
    for index, item in enumerate(clusters_raw, start=1):
        if not isinstance(item, dict):
            continue
        cluster_id = normalize_whitespace(item.get("cluster_id")) or f"C{index}"
        clusters.append(
            {
                "cluster_id": cluster_id,
                "name": normalize_whitespace(item.get("name")) or cluster_id,
                "definition": normalize_whitespace(item.get("definition")),
                "distinguishing_features": [
                    normalize_whitespace(v) for v in item.get("distinguishing_features", []) if normalize_whitespace(v)
                ],
                "paper_ids": [pid for pid in item.get("paper_ids", []) if pid in paper_cards_by_id],
                "representative_paper_ids": [
                    pid for pid in item.get("representative_paper_ids", []) if pid in paper_cards_by_id
                ],
            }
        )

    paper_assignment_log = clusters_payload.get("recluster_metadata", {}).get("paper_assignment_log", {})
    if not isinstance(paper_assignment_log, dict):
        paper_assignment_log = {}

    time_windows = search_result.get("time_windows") or []
    time_buckets = build_time_buckets(time_windows)
    if not time_buckets:
        # fallback: synthesize a single bucket
        time_buckets = [{"label": "all", "start": -10000, "end": 10000}]

    focus_cluster_ids = {
        normalize_whitespace(part)
        for part in (args.focus_clusters or "").split(",")
        if normalize_whitespace(part)
    }

    # Build cluster profiles using current memberships as seed corpus.
    clusters_for_profile = [
        {
            "cluster_id": c["cluster_id"],
            "name": c["name"],
            "definition": c["definition"],
            "distinguishing_features": c["distinguishing_features"],
            "seed_paper_ids": c["paper_ids"],
            "representative_seed_paper_ids": c["representative_paper_ids"],
            "missing_signals": [],
        }
        for c in clusters
    ]
    cluster_profiles = build_cluster_profiles(clusters_for_profile, paper_cards_by_id)
    cluster_by_id = {c["cluster_id"]: c for c in clusters}

    records = build_paper_records(
        paper_cards_by_id=paper_cards_by_id,
        clusters=clusters,
        paper_assignment_log=paper_assignment_log,
        time_buckets=time_buckets,
        focus_cluster_ids=focus_cluster_ids,
        current_year=args.current_year,
        cluster_profiles=cluster_profiles,
        cluster_by_id=cluster_by_id,
    )

    weights = {
        "cpy": args.w_cpy,
        "sim": args.w_similarity,
        "hits": args.w_hits,
        "conf": args.w_confidence,
    }
    compute_landmark_scores(
        records,
        w_cpy=weights["cpy"],
        w_sim=weights["sim"],
        w_hits=weights["hits"],
        w_conf=weights["conf"],
    )
    flag_landmarks(
        records,
        top_k_per_cluster=args.top_k_per_cluster,
        top_n_per_bucket=args.top_n_per_bucket,
    )

    topic_text = normalize_whitespace(
        (search_result.get("topic_profile") or {}).get("normalized_topic")
        or search_result.get("topic")
    )

    # Sort: cluster_id (focus first), then year asc, then landmark_score desc
    focus_order = {cid: 0 for cid in focus_cluster_ids} if focus_cluster_ids else {}
    records.sort(
        key=lambda r: (
            focus_order.get(r["cluster_id"], 1),
            r["cluster_id"],
            r.get("year") or 0,
            -r["landmark_score"],
        )
    )

    csv_path = output_dir / "papers_assigned.csv"
    landmarks_csv_path = output_dir / "landmarks.csv"
    md_path = output_dir / "method_timeline.md"
    json_path = output_dir / "method_timeline.json"

    write_csv(csv_path, records, CSV_FIELDS)
    write_csv(landmarks_csv_path, [r for r in records if r["is_landmark_any"]], CSV_FIELDS)
    write_text(
        md_path,
        render_markdown(
            records=records,
            clusters=clusters,
            time_buckets=time_buckets,
            focus_cluster_ids=focus_cluster_ids,
            topic_text=topic_text,
            weights=weights,
            top_k_per_cluster=args.top_k_per_cluster,
            top_n_per_bucket=args.top_n_per_bucket,
        ),
    )
    write_json(
        json_path,
        assemble_json(
            records=records,
            clusters=clusters,
            time_buckets=time_buckets,
            focus_cluster_ids=focus_cluster_ids,
            topic_text=topic_text,
            weights=weights,
            top_k_per_cluster=args.top_k_per_cluster,
            top_n_per_bucket=args.top_n_per_bucket,
            current_year=args.current_year,
        ),
    )

    summary = {
        "papers_assigned_csv": str(csv_path),
        "landmarks_csv": str(landmarks_csv_path),
        "method_timeline_md": str(md_path),
        "method_timeline_json": str(json_path),
        "paper_count": len(records),
        "landmark_count": sum(1 for r in records if r["is_landmark_any"]),
        "cluster_count": len(clusters),
        "focus_cluster_ids": sorted(focus_cluster_ids),
        "time_buckets": [b["label"] for b in time_buckets],
        "current_year": args.current_year,
        "scoring_weights": weights,
        "landmark_caps": {
            "top_k_per_cluster": args.top_k_per_cluster,
            "top_n_per_bucket": args.top_n_per_bucket,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
