#!/usr/bin/env python3
"""Post-search organization for literature-review search results.

This script consumes an existing search_result.json plus optional clusters_final.json
and produces a review-ready evidence map that uses every retrieved paper:

- primary method-cluster assignment for every paper;
- time-window assignment for every paper;
- lightweight paper role labels;
- method x time evidence matrix;
- representative paper selection.

It is intentionally deterministic and API-free. The LLM-generated clusters are used
as a taxonomy seed; all remaining papers are assigned by transparent lexical scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "large",
    "language",
    "llm",
    "llms",
    "model",
    "models",
    "of",
    "on",
    "or",
    "our",
    "paper",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "we",
    "with",
}

SURVEY_TERMS = {
    "survey",
    "review",
    "taxonomy",
    "overview",
    "roadmap",
    "landscape",
    "tutorial",
}
BENCHMARK_TERMS = {
    "benchmark",
    "benchmarks",
    "benchmarking",
    "dataset",
    "datasets",
    "evaluation",
    "evaluating",
    "eval",
    "leaderboard",
    "gsm8k",
    "math",
}
APPLICATION_TERMS = {
    "medical",
    "medicine",
    "clinical",
    "healthcare",
    "legal",
    "chart",
    "visual",
    "vision",
    "multimodal",
    "vqa",
    "sql",
    "code",
    "robot",
    "robotics",
}
METHOD_TERMS = {
    "prompting",
    "chain",
    "thought",
    "cot",
    "self",
    "consistency",
    "decoding",
    "verification",
    "refinement",
    "planning",
    "tree",
    "graph",
    "program",
    "symbolic",
    "retrieval",
    "rag",
    "reinforcement",
    "learning",
    "reasoning",
    "tool",
    "agent",
}


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def tokenize(text: Any) -> set[str]:
    normalized = normalize_whitespace(text).casefold()
    tokens = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_-]*", normalized):
        token = token.replace("_", "-")
        if len(token) <= 2 or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def paper_text(paper: dict[str, Any]) -> str:
    return " ".join(
        [
            normalize_whitespace(paper.get("title")),
            normalize_whitespace(paper.get("abstract")),
            normalize_whitespace(paper.get("venue")),
        ]
    )


def load_clusters(search_result: dict[str, Any], clusters_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    source_clusters = []
    if clusters_payload and isinstance(clusters_payload.get("clusters"), list):
        source_clusters = clusters_payload["clusters"]
    elif isinstance(search_result.get("method_clusters"), list):
        source_clusters = search_result["method_clusters"]

    clusters = []
    seen = set()
    for index, item in enumerate(source_clusters, start=1):
        if not isinstance(item, dict):
            continue
        cluster_id = normalize_whitespace(item.get("cluster_id")) or f"C{index}"
        if cluster_id in seen:
            continue
        seen.add(cluster_id)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "name": normalize_whitespace(item.get("name")) or cluster_id,
                "definition": normalize_whitespace(item.get("definition")),
                "distinguishing_features": [
                    normalize_whitespace(value)
                    for value in item.get("distinguishing_features", [])
                    if normalize_whitespace(value)
                ],
                "seed_paper_ids": [
                    normalize_whitespace(value)
                    for value in item.get("paper_ids", [])
                    if normalize_whitespace(value)
                ],
                "representative_seed_paper_ids": [
                    normalize_whitespace(value)
                    for value in item.get("representative_paper_ids", [])
                    if normalize_whitespace(value)
                ],
                "missing_signals": [
                    normalize_whitespace(value)
                    for value in item.get("missing_signals", [])
                    if normalize_whitespace(value)
                ],
            }
        )
    return clusters


def build_cluster_profiles(
    clusters: list[dict[str, Any]],
    papers_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    profiles = {}
    for cluster in clusters:
        text_parts = [
            cluster["name"],
            cluster["definition"],
            " ".join(cluster["distinguishing_features"]),
        ]
        seed_tokens: set[str] = set()
        for paper_id in cluster["seed_paper_ids"][:40]:
            paper = papers_by_id.get(paper_id)
            if paper:
                seed_tokens |= tokenize(paper_text(paper))
        profile_tokens = tokenize(" ".join(text_parts)) | seed_tokens
        profiles[cluster["cluster_id"]] = {
            "tokens": profile_tokens,
            "seed_paper_ids": set(cluster["seed_paper_ids"]),
        }
    return profiles


def score_cluster(
    paper: dict[str, Any],
    cluster_id: str,
    profile: dict[str, Any],
    paper_tokens: set[str],
) -> float:
    if not paper_tokens or not profile["tokens"]:
        return 0.0
    overlap = paper_tokens & profile["tokens"]
    cosine_like = len(overlap) / math.sqrt(len(paper_tokens) * len(profile["tokens"]))
    method_overlap = len(overlap & METHOD_TERMS) * 0.03
    seed_bonus = 0.08 if paper.get("paper_id") in profile["seed_paper_ids"] else 0.0
    title_tokens = tokenize(paper.get("title"))
    title_overlap = len(title_tokens & profile["tokens"]) * 0.015
    return round(cosine_like + method_overlap + seed_bonus + title_overlap, 5)


def classify_role(paper: dict[str, Any], *, outside_time: bool) -> tuple[str, list[str]]:
    title_tokens = tokenize(paper.get("title"))
    all_tokens = tokenize(paper_text(paper))
    reasons: list[str] = []

    if outside_time:
        reasons.append("Paper year falls outside generated topic time windows.")
        return "background", reasons

    if title_tokens & SURVEY_TERMS or "survey" in normalize_whitespace(paper.get("title")).casefold():
        reasons.append("Title indicates survey/review/taxonomy literature.")
        return "survey", reasons

    if title_tokens & BENCHMARK_TERMS or len(all_tokens & BENCHMARK_TERMS) >= 2:
        reasons.append("Title or abstract emphasizes benchmark, dataset, or evaluation.")
        return "benchmark_evaluation", reasons

    if title_tokens & APPLICATION_TERMS or len(all_tokens & APPLICATION_TERMS) >= 2:
        reasons.append("Paper is primarily domain-specific or multimodal/application-focused.")
        return "application_domain", reasons

    if len(all_tokens & METHOD_TERMS) >= 3:
        reasons.append("Paper appears to contribute or analyze a reasoning method.")
        return "core_method", reasons

    reasons.append("Paper is relevant but lacks strong method, benchmark, survey, or application signals.")
    return "supporting", reasons


def assign_time_bucket(
    paper: dict[str, Any],
    time_windows: list[dict[str, Any]],
) -> tuple[str, str]:
    year = paper.get("year")
    if not isinstance(year, int):
        return "unknown", "missing_year"
    for window in time_windows:
        start = window.get("start_year")
        end = window.get("end_year")
        if isinstance(start, int) and isinstance(end, int) and start <= year <= end:
            return normalize_whitespace(window.get("label")) or f"{start}_{end}", "in_window"
    return "outside_windows", "outside_window"


def representative_sort_key(paper: dict[str, Any]) -> tuple[Any, ...]:
    tier = normalize_whitespace(paper.get("evidence_source_tier"))
    if not tier:
        sources = set(paper.get("source", [])) if isinstance(paper.get("source"), list) else set()
        if paper.get("expert_seed") is True or "expert_seed" in sources:
            tier = "expert_seed"
        elif paper.get("expert_neighbor") is True or "expert_recommendation" in sources:
            tier = "expert_recommendation"
        else:
            tier = "pipeline"
    tier_rank = {"expert_seed": 0, "expert_recommendation": 1, "pipeline": 2}.get(tier, 2)
    return (
        tier_rank,
        -(paper.get("citation_count") or 0),
        -len(paper.get("source_actions", [])),
        paper.get("year") is None,
        -(paper.get("year") or 0),
        normalize_whitespace(paper.get("title")).casefold(),
    )


def role_counts(assignments: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(item["role"] for item in assignments))


def time_counts(assignments: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(item["time_bucket_label"] for item in assignments))


def organize(
    search_result: dict[str, Any],
    clusters_payload: dict[str, Any] | None,
    *,
    min_cluster_score: float,
    representatives_per_group: int,
) -> dict[str, Any]:
    papers = [paper for paper in search_result.get("paper_cards", []) if isinstance(paper, dict)]
    papers_by_id = {paper["paper_id"]: paper for paper in papers if paper.get("paper_id")}
    clusters = load_clusters(search_result, clusters_payload)
    profiles = build_cluster_profiles(clusters, papers_by_id)
    time_windows = [w for w in search_result.get("time_windows", []) if isinstance(w, dict)]
    explicit_outliers = set()
    if clusters_payload and isinstance(clusters_payload.get("hard_excluded_paper_ids"), list):
        explicit_outliers = {
            normalize_whitespace(item)
            for item in clusters_payload.get("hard_excluded_paper_ids", [])
            if normalize_whitespace(item) in papers_by_id
        }

    cluster_names = {cluster["cluster_id"]: cluster["name"] for cluster in clusters}
    assignments: list[dict[str, Any]] = []
    method_assignments: dict[str, list[str]] = defaultdict(list)
    time_assignments: dict[str, list[str]] = defaultdict(list)
    role_assignments: dict[str, list[str]] = defaultdict(list)

    for paper in papers:
        paper_id = paper["paper_id"]
        if paper_id in explicit_outliers:
            best_cluster_id = "OUT"
            best_score = 0.0
            secondary = []
        else:
            tokens = tokenize(paper_text(paper))
            scored = [
                (cluster_id, score_cluster(paper, cluster_id, profile, tokens))
                for cluster_id, profile in profiles.items()
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            best_cluster_id, best_score = scored[0] if scored else ("OUT", 0.0)
            second_score = scored[1][1] if len(scored) > 1 else 0.0
            secondary = [
                {"cluster_id": cluster_id, "score": score}
                for cluster_id, score in scored[1:4]
                if score >= max(min_cluster_score, best_score * 0.75)
            ]
            best_tokens = profiles.get(best_cluster_id, {}).get("tokens", set()) if best_cluster_id != "OUT" else set()
            token_overlap = len(tokens & best_tokens) if best_tokens else 0
            margin_ok = (best_score - second_score) >= 0.03 if best_cluster_id != "OUT" else False
            if (
                best_score < min_cluster_score
                or token_overlap < 2
                or (best_cluster_id != "OUT" and not margin_ok and best_score < (min_cluster_score + 0.08))
            ):
                best_cluster_id = "OUT"

        time_label, time_reason = assign_time_bucket(paper, time_windows)
        outside_time = time_label == "outside_windows"
        role, role_reasons = classify_role(paper, outside_time=outside_time)
        if best_cluster_id == "OUT" and role == "core_method":
            role = "supporting"
            role_reasons.append("Method signal exists, but no method cluster matched strongly enough.")

        assignment = {
            "paper_id": paper_id,
            "title": paper.get("title"),
            "year": paper.get("year"),
            "method_cluster_id": best_cluster_id,
            "method_cluster_name": cluster_names.get(best_cluster_id, "Peripheral or unassigned"),
            "method_score": best_score,
            "secondary_method_clusters": secondary,
            "time_bucket_label": time_label,
            "time_bucket_reason": time_reason,
            "role": role,
            "role_reasons": role_reasons,
            "citation_count": paper.get("citation_count"),
            "source_actions": paper.get("source_actions", []),
            "source": paper.get("source", []),
            "evidence_source_tier": paper.get("evidence_source_tier", "pipeline"),
            "expert_seed": paper.get("expert_seed", False),
            "expert_neighbor": paper.get("expert_neighbor", False),
        }
        assignments.append(assignment)
        method_assignments[best_cluster_id].append(paper_id)
        time_assignments[time_label].append(paper_id)
        role_assignments[role].append(paper_id)

    assignment_by_id = {item["paper_id"]: item for item in assignments}

    organized_clusters = []
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        paper_ids = method_assignments.get(cluster_id, [])
        cluster_assignment_rows = [assignment_by_id[paper_id] for paper_id in paper_ids]
        representative_ids = [
            paper["paper_id"]
            for paper in sorted((papers_by_id[paper_id] for paper_id in paper_ids), key=representative_sort_key)[
                :representatives_per_group
            ]
        ]
        organized_clusters.append(
            {
                "cluster_id": cluster_id,
                "name": cluster["name"],
                "definition": cluster["definition"],
                "paper_count": len(paper_ids),
                "paper_ids": paper_ids,
                "representative_paper_ids": representative_ids,
                "role_counts": role_counts(cluster_assignment_rows),
                "time_counts": time_counts(cluster_assignment_rows),
                "seed_paper_ids": cluster["seed_paper_ids"],
                "missing_signals": cluster["missing_signals"],
            }
        )

    if method_assignments.get("OUT"):
        out_ids = method_assignments["OUT"]
        out_rows = [assignment_by_id[paper_id] for paper_id in out_ids]
        organized_clusters.append(
            {
                "cluster_id": "OUT",
                "name": "Peripheral or unassigned",
                "definition": "Papers retained in the evidence map but not strongly matched to the induced method taxonomy.",
                "paper_count": len(out_ids),
                "paper_ids": out_ids,
                "representative_paper_ids": [
                    paper["paper_id"]
                    for paper in sorted((papers_by_id[paper_id] for paper_id in out_ids), key=representative_sort_key)[
                        :representatives_per_group
                    ]
                ],
                "role_counts": role_counts(out_rows),
                "time_counts": time_counts(out_rows),
                "seed_paper_ids": [],
                "missing_signals": ["No strong lexical match to the induced method clusters."],
            }
        )

    time_buckets = []
    for label, paper_ids in sorted(time_assignments.items(), key=lambda item: time_bucket_sort_key(item[0], time_windows)):
        rows = [assignment_by_id[paper_id] for paper_id in paper_ids]
        time_buckets.append(
            {
                "label": label,
                "paper_count": len(paper_ids),
                "paper_ids": paper_ids,
                "representative_paper_ids": [
                    paper["paper_id"]
                    for paper in sorted((papers_by_id[paper_id] for paper_id in paper_ids), key=representative_sort_key)[
                        :representatives_per_group
                    ]
                ],
                "method_counts": dict(Counter(row["method_cluster_id"] for row in rows)),
                "role_counts": role_counts(rows),
            }
        )

    method_time_matrix = []
    for cluster in organized_clusters:
        cluster_id = cluster["cluster_id"]
        cluster_name = cluster["name"]
        for time_bucket in time_buckets:
            label = time_bucket["label"]
            paper_ids = [
                assignment["paper_id"]
                for assignment in assignments
                if assignment["method_cluster_id"] == cluster_id and assignment["time_bucket_label"] == label
            ]
            if not paper_ids:
                continue
            rows = [assignment_by_id[paper_id] for paper_id in paper_ids]
            method_time_matrix.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "time_bucket_label": label,
                    "paper_count": len(paper_ids),
                    "paper_ids": paper_ids,
                    "representative_paper_ids": [
                        paper["paper_id"]
                        for paper in sorted(
                            (papers_by_id[paper_id] for paper_id in paper_ids),
                            key=representative_sort_key,
                        )[:representatives_per_group]
                    ],
                    "role_counts": role_counts(rows),
                }
            )

    low_confidence = [
        item["paper_id"]
        for item in assignments
        if item["method_cluster_id"] == "OUT" or item["method_score"] < min_cluster_score * 1.5
    ]

    return {
        "topic": search_result.get("topic"),
        "source_search_result": search_result.get("diagnostics", {}).get("output_dir"),
        "paper_count": len(papers),
        "organization_policy": {
            "method": "seeded_lexical_assignment",
            "min_cluster_score": min_cluster_score,
            "representatives_per_group": representatives_per_group,
            "uses_llm": False,
            "explicit_outliers_respected": bool(explicit_outliers),
        },
        "method_clusters": organized_clusters,
        "time_buckets": time_buckets,
        "method_time_matrix": method_time_matrix,
        "role_groups": {role: ids for role, ids in sorted(role_assignments.items())},
        "paper_assignments": assignments,
        "retained_cluster_paper_ids": sorted(
            {
                paper_id
                for cluster in organized_clusters
                if cluster["cluster_id"] != "OUT"
                for paper_id in cluster.get("paper_ids", [])
            }
        ),
        "excluded_outlier_paper_ids": sorted(explicit_outliers),
        "quality_notes": {
            "all_papers_assigned": len(assignments) == len(papers),
            "low_confidence_or_peripheral_paper_count": len(low_confidence),
            "low_confidence_or_peripheral_paper_ids": low_confidence,
            "method_cluster_count": len([cluster for cluster in organized_clusters if cluster["cluster_id"] != "OUT"]),
            "time_bucket_count": len(time_buckets),
        },
    }


def time_bucket_sort_key(label: str, time_windows: list[dict[str, Any]]) -> tuple[int, str]:
    if label == "unknown":
        return (9998, label)
    if label == "outside_windows":
        return (9999, label)
    for index, window in enumerate(time_windows):
        if normalize_whitespace(window.get("label")) == label:
            return (index, label)
    return (9000, label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Organize an lr_search search_result.json into a full evidence map.")
    parser.add_argument("--search-result", required=True, help="Path to search_result.json.")
    parser.add_argument(
        "--clusters",
        default=None,
        help="Optional clusters_final.json. Defaults to clusters_final.json next to search_result.json when present.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to organized_search_result.json next to search_result.json.",
    )
    parser.add_argument("--min-cluster-score", type=float, default=0.18, help="Minimum lexical score for assignment.")
    parser.add_argument("--representatives-per-group", type=int, default=5, help="Representative paper count per group.")
    parser.add_argument("--pretty", action="store_true", help="Print the full organized JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    search_path = Path(args.search_result).expanduser().resolve()
    search_result = read_json(search_path)

    clusters_path = Path(args.clusters).expanduser().resolve() if args.clusters else search_path.parent / "clusters_final.json"
    clusters_payload = read_json(clusters_path) if clusters_path.exists() else None

    output_path = Path(args.output).expanduser().resolve() if args.output else search_path.parent / "organized_search_result.json"
    organized = organize(
        search_result,
        clusters_payload,
        min_cluster_score=args.min_cluster_score,
        representatives_per_group=args.representatives_per_group,
    )
    write_json(output_path, organized)

    summary = {
        "output": str(output_path),
        "paper_count": organized["paper_count"],
        "method_cluster_count": organized["quality_notes"]["method_cluster_count"],
        "time_bucket_count": organized["quality_notes"]["time_bucket_count"],
        "low_confidence_or_peripheral_paper_count": organized["quality_notes"][
            "low_confidence_or_peripheral_paper_count"
        ],
        "all_papers_assigned": organized["quality_notes"]["all_papers_assigned"],
    }
    print(json.dumps(organized if args.pretty else summary, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
