#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from review.common import first_non_empty, load_env_values, normalize_whitespace, slugify, write_json
from review.evaluation import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, DEFAULT_LLM_TEMPERATURE
from review.llm_retry import call_llm_json_with_retry


DEFAULT_SELECTION_SUMMARY = (
    SCRIPT_DIR
    / "dataset_runs"
    / "pairs_v2_final"
    / "summary_reviewers_20260529_130404.json"
)
DEFAULT_BACKGROUND_SUMMARY = (
    SCRIPT_DIR
    / "dataset_runs"
    / "pairs_v2_final"
    / "_background_status"
    / "backgrounds_full_gpu1_8"
    / "summary.json"
)
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"
EVAL_CACHE_VERSION = "v1"
METHOD_KEYS = {
    "pipeline": "pipeline_reviewers",
    "baseline": "baseline_reviewers",
    "our_wo_keywords": "our_wo_keywords_reviewers",
    "our_wo_graphwalk": "our_wo_graphwalk_reviewers",
    "openreview": "openreview_reviewers",
}


RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "rationale": {"type": "string"},
        "matched_expertise": {"type": "array", "items": {"type": "string"}},
        "missing_expertise": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "rationale", "matched_expertise", "missing_expertise"],
}


DIVERSITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "rationale": {"type": "string"},
        "coverage_axes": {"type": "array", "items": {"type": "string"}},
        "redundancy_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "rationale", "coverage_axes", "redundancy_notes"],
}


@dataclass(slots=True)
class ReviewerContext:
    method: str
    reviewer_index: int
    reviewer_key: str
    author_id: str
    author_name: str
    reviewer_payload: dict[str, Any]
    summary_path: Path
    relevant_papers_path: Path | None
    academic_background: dict[str, Any]
    paper_count: float
    citation_count: float
    authority_score: float = 0.0


@dataclass(slots=True)
class MethodContext:
    method: str
    output_dir: Path
    eval_dir: Path
    reviewers: list[ReviewerContext]


@dataclass(slots=True)
class PaperContext:
    selection_index: int
    paper: dict[str, Any]
    output_dir: Path
    reviewer_lists_path: Path
    methods: dict[str, MethodContext]
    skipped_reasons: list[str]


@dataclass(slots=True)
class LLMTask:
    task_type: str
    cache_path: Path
    label: str
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def clamp_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(10.0, score))


def truncate(text: Any, max_chars: int) -> str:
    value = normalize_whitespace(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def now_run_id(prefix: str) -> str:
    return f"{slugify(prefix, limit=40)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def method_key(method: str) -> str:
    if method not in METHOD_KEYS:
        raise ValueError(f"Unknown method: {method}")
    return METHOD_KEYS[method]


def reviewer_identity(reviewer: dict[str, Any]) -> tuple[str, str, str]:
    author_id = normalize_whitespace(reviewer.get("author_id"))
    author_name = normalize_whitespace(reviewer.get("author_name")) or normalize_whitespace(
        reviewer.get("display_name")
    )
    reviewer_key = normalize_whitespace(reviewer.get("reviewer_key")) or author_id or author_name
    return reviewer_key, author_id, author_name


def output_dir_from_work_dir(work_dir: Path) -> Path | None:
    # work_dir = <output_dir>/reviewer_backgrounds/<method>/<reviewer_slug>
    try:
        return work_dir.parents[2]
    except IndexError:
        return None


def load_background_index(background_summary_path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = read_json(background_summary_path)
    candidates: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        method = normalize_whitespace(row.get("method"))
        try:
            reviewer_index = int(row.get("reviewer_index"))
        except (TypeError, ValueError):
            continue
        work_dir_text = normalize_whitespace(row.get("work_dir"))
        if not method or not work_dir_text:
            continue
        output_dir = output_dir_from_work_dir(Path(work_dir_text).expanduser().resolve())
        if output_dir is None:
            continue
        candidates[(str(output_dir), method, reviewer_index)].append(row)

    chosen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, rows in candidates.items():
        rows = sorted(
            rows,
            key=lambda item: (
                normalize_whitespace(item.get("status")) == "ok",
                bool(normalize_whitespace(item.get("summary_path"))),
            ),
            reverse=True,
        )
        chosen[key] = rows[0]
    return chosen


def summary_has_profile(summary_path: Path) -> bool:
    try:
        payload = read_json(summary_path)
    except Exception:
        return False
    return bool(normalize_whitespace(payload.get("overall_academic_profile")))


def find_background_row(
    *,
    background_index: dict[tuple[str, str, int], dict[str, Any]],
    output_dir: Path,
    method: str,
    reviewer_index: int,
) -> dict[str, Any] | None:
    key = (str(output_dir.resolve()), method, reviewer_index)
    row = background_index.get(key)
    if row is not None:
        return row

    # Fallback for runs without a background summary.
    method_dir = output_dir / "reviewer_backgrounds" / method
    for result_path in sorted(method_dir.glob(f"{reviewer_index:02d}_*/result.json")):
        try:
            payload = read_json(result_path)
        except Exception:
            continue
        payload.setdefault("work_dir", str(result_path.parent.resolve()))
        payload.setdefault("method", method)
        payload.setdefault("reviewer_index", reviewer_index)
        return payload
    return None


def load_relevant_papers(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = read_json(path)
    except Exception:
        return []
    papers = payload.get("papers")
    return [paper for paper in papers if isinstance(paper, dict)] if isinstance(papers, list) else []


def reviewer_paper_count(reviewer: dict[str, Any], relevant_papers: list[dict[str, Any]]) -> float:
    for key in ("works_count_sum", "works_count"):
        try:
            value = float(reviewer.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return float(len(relevant_papers))


def reviewer_citation_count(relevant_papers: list[dict[str, Any]]) -> float:
    total = 0.0
    for paper in relevant_papers:
        try:
            value = float(paper.get("citations"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            total += value
    return total


def collect_paper_contexts(
    *,
    selection_summary_path: Path,
    background_summary_path: Path,
    methods: list[str],
    include_incomplete: bool,
    max_papers: int,
    only_domain: str,
    only_pattern: str,
) -> tuple[list[PaperContext], list[dict[str, Any]], Counter[str]]:
    selection = read_json(selection_summary_path)
    background_index = load_background_index(background_summary_path) if background_summary_path.exists() else {}
    contexts: list[PaperContext] = []
    skipped: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    domain_filter = normalize_whitespace(only_domain).casefold()
    pattern_filter = normalize_whitespace(only_pattern).casefold()

    for selection_index, result in enumerate(selection.get("results", []), start=1):
        if not isinstance(result, dict):
            continue
        paper = result.get("paper") if isinstance(result.get("paper"), dict) else {}
        output_dir = Path(normalize_whitespace(result.get("output_dir"))).expanduser().resolve()
        reviewer_lists_path = Path(normalize_whitespace(result.get("reviewer_lists_path"))).expanduser().resolve()
        title = normalize_whitespace(paper.get("title"))
        domain = normalize_whitespace(paper.get("domain"))
        match_text = "\n".join(
            [
                normalize_whitespace(paper.get("pair_id")),
                normalize_whitespace(paper.get("paper_key")),
                title,
                domain,
                normalize_whitespace(paper.get("pdf_path")),
            ]
        ).casefold()
        if domain_filter and domain_filter != domain.casefold():
            continue
        if pattern_filter and pattern_filter not in match_text:
            continue
        if normalize_whitespace(result.get("status")) != "ok":
            skipped.append({"paper": paper, "reason": "selection_not_ok", "selection_status": result.get("status")})
            counters["selection_not_ok"] += 1
            continue
        if not reviewer_lists_path.exists():
            skipped.append({"paper": paper, "reason": "missing_reviewer_lists", "path": str(reviewer_lists_path)})
            counters["missing_reviewer_lists"] += 1
            continue

        reviewer_lists = read_json(reviewer_lists_path)
        method_contexts: dict[str, MethodContext] = {}
        paper_reasons: list[str] = []
        for method in methods:
            reviewers = reviewer_lists.get(method_key(method))
            if not isinstance(reviewers, list) or len(reviewers) != 10:
                paper_reasons.append(f"{method}: reviewer_count={len(reviewers) if isinstance(reviewers, list) else 'missing'}")
                counters["bad_reviewer_count"] += 1
                continue

            reviewer_contexts: list[ReviewerContext] = []
            for reviewer_index, reviewer in enumerate(reviewers, start=1):
                if not isinstance(reviewer, dict):
                    paper_reasons.append(f"{method}:{reviewer_index}: reviewer_not_object")
                    counters["bad_reviewer_record"] += 1
                    continue
                reviewer_key, author_id, author_name = reviewer_identity(reviewer)
                row = find_background_row(
                    background_index=background_index,
                    output_dir=output_dir,
                    method=method,
                    reviewer_index=reviewer_index,
                )
                if row is None:
                    paper_reasons.append(f"{method}:{reviewer_index}: missing_background_result")
                    counters["missing_background_result"] += 1
                    continue
                if normalize_whitespace(row.get("status")) != "ok":
                    paper_reasons.append(
                        f"{method}:{reviewer_index}: background_status={normalize_whitespace(row.get('status')) or 'unknown'}"
                    )
                    counters["background_not_ok"] += 1
                    continue
                summary_path = Path(normalize_whitespace(row.get("summary_path"))).expanduser().resolve()
                if not summary_path.exists() or not summary_has_profile(summary_path):
                    paper_reasons.append(f"{method}:{reviewer_index}: missing_or_empty_author_summary")
                    counters["missing_or_empty_author_summary"] += 1
                    continue
                relevant_papers_text = normalize_whitespace(row.get("relevant_papers_path"))
                relevant_papers_path = Path(relevant_papers_text).expanduser().resolve() if relevant_papers_text else None
                if relevant_papers_path is None:
                    sibling = summary_path.parent / "relevant_papers.json"
                    relevant_papers_path = sibling if sibling.exists() else None
                academic_background = read_json(summary_path)
                relevant_papers = load_relevant_papers(relevant_papers_path)
                reviewer_contexts.append(
                    ReviewerContext(
                        method=method,
                        reviewer_index=reviewer_index,
                        reviewer_key=reviewer_key,
                        author_id=author_id,
                        author_name=author_name,
                        reviewer_payload=reviewer,
                        summary_path=summary_path,
                        relevant_papers_path=relevant_papers_path,
                        academic_background=academic_background,
                        paper_count=reviewer_paper_count(reviewer, relevant_papers),
                        citation_count=reviewer_citation_count(relevant_papers),
                    )
                )

            if len(reviewer_contexts) == 10:
                eval_dir = output_dir / "reviewer_list_evaluation" / EVAL_CACHE_VERSION / method
                method_contexts[method] = MethodContext(
                    method=method,
                    output_dir=output_dir,
                    eval_dir=eval_dir,
                    reviewers=reviewer_contexts,
                )
            else:
                counters["incomplete_method"] += 1

        complete = all(method in method_contexts and len(method_contexts[method].reviewers) == 10 for method in methods)
        if not complete and not include_incomplete:
            skipped.append({"paper": paper, "reason": "incomplete_backgrounds", "details": paper_reasons[:50]})
            counters["incomplete_paper"] += 1
            continue
        if not method_contexts:
            skipped.append({"paper": paper, "reason": "no_evaluable_methods", "details": paper_reasons[:50]})
            counters["no_evaluable_methods"] += 1
            continue
        contexts.append(
            PaperContext(
                selection_index=selection_index,
                paper=paper,
                output_dir=output_dir,
                reviewer_lists_path=reviewer_lists_path,
                methods=method_contexts,
                skipped_reasons=paper_reasons,
            )
        )
        if max_papers and len(contexts) >= max_papers:
            break

    return contexts, skipped, counters


def percentile(values: list[float], q: float) -> float:
    cleaned = sorted(value for value in values if math.isfinite(value) and value >= 0)
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return cleaned[0]
    position = (len(cleaned) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return cleaned[lower]
    weight = position - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def normalized_log_score(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log1p(max(0.0, value)) / math.log1p(reference)))


def assign_authority_scores(
    contexts: list[PaperContext],
    *,
    paper_weight: float,
    citation_weight: float,
) -> dict[str, Any]:
    reviewers = [
        reviewer
        for context in contexts
        for method_context in context.methods.values()
        for reviewer in method_context.reviewers
    ]
    p95_papers = percentile([reviewer.paper_count for reviewer in reviewers], 0.95)
    p95_citations = percentile([reviewer.citation_count for reviewer in reviewers], 0.95)
    total_weight = paper_weight + citation_weight
    if total_weight <= 0:
        paper_weight, citation_weight, total_weight = 0.4, 0.6, 1.0
    paper_weight /= total_weight
    citation_weight /= total_weight

    for reviewer in reviewers:
        paper_score = normalized_log_score(reviewer.paper_count, p95_papers)
        citation_score = normalized_log_score(reviewer.citation_count, p95_citations)
        reviewer.authority_score = 10.0 * (paper_weight * paper_score + citation_weight * citation_score)

    return {
        "paper_count_p95": p95_papers,
        "citation_count_p95": p95_citations,
        "paper_weight": paper_weight,
        "citation_weight": citation_weight,
    }


def paper_idea_text(paper: dict[str, Any], *, max_chars: int) -> str:
    parts = [
        f"Title: {normalize_whitespace(paper.get('title'))}",
        f"Domain: {normalize_whitespace(paper.get('domain'))}",
    ]
    abstract = normalize_whitespace(paper.get("abstract"))
    if abstract:
        parts.append(f"Abstract: {abstract}")
    doi = normalize_whitespace(paper.get("doi"))
    if doi:
        parts.append(f"DOI: {doi}")
    return truncate("\n".join(parts), max_chars)


def format_background(background: dict[str, Any], *, max_chars: int) -> str:
    trajectory = background.get("relevant_research_trajectory")
    arsenal = background.get("technical_arsenal")
    payload = {
        "overall_academic_profile": normalize_whitespace(background.get("overall_academic_profile")),
        "relevant_research_trajectory": trajectory if isinstance(trajectory, list) else [],
        "technical_arsenal": arsenal if isinstance(arsenal, list) else [],
    }
    return truncate(json.dumps(payload, ensure_ascii=False, indent=2), max_chars)


def build_relevance_task(
    *,
    context: PaperContext,
    method_context: MethodContext,
    reviewer: ReviewerContext,
    idea_chars: int,
    background_chars: int,
) -> LLMTask:
    paper_slug = slugify(normalize_whitespace(context.paper.get("title")), limit=60)
    reviewer_slug = slugify(reviewer.reviewer_key or reviewer.author_name, limit=80)
    cache_path = method_context.eval_dir / "relevance" / f"{reviewer.reviewer_index:02d}_{reviewer_slug}.json"
    system_prompt = (
        "You are an expert meta-reviewer evaluating whether an academic reviewer is relevant "
        "for a target research paper. Score only expertise relevance, not seniority or diversity. "
        "Return calibrated JSON with a numeric score from 0 to 10."
    )
    user_prompt = f"""
Evaluate the relevance between the target paper/idea and one candidate reviewer's academic background.

Scoring rubric:
0-2: almost unrelated.
3-4: broad same area but weak technical/problem match.
5-6: partially relevant expertise.
7-8: clearly relevant expertise for core review.
9-10: highly precise match to the paper's topic, methods, and evaluation needs.

Target paper / idea:
{paper_idea_text(context.paper, max_chars=idea_chars)}

Reviewer:
name: {reviewer.author_name}
author_id: {reviewer.author_id}

Academic background:
{format_background(reviewer.academic_background, max_chars=background_chars)}
""".strip()
    return LLMTask(
        task_type="relevance",
        cache_path=cache_path,
        label=f"relevance_{paper_slug}_{method_context.method}_{reviewer.reviewer_index:02d}",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=RELEVANCE_SCHEMA,
    )


def build_diversity_task(
    *,
    context: PaperContext,
    method_context: MethodContext,
    idea_chars: int,
    background_chars: int,
) -> LLMTask:
    paper_slug = slugify(normalize_whitespace(context.paper.get("title")), limit=60)
    cache_path = method_context.eval_dir / "diversity.json"
    reviewer_blocks = []
    for reviewer in method_context.reviewers:
        reviewer_blocks.append(
            "\n".join(
                [
                    f"Reviewer {reviewer.reviewer_index}",
                    f"name: {reviewer.author_name}",
                    f"author_id: {reviewer.author_id}",
                    "academic_background:",
                    format_background(reviewer.academic_background, max_chars=background_chars),
                ]
            )
        )
    system_prompt = (
        "You are an expert evaluator measuring only the background diversity of a 10-person "
        "reviewer panel. Score how different the reviewers are from each other in disciplines, "
        "research areas, methods, application domains, intellectual traditions, and perspectives. "
        "Do not evaluate whether the reviewers are relevant to any target paper. Do not reward "
        "a panel because many reviewers are close experts for the same topic, and do not penalize "
        "a reviewer because their background is unrelated to the target paper. "
        "Return JSON only."
    )
    user_prompt = f"""
Evaluate only the background diversity of this 10-reviewer panel.

The target paper/idea is intentionally not provided. Do not infer or use paper relevance. This task
is not relevance evaluation, not authority evaluation, and not reviewer usefulness evaluation.
Assume the only question is: how varied are these 10 reviewers' academic backgrounds relative to
one another?

Score 0-10:
0-2: reviewers are almost all from the same narrow field, method family, and application area.
3-4: one field or method cluster strongly dominates, with only minor background differences.
5-6: several distinct backgrounds are present, but there is still substantial concentration or redundancy.
7-8: strong background diversity across multiple disciplines, methods, applications, or viewpoints.
9-10: exceptionally broad diversity; reviewers span many clearly different fields, methods,
applications, and intellectual perspectives with little background redundancy.

Important calibration rules:
- Ignore target-paper fit completely. A reviewer can increase diversity even if their field would
  be irrelevant for reviewing the target paper.
- A panel of ten highly relevant experts from the same field should receive a low diversity score
  if their backgrounds are similar.
- A panel spanning unrelated areas such as medicine, physics, chemistry, AI, biology, engineering,
  and social or clinical applications should receive a high diversity score, even if some areas
  would not be relevant to a target paper.
- Do not use phrases such as "no plausible reviewing contribution", "lacks relevance", or
  "outside the paper's scope" as reasons to lower the diversity score.
- Base the rationale, coverage_axes, and redundancy_notes only on similarities and differences
  among reviewer backgrounds.

Reviewer panel:
{chr(10).join(reviewer_blocks)}
""".strip()
    return LLMTask(
        task_type="diversity",
        cache_path=cache_path,
        label=f"diversity_{paper_slug}_{method_context.method}",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=DIVERSITY_SCHEMA,
    )


def cached_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if payload.get("smoke"):
        return False
    return normalize_whitespace(payload.get("status")) == "ok" and clamp_score(payload.get("score")) is not None


def write_smoke_result(task: LLMTask) -> None:
    write_json(
        task.cache_path,
        {
            "status": "ok",
            "score": 5.0,
            "rationale": "Smoke-mode placeholder; no LLM request was sent.",
            "matched_expertise": [],
            "missing_expertise": [],
            "coverage_axes": [],
            "redundancy_notes": [],
            "task_type": task.task_type,
            "cache_path": str(task.cache_path),
            "smoke": True,
        },
    )


def run_llm_task(
    task: LLMTask,
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    max_tokens: int,
    rerun: bool,
    smoke: bool,
) -> dict[str, Any]:
    if not rerun and cached_ok(task.cache_path):
        return {"status": "cached", "task_type": task.task_type, "cache_path": str(task.cache_path)}
    started_at = time.perf_counter()
    task.cache_path.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        write_smoke_result(task)
        return {"status": "ok", "task_type": task.task_type, "cache_path": str(task.cache_path), "smoke": True}

    try:
        payload = call_llm_json_with_retry(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            system_prompt=task.system_prompt,
            user_content=task.user_prompt,
            response_model_schema=task.schema,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            helper_path=REPO_ROOT / "review" / "llm_json_call_worker.py",
            label=task.label,
            debug_dir=task.cache_path.parent.parent / "llm_failures",
            raise_on_error=False,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("LLM returned no JSON payload")
        score = clamp_score(payload.get("score"))
        if score is None:
            raise RuntimeError(f"LLM JSON missing valid score: {payload}")
        result = {
            "status": "ok",
            "task_type": task.task_type,
            "score": score,
            "payload": payload,
            "model": model_name,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "cache_path": str(task.cache_path),
        }
        for key in ("rationale", "matched_expertise", "missing_expertise", "coverage_axes", "redundancy_notes"):
            if key in payload:
                result[key] = payload[key]
        write_json(task.cache_path, result)
        return {"status": "ok", "task_type": task.task_type, "cache_path": str(task.cache_path)}
    except Exception as exc:
        error_payload = {
            "status": "error",
            "task_type": task.task_type,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "cache_path": str(task.cache_path),
        }
        write_json(task.cache_path, error_payload)
        return error_payload


def build_tasks(
    contexts: list[PaperContext],
    *,
    idea_chars: int,
    relevance_background_chars: int,
    diversity_background_chars: int,
    run_relevance: bool,
    run_diversity: bool,
) -> list[LLMTask]:
    tasks: list[LLMTask] = []
    seen_paths: set[Path] = set()
    for context in contexts:
        for method_context in context.methods.values():
            if run_relevance:
                for reviewer in method_context.reviewers:
                    task = build_relevance_task(
                        context=context,
                        method_context=method_context,
                        reviewer=reviewer,
                        idea_chars=idea_chars,
                        background_chars=relevance_background_chars,
                    )
                    if task.cache_path not in seen_paths:
                        seen_paths.add(task.cache_path)
                        tasks.append(task)
            if run_diversity:
                task = build_diversity_task(
                    context=context,
                    method_context=method_context,
                    idea_chars=idea_chars,
                    background_chars=diversity_background_chars,
                )
                if task.cache_path not in seen_paths:
                    seen_paths.add(task.cache_path)
                    tasks.append(task)
    return tasks


def run_tasks_parallel(
    tasks: list[LLMTask],
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    max_tokens: int,
    max_workers: int,
    rerun: bool,
    smoke: bool,
) -> Counter[str]:
    counters: Counter[str] = Counter()
    if not tasks:
        return counters
    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_llm_task,
                task,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                max_tokens=max_tokens,
                rerun=rerun,
                smoke=smoke,
            )
            for task in tasks
        ]
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            status = normalize_whitespace(result.get("status")) or "unknown"
            task_type = normalize_whitespace(result.get("task_type")) or "unknown"
            counters[f"{task_type}_{status}"] += 1
            counters[status] += 1
            if done_count == 1 or done_count % 50 == 0 or done_count == len(tasks):
                elapsed = time.perf_counter() - started_at
                print(
                    f"llm_progress={done_count}/{len(tasks)} elapsed_s={elapsed:.1f} "
                    f"ok={counters['ok']} cached={counters['cached']} error={counters['error']}",
                    flush=True,
                )
    return counters


def list_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def list_median(values: list[float]) -> float | None:
    return median(values) if values else None


def read_score(path: Path) -> tuple[float | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    try:
        payload = read_json(path)
    except Exception:
        return None, None
    if normalize_whitespace(payload.get("status")) != "ok":
        return None, payload
    return clamp_score(payload.get("score")), payload


def write_authority(method_context: MethodContext) -> dict[str, Any]:
    reviewer_scores = [
        {
            "reviewer_index": reviewer.reviewer_index,
            "reviewer_key": reviewer.reviewer_key,
            "author_id": reviewer.author_id,
            "author_name": reviewer.author_name,
            "paper_count": reviewer.paper_count,
            "citation_count": reviewer.citation_count,
            "authority_score": reviewer.authority_score,
        }
        for reviewer in method_context.reviewers
    ]
    scores = [item["authority_score"] for item in reviewer_scores]
    payload = {
        "status": "ok",
        "score": list_mean(scores),
        "authority_mean": list_mean(scores),
        "authority_median": list_median(scores),
        "authority_top1": max(scores) if scores else None,
        "authority_min": min(scores) if scores else None,
        "reviewers": reviewer_scores,
    }
    write_json(method_context.eval_dir / "authority.json", payload)
    return payload


def write_method_metrics(context: PaperContext, method_context: MethodContext) -> dict[str, Any]:
    relevance_items: list[dict[str, Any]] = []
    relevance_scores: list[float] = []
    for reviewer in method_context.reviewers:
        reviewer_slug = slugify(reviewer.reviewer_key or reviewer.author_name, limit=80)
        relevance_path = method_context.eval_dir / "relevance" / f"{reviewer.reviewer_index:02d}_{reviewer_slug}.json"
        score, payload = read_score(relevance_path)
        item = {
            "reviewer_index": reviewer.reviewer_index,
            "reviewer_key": reviewer.reviewer_key,
            "author_id": reviewer.author_id,
            "author_name": reviewer.author_name,
            "score": score,
            "path": str(relevance_path),
        }
        if payload is not None and normalize_whitespace(payload.get("status")) != "ok":
            item["error"] = payload.get("error")
        if score is not None:
            relevance_scores.append(score)
            item["rationale"] = payload.get("rationale") if payload else None
        relevance_items.append(item)

    diversity_score, diversity_payload = read_score(method_context.eval_dir / "diversity.json")
    authority_payload = write_authority(method_context)
    authority_score = clamp_score(authority_payload.get("score"))

    best_relevance = None
    if relevance_scores:
        best_relevance = max(
            (item for item in relevance_items if item.get("score") is not None),
            key=lambda item: float(item["score"]),
        )

    missing = []
    if len(relevance_scores) != len(method_context.reviewers):
        missing.append(f"relevance_complete={len(relevance_scores)}/{len(method_context.reviewers)}")
    if diversity_score is None:
        missing.append("diversity_missing")
    if authority_score is None:
        missing.append("authority_missing")

    metrics = {
        "status": "ok" if not missing else "partial_error",
        "paper": context.paper,
        "method": method_context.method,
        "relevance": max(relevance_scores) if relevance_scores else None,
        "diversity": diversity_score,
        "authority": authority_score,
        "best_relevance_reviewer": best_relevance,
        "reviewer_relevance": relevance_items,
        "diversity_result": diversity_payload,
        "authority_result": authority_payload,
        "missing": missing,
        "paths": {
            "metrics": str((method_context.eval_dir / "metrics.json").resolve()),
            "authority": str((method_context.eval_dir / "authority.json").resolve()),
            "diversity": str((method_context.eval_dir / "diversity.json").resolve()),
        },
    }
    write_json(method_context.eval_dir / "metrics.json", metrics)
    return metrics


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    return clamp_score(metrics.get(key))


def aggregate_rows(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    if len(methods) < 2:
        return {}
    left, right = methods[0], methods[1]

    def aggregate_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"paper_count": len(subset)}
        for metric in ("relevance", "diversity", "authority"):
            left_values: list[float] = []
            right_values: list[float] = []
            deltas: list[float] = []
            left_wins = 0
            ties = 0
            comparable = 0
            for row in subset:
                left_score = metric_value(row.get(left, {}), metric)
                right_score = metric_value(row.get(right, {}), metric)
                if left_score is None or right_score is None:
                    continue
                left_values.append(left_score)
                right_values.append(right_score)
                delta = left_score - right_score
                deltas.append(delta)
                comparable += 1
                if delta > 0:
                    left_wins += 1
                elif delta == 0:
                    ties += 1
            payload[metric] = {
                f"{left}_mean": list_mean(left_values),
                f"{left}_median": list_median(left_values),
                f"{right}_mean": list_mean(right_values),
                f"{right}_median": list_median(right_values),
                "delta_mean": list_mean(deltas),
                "delta_median": list_median(deltas),
                f"{left}_win_rate": (left_wins / comparable) if comparable else None,
                "tie_rate": (ties / comparable) if comparable else None,
                "comparable_count": comparable,
            }
        return payload

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = normalize_whitespace((row.get("paper") or {}).get("domain")) or "unknown"
        by_domain[domain].append(row)

    return {
        "overall": aggregate_subset(rows),
        "by_domain": {domain: aggregate_subset(items) for domain, items in sorted(by_domain.items())},
    }


def write_csv_summary(path: Path, rows: list[dict[str, Any]], methods: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "paper_key",
        "domain",
        "title",
    ]
    for method in methods:
        fieldnames.extend(
            [
                f"{method}_status",
                f"{method}_relevance",
                f"{method}_diversity",
                f"{method}_authority",
                f"{method}_diversity_rationale",
                f"{method}_diversity_coverage_axes",
                f"{method}_diversity_redundancy_notes",
            ]
        )
    if len(methods) >= 2:
        left, right = methods[0], methods[1]
        fieldnames.extend(["delta_relevance", "delta_diversity", "delta_authority"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            paper = row.get("paper") or {}
            item = {
                "pair_id": paper.get("pair_id"),
                "paper_key": paper.get("paper_key"),
                "domain": paper.get("domain"),
                "title": paper.get("title"),
            }
            for method in methods:
                metrics = row.get(method, {})
                item[f"{method}_status"] = metrics.get("status")
                item[f"{method}_relevance"] = metrics.get("relevance")
                item[f"{method}_diversity"] = metrics.get("diversity")
                item[f"{method}_authority"] = metrics.get("authority")
                diversity_result = metrics.get("diversity_result") if isinstance(metrics, dict) else None
                if isinstance(diversity_result, dict):
                    item[f"{method}_diversity_rationale"] = normalize_whitespace(diversity_result.get("rationale"))
                    coverage_axes = diversity_result.get("coverage_axes")
                    redundancy_notes = diversity_result.get("redundancy_notes")
                    item[f"{method}_diversity_coverage_axes"] = " | ".join(
                        normalize_whitespace(item)
                        for item in coverage_axes
                        if normalize_whitespace(item)
                    ) if isinstance(coverage_axes, list) else ""
                    item[f"{method}_diversity_redundancy_notes"] = " | ".join(
                        normalize_whitespace(item)
                        for item in redundancy_notes
                        if normalize_whitespace(item)
                    ) if isinstance(redundancy_notes, list) else ""
            if len(methods) >= 2:
                left, right = methods[0], methods[1]
                for metric in ("relevance", "diversity", "authority"):
                    left_score = metric_value(row.get(left, {}), metric)
                    right_score = metric_value(row.get(right, {}), metric)
                    item[f"delta_{metric}"] = (
                        left_score - right_score if left_score is not None and right_score is not None else None
                    )
            writer.writerow(item)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reviewer lists with LLM relevance/diversity and formula-based authority."
    )
    parser.add_argument("--summary-json", default=str(DEFAULT_SELECTION_SUMMARY), help="run_pairs_dataset summary JSON.")
    parser.add_argument(
        "--background-summary-json",
        default=str(DEFAULT_BACKGROUND_SUMMARY),
        help="run_reviewer_backgrounds summary JSON.",
    )
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with DMX/OPENAI API key.")
    parser.add_argument("--run-id", default=None, help="Summary run id. Defaults to eval_<timestamp>.")
    parser.add_argument("--methods", default="pipeline,baseline", help="Comma-separated methods to evaluate.")
    parser.add_argument("--max-workers", type=int, default=128, help="Concurrent LLM requests.")
    parser.add_argument("--max-papers", type=int, default=0, help="Evaluate at most N papers after filtering.")
    parser.add_argument("--only-domain", default="", help="Only evaluate a single domain.")
    parser.add_argument("--only-pattern", default="", help="Only evaluate papers whose title/path/domain contains text.")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Evaluate methods that have complete backgrounds even if the other method is incomplete.",
    )
    parser.add_argument("--skip-relevance", action="store_true", help="Do not run relevance LLM calls.")
    parser.add_argument("--skip-diversity", action="store_true", help="Do not run diversity LLM calls.")
    parser.add_argument("--rerun", action="store_true", help="Rerun LLM calls even when cache files are ok.")
    parser.add_argument("--dry-run", action="store_true", help="Collect tasks and print counts without running LLMs.")
    parser.add_argument("--smoke", action="store_true", help="Write placeholder LLM scores instead of calling the LLM.")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model-name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    parser.add_argument("--llm-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--idea-max-chars", type=int, default=5000)
    parser.add_argument("--relevance-background-max-chars", type=int, default=3000)
    parser.add_argument("--diversity-background-max-chars", type=int, default=1800)
    parser.add_argument("--authority-paper-weight", type=float, default=0.4)
    parser.add_argument("--authority-citation-weight", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.perf_counter()
    summary_path = Path(args.summary_json).expanduser().resolve()
    background_summary_path = Path(args.background_summary_json).expanduser().resolve()
    env_path = Path(args.env).expanduser().resolve() if args.env else None
    methods = [normalize_whitespace(item) for item in args.methods.split(",") if normalize_whitespace(item)]
    if not methods:
        raise ValueError("--methods must include at least one method")
    for method in methods:
        method_key(method)

    contexts, skipped, collect_counters = collect_paper_contexts(
        selection_summary_path=summary_path,
        background_summary_path=background_summary_path,
        methods=methods,
        include_incomplete=bool(args.include_incomplete),
        max_papers=max(0, int(args.max_papers)),
        only_domain=args.only_domain,
        only_pattern=args.only_pattern,
    )
    authority_config = assign_authority_scores(
        contexts,
        paper_weight=float(args.authority_paper_weight),
        citation_weight=float(args.authority_citation_weight),
    )
    tasks = build_tasks(
        contexts,
        idea_chars=max(1000, int(args.idea_max_chars)),
        relevance_background_chars=max(500, int(args.relevance_background_max_chars)),
        diversity_background_chars=max(500, int(args.diversity_background_max_chars)),
        run_relevance=not args.skip_relevance,
        run_diversity=not args.skip_diversity,
    )
    pending_tasks = [task for task in tasks if args.rerun or not cached_ok(task.cache_path)]

    print(f"summary_json={summary_path}")
    print(f"background_summary_json={background_summary_path}")
    print(f"methods={','.join(methods)}")
    print(f"evaluable_papers={len(contexts)} skipped_papers={len(skipped)}")
    print(f"llm_tasks_total={len(tasks)} pending={len(pending_tasks)} cached={len(tasks) - len(pending_tasks)}")
    print(f"max_workers={args.max_workers}")
    print(f"authority_config={json.dumps(authority_config, ensure_ascii=False)}")

    if args.dry_run:
        return 0

    env_values = load_env_values(env_path)
    api_key = first_non_empty(
        args.llm_api_key,
        env_values.get("DMX-API-KEY"),
        env_values.get("DMX_API_KEY"),
        env_values.get("OPENAI_API_KEY"),
    )
    if not api_key and not args.smoke and pending_tasks:
        raise ValueError(f"Missing LLM API key. Pass --llm-api-key or set it in {env_path}")
    base_url = first_non_empty(args.llm_base_url, env_values.get("OPENAI_BASE_URL"), DEFAULT_LLM_BASE_URL)
    model_name = first_non_empty(args.llm_model_name, DEFAULT_LLM_MODEL_NAME)

    task_counters = run_tasks_parallel(
        pending_tasks,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=float(args.temperature),
        timeout_seconds=max(30, int(args.llm_timeout_seconds)),
        max_retries=max(1, int(args.max_retries)),
        max_tokens=max(256, int(args.llm_max_tokens)),
        max_workers=max(1, int(args.max_workers)),
        rerun=bool(args.rerun),
        smoke=bool(args.smoke),
    )

    rows: list[dict[str, Any]] = []
    metrics_counters: Counter[str] = Counter()
    for context in contexts:
        row: dict[str, Any] = {
            "paper": context.paper,
            "selection_index": context.selection_index,
            "output_dir": str(context.output_dir),
            "reviewer_lists_path": str(context.reviewer_lists_path),
            "skipped_reasons": context.skipped_reasons[:50],
        }
        for method in methods:
            method_context = context.methods.get(method)
            if method_context is None:
                continue
            metrics = write_method_metrics(context, method_context)
            diversity_result = metrics.get("diversity_result")
            row[method] = {
                "status": metrics.get("status"),
                "relevance": metrics.get("relevance"),
                "diversity": metrics.get("diversity"),
                "authority": metrics.get("authority"),
                "best_relevance_reviewer": metrics.get("best_relevance_reviewer"),
                "diversity_result": diversity_result,
                "metrics_path": metrics.get("paths", {}).get("metrics"),
            }
            metrics_counters[f"{method}_{metrics.get('status')}"] += 1
        if len(methods) >= 2 and methods[0] in row and methods[1] in row:
            left, right = methods[0], methods[1]
            row["delta"] = {}
            for metric in ("relevance", "diversity", "authority"):
                left_score = metric_value(row[left], metric)
                right_score = metric_value(row[right], metric)
                row["delta"][metric] = (
                    left_score - right_score if left_score is not None and right_score is not None else None
                )
        rows.append(row)

    run_id = normalize_whitespace(args.run_id) or now_run_id("reviewer_list_eval")
    result_root = summary_path.parent
    summary_output_path = result_root / f"evaluation_summary_{run_id}.json"
    csv_output_path = result_root / f"evaluation_summary_{run_id}.csv"
    summary_payload = {
        "status": "ok" if not task_counters.get("error") else "partial_error",
        "run_id": run_id,
        "summary_json": str(summary_path),
        "background_summary_json": str(background_summary_path),
        "methods": methods,
        "model": None if args.smoke else model_name,
        "base_url": None if args.smoke else base_url,
        "max_workers": max(1, int(args.max_workers)),
        "evaluable_paper_count": len(contexts),
        "skipped_paper_count": len(skipped),
        "llm_task_count": len(tasks),
        "llm_pending_task_count": len(pending_tasks),
        "collection_counters": dict(collect_counters),
        "task_counters": dict(task_counters),
        "metrics_counters": dict(metrics_counters),
        "authority_config": authority_config,
        "aggregate": aggregate_rows(rows, methods),
        "results": rows,
        "skipped": skipped,
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }
    write_json(summary_output_path, summary_payload)
    write_csv_summary(csv_output_path, rows, methods)
    print(f"summary_output={summary_output_path}")
    print(f"csv_output={csv_output_path}")
    print(json.dumps(summary_payload["aggregate"].get("overall", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
