from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.common import ensure_dir, normalize_whitespace, truncate_text
from ..core.schemas import SciAtlasRequest, TASK_IDEA_GENERATION, merge_task_params


_PIPELINE_PARAM_KEYS = {
    "anchor_top_k",
    "keyword_min_papers",
    "k_step1",
    "seed_num_queries",
    "seed_per_query",
    "seed_recent_years",
    "seed_min_citations",
    "graph_budget_ratio",
    "graph_budget_min",
    "graph_budget_max",
    "graph_max_predecessors_per_paper",
    "graph_min_forward_papers",
    "num_candidate_step2",
    "num_expansion_step2",
    "num_cross_domains",
    "inspiration_top_k_same_field",
    "inspiration_top_k_per_domain",
    "max_novelty_feedback_rounds",
    "idea_count",
}


def _apply_pipeline_param_overrides(cfg: Any, params: dict[str, Any]) -> None:
    for key in _PIPELINE_PARAM_KEYS:
        if key in params and params[key] is not None:
            setattr(cfg.pipeline, key, params[key])


def _artifact_if_exists(run_dir: Path, name: str) -> str | None:
    path = run_dir / name
    return str(path.resolve()) if path.exists() else None


def _idea_cards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    ideas = summary.get("ideas", [])
    if not isinstance(ideas, list):
        return []
    cards: list[dict[str, Any]] = []
    for item in ideas:
        if not isinstance(item, dict):
            continue
        cards.append(
            {
                "title": normalize_whitespace(item.get("title") or "Untitled"),
                "description": normalize_whitespace(item.get("description") or ""),
                "motivation": normalize_whitespace(item.get("motivation") or ""),
                "methods": normalize_whitespace(item.get("methods") or ""),
                "inspiration_sources": item.get("inspiration_sources") or [],
                "key_references": item.get("key_references") or [],
                "references": item.get("references") or [],
                "novelty_level": normalize_whitespace(item.get("novelty_level")),
                "novelty_justification": normalize_whitespace(item.get("novelty_justification")),
                "improvement_suggestions": item.get("improvement_suggestions") or [],
            }
        )
    return cards


def execute_current_idea_generation(request: SciAtlasRequest, run_dir: Path, _client: Any = None) -> dict[str, Any]:
    """Run the current SciAtlas idea-generation workflow.

    This is the only repository task implementation for `TASK_IDEA_GENERATION`.
    """
    from sciatlas_idea_gen.config import load_config
    from sciatlas_idea_gen.pipeline import run_pipeline

    params = merge_task_params(TASK_IDEA_GENERATION, request.params)
    topic_text = normalize_whitespace(request.input_payload.get("topic_text") or request.input_payload.get("idea_text"))
    if not topic_text and not request.input_payload.get("pdf_path") and not request.input_payload.get("pdf_paths"):
        raise ValueError(f"{TASK_IDEA_GENERATION} requires topic_text, idea_text, or pdf_path.")

    cfg = load_config()
    _apply_pipeline_param_overrides(cfg, params)
    workflow_dir = ensure_dir(run_dir / "artifacts" / "idea_generation_workflow")
    pdf_paths = request.input_payload.get("pdf_paths")
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
    elif not isinstance(pdf_paths, list):
        pdf_path = normalize_whitespace(request.input_payload.get("pdf_path"))
        pdf_paths = [pdf_path] if pdf_path else None

    summary = run_pipeline(
        topic_text or None,
        seed_keywords=params.get("seed_keywords") if isinstance(params.get("seed_keywords"), list) else None,
        domain=normalize_whitespace(params.get("target_field") or request.input_payload.get("domain")) or None,
        pdf_paths=pdf_paths,
        config=cfg,
        output_dir=workflow_dir,
        smoke_mode=bool(params.get("smoke_mode", False)),
        resume_from_run_dir=params.get("resume_from_run_dir"),
        save_intermediate_json=bool(params.get("save_intermediate_json", True)),
    )

    artifacts = {
        "workflow_run_dir": str(workflow_dir.resolve()),
        "summary_path": _artifact_if_exists(workflow_dir, "summary.json"),
        "retrieval_trace_path": _artifact_if_exists(workflow_dir, "retrieval_trace.json"),
        "llm_trace_path": _artifact_if_exists(workflow_dir, "llm_trace.jsonl"),
        "step1_seed_papers_path": _artifact_if_exists(workflow_dir, "step1_seed_papers.json"),
        "step2_research_graph_path": _artifact_if_exists(workflow_dir, "step2_research_graph.json"),
        "step3_trend_path": _artifact_if_exists(workflow_dir, "step3_trend.txt"),
        "step8_selected_inspirations_path": _artifact_if_exists(workflow_dir, "step8_selected_inspirations.json"),
        "step9_ideas_path": _artifact_if_exists(workflow_dir, "step9_ideas.json"),
        "step9_ideas_markdown_path": _artifact_if_exists(workflow_dir, "step9_ideas.md"),
    }
    artifacts = {key: value for key, value in artifacts.items() if value}

    return {
        "status": "ok" if summary.get("ok", True) else "partial",
        "input_summary": {
            "topic_text": truncate_text(topic_text, max_chars=220),
            "pdf_paths": pdf_paths or [],
        },
        "params_effective": {
            **params,
            "workflow": "sciatlas_idea_gen",
            "workflow_config": asdict(cfg.pipeline),
        },
        "artifacts": artifacts,
        "result": {
            "query_text": topic_text,
            "workflow": "sciatlas_idea_gen",
            "seed_paper_count": summary.get("seed_paper_count"),
            "graph_paper_count": summary.get("graph_paper_count"),
            "candidate_inspiration_count": summary.get("candidate_inspiration_count"),
            "selected_inspiration_count": summary.get("selected_inspiration_count"),
            "idea_count": summary.get("idea_count"),
            "ideas": _idea_cards(summary),
            "summary": summary,
        },
    }
