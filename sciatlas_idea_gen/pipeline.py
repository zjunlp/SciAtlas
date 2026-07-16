"""End-to-end SciAtlas idea-generation pipeline orchestrator."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import traceback
from typing import Any

from .clients.llm_client import LLMClient
from .clients.sciatlas_client import SciAtlasClient
from .config import Config, load_config
from .models import GraphEdge, Idea, Inspiration, Paper, ResearchGraph, SeedSet
from .steps import step1_anchor_retrieval, step2_graph_construction, step3_trend_summarizer, step4_rss_extraction
from .steps import step5_inspiration_gate, step6_inspiration_retrieval, step7_analogy_extraction, step8_inspiration_selection, step9_idea_generation, step9_novelty_check
from .utils import get_logger, read_json, save_json, save_text

log = get_logger("sciatlas.pipeline")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ideas_to_json(ideas: list[Idea]) -> list[dict[str, Any]]:
    return [
        {
            "title": idea.title,
            "motivation": idea.motivation,
            "methods": idea.methods,
            "method_overview": idea.method_overview,
            "proposed_method": idea.proposed_method,
            "analogy_mapping": idea.analogy_mapping,
            "experiment_plan": idea.experiment_plan,
            "description": idea.description,
            "inspiration_sources": idea.inspiration_sources,
            "key_references": idea.key_references,
            "references": idea.references,
            "novelty_level": idea.novelty_level,
            "novelty_justification": idea.novelty_justification,
            "improvement_suggestions": idea.improvement_suggestions,
        }
        for idea in ideas
    ]


def _ideas_to_markdown(ideas: list[Idea]) -> str:
    lines: list[str] = []
    for index, idea in enumerate(ideas, start=1):
        title = (idea.title or f"Idea {index}").strip()
        motivation = (idea.motivation or "").strip()
        if index > 1:
            lines.append("---")
            lines.append("")
        lines.append(f"# {title}")
        lines.append("")
        lines.append("## Motivation")
        lines.append("")
        lines.append(motivation or "N/A")
        lines.append("")
        lines.append("## Method")
        lines.append("")
        overview = (idea.method_overview or "").strip()
        if overview:
            lines.append(overview)
            lines.append("")
        if idea.proposed_method:
            for comp_index, component in enumerate(idea.proposed_method, start=1):
                name = str(component.get("name", "")).strip()
                description = str(component.get("description", "")).strip()
                heading = f"**{comp_index}. {name}**" if name else f"**{comp_index}.**"
                lines.append(heading)
                if description:
                    lines.append("")
                    lines.append(description)
                lines.append("")
            if lines[-1] == "":
                lines.pop()
        else:
            lines.append((idea.methods or "").strip() or "N/A")
        lines.append("")
        lines.append("## References")
        lines.append("")
        if idea.references:
            for ref_index, ref in enumerate(idea.references, start=1):
                full = str(ref.get("full", "")).strip()
                citation = str(ref.get("citation", "")).strip() or "Reference"
                url = str(ref.get("url", "")).strip()
                if full:
                    lines.append(f"{ref_index}. {full}")
                elif url:
                    lines.append(f"{ref_index}. {citation}. {url}")
                else:
                    lines.append(f"{ref_index}. {citation}")
        else:
            lines.append("N/A")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _inspirations_to_json(items: list[Inspiration]) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for item in items:
        obj = asdict(item)
        source_paper = obj.pop("source_paper", None)
        if source_paper:
            obj["source_paper"] = source_paper
        data.append(obj)
    return data


def _save_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    save_json(summary, run_dir / "summary.json")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _redact_config(data: dict[str, Any]) -> dict[str, Any]:
    redacted = _to_jsonable(data)
    for key in ("api_key", "token", "authorization"):
        if key in redacted and redacted[key]:
            redacted[key] = "***REDACTED***"
    return redacted


def _load_summary(run_dir: Path, *, topic: str | None, smoke_mode: bool) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if path.exists():
        summary = read_json(path)
        summary.setdefault("steps", {})
        summary["run_dir"] = str(run_dir)
        if topic:
            summary["topic"] = topic
        summary["smoke_mode"] = smoke_mode
        return summary
    return {
        "ok": True,
        "run_dir": str(run_dir),
        "topic": topic,
        "smoke_mode": smoke_mode,
        "steps": {},
    }


def _resume_step(summary: dict[str, Any], step_name: str, **payload: Any) -> None:
    summary["steps"][step_name] = {"status": "resumed", **payload}


def _paper_from_saved(data: dict[str, Any]) -> Paper:
    return Paper.from_api(data)


def _load_seedset(path: Path) -> SeedSet:
    data = read_json(path)
    return SeedSet(
        seed_papers=[_paper_from_saved(item) for item in data.get("seed_papers", [])],
        research_problem=str(data.get("research_problem", "") or ""),
        keywords=list(data.get("keywords", []) or []),
        refined_query=str(data.get("refined_query", "") or ""),
    )


def _load_graph(path: Path) -> ResearchGraph:
    data = read_json(path)
    graph = ResearchGraph(seed_paper_id=data.get("seed_paper_id"))
    graph.papers = {
        pid: _paper_from_saved(item) for pid, item in (data.get("papers", {}) or {}).items()
    }
    graph.edges = [GraphEdge(**edge) for edge in data.get("edges", []) or []]
    graph.evidence_scores = {
        str(k): float(v) for k, v in (data.get("evidence_scores", {}) or {}).items()
    }
    graph.foundational_ids = list(data.get("foundational_ids", []) or [])
    graph.phases = [list(layer) for layer in data.get("phases", []) or []]
    graph.seed_paper_ids = list(data.get("seed_paper_ids", []) or [])
    return graph


def _load_inspirations(path: Path) -> list[Inspiration]:
    data = read_json(path)
    items: list[Inspiration] = []
    for obj in data if isinstance(data, list) else []:
        if not isinstance(obj, dict):
            continue
        source_paper = obj.get("source_paper")
        items.append(
            Inspiration(
                domain=str(obj.get("domain", "")),
                paper_title=str(obj.get("paper_title", "")),
                paper_abstract=str(obj.get("paper_abstract", "")),
                radius=str(obj.get("radius", "")),
                is_combination_plausible=bool(obj.get("is_combination_plausible", False)),
                combination_points=list(obj.get("combination_points", []) or []),
                combination_plan=str(obj.get("combination_plan", "") or ""),
                justification=str(obj.get("justification", "") or ""),
                source_paper=_paper_from_saved(source_paper) if isinstance(source_paper, dict) else None,
                candidate_id=int(obj.get("candidate_id")) if obj.get("candidate_id") not in (None, "") else None,
            )
        )
    return items


def _load_ideas(path: Path) -> list[Idea]:
    return [Idea(**obj) for obj in read_json(path)]


def _is_flash_workflow(cfg: Config) -> bool:
    return str(getattr(cfg.pipeline, "workflow_mode", "") or "").strip().lower() == "flash"


def _flash_radius_plan(cfg: Config) -> dict[str, Any]:
    """Deterministic inspiration plan for the fast interactive workflow."""
    same_field_top_k = max(0, int(getattr(cfg.pipeline, "inspiration_top_k_same_field", 0)))
    num_cross_domains = max(0, int(getattr(cfg.pipeline, "num_cross_domains", 0)))
    cross_domain_top_k_per_domain = max(
        0,
        int(getattr(cfg.pipeline, "inspiration_top_k_per_domain", 0)),
    )
    use_same_field = same_field_top_k > 0
    use_cross_domain = num_cross_domains > 0 and cross_domain_top_k_per_domain > 0
    if not use_same_field and not use_cross_domain:
        use_same_field = True
        same_field_top_k = 1
    return {
        "use_same_field": use_same_field,
        "use_cross_domain": use_cross_domain,
        "preferred_radii": [
            radius
            for radius, enabled in (("R1", use_same_field), ("R2", use_cross_domain))
            if enabled
        ],
        "same_field_top_k": same_field_top_k if use_same_field else 0,
        "num_cross_domains": num_cross_domains if use_cross_domain else 0,
        "cross_domain_top_k_per_domain": (
            cross_domain_top_k_per_domain if use_cross_domain else 0
        ),
        "rationale": (
            "flash workflow: skip the LLM radius gate and use the compact default "
            "plan to keep one same-field and one cross-domain evidence path when available"
        ),
        "compressed_stage": "step5",
    }


def _flash_select_inspirations(cfg: Config, inspirations: list[Inspiration]) -> list[Inspiration]:
    """Deterministic replacement for Step 8 in flash mode."""
    if not inspirations:
        return []
    max_selected = max(
        1,
        int(getattr(cfg.pipeline, "inspiration_top_k_same_field", 0))
        + int(getattr(cfg.pipeline, "num_cross_domains", 0))
        * int(getattr(cfg.pipeline, "inspiration_top_k_per_domain", 0)),
    )
    ordered = sorted(
        enumerate(inspirations),
        key=lambda item: (
            0
            if item[1].radius == "R2" or (item[1].domain and item[1].domain != "same_field")
            else 1,
            item[0],
        ),
    )
    return [item for _, item in ordered[:max_selected]]


def _collect_novelty_feedback(ideas: list[Idea]) -> str:
    lines: list[str] = []
    for index, idea in enumerate(ideas, start=1):
        level = (idea.novelty_level or "unknown").strip()
        if level not in {"medium", "low", "none"}:
            continue
        lines.append(f"Idea {index}: {idea.title}")
        lines.append(f"Novelty level: {level}")
        if idea.novelty_justification:
            lines.append(f"Justification: {idea.novelty_justification}")
        if idea.improvement_suggestions:
            lines.append(
                "Improvement suggestions: "
                + "; ".join(str(item) for item in idea.improvement_suggestions if item)
            )
        lines.append("")
    return "\n".join(lines).strip()


def _should_retry_for_novelty(ideas: list[Idea]) -> bool:
    return any((idea.novelty_level or "").strip() in {"medium", "low", "none"} for idea in ideas)


def _fail_or_raise(
    *,
    smoke_mode: bool,
    run_dir: Path,
    summary: dict[str, Any],
    step_name: str,
    exc: Exception,
    save_intermediate_json: bool,
) -> dict[str, Any]:
    error = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    summary["ok"] = False
    summary["failed_step"] = step_name
    summary["steps"][step_name] = {"status": "failed", "error": error}
    if save_intermediate_json:
        save_json(error, run_dir / f"{step_name}_error.json")
    _save_summary(run_dir, summary)
    if smoke_mode:
        return summary
    raise exc


def run_pipeline(
    topic: str | None,
    *,
    seed_keywords: list[dict[str, Any]] | None = None,
    domain: str | None = None,
    pdf_paths: list[str] | None = None,
    config: Config | None = None,
    output_dir: str | Path | None = None,
    smoke_mode: bool = False,
    resume_from_run_dir: str | Path | None = None,
    save_intermediate_json: bool = True,
) -> dict[str, Any]:
    """Run the full idea-generation pipeline and persist artifacts to disk."""
    cfg = config or load_config()
    run_dir = (
        Path(output_dir) if output_dir
        else Path(resume_from_run_dir) if resume_from_run_dir
        else cfg.pipeline.runs_dir / _stamp()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Pipeline run dir: %s", run_dir)
    sciatlas = SciAtlasClient(cfg.sciatlas, trace_path=run_dir / "retrieval_trace.json")
    llm = LLMClient(cfg.llm, trace_path=run_dir / "llm_trace.jsonl")
    summary = _load_summary(run_dir, topic=topic, smoke_mode=smoke_mode)
    workflow_mode = str(getattr(cfg.pipeline, "workflow_mode", "default") or "default")
    topic = str(summary.get("topic") or topic or "")
    if not topic and not pdf_paths:
        raise ValueError(
            "topic is required unless --pdf is given or it can be recovered from summary.json"
        )
    pdf_paths = [str(p) for p in (pdf_paths or [])]
    summary["input"] = {
        "topic": topic,
        "seed_keywords": seed_keywords,
        "domain": domain,
        "pdf_paths": pdf_paths,
        "resume_from_run_dir": str(resume_from_run_dir) if resume_from_run_dir else None,
        "workflow_mode": workflow_mode,
    }
    summary["workflow_mode"] = workflow_mode
    summary["workflow_path"] = (
        "flash"
        if workflow_mode == "flash"
        else "full" if workflow_mode == "full" else workflow_mode
    )
    summary["effective_config"] = {
        "sciatlas": _redact_config(asdict(cfg.sciatlas)),
        "llm": {
            "base_url": cfg.llm.base_url,
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "max_retries": cfg.llm.max_retries,
            "request_timeout": cfg.llm.request_timeout,
        },
        "pipeline": _to_jsonable(asdict(cfg.pipeline)),
    }
    _save_summary(run_dir, summary)

    step1_path = run_dir / "step1_seed_papers.json"
    step2_path = run_dir / "step2_research_graph.json"
    step3_graph_path = run_dir / "step3_serialized_graph.txt"
    step3_trend_path = run_dir / "step3_trend.txt"
    step4_path = run_dir / "step4_rss.json"
    step5_path = run_dir / "step5_radius_plan.json"
    step6_path = run_dir / "step6_inspiration_candidates.json"
    step7_path = run_dir / "step7_inspirations.json"
    step8_path = run_dir / "step8_selected_inspirations.json"
    step9_path = run_dir / "step9_ideas.json"
    step9_markdown_path = run_dir / "step9_ideas.md"

    if step1_path.exists():
        seeds = _load_seedset(step1_path)
        _resume_step(summary, "step1", seed_paper_count=len(seeds.seed_papers))
        _save_summary(run_dir, summary)
    else:
        try:
            sciatlas.set_trace_context("step1")
            llm.set_trace_context("step1")
            seeds = step1_anchor_retrieval.run(
                sciatlas,
                llm,
                topic,
                cfg.pipeline,
                seed_keywords=seed_keywords,
                domain=domain,
                pdf_paths=pdf_paths,
            )
            if save_intermediate_json:
                save_json(asdict(seeds), step1_path)
            summary["steps"]["step1"] = {
                "status": "completed",
                "seed_paper_count": len(seeds.seed_papers),
            }
            _save_summary(run_dir, summary)
        except Exception as exc:
            return _fail_or_raise(
                smoke_mode=smoke_mode,
                run_dir=run_dir,
                summary=summary,
                step_name="step1",
                exc=exc,
                save_intermediate_json=save_intermediate_json,
            )
        finally:
            sciatlas.set_trace_context(None)
            llm.set_trace_context(None)

    if step2_path.exists():
        graph = _load_graph(step2_path)
        _resume_step(summary, "step2", graph_paper_count=len(graph.papers))
        _save_summary(run_dir, summary)
    else:
        try:
            sciatlas.set_trace_context("step2")
            llm.set_trace_context("step2")
            graph = step2_graph_construction.run(sciatlas, llm, seeds, cfg.pipeline)
            if save_intermediate_json:
                save_json(asdict(graph), step2_path)
            summary["steps"]["step2"] = {
                "status": "completed",
                "graph_paper_count": len(graph.papers),
                "edge_count": len(graph.edges),
            }
            _save_summary(run_dir, summary)
        except Exception as exc:
            return _fail_or_raise(
                smoke_mode=smoke_mode,
                run_dir=run_dir,
                summary=summary,
                step_name="step2",
                exc=exc,
                save_intermediate_json=save_intermediate_json,
            )
        finally:
            sciatlas.set_trace_context(None)
            llm.set_trace_context(None)

    if step3_graph_path.exists() and step3_trend_path.exists():
        serialized_graph = step3_graph_path.read_text(encoding="utf-8")
        trend = step3_trend_path.read_text(encoding="utf-8")
        _resume_step(
            summary,
            "step3",
            serialized_chars=len(serialized_graph),
            trend_chars=len(trend),
        )
        _save_summary(run_dir, summary)
    else:
        try:
            llm.set_trace_context("step3")
            serialized_graph = step3_trend_summarizer.render_graph_markdown(graph, topic)
            trend = step3_trend_summarizer.run(llm, graph, topic, serialized_graph=serialized_graph)
            save_text(serialized_graph, step3_graph_path)
            save_text(trend, step3_trend_path)
            summary["steps"]["step3"] = {
                "status": "completed",
                "serialized_chars": len(serialized_graph),
                "trend_chars": len(trend),
            }
            _save_summary(run_dir, summary)
        except Exception as exc:
            return _fail_or_raise(
                smoke_mode=smoke_mode,
                run_dir=run_dir,
                summary=summary,
                step_name="step3",
                exc=exc,
                save_intermediate_json=save_intermediate_json,
            )
        finally:
            llm.set_trace_context(None)

    if step4_path.exists():
        rss = read_json(step4_path)
        _resume_step(summary, "step4", rss_keys=sorted(rss.keys()))
        _save_summary(run_dir, summary)
    else:
        try:
            llm.set_trace_context("step4")
            rss = step4_rss_extraction.run(llm, graph, topic, trend)
            if save_intermediate_json:
                save_json(rss, step4_path)
            summary["steps"]["step4"] = {"status": "completed", "rss_keys": sorted(rss.keys())}
            _save_summary(run_dir, summary)
        except Exception as exc:
            return _fail_or_raise(
                smoke_mode=smoke_mode,
                run_dir=run_dir,
                summary=summary,
                step_name="step4",
                exc=exc,
                save_intermediate_json=save_intermediate_json,
            )
        finally:
            llm.set_trace_context(None)

    if step5_path.exists():
        radius_plan = read_json(step5_path)
        _resume_step(summary, "step5", preferred_radii=radius_plan.get("preferred_radii", []))
        _save_summary(run_dir, summary)
    else:
        try:
            if _is_flash_workflow(cfg):
                radius_plan = _flash_radius_plan(cfg)
                summary["steps"]["step5"] = {
                    "status": "compressed",
                    "preferred_radii": radius_plan.get("preferred_radii", []),
                    "compression": "deterministic flash radius plan; skipped LLM gate",
                }
                log.info("Step 5: compressed flash radius plan: %s", radius_plan)
            else:
                llm.set_trace_context("step5")
                radius_plan = step5_inspiration_gate.run(llm, rss, trend, cfg.pipeline)
                summary["steps"]["step5"] = {
                    "status": "completed",
                    "preferred_radii": radius_plan.get("preferred_radii", []),
                }
            if save_intermediate_json:
                save_json(radius_plan, step5_path)
            _save_summary(run_dir, summary)
        except Exception as exc:
            return _fail_or_raise(
                smoke_mode=smoke_mode,
                run_dir=run_dir,
                summary=summary,
                step_name="step5",
                exc=exc,
                save_intermediate_json=save_intermediate_json,
            )
        finally:
            llm.set_trace_context(None)

    candidate_inspirations = _load_inspirations(step6_path) if step6_path.exists() else []
    inspirations = _load_inspirations(step7_path) if step7_path.exists() else []
    selected_inspirations = _load_inspirations(step8_path) if step8_path.exists() else []

    if step9_path.exists():
        ideas = _load_ideas(step9_path)
        save_text(_ideas_to_markdown(ideas), step9_markdown_path)
        _resume_step(summary, "step6", candidate_count=len(candidate_inspirations))
        _resume_step(summary, "step7", inspiration_count=len(inspirations))
        _resume_step(summary, "step8", selected_count=len(selected_inspirations))
        _resume_step(summary, "step9", idea_count=len(ideas))
        summary["post_step9_novelty"] = {
            "status": "resumed",
            "novelty_levels": [idea.novelty_level for idea in ideas],
        }
        _save_summary(run_dir, summary)
    else:
        max_rounds = max(0, int(cfg.pipeline.max_novelty_feedback_rounds))
        novelty_feedback = ""
        ideas: list[Idea] = []
        problem_text = str(rss.get("core_gap") or seeds.refined_query)
        for round_idx in range(max_rounds + 1):
            try:
                if round_idx == 0 and step6_path.exists():
                    candidate_inspirations = _load_inspirations(step6_path)
                else:
                    sciatlas.set_trace_context("step6")
                    llm.set_trace_context("step6")
                    candidate_inspirations = step6_inspiration_retrieval.run(
                        sciatlas,
                        llm,
                        seeds,
                        rss,
                        cfg.pipeline,
                        plan=radius_plan,
                    )
                    if save_intermediate_json:
                        save_json(_inspirations_to_json(candidate_inspirations), step6_path)
                summary["steps"]["step6"] = {
                    "status": "completed",
                    "candidate_count": len(candidate_inspirations),
                    "round": round_idx + 1,
                }
                _save_summary(run_dir, summary)
            except Exception as exc:
                return _fail_or_raise(
                    smoke_mode=smoke_mode,
                    run_dir=run_dir,
                    summary=summary,
                    step_name="step6",
                    exc=exc,
                    save_intermediate_json=save_intermediate_json,
                )
            finally:
                sciatlas.set_trace_context(None)
                llm.set_trace_context(None)

            try:
                if round_idx == 0 and step7_path.exists():
                    inspirations = _load_inspirations(step7_path)
                else:
                    llm.set_trace_context("step7")
                    inspirations = step7_analogy_extraction.run(llm, problem_text, candidate_inspirations)
                    if save_intermediate_json:
                        save_json(_inspirations_to_json(inspirations), step7_path)
                summary["steps"]["step7"] = {
                    "status": "completed",
                    "inspiration_count": len(inspirations),
                    "round": round_idx + 1,
                }
                _save_summary(run_dir, summary)
            except Exception as exc:
                return _fail_or_raise(
                    smoke_mode=smoke_mode,
                    run_dir=run_dir,
                    summary=summary,
                    step_name="step7",
                    exc=exc,
                    save_intermediate_json=save_intermediate_json,
                )
            finally:
                llm.set_trace_context(None)

            try:
                if round_idx == 0 and step8_path.exists():
                    selected_inspirations = _load_inspirations(step8_path)
                else:
                    if _is_flash_workflow(cfg):
                        selected_inspirations = _flash_select_inspirations(cfg, inspirations)
                    else:
                        llm.set_trace_context("step8")
                        selected_inspirations = step8_inspiration_selection.run(llm, rss, trend, inspirations)
                    if save_intermediate_json:
                        save_json(_inspirations_to_json(selected_inspirations), step8_path)
                summary["steps"]["step8"] = {
                    "status": "compressed" if _is_flash_workflow(cfg) else "completed",
                    "selected_count": len(selected_inspirations),
                    "round": round_idx + 1,
                }
                if _is_flash_workflow(cfg):
                    summary["steps"]["step8"]["compression"] = (
                        "deterministic flash selection; skipped LLM inspiration selector"
                    )
                _save_summary(run_dir, summary)
            except Exception as exc:
                return _fail_or_raise(
                    smoke_mode=smoke_mode,
                    run_dir=run_dir,
                    summary=summary,
                    step_name="step8",
                    exc=exc,
                    save_intermediate_json=save_intermediate_json,
                )
            finally:
                llm.set_trace_context(None)

            try:
                llm.set_trace_context("step9")
                ideas = step9_idea_generation.run(
                    llm,
                    graph,
                    rss,
                    trend,
                    selected_inspirations,
                    cfg.pipeline,
                    topic,
                    novelty_feedback=novelty_feedback,
                    sciatlas=sciatlas,
                )
            except Exception as exc:
                return _fail_or_raise(
                    smoke_mode=smoke_mode,
                    run_dir=run_dir,
                    summary=summary,
                    step_name="step9",
                    exc=exc,
                    save_intermediate_json=save_intermediate_json,
                )
            finally:
                llm.set_trace_context(None)

            try:
                llm.set_trace_context("post_step9_novelty")
                ideas = step9_novelty_check.run(llm, ideas)
            except Exception as exc:
                return _fail_or_raise(
                    smoke_mode=smoke_mode,
                    run_dir=run_dir,
                    summary=summary,
                    step_name="post_step9_novelty",
                    exc=exc,
                    save_intermediate_json=save_intermediate_json,
                )
            finally:
                llm.set_trace_context(None)

            summary["steps"]["step9"] = {
                "status": "completed",
                "idea_count": len(ideas),
                "round": round_idx + 1,
            }
            summary["post_step9_novelty"] = {
                "status": "completed",
                "round": round_idx + 1,
                "novelty_levels": [idea.novelty_level for idea in ideas],
            }
            _save_summary(run_dir, summary)

            if save_intermediate_json:
                save_json(_ideas_to_json(ideas), step9_path)
                save_text(_ideas_to_markdown(ideas), step9_markdown_path)

            if round_idx < max_rounds and _should_retry_for_novelty(ideas):
                novelty_feedback = _collect_novelty_feedback(ideas)
                continue
            break

    summary.update(
        {
            "seed_paper_count": len(seeds.seed_papers),
            "graph_paper_count": len(graph.papers),
            "candidate_inspiration_count": len(candidate_inspirations),
            "inspiration_count": len(inspirations),
            "selected_inspiration_count": len(selected_inspirations),
            "idea_count": len(ideas),
            "ideas": _ideas_to_json(ideas),
        }
    )
    _save_summary(run_dir, summary)
    return summary
