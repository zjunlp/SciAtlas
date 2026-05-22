from __future__ import annotations

from typing import Any

from ..core.common import normalize_whitespace, truncate_text
from ..core.schemas import (
    TASK_AUTHOR_PROFILE,
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_RELATED_AUTHORS,
    TASK_TOPIC_TREND_REVIEW,
)


def _render_mapping_lines(mapping: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            value = str(value)
        text = normalize_whitespace(value)
        if not text:
            continue
        lines.append(f"- `{key}`: {text}")
    return lines


def _render_paper_table(papers: list[dict[str, Any]], *, title: str) -> list[str]:
    if not papers:
        return [f"## {title}", "", "_No papers._", ""]

    lines = [
        f"## {title}",
        "",
        "| Rank | Year | Title | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for index, paper in enumerate(papers, start=1):
        year = paper.get("year") or paper.get("publication_year") or ""
        paper_title = normalize_whitespace(paper.get("title")) or "Untitled"
        notes = normalize_whitespace(
            paper.get("why_representative")
            or paper.get("summary_note")
            or paper.get("source")
            or paper.get("reason")
        )
        lines.append(
            f"| {index} | {year} | {paper_title.replace('|', ' ')} | {truncate_text(notes, max_chars=140).replace('|', ' ')} |"
        )
    lines.append("")
    return lines


def _render_author_table(authors: list[dict[str, Any]]) -> list[str]:
    if not authors:
        return ["## Related Authors", "", "_No authors returned._", ""]

    lines = [
        "## Related Authors",
        "",
        "| Rank | Author | Score | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for index, author in enumerate(authors, start=1):
        evidence_titles = ", ".join(
            normalize_whitespace(paper.get("title"))
            for paper in author.get("support_papers", [])[:2]
            if normalize_whitespace(paper.get("title"))
        )
        evidence = evidence_titles or normalize_whitespace(author.get("selection_rationale")) or "-"
        score = author.get("score")
        if isinstance(score, float):
            score_text = f"{score:.4f}"
        else:
            score_text = normalize_whitespace(score) or "-"
        lines.append(
            f"| {index} | {normalize_whitespace(author.get('name')) or 'Unknown'} | {score_text} | {truncate_text(evidence, max_chars=120).replace('|', ' ')} |"
        )
    lines.append("")
    return lines


def _render_grounding_result(response: dict[str, Any]) -> str:
    result = response["result"]
    lines = [
        f"# {TASK_GROUNDED_REVIEW}",
        "",
        "## Input",
        "",
        *_render_mapping_lines(response["input_summary"]),
        "",
        "## Effective Params",
        "",
        *_render_mapping_lines(response["params_effective"]),
        "",
        "## Executive Summary",
        "",
    ]
    summary = normalize_whitespace(result.get("summary"))
    if summary:
        lines.append(summary)
    else:
        lines.append("_No summary generated._")
    lines.append("")

    lines.extend(_render_paper_table(result.get("retrieved_papers", []), title="Retrieved Papers"))

    top_matches = result.get("top_matches", [])
    lines.append("## Top Grounding Matches")
    lines.append("")
    if not top_matches:
        lines.append("_No grounding matches._")
        lines.append("")
    else:
        for index, match in enumerate(top_matches, start=1):
            lines.append(
                f"{index}. **{normalize_whitespace(match.get('paper_title')) or 'Unknown paper'}**"
            )
            lines.append(
                f"   Query: {normalize_whitespace(match.get('query_sentence')) or normalize_whitespace(match.get('focus_aspect')) or '-'}"
            )
            grounded = truncate_text(match.get('grounded_passage') or match.get('why_this_matches') or '', max_chars=320)
            original = truncate_text(match.get('text') or '', max_chars=400)
            if grounded:
                lines.append(f"   Passage: {grounded}")
            if original:
                lines.append(f"   Original: {original}")
    lines.append("")

    per_paper = result.get("per_paper", {})
    if per_paper:
        lines.append("## Per-Paper Grounding Details")
        lines.append("")
        for paper_title, pdata in per_paper.items():
            lines.append(f"### {normalize_whitespace(paper_title) or 'Unknown paper'}")
            lines.append("")

            matches = pdata.get("matches", [])
            if matches:
                lines.append("**Matched Paragraphs:**")
                for m in matches:
                    query_s = normalize_whitespace(m.get("query_sentence")) or "-"
                    grounded = truncate_text(m.get("grounded_passage") or "", max_chars=300)
                    original = truncate_text(m.get("text") or "", max_chars=400)
                    aspect = normalize_whitespace(m.get("focus_aspect")) or ""
                    lines.append(f"- Query: _{query_s}_")
                    if aspect:
                        lines.append(f"  Aspect: {aspect}")
                    if grounded:
                        lines.append(f"  Passage: {grounded}")
                    if original:
                        lines.append(f"  Original: {original}")
                lines.append("")

            similar_points = pdata.get("similar_points", [])
            if similar_points:
                lines.append(f"**Similar Points ({len(similar_points)}):**")
                for item in similar_points:
                    lines.append(f"- {item}")
            else:
                lines.append("**Similar Points:** _None._")
            lines.append("")

            different_points = pdata.get("different_points", [])
            if different_points:
                lines.append(f"**Different Points ({len(different_points)}):**")
                for item in different_points:
                    lines.append(f"- {item}")
            else:
                lines.append("**Different Points:** _None._")
            lines.append("")
    else:
        lines.append("## Per-Paper Grounding Details")
        lines.append("")
        lines.append("_No per-paper grounding data available._")
        lines.append("")

    # Idea Evaluation section
    evaluation = result.get("idea_evaluation")
    if evaluation:
        lines.append("## Idea Evaluation")
        lines.append("")
        dims = [
            ("clarity", "Clarity"),
            ("novelty", "Novelty"),
            ("validity", "Validity"),
            ("feasibility", "Feasibility"),
            ("significance", "Significance"),
        ]
        for key, label in dims:
            dim = evaluation.get(key, {})
            if isinstance(dim, dict):
                score = dim.get("score")
                reason = normalize_whitespace(dim.get("reason") or "")
                score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score or "-")
                lines.append(f"- **{label}**: {score_str} — {reason}")
        overall = evaluation.get("overall", {})
        if isinstance(overall, dict):
            lines.append("")
            strengths = overall.get("strengths", [])
            if isinstance(strengths, list) and strengths:
                lines.append("**Strengths:**")
                for s in strengths:
                    lines.append(f"- {normalize_whitespace(s)}")
            weaknesses = overall.get("weaknesses", [])
            if isinstance(weaknesses, list) and weaknesses:
                lines.append("**Weaknesses:**")
                for w in weaknesses:
                    lines.append(f"- {normalize_whitespace(w)}")
            suggestions = overall.get("suggestions", [])
            if isinstance(suggestions, list) and suggestions:
                lines.append("**Suggestions:**")
                for s in suggestions:
                    lines.append(f"- {normalize_whitespace(s)}")
            rec = normalize_whitespace(overall.get("recommendation") or "")
            if rec:
                lines.append(f"**Recommendation:** {rec}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_trend_review(response: dict[str, Any]) -> str:
    result = response["result"]
    trend_summary = result.get("trend_summary", {})
    lines = [
        f"# {TASK_TOPIC_TREND_REVIEW}",
        "",
        "## Input",
        "",
        *_render_mapping_lines(response["input_summary"]),
        "",
        "## Effective Params",
        "",
        *_render_mapping_lines(response["params_effective"]),
        "",
        "## Executive Summary",
        "",
        normalize_whitespace(trend_summary.get("one_sentence_summary") or trend_summary.get("trend_summary"))
        or "_No summary generated._",
        "",
    ]

    stage_summary = trend_summary.get("stage_summary", [])
    lines.append("## Stage Summary")
    lines.append("")
    if stage_summary:
        for index, stage in enumerate(stage_summary, start=1):
            lines.append(
                f"{index}. **{normalize_whitespace(stage.get('period')) or 'Unknown period'}**: {normalize_whitespace(stage.get('theme')) or normalize_whitespace(stage.get('description')) or '-'}"
            )
            description = normalize_whitespace(stage.get("description"))
            if description:
                lines.append(f"   {description}")
    else:
        lines.append("_No stage summary generated._")
    lines.append("")

    for section_title, key in (
        ("Methodological Shifts", "methodological_shifts"),
        ("Emerging Topics", "emerging_topics"),
        ("Open Gaps", "open_gaps"),
    ):
        lines.append(f"## {section_title}")
        lines.append("")
        values = trend_summary.get(key, [])
        if values:
            for item in values:
                lines.append(f"- {normalize_whitespace(item)}")
        else:
            lines.append("_None._")
        lines.append("")

    lines.extend(_render_paper_table(result.get("representative_papers", []), title="Representative Papers"))
    lines.extend(_render_paper_table(result.get("papers_by_year", []), title="Chronological Paper List"))
    return "\n".join(lines).rstrip() + "\n"


def _render_related_authors(response: dict[str, Any]) -> str:
    result = response["result"]
    lines = [
        f"# {TASK_RELATED_AUTHORS}",
        "",
        "## Input",
        "",
        *_render_mapping_lines(response["input_summary"]),
        "",
        "## Effective Params",
        "",
        *_render_mapping_lines(response["params_effective"]),
        "",
        "## Executive Summary",
        "",
        normalize_whitespace(result.get("summary")) or "_No summary generated._",
        "",
    ]
    lines.extend(_render_author_table(result.get("authors", [])))
    lines.extend(_render_paper_table(result.get("supporting_papers", []), title="Supporting Papers"))
    return "\n".join(lines).rstrip() + "\n"


def _render_author_profile(response: dict[str, Any]) -> str:
    result = response["result"]
    lines = [
        f"# {TASK_AUTHOR_PROFILE}",
        "",
        "## Input",
        "",
        *_render_mapping_lines(response["input_summary"]),
        "",
        "## Effective Params",
        "",
        *_render_mapping_lines(response["params_effective"]),
        "",
        "## Executive Summary",
        "",
        normalize_whitespace(result.get("overall_academic_profile")) or "_No summary generated._",
        "",
    ]
    author_stats = result.get("author_stats", {})
    if isinstance(author_stats, dict) and any(author_stats.values()):
        lines.append("## Author Stats")
        lines.append("")
        for key in ("name", "h_index", "total_works", "total_citations"):
            value = author_stats.get(key)
            text = normalize_whitespace(value)
            if text:
                lines.append(f"- `{key}`: {text}")
        lines.append("")

    lines.append("## Research Trajectory")
    lines.append("")
    trajectory = result.get("main_research_directions") or result.get("relevant_research_trajectory") or []
    if trajectory:
        for index, item in enumerate(trajectory, start=1):
            lines.append(
                f"{index}. **{normalize_whitespace(item.get('research_theme') or item.get('theme')) or 'Unknown theme'}** ({normalize_whitespace(item.get('active_years')) or 'years not specified'})"
            )
            description = normalize_whitespace(item.get("description"))
            if description:
                lines.append(f"   {description}")
    else:
        lines.append("_No trajectory data generated._")
    lines.append("")

    lines.append("## Technical Arsenal")
    lines.append("")
    technical_arsenal = result.get("technical_arsenal", [])
    if technical_arsenal:
        for item in technical_arsenal:
            lines.append(f"- {normalize_whitespace(item)}")
    else:
        lines.append("_No technical arsenal generated._")
    lines.append("")

    lines.extend(_render_paper_table(result.get("representative_papers", []), title="Representative Papers"))
    return "\n".join(lines).rstrip() + "\n"


def _render_idea_generation(response: dict[str, Any]) -> str:
    result = response["result"]
    lines = [
        f"# {TASK_IDEA_GENERATION}",
        "",
        "## Input",
        "",
        *_render_mapping_lines(response["input_summary"]),
        "",
        "## Effective Params",
        "",
        *_render_mapping_lines(response["params_effective"]),
        "",
    ]
    ideas = result.get("ideas", [])
    if not ideas:
        lines.append("_No ideas generated._")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    lines.append(f"## Generated Ideas ({len(ideas)})")
    lines.append("")
    for index, idea in enumerate(ideas, start=1):
        lines.append(f"### {index}. {normalize_whitespace(idea.get('title') or 'Untitled')}")
        lines.append("")
        desc = normalize_whitespace(idea.get("description") or "")
        if desc:
            lines.append(f"**Description:** {desc}")
        novelty = normalize_whitespace(idea.get("novelty") or "")
        if novelty:
            lines.append(f"**Novelty:** {novelty}")
        significance = normalize_whitespace(idea.get("significance") or "")
        if significance:
            lines.append(f"**Significance:** {significance}")
        related = idea.get("related_papers", [])
        if related:
            lines.append("**Key References:**")
            for p in related:
                title = normalize_whitespace(p.get("title") or "Untitled")
                year = p.get("year") or ""
                lines.append(f"- [{year}] {title}" if year else f"- {title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_response_markdown(response: dict[str, Any]) -> str:
    task_type = response["task_type"]
    if task_type == TASK_GROUNDED_REVIEW:
        return _render_grounding_result(response)
    if task_type == TASK_TOPIC_TREND_REVIEW:
        return _render_trend_review(response)
    if task_type == TASK_RELATED_AUTHORS:
        return _render_related_authors(response)
    if task_type == TASK_AUTHOR_PROFILE:
        return _render_author_profile(response)
    if task_type == TASK_IDEA_GENERATION:
        return _render_idea_generation(response)
    raise ValueError(f"Unsupported task_type: {task_type}")
