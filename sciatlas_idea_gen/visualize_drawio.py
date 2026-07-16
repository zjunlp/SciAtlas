"""Export a pipeline run directory as a draw.io flowchart."""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import wrap
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from .utils import read_json, save_text


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _wrap(text: str, width: int = 58) -> str:
    lines: list[str] = []
    for chunk in str(text).splitlines() or [""]:
        wrapped = wrap(chunk, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def _node_height(
    text: str,
    *,
    base: int = 90,
    chars_per_line: int = 58,
    line_px: int = 18,
    padding: int = 24,
) -> int:
    line_count = 0
    for chunk in text.splitlines() or [""]:
        line_count += max(1, (len(chunk) // chars_per_line) + 1)
    return max(base, line_px * line_count + padding)


def _node_width(
    text: str,
    *,
    min_width: int = 180,
    max_width: int = 760,
    char_px: int = 7,
    padding: int = 36,
) -> int:
    longest = max((len(line) for line in text.splitlines()), default=0)
    estimated = longest * char_px + padding
    return max(min_width, min(max_width, estimated))


def _mx_cell(
    root: Element,
    *,
    cell_id: str,
    value: str,
    x: int,
    y: int,
    w: int,
    h: int,
    style: str,
    parent: str = "1",
    edge: bool = False,
    source: str | None = None,
    target: str | None = None,
) -> None:
    attrs = {"id": cell_id, "value": value, "style": style, "vertex": "1", "parent": parent}
    if edge:
        attrs = {
            "id": cell_id,
            "value": value,
            "style": style,
            "edge": "1",
            "parent": parent,
        }
        if source:
            attrs["source"] = source
        if target:
            attrs["target"] = target
    cell = SubElement(root, "mxCell", attrs)
    geometry_attrs = {"as": "geometry"}
    if edge:
        geometry_attrs["relative"] = "1"
        SubElement(cell, "mxGeometry", geometry_attrs)
    else:
        geometry_attrs.update({"x": str(x), "y": str(y), "width": str(w), "height": str(h)})
        SubElement(cell, "mxGeometry", geometry_attrs)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found under {run_dir}")
    return read_json(path)


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return read_json(path)


def _load_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _paper_direction(paper: dict[str, Any]) -> str:
    raw = paper.get("raw", {}) if isinstance(paper.get("raw"), dict) else {}
    return str(raw.get("expansion_direction", "unknown") or "unknown")


def _direction_style(direction: str, *, foundational: bool) -> tuple[str, str]:
    palette = {
        "anchor": ("#d9ead3", "#6aa84f"),
        "backward": ("#fff2cc", "#d6b656"),
        "forward": ("#d9eaf7", "#5b9bd5"),
        "semantic": ("#fce5cd", "#c27c0e"),
        "unknown": ("#f3f3f3", "#666666"),
    }
    fill, stroke = palette.get(direction, palette["unknown"])
    if foundational:
        stroke = "#38761d"
    return fill, stroke


def _load_trace(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "retrieval_trace.json"
    if not path.exists():
        return []
    data = read_json(path)
    events = data.get("events", []) if isinstance(data, dict) else []
    return [event for event in events if isinstance(event, dict)]


def _format_payload(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if value in (None, [], {}, ""):
            continue
        if key.lower() in {"api_key", "token", "authorization"}:
            value = "***REDACTED***"
        lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "No explicit payload fields recorded."


def _paper_lines(papers: list[dict[str, Any]], abstract_chars: int) -> str:
    if not papers:
        return "No papers recorded."
    lines: list[str] = []
    for i, paper in enumerate(papers, start=1):
        title = paper.get("title") or "Untitled"
        abstract = _clip(str(paper.get("abstract") or ""), abstract_chars)
        year = paper.get("year")
        score = paper.get("score")
        header = f"{i}. {title}"
        if year:
            header += f" ({year})"
        if score is not None:
            header += f" [score={score}]"
        lines.append(header)
        lines.append(f"Abstract: {abstract or 'N/A'}")
        lines.append("")
    return "\n".join(lines).strip()


def _fallback_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    step1 = run_dir / "step1_seed_papers.json"
    if step1.exists():
        data = read_json(step1)
        events.append(
            {
                "index": len(events) + 1,
                "step": "step1",
                "command_name": "step1_saved_seeds",
                "cache_hit": None,
                "payload": {"query_text": data.get("refined_query")},
                "paper_count": len(data.get("seed_papers", []) or []),
                "papers": data.get("seed_papers", []) or [],
            }
        )
    step2 = run_dir / "step2_research_graph.json"
    if step2.exists():
        data = read_json(step2)
        papers = list((data.get("papers", {}) or {}).values())
        events.append(
            {
                "index": len(events) + 1,
                "step": "step2",
                "command_name": "step2_saved_graph",
                "cache_hit": None,
                "payload": {"graph_paper_count": len(papers)},
                "paper_count": len(papers),
                "papers": papers,
            }
        )
    step6 = run_dir / "step6_inspiration_candidates.json"
    if step6.exists():
        data = read_json(step6)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            key = f"step6_{item.get('radius', 'R?')}_{item.get('domain', 'unknown')}"
            grouped.setdefault(key, [])
            source_paper = item.get("source_paper")
            if isinstance(source_paper, dict):
                grouped[key].append(source_paper)
        for key, papers in grouped.items():
            events.append(
                {
                    "index": len(events) + 1,
                    "step": "step6",
                    "command_name": key,
                    "cache_hit": None,
                    "payload": {"group": key},
                    "paper_count": len(papers),
                    "papers": papers,
                }
            )
    return events


def _render_text_panel(
    root: Element,
    *,
    title: str,
    body: str,
    cell_id: str,
    source_id: str,
    start_y: int,
    fill: str,
    stroke: str,
    x: int = 80,
    min_width: int = 560,
    max_width: int = 980,
    wrap_width: int = 62,
) -> int:
    text = _wrap("\n".join([title, body]).strip(), width=wrap_width)
    height = _node_height(text, base=160, chars_per_line=wrap_width)
    width = _node_width(text, min_width=min_width, max_width=max_width)
    _mx_cell(
        root,
        cell_id=cell_id,
        value=text,
        x=x,
        y=start_y,
        w=width,
        h=height,
        style=f"rounded=0;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontSize=11;align=left;verticalAlign=top;",
    )
    _mx_cell(
        root,
        cell_id=f"edge_{cell_id}",
        value="",
        x=0,
        y=0,
        w=0,
        h=0,
        style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;endArrow=block;",
        edge=True,
        source=source_id,
        target=cell_id,
    )
    return start_y + height + 30


def _render_step2_graph(
    root: Element,
    *,
    run_dir: Path,
    step2_cell_id: str,
    start_y: int,
    x_offset: int = 60,
    render_edges: bool = True,
) -> int:
    graph = _load_json_if_exists(run_dir / "step2_research_graph.json")
    if not isinstance(graph, dict):
        return start_y
    papers = graph.get("papers", {}) if isinstance(graph.get("papers"), dict) else {}
    if not papers:
        return start_y
    phases = graph.get("phases", []) if isinstance(graph.get("phases"), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    foundational = set(graph.get("foundational_ids", []) or [])
    seed_paper_id = str(graph.get("seed_paper_id") or "")
    # All Step-1 seeds, not just the primary one. Prefer the graph's own list;
    # fall back to step1_seed_papers.json for runs produced before that field
    # existed; finally fall back to the single primary seed id.
    seed_ids = [str(pid) for pid in (graph.get("seed_paper_ids") or []) if str(pid) in papers]
    if not seed_ids:
        step1 = _load_json_if_exists(run_dir / "step1_seed_papers.json")
        seed_ids = [
            str(p.get("paper_id"))
            for p in (step1.get("seed_papers", []) or [])
            if isinstance(p, dict) and str(p.get("paper_id")) in papers
        ]
    if not seed_ids and seed_paper_id:
        seed_ids = [seed_paper_id]
    if seed_paper_id and seed_paper_id not in seed_ids and seed_paper_id in papers:
        seed_ids.insert(0, seed_paper_id)
    seed_id_set = set(seed_ids)
    evidence_scores = graph.get("evidence_scores", {}) if isinstance(graph.get("evidence_scores"), dict) else {}
    ordered_non_seed_ids = [
        paper_id
        for paper_id, paper in sorted(
            papers.items(),
            key=lambda item: ((item[1] or {}).get("year") or 0, str((item[1] or {}).get("title") or "")),
        )
        if paper_id not in seed_id_set and isinstance(paper, dict)
    ]
    phase_groups: list[list[str]] = []
    assigned_to_phase: set[str] = set()
    for phase in phases:
        if not isinstance(phase, list):
            continue
        clean_ids = [
            str(paper_id)
            for paper_id in phase
            if str(paper_id) in papers
            and str(paper_id) not in seed_id_set
            and str(paper_id) not in assigned_to_phase
        ]
        if clean_ids:
            phase_groups.append(clean_ids)
            assigned_to_phase.update(clean_ids)
    missing_ids = [paper_id for paper_id in ordered_non_seed_ids if paper_id not in assigned_to_phase]
    if missing_ids:
        phase_groups.append(missing_ids)
    phase_by_paper: dict[str, int] = {}
    for phase_index, phase_paper_ids in enumerate(phase_groups, start=1):
        for paper_id in phase_paper_ids:
            phase_by_paper[paper_id] = phase_index

    direction_counts: dict[str, int] = {"anchor": 0, "backward": 0, "forward": 0, "unknown": 0}
    pred_counts: dict[str, int] = {}
    for paper_id, paper in papers.items():
        if not isinstance(paper, dict):
            continue
        direction = _paper_direction(paper)
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if str(edge.get("relation") or "") not in {"explicit_pred", "parallel_pred", "direct_to_seed"}:
            continue
        target = str(edge.get("target") or "")
        if target:
            pred_counts[target] = pred_counts.get(target, 0) + 1
    title_text = _wrap(
        "\n".join(
            [
                "Step 2 Annotated Research Graph",
                f"paper_count: {len(papers)}",
                f"edge_count: {len(edges)}",
                f"phase_count: {len(phase_groups) or 1}",
                (
                    "directions: "
                    f"anchor={direction_counts.get('anchor', 0)}, "
                    f"backward={direction_counts.get('backward', 0)}, "
                    f"forward={direction_counts.get('forward', 0)}"
                ),
            ]
        ),
        width=42,
    )
    title_height = _node_height(title_text, base=120, chars_per_line=42)
    _mx_cell(
        root,
        cell_id="step2_graph_title",
        value=title_text,
        x=x_offset,
        y=start_y,
        w=_node_width(title_text, min_width=260, max_width=360),
        h=title_height,
        style="rounded=0;whiteSpace=wrap;html=0;fillColor=#cfe2f3;strokeColor=#3d85c6;fontSize=13;align=left;verticalAlign=top;",
    )
    _mx_cell(
        root,
        cell_id="edge_step2_graph_title",
        value="",
        x=0,
        y=0,
        w=0,
        h=0,
        style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;endArrow=block;",
        edge=True,
        source=step2_cell_id,
        target="step2_graph_title",
    )
    phase_top = start_y + title_height + 40
    phase_x = x_offset
    phase_gap = 280
    graph_node_ids: dict[str, str] = {}
    phase_cell_ids: dict[int, str] = {}
    max_bottom = phase_top
    rendered_seed_ids = [pid for pid in seed_ids if isinstance(papers.get(pid), dict)]
    seed_gap = 280
    seed_row_width = max(0, (len(rendered_seed_ids) - 1) * seed_gap)
    # Centre the seed row over the phase columns below it.
    seed_row_x = phase_x + max(0, (len(phase_groups) - 1) * phase_gap // 2) - seed_row_width // 2
    seed_x = max(phase_x, seed_row_x)
    max_seed_h = 0
    for seed_index, sid in enumerate(rendered_seed_ids):
        seed_paper = papers[sid]
        seed_text = _wrap(
            "\n".join(
                [
                    f"Seed Paper {seed_index + 1}/{len(rendered_seed_ids)}",
                    str(seed_paper.get("title") or "Untitled"),
                    f"year: {seed_paper.get('year') or 'n.d.'}",
                    f"paper_id: {sid}",
                    f"direction: {_paper_direction(seed_paper)}",
                    f"evidence_score: {evidence_scores.get(sid, 1.0)}",
                ]
            ),
            width=34,
        )
        seed_h = _node_height(seed_text, base=110, chars_per_line=34, line_px=16, padding=20)
        max_seed_h = max(max_seed_h, seed_h)
        seed_cell_id = "graph_seed" if seed_index == 0 else f"graph_seed_{seed_index}"
        _mx_cell(
            root,
            cell_id=seed_cell_id,
            value=seed_text,
            x=seed_x,
            y=phase_top,
            w=260,
            h=seed_h,
            style="rounded=0;whiteSpace=wrap;html=0;fillColor=#cfe2f3;strokeColor=#3d85c6;fontSize=11;align=left;verticalAlign=top;",
        )
        graph_node_ids[sid] = seed_cell_id
        _mx_cell(
            root,
            cell_id=f"edge_step2_graph_seed_{seed_index}",
            value="seed",
            x=0,
            y=0,
            w=0,
            h=0,
            style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;endArrow=block;strokeColor=#3d85c6;fontSize=10;",
            edge=True,
            source="step2_graph_title",
            target=seed_cell_id,
        )
        seed_x += seed_gap
        max_bottom = max(max_bottom, phase_top + seed_h)
    phase_top += (max_seed_h + 50) if rendered_seed_ids else 0
    for phase_index, phase_paper_ids in enumerate(phase_groups, start=1):
        phase_label = _wrap(f"Phase {phase_index}\n{len(phase_paper_ids)} papers", width=20)
        phase_cell_ids[phase_index] = f"phase_{phase_index}"
        _mx_cell(
            root,
            cell_id=phase_cell_ids[phase_index],
            value=phase_label,
            x=phase_x,
            y=phase_top,
            w=200,
            h=70,
            style="rounded=0;whiteSpace=wrap;html=0;fillColor=#f3f3f3;strokeColor=#666666;fontSize=12;",
        )
        node_y = phase_top + 90
        for local_index, paper_id in enumerate(phase_paper_ids, start=1):
            paper = papers.get(paper_id)
            if not isinstance(paper, dict):
                continue
            node_id = f"graph_paper_{phase_index}_{local_index}"
            graph_node_ids[paper_id] = node_id
            idea_fields = paper.get("idea_fields", {}) if isinstance(paper.get("idea_fields"), dict) else {}
            idea_preview = "; ".join(
                f"{key}: {_clip(str(value), 80)}"
                for key, value in list(idea_fields.items())[:2]
                if value
            )
            direction = _paper_direction(paper)
            text = _wrap(
                "\n".join(
                    [
                        str(paper.get("title") or "Untitled"),
                        f"year: {paper.get('year') or 'n.d.'}",
                        f"paper_id: {paper_id}",
                        f"direction: {direction}",
                        f"foundational: {paper_id in foundational}",
                        f"pred_count: {pred_counts.get(paper_id, 0)}",
                        f"evidence_score: {evidence_scores.get(paper_id, 'n/a')}",
                        f"keywords: {', '.join((paper.get('keywords') or [])[:4])}",
                        f"idea: {idea_preview or 'n/a'}",
                        f"abstract: {_clip(str(paper.get('abstract') or ''), 180)}",
                    ]
                ),
                width=34,
            )
            height = _node_height(text, base=120, chars_per_line=34, line_px=16, padding=20)
            fill, stroke = _direction_style(direction, foundational=paper_id in foundational)
            _mx_cell(
                root,
                cell_id=node_id,
                value=text,
                x=phase_x,
                y=node_y,
                w=250,
                h=height,
                style=f"rounded=0;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontSize=11;align=left;verticalAlign=top;",
            )
            node_y += height + 24
            max_bottom = max(max_bottom, node_y)
        phase_x += phase_gap
    if not render_edges:
        return max_bottom + 30
    for edge_index, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            continue
        source = graph_node_ids.get(str(edge.get("source") or ""))
        target = graph_node_ids.get(str(edge.get("target") or ""))
        if not source or not target:
            continue
        label = str(edge.get("relation") or edge.get("edge_type") or "")
        _mx_cell(
            root,
            cell_id=f"graph_edge_{edge_index}",
            value=label,
            x=0,
            y=0,
            w=0,
            h=0,
            style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;endArrow=block;strokeColor=#555555;fontSize=10;",
            edge=True,
            source=source,
            target=target,
        )
    phase_edge_counts: dict[tuple[int, int], int] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_phase = phase_by_paper.get(str(edge.get("source") or ""))
        target_phase = phase_by_paper.get(str(edge.get("target") or ""))
        if source_phase is None or target_phase is None or source_phase == target_phase:
            continue
        phase_edge_counts[(source_phase, target_phase)] = phase_edge_counts.get((source_phase, target_phase), 0) + 1
    for phase_edge_index, ((source_phase, target_phase), edge_count) in enumerate(sorted(phase_edge_counts.items()), start=1):
        _mx_cell(
            root,
            cell_id=f"phase_edge_{phase_edge_index}",
            value=f"{edge_count} cross-phase edge(s)",
            x=0,
            y=0,
            w=0,
            h=0,
            style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;dashed=1;endArrow=block;strokeColor=#999999;fontSize=10;",
            edge=True,
            source=phase_cell_ids[source_phase],
            target=phase_cell_ids[target_phase],
        )
    return max_bottom + 30


def _render_inspiration_panel(
    root: Element,
    *,
    title: str,
    items: list[dict[str, Any]],
    source_id: str,
    cell_id: str,
    start_y: int,
    abstract_chars: int,
    fill: str,
    stroke: str,
    x: int = 80,
) -> int:
    if not items:
        return start_y
    lines = [f"count: {len(items)}", ""]
    for item in items:
        source = item.get("source_paper") if isinstance(item.get("source_paper"), dict) else {}
        lines.extend(
            [
                f"candidate_id: {item.get('candidate_id', '')}",
                f"domain: {item.get('domain', '')}",
                f"radius: {item.get('radius', '')}",
                f"title: {item.get('paper_title', source.get('title', ''))}",
            ]
        )
        if "is_combination_plausible" in item:
            lines.append(f"plausible: {item.get('is_combination_plausible')}")
        if item.get("combination_points"):
            lines.append(f"combination_points: {item.get('combination_points')}")
        if item.get("combination_plan"):
            lines.append(f"combination_plan: {_clip(str(item.get('combination_plan')), 280)}")
        if item.get("justification"):
            lines.append(f"justification: {_clip(str(item.get('justification')), 280)}")
        abstract = str(item.get("paper_abstract", source.get("abstract", "")))
        lines.append(f"abstract: {_clip(abstract, abstract_chars)}")
        lines.append("")
    return _render_text_panel(
        root,
        title=title,
        body="\n".join(lines),
        cell_id=cell_id,
        source_id=source_id,
        start_y=start_y,
        x=x,
        fill=fill,
        stroke=stroke,
        min_width=620,
        max_width=1100,
        wrap_width=62,
    )


def _render_ideas_panel(
    root: Element,
    *,
    ideas: list[dict[str, Any]],
    source_id: str,
    start_y: int,
    x: int = 80,
) -> int:
    if not ideas:
        return start_y
    lines = [f"count: {len(ideas)}", ""]
    for index, idea in enumerate(ideas, start=1):
        lines.append(f"{index}. {idea.get('title', 'Untitled')}")
        if idea.get("novelty_level"):
            lines.append(f"novelty_level: {idea.get('novelty_level')}")
        if idea.get("motivation"):
            lines.append(f"motivation: {_clip(str(idea.get('motivation')), 420)}")
        if idea.get("methods"):
            lines.append(f"methods: {_clip(str(idea.get('methods')), 520)}")
        if idea.get("novelty_justification"):
            lines.append(f"novelty_justification: {_clip(str(idea.get('novelty_justification')), 420)}")
        if idea.get("improvement_suggestions"):
            lines.append(f"improvement_suggestions: {idea.get('improvement_suggestions')}")
        lines.append("")
    return _render_text_panel(
        root,
        title="Step 9 Idea Generation + Novelty Check",
        body="\n".join(lines),
        cell_id="step9_ideas_panel",
        source_id=source_id,
        start_y=start_y,
        x=x,
        fill="#e1d5e7",
        stroke="#9673a6",
        min_width=680,
        max_width=1150,
        wrap_width=64,
    )


def export_drawio(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    abstract_chars: int = 280,
) -> Path:
    run_dir = Path(run_dir)
    summary = _load_summary(run_dir)
    events = _load_trace(run_dir) or _fallback_events(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "pipeline.drawio"

    mxfile = Element("mxfile", host="app.diagrams.net", modified="2026-06-11T00:00:00.000Z", agent="Trae")
    diagram = SubElement(mxfile, "diagram", id="pipeline", name="Pipeline")
    model = SubElement(
        diagram,
        "mxGraphModel",
        dx="1800",
        dy="2600",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="2200",
        pageHeight="3200",
        math="0",
        shadow="0",
    )
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")

    header_text = _wrap(
        "\n".join(
            [
                "SciAtlas Idea Generation Pipeline",
                f"Run dir: {summary.get('run_dir', run_dir)}",
                f"Topic: {summary.get('topic', 'N/A')}",
                f"Status: {'ok' if summary.get('ok') else 'failed'}",
                f"Failed step: {summary.get('failed_step', 'N/A')}",
            ]
        )
    )
    params_text = _wrap(
        "\n".join(
            [
                "Input Parameters",
                _format_payload(summary.get("input", {})),
                "",
                "Effective Pipeline Config",
                _format_payload((summary.get("effective_config") or {}).get("pipeline", {})),
                "",
                "Effective SciAtlas Config",
                _format_payload((summary.get("effective_config") or {}).get("sciatlas", {})),
            ]
        )
    )

    _mx_cell(
        root,
        cell_id="header",
        value=header_text,
        x=40,
        y=40,
        w=_node_width(header_text, min_width=340, max_width=540),
        h=_node_height(header_text, base=140),
        style="rounded=0;whiteSpace=wrap;html=0;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=14;",
    )
    _mx_cell(
        root,
        cell_id="params",
        value=params_text,
        x=920,
        y=40,
        w=_node_width(params_text, min_width=360, max_width=660),
        h=_node_height(params_text, base=240),
        style="rounded=0;whiteSpace=wrap;html=0;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;",
    )

    step_order = [f"step{i}" for i in range(1, 10)]
    step_titles = {
        "step1": "Step 1 Anchor Retrieval",
        "step2": "Step 2 Research Graph Construction",
        "step3": "Step 3 Graph Serialization + Trend",
        "step4": "Step 4 RSS Extraction",
        "step5": "Step 5 Inspiration Radius Gate",
        "step6": "Step 6 Multi-radius Retrieval",
        "step7": "Step 7 Analogy Extraction",
        "step8": "Step 8 Inspiration Selection",
        "step9": "Step 9 Idea Generation",
    }
    # ---- Layout: each step occupies one row. The step node sits in a fixed
    # left column; its result panel is rendered immediately to the right at the
    # same y, so every step is visually grouped with its own output. Rows are
    # spaced apart vertically (block_gap) and the next row starts below the
    # tallest element of the current one, so nothing piles up at the bottom.
    step_ids: dict[str, str] = {}
    step_y: dict[str, int] = {}
    step_x = 60
    panel_x = 360
    block_gap = 70
    y_cursor = 340

    # Preload all per-step artifacts up front so the row loop can dispatch.
    serialized_graph = _load_text_if_exists(run_dir / "step3_serialized_graph.txt")
    trend = _load_text_if_exists(run_dir / "step3_trend.txt")
    rss = _load_json_if_exists(run_dir / "step4_rss.json")
    radius_plan = _load_json_if_exists(run_dir / "step5_radius_plan.json")
    step6_items = _load_json_if_exists(run_dir / "step6_inspiration_candidates.json")
    step7_items = _load_json_if_exists(run_dir / "step7_inspirations.json")
    step8_items = _load_json_if_exists(run_dir / "step8_selected_inspirations.json")
    ideas = _load_json_if_exists(run_dir / "step9_ideas.json")
    if not isinstance(ideas, list):
        ideas = summary.get("ideas", []) if isinstance(summary.get("ideas"), list) else []

    def _render_panel_for(step_name: str, start_y: int) -> int:
        """Render the result panel(s) for one step at panel_x; return its bottom y."""
        cell = step_ids[step_name]
        if step_name == "step2":
            return _render_step2_graph(
                root, run_dir=run_dir, step2_cell_id=cell, start_y=start_y, x_offset=panel_x
            )
        if step_name == "step3" and (serialized_graph or trend):
            body = "\n\n".join(
                part for part in [
                    f"Serialized Graph:\n{_clip(serialized_graph, 3200)}" if serialized_graph else "",
                    f"Trend Summary:\n{_clip(trend, 2400)}" if trend else "",
                ] if part
            )
            return _render_text_panel(
                root, title="Step 3 Serialization + Trend", body=body, cell_id="step3_panel",
                source_id=cell, start_y=start_y, x=panel_x, fill="#d9eaf7", stroke="#5b9bd5",
            )
        if step_name == "step4" and isinstance(rss, dict):
            rss_lines: list[str] = []
            for key in (
                "target_domain", "core_gap", "objects", "relations", "bottlenecks",
                "unresolved_tensions", "desired_solution_properties",
                "possible_mechanism_types", "inspiration_search_hints",
            ):
                value = rss.get(key)
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, list):
                    rss_lines.append(f"{key}:")
                    rss_lines.extend(f"- {item}" for item in value)
                else:
                    rss_lines.append(f"{key}: {value}")
                rss_lines.append("")
            return _render_text_panel(
                root, title="Step 4 RSS", body="\n".join(rss_lines).strip(),
                cell_id="step4_rss_panel", source_id=cell, start_y=start_y, x=panel_x,
                fill="#e2f0d9", stroke="#70ad47",
            )
        if step_name == "step5" and isinstance(radius_plan, dict):
            return _render_text_panel(
                root, title="Step 5 Inspiration Radius Gate", body=_format_payload(radius_plan),
                cell_id="step5_gate_panel", source_id=cell, start_y=start_y, x=panel_x,
                fill="#fce4d6", stroke="#c55a11",
            )
        if step_name == "step6" and isinstance(step6_items, list):
            return _render_inspiration_panel(
                root, title="Step 6 Retrieval Candidates", items=step6_items, source_id=cell,
                cell_id="step6_candidates_panel", start_y=start_y, x=panel_x,
                abstract_chars=abstract_chars, fill="#fde9d9", stroke="#c55a11",
            )
        if step_name == "step7" and isinstance(step7_items, list):
            return _render_inspiration_panel(
                root, title="Step 7 Analogy Results", items=step7_items, source_id=cell,
                cell_id="step7_analogy_panel", start_y=start_y, x=panel_x,
                abstract_chars=abstract_chars, fill="#f9cb9c", stroke="#b45f06",
            )
        if step_name == "step8" and isinstance(step8_items, list):
            return _render_inspiration_panel(
                root, title="Step 8 Selected Inspiration", items=step8_items, source_id=cell,
                cell_id="step8_selected_panel", start_y=start_y, x=panel_x,
                abstract_chars=abstract_chars, fill="#ead1dc", stroke="#a64d79",
            )
        if step_name == "step9":
            return _render_ideas_panel(root, ideas=ideas, source_id=cell, start_y=start_y, x=panel_x)
        return start_y

    prev_cell: str | None = None
    for step_name in step_order:
        step_info = (summary.get("steps") or {}).get(step_name, {})
        body = _wrap(
            "\n".join(
                [
                    step_titles[step_name],
                    f"status: {step_info.get('status', 'missing')}",
                    *[f"{k}: {v}" for k, v in step_info.items() if k != "status"],
                ]
            ),
            width=34,
        )
        cell_id = f"step_{step_name}"
        step_ids[step_name] = cell_id
        step_y[step_name] = y_cursor
        node_h = _node_height(body, base=90, chars_per_line=34)
        _mx_cell(
            root,
            cell_id=cell_id,
            value=body,
            x=step_x,
            y=y_cursor,
            w=_node_width(body, min_width=220, max_width=280),
            h=node_h,
            style="rounded=0;whiteSpace=wrap;html=0;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;align=left;verticalAlign=top;",
        )
        if prev_cell:
            _mx_cell(
                root,
                cell_id=f"edge_{prev_cell}_{cell_id}",
                value="",
                x=0,
                y=0,
                w=0,
                h=0,
                style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;endArrow=block;",
                edge=True,
                source=prev_cell,
                target=cell_id,
            )
        prev_cell = cell_id
        panel_bottom = _render_panel_for(step_name, y_cursor)
        y_cursor = max(y_cursor + node_h, panel_bottom) + block_gap

    # Retrieval-trace boxes go in a far-right column, each aligned to its step's
    # row (but never overlapping the previous box).
    event_x = 2400
    event_gap_y = 40
    event_cursor = step_y.get("step1", 340)
    for event_index, event in enumerate(events):
        step_name = str(event.get("step") or "step1")
        step_cell = step_ids.get(step_name, "step_step1")
        payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
        papers = event.get("papers", []) if isinstance(event.get("papers"), list) else []
        event_text = _wrap(
            "\n".join(
                [
                    f"Retrieval {event.get('index', event_index + 1)}",
                    f"step: {step_name}",
                    f"command: {event.get('command_name', 'unknown')}",
                    f"cache_hit: {event.get('cache_hit', 'n/a')}",
                    f"paper_count: {event.get('paper_count', len(papers))}",
                    "",
                    "Parameters",
                    _format_payload(payload),
                    "",
                    "Returned Papers",
                    _paper_lines(papers, abstract_chars),
                ]
            )
        )
        cell_id = f"retrieval_{event_index + 1}"
        height = _node_height(event_text, base=200)
        width = _node_width(event_text, min_width=420, max_width=620)
        event_y = max(step_y.get(step_name, event_cursor), event_cursor)
        _mx_cell(
            root,
            cell_id=cell_id,
            value=event_text,
            x=event_x,
            y=event_y,
            w=width,
            h=height,
            style="rounded=0;whiteSpace=wrap;html=0;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;align=left;verticalAlign=top;",
        )
        _mx_cell(
            root,
            cell_id=f"edge_event_{event_index + 1}",
            value="",
            x=0,
            y=0,
            w=0,
            h=0,
            style="edgeStyle=none;rounded=0;orthogonalLoop=0;jettySize=0;html=0;dashed=1;endArrow=block;",
            edge=True,
            source=step_cell,
            target=cell_id,
        )
        event_cursor = event_y + height + event_gap_y

    xml = tostring(mxfile, encoding="unicode")
    save_text(xml, output_path)
    return output_path


def export_step2_drawio(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Export ONLY the Step-2 annotated research graph to a .drawio file.

    Unlike :func:`export_drawio`, this skips the full pipeline scaffolding
    (header, params, per-step rows, retrieval trace) and renders just the
    research graph from ``step2_research_graph.json``.
    """
    run_dir = Path(run_dir)
    graph_path = run_dir / "step2_research_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"step2_research_graph.json not found under {run_dir}")
    output_path = Path(output_path) if output_path else run_dir / "step2_graph.drawio"

    mxfile = Element("mxfile", host="app.diagrams.net", modified="2026-06-11T00:00:00.000Z", agent="Trae")
    diagram = SubElement(mxfile, "diagram", id="step2_graph", name="Step 2 Research Graph")
    model = SubElement(
        diagram,
        "mxGraphModel",
        dx="1800",
        dy="2600",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="2200",
        pageHeight="3200",
        math="0",
        shadow="0",
    )
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")

    # A minimal anchor node acts as the source for the graph title edge so
    # _render_step2_graph (which expects a step-2 cell to link from) works.
    anchor_text = _wrap("\n".join(["Step 2 Research Graph Construction", f"Run dir: {run_dir}"]))
    _mx_cell(
        root,
        cell_id="step2_anchor",
        value=anchor_text,
        x=40,
        y=40,
        w=_node_width(anchor_text, min_width=300, max_width=520),
        h=_node_height(anchor_text, base=90),
        style="rounded=0;whiteSpace=wrap;html=0;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;align=left;verticalAlign=top;",
    )

    # Concise mode: render nodes/phases only, no relation edges.
    _render_step2_graph(
        root,
        run_dir=run_dir,
        step2_cell_id="step2_anchor",
        start_y=180,
        x_offset=60,
        render_edges=False,
    )

    xml = tostring(mxfile, encoding="unicode")
    save_text(xml, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a pipeline run directory to draw.io format.")
    parser.add_argument("run_dir", help="Run directory containing summary.json and step artifacts.")
    parser.add_argument("--output", default=None, help="Output .drawio path. Defaults to <run_dir>/pipeline.drawio")
    parser.add_argument(
        "--abstract-chars",
        type=int,
        default=280,
        help="Maximum abstract characters to embed for each returned paper.",
    )
    parser.add_argument(
        "--step2-only",
        action="store_true",
        help="Export ONLY the Step-2 research graph (skip pipeline scaffolding). "
        "Defaults output to <run_dir>/step2_graph.drawio.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.step2_only:
        output = export_step2_drawio(args.run_dir, output_path=args.output)
    else:
        output = export_drawio(
            args.run_dir,
            output_path=args.output,
            abstract_chars=args.abstract_chars,
        )
    print(output)


if __name__ == "__main__":
    main()
