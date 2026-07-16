#!/usr/bin/env python3
"""Render a swimming-lane method-timeline figure from a YAML / JSON spec.

Output: a single, self-contained HTML file with inline SVG. No external CSS
or JS dependencies. Open it in any modern browser; export to PDF via the
browser's print dialog; or embed the standalone SVG (--output-svg) into a
LaTeX paper after light editing in Inkscape / Illustrator.

The spec is intentionally human-editable. Edit anchors, citation counts,
colors, etc. and re-run this script to refresh the figure in seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any


# ----------------------------- Spec loading -----------------------------


def load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PyYAML required for .yaml specs; install pyyaml or convert to .json") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit(f"Spec at {path} must be a top-level mapping/object")
    return payload


# ----------------------------- Geometry -----------------------------


def citation_to_radius(cites: Any, *, min_r: float = 5.0, max_r: float = 11.0) -> float:
    try:
        cites_val = float(cites)
    except (TypeError, ValueError):
        cites_val = 0.0
    if cites_val <= 0:
        return min_r
    log_cites = math.log10(max(1.0, cites_val))
    log_max = math.log10(300_000.0)
    frac = max(0.0, min(1.0, log_cites / log_max))
    return min_r + (max_r - min_r) * frac


def estimate_label_width(text: str, *, font_px: float = 10.5, char_factor: float = 0.58) -> float:
    """Very rough width estimate. Errs on the wide side to leave breathing room."""
    width_per_char = font_px * char_factor
    return max(28.0, len(text) * width_per_char + 4.0)


def wrap_text_lines(text: str, *, max_chars: int) -> list[str]:
    """Wrap plain text into short SVG-friendly lines by word count."""
    words = str(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def wrap_text_lines_by_width(
    text: str,
    *,
    max_width: float,
    font_px: float,
    char_factor: float = 0.56,
) -> list[str]:
    """Wrap plain text into lines using rough width estimation."""
    words = str(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if estimate_label_width(candidate, font_px=font_px, char_factor=char_factor) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def shorten_future_lane_label(label: str) -> str:
    """Shorten long lane names for future-work cards."""
    text = str(label).strip()
    replacements = {
        "RL for LLMs / Alignment": "RL for LLMs",
        "Exploration / Intrinsic Motivation": "Exploration",
        "Actor-Critic / Continuous Control": "Actor-Critic",
        "Trust Region / PPO": "PPO",
        "Deep Value-Based RL": "Value-Based RL",
        "Model-Based RL": "Model-Based RL",
        "Offline RL": "Offline RL",
        "Multi-Agent RL": "Multi-Agent RL",
        "Classical Foundations": "Foundations",
        "MLP / Foundational": "MLP",
        "CNN Architectures": "CNNs",
        "RNN / Sequence": "RNNs",
        "Transformer Family": "Transformers",
        "State-Space Models / Mamba": "SSMs",
        "Diffusion / Score-Based": "Diffusion",
        "Generative Adversarial Nets": "GANs",
        "Self-Supervised": "Self-Supervised",
        "Graph Neural Networks": "GNNs",
    }
    return replacements.get(text, text)


def stagger_label_positions(geoms: list[dict[str, Any]], *, base_offset: float, row_height: float) -> None:
    """Greedy collision-avoidance: assign each label to the lowest row that doesn't overlap.

    Rows alternate below the dot. Each row adds `row_height` to the y offset.
    """
    rows: list[list[tuple[float, float]]] = []
    for geom in geoms:
        label_w = geom["label_width"]
        x_min = geom["x"] - label_w / 2.0
        x_max = geom["x"] + label_w / 2.0
        placed = None
        for r_idx, occupied in enumerate(rows):
            conflict = any(not (x_max < lo - 3.0 or x_min > hi + 3.0) for lo, hi in occupied)
            if not conflict:
                occupied.append((x_min, x_max))
                placed = r_idx
                break
        if placed is None:
            rows.append([(x_min, x_max)])
            placed = len(rows) - 1
        geom["label_row"] = placed
        geom["label_y_offset"] = base_offset + placed * row_height


def stagger_label_positions_twosided(
    geoms: list[dict[str, Any]],
    *,
    base_below: float = 16.0,
    base_above: float = -12.0,
    step: float = 13.5,
) -> None:
    """Place labels on both sides of the lane center line, picking the side
    whose lowest available row is shallower. Halves vertical footprint
    compared to one-sided stagger, which lets demo mode use ~90px row_height
    for dense lanes.
    """
    rows_below: list[list[tuple[float, float]]] = []
    rows_above: list[list[tuple[float, float]]] = []

    def find_row(rows: list[list[tuple[float, float]]], x_min: float, x_max: float) -> int:
        for r_idx, occupied in enumerate(rows):
            if not any(not (x_max < lo - 3.0 or x_min > hi + 3.0) for lo, hi in occupied):
                return r_idx
        return len(rows)  # would need a new row at this index

    for geom in geoms:
        label_w = geom["label_width"]
        x_min = geom["x"] - label_w / 2.0
        x_max = geom["x"] + label_w / 2.0
        row_below = find_row(rows_below, x_min, x_max)
        row_above = find_row(rows_above, x_min, x_max)
        # Prefer the side whose target row is shallower; tiebreak: below first.
        if row_below <= row_above:
            if row_below == len(rows_below):
                rows_below.append([])
            rows_below[row_below].append((x_min, x_max))
            geom["label_y_offset"] = base_below + row_below * step
            geom["label_side"] = "below"
        else:
            if row_above == len(rows_above):
                rows_above.append([])
            rows_above[row_above].append((x_min, x_max))
            geom["label_y_offset"] = base_above - row_above * step
            geom["label_side"] = "above"


# ----------------------------- SVG primitives -----------------------------


SVG_CSS_TEMPLATE = """
:root {{
    --fg: #1a1a1a;
    --muted: #6b7280;
    --grid: #d1d5db;
    --axis: #374151;
}}
.figure-title {{ font: 700 22px "Helvetica Neue", "Inter", "SF Pro Text", system-ui, sans-serif; fill: var(--fg); }}
.figure-subtitle {{ font: 400 13px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--muted); }}
.cluster-label {{ font: 600 {cluster_font_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; }}
.cluster-sub {{ font: 500 10.5px "Helvetica Neue", "Inter", system-ui, sans-serif; }}
.anchor-label {{ font: 500 {anchor_font_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; pointer-events: none; }}
.anchor {{ cursor: default; }}
.anchor-circle {{ transition: stroke-width 120ms ease, transform 120ms ease; transform-box: fill-box; transform-origin: center; }}
.anchor:hover .anchor-circle {{ stroke-width: 3.2; transform: scale(1.35); }}
.anchor:hover .anchor-label {{ font-weight: 700; }}
.time-axis {{ stroke: var(--axis); stroke-width: 1.2; }}
.tick-major {{ stroke: var(--axis); stroke-width: 1.2; }}
.tick-minor {{ stroke: var(--muted); stroke-width: 0.6; opacity: 0.55; }}
.tick-label-major {{ font: 600 {tick_font_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--fg); }}
.year-grid {{ stroke: var(--grid); stroke-width: 0.6; stroke-dasharray: 2,3; opacity: 0.7; }}
.legend-text {{ font: 500 12px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--fg); }}
.legend-sub {{ font: 400 11px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--muted); }}
.lane-band:nth-child(odd) {{ fill: #f7f8fb; }}
.lane-band:nth-child(even) {{ fill: #ffffff; }}
.future-rail-title {{ font: 700 {future_title_font_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--fg); }}
.future-rail-bg {{ fill: #f8fafc; stroke: #dbe4ee; stroke-width: 1; }}
.future-rail-divider {{ stroke: #cbd5e1; stroke-width: 1.2; stroke-dasharray: 4,4; }}
.future-connector {{ fill: none; stroke: #94a3b8; stroke-width: 1.2; stroke-opacity: 0.42; }}
.future-connector-stem {{ stroke: #94a3b8; stroke-width: 1.2; stroke-opacity: 0.35; }}
.future-card {{ fill: white; stroke: #d7dee8; stroke-width: 1.1; }}
.future-card-title {{ font: 700 {future_card_title_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--fg); }}
.future-card-body {{ font: 500 {future_card_body_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: #334155; }}
.future-card-lanes {{ font: 600 {future_card_lane_px}px "Helvetica Neue", "Inter", system-ui, sans-serif; fill: var(--muted); letter-spacing: 0.04em; }}
"""


def make_svg_css(*, demo: bool) -> str:
    if demo:
        return SVG_CSS_TEMPLATE.format(
            cluster_font_px=16,
            anchor_font_px=12.5,
            tick_font_px=12.5,
            future_title_font_px=21,
            future_sub_font_px=10.5,
            future_card_title_px=14,
            future_card_body_px=10.4,
            future_card_lane_px=9.8,
        )
    return SVG_CSS_TEMPLATE.format(
        cluster_font_px=14,
        anchor_font_px=10.5,
        tick_font_px=11.5,
            future_title_font_px=20,
            future_sub_font_px=11,
            future_card_title_px=13.8,
            future_card_body_px=10.6,
            future_card_lane_px=10.0,
    )


def short_label(anchor: dict[str, Any]) -> str:
    return str(anchor.get("label") or anchor.get("full_title") or "?")


def build_piecewise_time_scale(
    topic: dict[str, Any],
    *,
    year_min: int,
    year_max: int,
    plot_x0: float,
    plot_x1: float,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a piecewise-linear year→x mapping.

    Read from ``topic.time_segments`` if present, otherwise use a sensible default
    that compresses the pre-2005 era and expands 2012–2025 where most of the
    dots live. Each segment has ``{start, end, weight}``; weights are normalized.
    """
    raw_segments = topic.get("time_segments")
    if not raw_segments:
        raw_segments = [
            {"start": year_min, "end": 1985, "weight": 0.08},
            {"start": 1985,     "end": 2005, "weight": 0.14},
            {"start": 2005,     "end": 2012, "weight": 0.13},
            {"start": 2012,     "end": 2016, "weight": 0.16},
            {"start": 2016,     "end": 2019, "weight": 0.16},
            {"start": 2019,     "end": 2021, "weight": 0.14},
            {"start": 2021,     "end": year_max, "weight": 0.19},
        ]
    segments = []
    for s in raw_segments:
        try:
            seg_start = int(s["start"])
            seg_end = int(s["end"])
            weight = float(s.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if seg_end <= seg_start or weight <= 0:
            continue
        segments.append({"start": seg_start, "end": seg_end, "weight": weight})
    if not segments:
        segments.append({"start": year_min, "end": year_max, "weight": 1.0})

    total_weight = sum(s["weight"] for s in segments)
    cumulative = 0.0
    total_span = plot_x1 - plot_x0
    for s in segments:
        norm_w = s["weight"] / total_weight
        s["x_start"] = plot_x0 + cumulative * total_span
        s["x_end"] = plot_x0 + (cumulative + norm_w) * total_span
        cumulative += norm_w

    def x_for(year: float) -> float:
        for s in segments:
            if s["start"] <= year <= s["end"]:
                frac = (year - s["start"]) / max(1.0, (s["end"] - s["start"]))
                return s["x_start"] + frac * (s["x_end"] - s["x_start"])
        if year < segments[0]["start"]:
            return segments[0]["x_start"]
        return segments[-1]["x_end"]

    return x_for, segments


def render_svg(spec: dict[str, Any], *, width: int = 1500, demo: bool = False) -> tuple[str, int]:
    topic = spec.get("topic") or {}
    clusters = list(spec.get("clusters") or [])
    future_items = list(spec.get("future_work") or topic.get("future_work") or [])
    if not clusters:
        raise SystemExit("Spec contains no clusters")
    cluster_name_by_id = {
        str(cluster.get("id") or ""): str(cluster.get("name") or cluster.get("id") or "")
        for cluster in clusters
    }

    year_range = topic.get("time_range") or [1958, 2026]
    year_min, year_max = int(year_range[0]), int(year_range[1])

    if demo:
        # Auto-fit left margin to the widest lane name so labels never clip.
        max_name_chars = max(
            len(shorten_future_lane_label(str(c.get("name") or c.get("id") or ""))) for c in clusters
        )
        # cluster-label font is 16px in demo (set via CSS); ~0.62 width factor.
        left_margin = int(max(150, max_name_chars * 16 * 0.62 + 28))
        # No SVG figure-title in demo (HTML <h1> shows it), so top margin is tight.
        margin = {"top": 32, "right": 30, "bottom": 56, "left": left_margin}
        row_height = int(topic.get("demo_row_height") or 86)
        anchor_label_font_px = 12.5
        dot_min_r = 7.0
        dot_max_r = 14.0
    else:
        margin = {"top": 100, "right": 80, "bottom": 110, "left": 230}
        row_height = int(topic.get("row_height") or 150)
        anchor_label_font_px = 10.5
        dot_min_r = 5.0
        dot_max_r = 11.0
    future_gap = 24 if demo else 30
    future_rail_width = 0
    if future_items:
        future_rail_width = int(topic.get("future_work_width") or (330 if demo else 350))
    plot_width = width - margin["left"] - margin["right"] - future_rail_width - (future_gap if future_items else 0)
    plot_height = len(clusters) * row_height
    svg_height = margin["top"] + plot_height + margin["bottom"]
    plot_x0 = margin["left"]
    plot_x1 = margin["left"] + plot_width
    future_x0 = plot_x1 + future_gap if future_items else None
    future_x1 = width - margin["right"] if future_items else None

    x_for, segments = build_piecewise_time_scale(
        topic,
        year_min=year_min,
        year_max=year_max,
        plot_x0=plot_x0,
        plot_x1=plot_x1,
    )

    elements: list[str] = []
    lane_centers: dict[str, float] = {}

    # ----- Alternating row bands for readability -----
    for i, _cluster in enumerate(clusters):
        y0 = margin["top"] + i * row_height
        fill = "#f7f8fb" if i % 2 == 0 else "#ffffff"
        elements.append(
            f'<rect class="lane-band" x="{plot_x0:.1f}" y="{y0:.1f}" '
            f'width="{plot_width:.1f}" height="{row_height:.1f}" fill="{fill}" />'
        )

    # ----- Vertical grid at segment boundaries (light) -----
    for s in segments:
        for boundary_year in (s["start"], s["end"]):
            x = x_for(boundary_year)
            elements.append(
                f'<line class="year-grid" x1="{x:.1f}" y1="{margin["top"]:.1f}" '
                f'x2="{x:.1f}" y2="{margin["top"] + plot_height:.1f}" />'
            )

    # ----- Per-cluster rows -----
    for i, cluster in enumerate(clusters):
        cy = margin["top"] + i * row_height + row_height / 2.0
        lane_centers[str(cluster.get("id") or "")] = cy
        color = str(cluster.get("color") or "#444")
        cluster_name = escape(shorten_future_lane_label(str(cluster.get("name") or cluster.get("id") or "?")))
        cluster_id = escape(str(cluster.get("id") or ""))
        anchors = sorted(cluster.get("anchors") or [], key=lambda a: a.get("year") or 0)

        # Lane center line
        elements.append(
            f'<line x1="{plot_x0:.1f}" y1="{cy:.1f}" '
            f'x2="{plot_x1:.1f}" y2="{cy:.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-opacity="0.18" />'
        )

        # Left-side cluster name + counts
        n_total = len(anchors)
        n_in_corpus = sum(1 for a in anchors if a.get("in_corpus"))
        if demo:
            # Demo: only the lane name, vertically centered, no debug subline.
            elements.append(
                f'<text class="cluster-label" x="{margin["left"] - 14:.1f}" y="{cy:.1f}" '
                f'text-anchor="end" fill="{color}" dominant-baseline="middle">{cluster_name}</text>'
            )
        else:
            elements.append(
                f'<text class="cluster-label" x="{margin["left"] - 18:.1f}" y="{cy - 6:.1f}" '
                f'text-anchor="end" fill="{color}" dominant-baseline="middle">{cluster_name}</text>'
            )
            elements.append(
                f'<text class="cluster-sub" x="{margin["left"] - 18:.1f}" y="{cy + 12:.1f}" '
                f'text-anchor="end" fill="{color}" fill-opacity="0.7" dominant-baseline="middle">'
                f'{cluster_id} · n={n_total} · {n_in_corpus}/{n_total} in corpus</text>'
            )

        # Pre-compute anchor geoms (raw center x at year)
        geoms = []
        for anchor in anchors:
            try:
                year = float(anchor.get("year") or 0)
            except (TypeError, ValueError):
                continue
            if year <= 0:
                continue
            x = x_for(year)
            label = short_label(anchor)
            geoms.append(
                {
                    "anchor": anchor,
                    "year": year,
                    "center_x": x,
                    "x": x,  # will be replaced after same-year spreading
                    "label": label,
                    "label_width": estimate_label_width(
                        label,
                        font_px=anchor_label_font_px,
                        char_factor=0.66 if demo else 0.58,
                    ),
                    "radius": citation_to_radius(anchor.get("citation_count"), min_r=dot_min_r, max_r=dot_max_r),
                }
            )

        # ---- Lane-level 1D relaxation: separates same-year and adjacent-year clashes ----
        # Initialize all anchors at their year-center x.
        for geom in geoms:
            geom["x"] = geom["center_x"]

        # First, explicitly fan out same-year anchors so newly added dense
        # frontier clusters are laid out in a visible left-to-right row before
        # the lighter global relaxation runs.
        year_groups: dict[int, list[dict[str, Any]]] = {}
        for geom in geoms:
            year_groups.setdefault(int(round(geom["year"])), []).append(geom)
        for _year, group in year_groups.items():
            if len(group) <= 1:
                continue
            group.sort(key=lambda g: (g["label"], g["anchor"].get("full_title", "")))
            max_r = max(g["radius"] for g in group)
            step = max(24.0, 2.0 * max_r + 8.0)
            center = sum(g["center_x"] for g in group) / len(group)
            start = center - step * (len(group) - 1) / 2.0
            for idx, geom in enumerate(group):
                geom["x"] = start + idx * step

        min_gap = 4.0  # additional pixel gap between circle edges
        pull_strength = 0.04  # gentle spring back toward center_x each iteration
        max_iter = 200
        for _ in range(max_iter):
            geoms.sort(key=lambda g: g["x"])
            collided = False
            for i in range(len(geoms) - 1):
                a, b = geoms[i], geoms[i + 1]
                need = a["radius"] + b["radius"] + min_gap
                dist = b["x"] - a["x"]
                if dist < need:
                    delta = (need - dist) / 2.0 * 1.02  # slight overshoot helps convergence
                    a["x"] -= delta
                    b["x"] += delta
                    collided = True
            # Gentle pull-back so anchors stay near their true year when possible.
            for geom in geoms:
                drift = geom["x"] - geom["center_x"]
                if abs(drift) > 0.5:
                    geom["x"] -= drift * pull_strength
            if not collided:
                break

        # Clamp relaxed anchors back into the visible plotting range so dense
        # right-edge clusters cannot drift outside the lane.
        right_safety = 10.0
        left_safety = 6.0
        for geom in geoms:
            geom["x"] = max(
                plot_x0 + geom["radius"] + left_safety,
                min(plot_x1 - geom["radius"] - right_safety, geom["x"]),
            )

        # Sort by final x for label-stagger pass (so deconfliction sweeps left-to-right).
        geoms.sort(key=lambda g: g["x"])
        if demo:
            # base_below / base_above are tuned to clear the dot top
            # (max dot radius = 14 → labels start ≥ 22px from lane center).
            stagger_label_positions_twosided(
                geoms, base_below=26.0, base_above=-20.0, step=16.0
            )
        else:
            stagger_label_positions(geoms, base_offset=20.0, row_height=16.0)

        def lane_y_for_geom(geom: dict[str, Any]) -> float:
            return cy

        # ----- Draw subtle connection line between adjacent anchors (by x) -----
        for g_a, g_b in zip(geoms, geoms[1:]):
            # only connect when reasonably close to avoid huge spans dominating the eye
            if abs(g_b["x"] - g_a["x"]) > 250:
                continue
            y_a = lane_y_for_geom(g_a)
            y_b = lane_y_for_geom(g_b)
            elements.append(
                f'<line x1="{g_a["x"]:.1f}" y1="{y_a:.1f}" '
                f'x2="{g_b["x"]:.1f}" y2="{y_b:.1f}" '
                f'stroke="{color}" stroke-width="1.4" stroke-opacity="0.30" />'
            )

        # NOTE: in v1 we drew a dashed leader from center_x to the relaxed x for each
        # nudged anchor. After switching to lane-wide relaxation almost every anchor in
        # the dense post-2015 region drifts a few pixels, so leaders become visual
        # clutter. We drop them; the year axis below is the single source of truth and
        # hovering a dot still shows its true year in the tooltip.

        # Draw dots + labels
        for geom in geoms:
            anchor = geom["anchor"]
            in_corpus = bool(anchor.get("in_corpus"))
            year = anchor.get("year")
            label = geom["label"]
            full_title = str(anchor.get("full_title") or label)
            first_author = str(anchor.get("first_author") or "")
            venue = str(anchor.get("venue") or "")
            cite_value = anchor.get("citation_count")
            try:
                cite_int = int(cite_value)
                cite_str = f"{cite_int:,}"
            except (TypeError, ValueError):
                cite_str = str(cite_value) if cite_value else "?"
            status = "in corpus" if in_corpus else "INJECTED (canonical anchor missing from retrieval)"
            notes = anchor.get("notes") or ""
            anchor_cy = lane_y_for_geom(geom)

            tooltip = "\n".join(
                [
                    f"{label} ({year})",
                    full_title,
                    f"First author: {first_author}" if first_author else "",
                    f"Venue: {venue}" if venue else "",
                    f"Citations: ~{cite_str}",
                    f"Status: {status}",
                    f"Note: {notes}" if notes else "",
                ]
            ).strip()

            if demo or in_corpus:
                circle = (
                    f'<circle class="anchor-circle" cx="{geom["x"]:.1f}" cy="{anchor_cy:.1f}" '
                    f'r="{geom["radius"]:.1f}" fill="{color}" stroke="{color}" stroke-width="1.5" />'
                )
            else:
                circle = (
                    f'<circle class="anchor-circle" cx="{geom["x"]:.1f}" cy="{anchor_cy:.1f}" '
                    f'r="{geom["radius"]:.1f}" fill="white" stroke="{color}" '
                    f'stroke-width="2.2" stroke-dasharray="3,2" />'
                )

            label_y = anchor_cy + geom["label_y_offset"]
            label_anchor = "middle"
            label_x = geom["x"]
            near_right = geom["x"] + geom["label_width"] / 2.0 > plot_x1 - 16.0
            near_left = geom["x"] - geom["label_width"] / 2.0 < plot_x0 + 10.0
            if near_right:
                label_anchor = "end"
                label_x = geom["x"] - 8.0
            elif near_left:
                label_anchor = "start"
                label_x = geom["x"] + 8.0
            elements.append(
                "<g class=\"anchor\">"
                f"<title>{escape(tooltip)}</title>"
                f"{circle}"
                f'<text class="anchor-label" x="{label_x:.1f}" y="{label_y:.1f}" '
                f'text-anchor="{label_anchor}" fill="{color}">{escape(label)}</text>'
                "</g>"
            )

    # ----- Time axis at bottom (piecewise; tick density per segment width) -----
    axis_y = margin["top"] + plot_height + 18
    elements.append(
        f'<line class="time-axis" x1="{plot_x0:.1f}" y1="{axis_y:.1f}" '
        f'x2="{plot_x1:.1f}" y2="{axis_y:.1f}" />'
    )
    drawn_ticks: set[int] = set()

    def draw_tick(year: int, major: bool) -> None:
        if year in drawn_ticks:
            return
        drawn_ticks.add(year)
        x = x_for(year)
        if major:
            elements.append(
                f'<line class="tick-major" x1="{x:.1f}" y1="{axis_y:.1f}" '
                f'x2="{x:.1f}" y2="{axis_y + 8:.1f}" />'
            )
            elements.append(
                f'<text class="tick-label-major" x="{x:.1f}" y="{axis_y + 24:.1f}" '
                f'text-anchor="middle">{year}</text>'
            )
        else:
            elements.append(
                f'<line class="tick-minor" x1="{x:.1f}" y1="{axis_y:.1f}" '
                f'x2="{x:.1f}" y2="{axis_y + 4:.1f}" />'
            )

    for seg in segments:
        seg_pixel_width = seg["x_end"] - seg["x_start"]
        seg_years = seg["end"] - seg["start"]
        pixels_per_year = seg_pixel_width / max(1, seg_years)
        # choose major-tick interval to keep ~60px between labels
        if pixels_per_year >= 60:
            major_step = 1
        elif pixels_per_year >= 30:
            major_step = 2
        elif pixels_per_year >= 15:
            major_step = 5
        elif pixels_per_year >= 7:
            major_step = 10
        else:
            major_step = 20
        # always draw segment boundaries as major
        draw_tick(seg["start"], major=True)
        draw_tick(seg["end"], major=True)
        for year in range(seg["start"], seg["end"] + 1):
            if year == seg["start"] or year == seg["end"]:
                continue
            if year % major_step == 0:
                draw_tick(year, major=True)
            elif pixels_per_year >= 30 and year % max(1, major_step // 2) == 0:
                draw_tick(year, major=False)

    # ----- Future-work rail at the right -----
    if future_items and future_x0 is not None and future_x1 is not None:
        rail_w = future_x1 - future_x0
        content_top = margin["top"] + 8
        content_bottom = margin["top"] + plot_height - 10
        inner_x = future_x0 + 14
        card_width = rail_w - 28
        stem_x = future_x0 - 10

        prepared_items: list[dict[str, Any]] = []
        card_title_font_px = 14.0 if demo else 13.8
        card_body_font_px = 10.4 if demo else 10.6
        card_lane_font_px = 9.8 if demo else 10.0
        for idx, item in enumerate(future_items):
            title = str(item.get("title") or item.get("label") or f"Future Direction {idx + 1}")
            body_text = str(item.get("text") or item.get("description") or "")
            text_padding_x = 24
            text_max_width = card_width - text_padding_x
            title_max_width = card_width - 24
            body_lines = wrap_text_lines_by_width(
                body_text,
                max_width=text_max_width,
                font_px=card_body_font_px,
                char_factor=0.54,
            )
            lane_ids = [str(lane) for lane in (item.get("lane_ids") or item.get("lanes") or []) if str(lane)]
            lane_labels = [str(label) for label in (item.get("lane_labels") or []) if str(label)]
            if not lane_labels:
                lane_labels = [cluster_name_by_id[lane_id] for lane_id in lane_ids if lane_id in cluster_name_by_id]
            lane_labels = [shorten_future_lane_label(label) for label in lane_labels]
            lane_text = " · ".join(lane_labels[:2])
            title_lines = wrap_text_lines_by_width(
                title,
                max_width=title_max_width,
                font_px=card_title_font_px,
                char_factor=0.53,
            )[:2]
            text_line_h = 12.5 if demo else 13
            title_line_h = 15.5 if demo else 15.5
            lane_line_h = 11
            card_h = 14
            card_h += len(title_lines) * title_line_h
            if body_lines:
                card_h += 5 + len(body_lines) * text_line_h
            if lane_text:
                card_h += 7 + lane_line_h
            card_h += 12
            lane_targets = [lane_centers[lane_id] for lane_id in lane_ids if lane_id in lane_centers]
            if lane_targets:
                target_y = sum(lane_targets) / len(lane_targets)
            else:
                target_y = content_top + (idx + 0.5) * ((content_bottom - content_top) / max(1, len(future_items)))
            prepared_items.append(
                {
                    "title": title,
                    "title_lines": title_lines,
                    "body_lines": body_lines,
                    "lane_ids": lane_ids,
                    "lane_text": lane_text,
                    "lane_targets": lane_targets,
                    "target_y": target_y,
                    "height": card_h,
                }
            )

        # Keep a stable semantic order in the card column, then pack tightly.
        prepared_items.sort(key=lambda item: item["target_y"])
        gap = 12 if demo else 14
        current_top = content_top
        for item in prepared_items:
            item["card_y"] = current_top
            current_top = item["card_y"] + item["height"] + gap

        if prepared_items and current_top - gap > content_bottom:
            current_bottom = content_bottom
            for item in reversed(prepared_items):
                item["card_y"] = min(item["card_y"], current_bottom - item["height"])
                current_bottom = item["card_y"] - gap
            if prepared_items[0]["card_y"] < content_top:
                current_top = content_top
                for item in prepared_items:
                    item["card_y"] = current_top
                    current_top += item["height"] + gap

        title_y = (prepared_items[0]["card_y"] - 6) if prepared_items else (content_top + 18)
        rail_y0 = max(18.0, title_y - 18)
        cards_bottom = max((item["card_y"] + item["height"]) for item in prepared_items) if prepared_items else title_y
        rail_y1 = cards_bottom + 12
        divider_y0 = min(
            rail_y0 + 6,
            min((min(item["lane_targets"]) for item in prepared_items if item["lane_targets"]), default=rail_y0 + 6),
        ) - 6
        divider_y1 = max(
            rail_y1 - 6,
            max((max(item["lane_targets"]) for item in prepared_items if item["lane_targets"]), default=rail_y1 - 6),
        ) + 6

        future_bg_elements = [
            f'<line class="future-rail-divider" x1="{plot_x1 + future_gap / 2.0:.1f}" '
            f'y1="{divider_y0:.1f}" x2="{plot_x1 + future_gap / 2.0:.1f}" '
            f'y2="{divider_y1:.1f}" />',
            f'<rect class="future-rail-bg" x="{future_x0:.1f}" y="{rail_y0:.1f}" '
            f'width="{rail_w:.1f}" height="{rail_y1 - rail_y0:.1f}" rx="18" ry="18" />',
            f'<text class="future-rail-title" x="{future_x0 + 18:.1f}" y="{title_y:.1f}">Future Work</text>',
        ]
        future_fg_elements: list[str] = []

        for item in prepared_items:
            card_x = inner_x
            card_y = item["card_y"]
            card_h = item["height"]
            card_cy = card_y + card_h / 2.0
            lane_targets = item["lane_targets"]

            if lane_targets:
                hub_y = sum(lane_targets) / len(lane_targets)
                if len(lane_targets) > 1:
                    future_fg_elements.append(
                        f'<line class="future-connector-stem" x1="{stem_x:.1f}" y1="{min(lane_targets):.1f}" '
                        f'x2="{stem_x:.1f}" y2="{max(lane_targets):.1f}" />'
                    )
                for lane_y in lane_targets:
                    control_x = plot_x1 + (stem_x - plot_x1) * 0.55
                    future_fg_elements.append(
                        f'<path class="future-connector" d="M {plot_x1:.1f} {lane_y:.1f} '
                        f'C {control_x:.1f} {lane_y:.1f}, {stem_x - 8:.1f} {lane_y:.1f}, {stem_x:.1f} {lane_y:.1f}" />'
                    )
                future_fg_elements.append(
                    f'<path class="future-connector" d="M {stem_x:.1f} {hub_y:.1f} '
                    f'C {stem_x + 10:.1f} {hub_y:.1f}, {card_x - 16:.1f} {card_cy:.1f}, {card_x:.1f} {card_cy:.1f}" />'
                )

            future_fg_elements.append(
                f'<rect class="future-card" x="{card_x:.1f}" y="{card_y:.1f}" '
                f'width="{card_width:.1f}" height="{card_h:.1f}" rx="14" ry="14" />'
            )
            text_x = card_x + 12
            cursor_y = card_y + 16
            for line_idx, line in enumerate(item["title_lines"]):
                line_y = cursor_y + line_idx * (15.5 if demo else 15.5)
                future_fg_elements.append(
                    f'<text class="future-card-title" x="{text_x:.1f}" y="{line_y:.1f}">{escape(line)}</text>'
                )
            cursor_y += len(item["title_lines"]) * (15.5 if demo else 15.5)
            if item["body_lines"]:
                cursor_y += 5
                for line_idx, line in enumerate(item["body_lines"]):
                    line_y = cursor_y + line_idx * (12.5 if demo else 13)
                    future_fg_elements.append(
                        f'<text class="future-card-body" x="{text_x:.1f}" y="{line_y:.1f}">{escape(line)}</text>'
                    )
                cursor_y += len(item["body_lines"]) * (12.5 if demo else 13)
            if item["lane_text"]:
                cursor_y += 7
                future_fg_elements.append(
                    f'<text class="future-card-lanes" x="{text_x:.1f}" y="{cursor_y:.1f}">{escape(item["lane_text"])}</text>'
                )

        elements.extend(future_bg_elements)
        elements.extend(future_fg_elements)

    # ----- Title and subtitle -----
    title_text = escape(str(topic.get("title") or ""))
    subtitle_text = escape(str(topic.get("subtitle") or ""))
    top_elems = []
    if title_text and not demo:
        # In demo mode the HTML <h1> shows the title; avoid duplication.
        top_elems.append(
            f'<text class="figure-title" x="{margin["left"]:.1f}" y="36">{title_text}</text>'
        )
    if subtitle_text and not demo:
        top_elems.append(
            f'<text class="figure-subtitle" x="{margin["left"]:.1f}" y="58">{subtitle_text}</text>'
        )

    # ----- Legend (top-right) — omitted in demo mode -----
    if not demo:
        total_anchors = sum(len(c.get("anchors") or []) for c in clusters)
        in_corpus_anchors = sum(
            1 for c in clusters for a in (c.get("anchors") or []) if a.get("in_corpus")
        )
        legend_x = margin["left"] + plot_width - 360
        legend_y = 30
        top_elems.append(
            f'<g class="legend" transform="translate({legend_x:.1f}, {legend_y:.1f})">'
            f'  <circle cx="10" cy="10" r="6" fill="#444" stroke="#444" stroke-width="1.5" />'
            f'  <text class="legend-text" x="22" y="14">retrieved by pipeline</text>'
            f'  <circle cx="180" cy="10" r="6" fill="white" stroke="#444" '
            f'stroke-width="2.2" stroke-dasharray="3,2" />'
            f'  <text class="legend-text" x="192" y="14">injected canonical anchor</text>'
            f'  <text class="legend-sub" x="0" y="32">'
            f'  dot size ∝ log(citation count). {in_corpus_anchors}/{total_anchors} anchors retrieved by pipeline; '
            f'rest injected to complete the canonical timeline.'
            f'  </text>'
            f'</g>'
        )

    if demo:
        # Responsive SVG: drop fixed width/height; CSS in wrapper controls size.
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {svg_height}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'role="img" aria-label="Method timeline">\n'
            f'<style><![CDATA[{make_svg_css(demo=True)}]]></style>\n'
            f'{"".join(top_elems)}\n'
            f'{"".join(elements)}\n'
            f'</svg>'
        )
    else:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{svg_height}" '
            f'viewBox="0 0 {width} {svg_height}" '
            f'role="img" aria-label="Method timeline">\n'
            f'<style><![CDATA[{make_svg_css(demo=False)}]]></style>\n'
            f'{"".join(top_elems)}\n'
            f'{"".join(elements)}\n'
            f'</svg>'
        )
    return svg, svg_height


# ----------------------------- HTML wrapper + reference table -----------------------------


HTML_CSS = """
body { font-family: "Helvetica Neue", "Inter", "SF Pro Text", system-ui, sans-serif;
       max-width: 1640px; margin: 24px auto 64px; padding: 0 24px;
       color: #1a1a1a; background: #fafafa; }
h1 { font-size: 24px; font-weight: 700; margin: 8px 0 4px; }
h2 { font-size: 18px; font-weight: 600; margin: 28px 0 10px; color: #111827; }
p.lede { color: #4b5563; font-size: 13px; margin: 0 0 16px; }
.svg-wrap { background: white; padding: 12px 8px; border-radius: 8px;
            border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            overflow-x: auto; }
.anchor-table { border-collapse: collapse; width: 100%; font-size: 12.5px;
                background: white; border: 1px solid #e5e7eb; border-radius: 6px;
                overflow: hidden; }
.anchor-table th, .anchor-table td { text-align: left; padding: 6px 10px;
                                      border-bottom: 1px solid #f1f1f4;
                                      vertical-align: top; }
.anchor-table th { background: #f3f4f6; font-weight: 600; font-size: 12px;
                   color: #374151; position: sticky; top: 0; }
.anchor-table tr:hover td { background: #f9fafb; }
.anchor-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.anchor-table .dot { display: inline-block; width: 10px; height: 10px;
                     border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.anchor-table .hollow { background: transparent !important; border: 2px dashed #888; }
.anchor-table .status-in { background: #ecfdf5; color: #047857; padding: 2px 6px;
                            border-radius: 3px; font-size: 10.5px; font-weight: 600; }
.anchor-table .status-inj { background: #fff7ed; color: #9a3412; padding: 2px 6px;
                             border-radius: 3px; font-size: 10.5px; font-weight: 600; }
.footer { margin-top: 36px; font-size: 12px; color: #6b7280; line-height: 1.5; }
code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
"""


DEMO_HTML_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; width: 100%;
             font-family: "Helvetica Neue", "Inter", "SF Pro Text", system-ui, sans-serif;
             color: #111; background: #fff; overflow: hidden; }
.demo-page { display: flex; flex-direction: column; height: 100vh; width: 100vw;
             padding: 14px 20px 10px; }
.demo-title { font-size: 22px; font-weight: 700; margin: 0 0 8px;
              flex: 0 0 auto; line-height: 1.2; }
.demo-figure { flex: 1 1 auto; display: flex; align-items: center;
               justify-content: center; min-height: 0; }
.demo-figure svg { display: block; max-width: 100%; max-height: 100%;
                   width: auto; height: auto; }
"""


def render_demo_html(spec: dict[str, Any], svg: str) -> str:
    topic = spec.get("topic") or {}
    title = escape(str(topic.get("title") or "Method Timeline"))
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{DEMO_HTML_CSS}</style>\n"
        "</head>\n<body>\n"
        '<div class="demo-page">\n'
        f'  <h1 class="demo-title">{title}</h1>\n'
        f'  <div class="demo-figure">{svg}</div>\n'
        "</div>\n"
        "</body></html>\n"
    )


def render_html(spec: dict[str, Any], svg: str, *, spec_path: str, include_table: bool) -> str:
    topic = spec.get("topic") or {}
    title = escape(str(topic.get("title") or "Method Timeline"))
    subtitle = escape(str(topic.get("subtitle") or ""))

    table_html = ""
    if include_table:
        rows_html: list[str] = []
        for cluster in spec.get("clusters", []):
            color = escape(str(cluster.get("color") or "#444"))
            cluster_name = escape(str(cluster.get("name") or cluster.get("id") or ""))
            for anchor in sorted(cluster.get("anchors") or [], key=lambda a: a.get("year") or 0):
                in_corpus = bool(anchor.get("in_corpus"))
                cite_value = anchor.get("citation_count")
                try:
                    cite_str = f"{int(cite_value):,}"
                except (TypeError, ValueError):
                    cite_str = str(cite_value) if cite_value else "—"
                status_cls = "status-in" if in_corpus else "status-inj"
                status_txt = "retrieved" if in_corpus else "injected"
                dot_cls = "dot" if in_corpus else "dot hollow"
                dot_style = f"background:{color}" if in_corpus else f"border-color:{color}"
                rows_html.append(
                    "<tr>"
                    f'<td><span class="{dot_cls}" style="{dot_style}"></span>{cluster_name}</td>'
                    f"<td class=\"num\">{anchor.get('year', '?')}</td>"
                    f"<td><b>{escape(str(anchor.get('label') or ''))}</b></td>"
                    f"<td>{escape(str(anchor.get('full_title') or ''))}</td>"
                    f"<td>{escape(str(anchor.get('first_author') or ''))}</td>"
                    f"<td>{escape(str(anchor.get('venue') or ''))}</td>"
                    f'<td class="num">~{cite_str}</td>'
                    f'<td><span class="{status_cls}">{status_txt}</span></td>'
                    "</tr>"
                )
        table_html = (
            "<h2>Anchor reference table</h2>"
            '<p class="lede">All anchors in the figure. Hover any dot in the SVG above to see the same metadata as a tooltip.</p>'
            '<table class="anchor-table"><thead><tr>'
            "<th>Lane</th><th>Year</th><th>Label</th><th>Full title</th>"
            "<th>First author</th><th>Venue</th><th>~Citations</th><th>Status</th>"
            "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        )

    spec_path_escaped = escape(spec_path)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>{HTML_CSS}</style>\n"
        "</head>\n<body>\n"
        f"<h1>{title}</h1>\n"
        + (f'<p class="lede">{subtitle}</p>\n' if subtitle else "")
        + '<div class="svg-wrap">\n'
        + svg
        + "\n</div>\n"
        + table_html
        + '\n<p class="footer">Generated from <code>' + spec_path_escaped + "</code> "
        + "via <code>render_method_timeline.py</code>. "
        + "Edit the YAML spec (anchors, citation counts, colors, time range) and re-run to refresh. "
        + "Hover any dot to see the full title and metadata. "
        + "Export to PDF via the browser's print dialog, or use the standalone SVG output for LaTeX inclusion.</p>\n"
        "</body></html>\n"
    )


# ----------------------------- Driver -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render method-timeline swimming-lane figure from a YAML/JSON spec.")
    parser.add_argument("--spec", required=True, help="Path to spec YAML/JSON.")
    parser.add_argument("--output-html", required=True, help="Output HTML path (self-contained).")
    parser.add_argument("--output-svg", default=None, help="Optional standalone SVG output path.")
    parser.add_argument("--width", type=int, default=1500, help="SVG width in pixels.")
    parser.add_argument("--no-table", action="store_true", help="Omit the anchor reference table.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Render a minimal demo HTML: no subtitle, legend, per-lane debug "
            "subline, anchor table, or hollow dashed dots. All dots filled. "
            "Compact row height. SVG scales to fit viewport (no scroll)."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    spec = load_spec(spec_path)
    svg, height = render_svg(spec, width=args.width, demo=args.demo)
    if args.demo:
        html = render_demo_html(spec, svg)
    else:
        html = render_html(spec, svg, spec_path=str(spec_path), include_table=not args.no_table)

    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    output_svg_path = None
    if args.output_svg:
        output_svg_path = Path(args.output_svg).expanduser().resolve()
        output_svg_path.parent.mkdir(parents=True, exist_ok=True)
        output_svg_path.write_text(f'<?xml version="1.0" encoding="utf-8"?>\n{svg}\n', encoding="utf-8")

    summary = {
        "output_html": str(output_html),
        "output_svg": str(output_svg_path) if output_svg_path else None,
        "svg_width": args.width,
        "svg_height": height,
        "cluster_count": len(spec.get("clusters") or []),
        "anchor_count": sum(len(c.get("anchors") or []) for c in (spec.get("clusters") or [])),
        "future_work_count": len(spec.get("future_work") or spec.get("topic", {}).get("future_work") or []),
        "in_corpus_count": sum(
            1 for c in (spec.get("clusters") or []) for a in (c.get("anchors") or []) if a.get("in_corpus")
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
