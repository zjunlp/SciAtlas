"""Step 4 — extract a Research Structure Signature (RSS) from the GoR graph."""
from __future__ import annotations

import json
from typing import Any

from ..clients.llm_client import LLMClient
from ..models import ResearchGraph
from ..prompts import RSS_EXTRACTION
from ..utils import get_logger
from .step3_trend_summarizer import render_graph_markdown

log = get_logger("sciatlas.step4")


def run(
    llm: LLMClient,
    graph: ResearchGraph,
    research_topic: str,
    trend: str,
    *,
    seed_paper_or_target_direction: str | None = None,
) -> dict[str, Any]:
    """Return the RSS JSON object extracted from the graph and trend summary."""
    log.info("Step 4: RSS extraction")
    graph_text = render_graph_markdown(graph, research_topic)
    prompt = RSS_EXTRACTION.format(
        research_topic=research_topic,
        seed_paper_or_target_direction=(seed_paper_or_target_direction or research_topic),
        gor_graph=graph_text,
        trend=trend,
    )
    rss = llm.chat_json(prompt, temperature=0.2)
    if not isinstance(rss, dict):
        raise ValueError("RSS extraction did not return a JSON object")
    log.info("  RSS extracted with keys: %s", ", ".join(sorted(rss.keys())))
    return rss


def render_rss_markdown(rss: dict[str, Any]) -> str:
    """Compact human-readable RSS rendering for prompts and saved artifacts."""
    lines = ["Research Structure Signature", "-------"]
    for key in (
        "target_domain",
        "core_gap",
        "objects",
        "relations",
        "bottlenecks",
        "unresolved_tensions",
        "desired_solution_properties",
        "possible_mechanism_types",
        "inspiration_search_hints",
    ):
        value = rss.get(key)
        lines.append(f"{key}:")
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(json.dumps(value, ensure_ascii=False))
    return "\n".join(lines)
