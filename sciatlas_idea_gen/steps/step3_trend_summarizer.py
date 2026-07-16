"""Step 3 — GoR-style graph serialization and trend summarization."""
from __future__ import annotations

from ..clients.llm_client import LLMClient
from ..models import GraphEdge, Paper, ResearchGraph
from ..prompts import TREND_SUMMARY
from ..utils import get_logger

log = get_logger("sciatlas.step3")

IDEA_FIELD_ORDER = (
    "Problem",
    "Existing Methods",
    "Motivation",
    "Proposed Method",
    "Experiment Plan",
)


def _seed_paper(graph: ResearchGraph) -> Paper | None:
    if graph.seed_paper_id and graph.seed_paper_id in graph.papers:
        return graph.papers[graph.seed_paper_id]
    if graph.phases:
        for layer in graph.phases:
            for pid in layer:
                paper = graph.papers.get(pid)
                if paper:
                    return paper
    return next(iter(graph.papers.values()), None)


def _paper_edges(graph: ResearchGraph) -> list[GraphEdge]:
    return [
        edge for edge in graph.edges
        if edge.relation in {"explicit_pred", "parallel_pred", "direct_to_seed"}
        and edge.source in graph.papers
        and edge.target in graph.papers
    ]


def _paper_cite_count(paper: Paper) -> int:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    value = raw.get("cited_by_count")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return len(paper.citations or [])


def render_graph_markdown(graph: ResearchGraph, research_topic: str | None = None) -> str:
    seed = _seed_paper(graph)
    if seed is None:
        return "# SEED\nMETA\ntopic: unknown\nseed_paper: unknown\n"

    ordered_ids: list[str] = []
    for layer in graph.phases:
        for pid in layer:
            if pid in graph.papers and pid != seed.paper_id:
                ordered_ids.append(pid)
    if not ordered_ids:
        ordered_ids = [pid for pid in graph.papers if pid != seed.paper_id]

    phase_lookup: dict[str, int] = {}
    for layer_index, layer in enumerate(graph.phases, start=1):
        for pid in layer:
            phase_lookup[pid] = layer_index

    incoming: dict[str, list[GraphEdge]] = {pid: [] for pid in ordered_ids}
    cited_by_subgraph: dict[str, int] = {pid: 0 for pid in ordered_ids}
    for edge in _paper_edges(graph):
        if edge.target in incoming:
            incoming[edge.target].append(edge)
        if edge.source in cited_by_subgraph:
            cited_by_subgraph[edge.source] += 1

    index_by_id = {pid: index for index, pid in enumerate(ordered_ids, start=1)}
    lines = [
        "# SEED",
        "META",
        f"topic: {research_topic or seed.title or 'unknown'}",
        f"seed_paper: {seed.title or seed.paper_id or 'unknown'}",
        "",
        f"# CITATION SUBGRAPH ({len(ordered_ids)} refs, temporally ordered by year)",
        "",
    ]

    for pid in ordered_ids:
        paper = graph.papers[pid]
        year = paper.year if paper.year is not None else "n.d."
        incoming_edges = incoming.get(pid, [])
        feature_source = incoming_edges[0].features if incoming_edges else {}
        cited_in_sections = feature_source.get("cited_in_sections", [])
        if not isinstance(cited_in_sections, list):
            cited_in_sections = []
        lines.append(f"## [{index_by_id[pid]}] {paper.title} ({year}, venue=N/A)")
        lines.append("authors: N/A")
        lines.append("")
        lines.append("[EDGE]")
        lines.append(f"layer_depth = {feature_source.get('layer_depth', phase_lookup.get(pid, 1))}")
        lines.append(f"cited_in_sections = {cited_in_sections}")
        lines.append(f"cite_count = {feature_source.get('cite_count', _paper_cite_count(paper))}")
        lines.append(f"section_weight = {feature_source.get('section_weight', 0.0)}")
        delta_year = feature_source.get("delta_year")
        lines.append(f"delta_year = {delta_year if delta_year is not None else 'unknown'}")
        lines.append(
            f"is_influential = {feature_source.get('is_influential_raw', paper.is_influential)}"
        )
        lines.append(f"low_confidence = {feature_source.get('low_confidence', True)}")
        lines.append(
            f"cited_by_subgraph = {feature_source.get('cited_by_subgraph', cited_by_subgraph.get(pid, 0))}"
        )
        lines.append("")
        lines.append("[PREDECESSORS]")
        preds = sorted(
            incoming_edges,
            key=lambda edge: (graph.papers[edge.source].year or 0, graph.papers[edge.source].title),
        )
        if preds:
            for edge in preds:
                pred = graph.papers[edge.source]
                if pred.year is not None and paper.year is not None:
                    delta_yr = paper.year - pred.year
                else:
                    delta_yr = "unknown"
                lines.append(
                    f"- ref_idx={index_by_id.get(edge.source, 0)} delta_yr={delta_yr} edge_type={edge.relation}"
                )
        else:
            lines.append("- ref_idx=[] delta_yr=unknown edge_type=none")
        lines.append("")
        lines.append("[IDEA]")
        for field_name in IDEA_FIELD_ORDER:
            value = (paper.idea_fields or {}).get(field_name, "")
            lines.append(f"{field_name}: {value}")
        lines.append("")
        lines.append(f"[ABSTRACT] {(paper.abstract or '').strip()}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run(
    llm: LLMClient,
    graph: ResearchGraph,
    topic: str,
    *,
    serialized_graph: str | None = None,
) -> str:
    log.info("Step 3: trend summarization")
    graph_text = serialized_graph or render_graph_markdown(graph, topic)
    prompt = TREND_SUMMARY.format(
        research_topic=topic,
        serialized_graph=graph_text,
    )
    trend = llm.chat(prompt, temperature=0.3)
    log.info("  trend summary: %d chars", len(trend))
    return trend
