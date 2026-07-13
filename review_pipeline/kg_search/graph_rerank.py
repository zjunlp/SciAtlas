from __future__ import annotations

import math
from collections import defaultdict

from .config import SearchConfig
from .models import CandidatePaper, GraphEdge, GraphNode
from .neo4j_repository import Neo4jSearchRepository
from .text_utils import percentile, resolve_importance


def rerank_with_graph(
    candidates: dict[str, CandidatePaper],
    matched_keyword_scores: dict[str, float],
    config: SearchConfig,
    repository: Neo4jSearchRepository,
) -> tuple[dict[str, float], dict[str, float], dict[str, GraphNode], dict[str, list[str]]]:
    if not candidates and not matched_keyword_scores:
        return {}, {}, {}, {}

    seed_papers = _select_seed_papers(candidates, config)
    seed_keywords = sorted(
        matched_keyword_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[: config.max_seed_keywords]

    graph_nodes, graph_edges = _build_subgraph(
        seed_papers=seed_papers,
        seed_keyword_ids=[keyword_id for keyword_id, _ in seed_keywords],
        repository=repository,
        config=config,
    )
    if not graph_nodes:
        return {}, {}, {}, {}

    paper_citations = [node.cited_by_count for node in graph_nodes.values() if node.node_type == "Paper"]
    citation_p95 = percentile(paper_citations, 0.95)
    importance_by_paper = {
        node_id: resolve_importance(
            node.cited_by_count,
            citation_p95,
            uniform_importance=config.uniform_importance,
        )
        for node_id, node in graph_nodes.items()
        if node.node_type == "Paper"
    }

    seed_distribution = _build_seed_distribution(
        candidates=candidates,
        seed_paper_ids={item.paper_id for item in seed_papers},
        matched_keyword_scores=matched_keyword_scores,
        importance_by_paper=importance_by_paper,
        config=config,
    )
    seed_distribution = {node_id: score for node_id, score in seed_distribution.items() if node_id in graph_nodes}
    seed_total = sum(seed_distribution.values())
    if seed_total > 0:
        seed_distribution = {node_id: score / seed_total for node_id, score in seed_distribution.items()}
    else:
        seed_distribution = {}
    adjacency = _build_transition_graph(
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        matched_keyword_scores=matched_keyword_scores,
        config=config,
    )
    if not adjacency or not seed_distribution:
        return {}, {}, graph_nodes, {}

    if config.graph_method == "A":
        node_scores = _propagate_method_a(
            graph_nodes=graph_nodes,
            adjacency=adjacency,
            seed_distribution=seed_distribution,
            config=config,
        )
    elif config.graph_method == "B":
        node_scores = _propagate_method_b(
            graph_nodes=graph_nodes,
            adjacency=adjacency,
            seed_distribution=seed_distribution,
            config=config,
        )
    else:
        node_scores = {paper_id: item.pre_graph_score for paper_id, item in candidates.items() if item.pre_graph_score > 0}

    paper_scores = _extract_positive_paper_scores(node_scores, graph_nodes)

    explanations = _build_path_explanations(
        graph_nodes=graph_nodes,
        adjacency=adjacency,
        seed_distribution=seed_distribution,
        target_paper_ids=set(paper_scores),
        config=config,
    )
    return paper_scores, node_scores, graph_nodes, explanations


def _select_seed_papers(candidates: dict[str, CandidatePaper], config: SearchConfig) -> list[CandidatePaper]:
    positive_candidates = [item for item in candidates.values() if item.pre_graph_score > 0]
    title_hits = sorted(
        [item for item in positive_candidates if item.title_evidence],
        key=lambda item: item.pre_graph_score,
        reverse=True,
    )
    title_hit_ids = {item.paper_id for item in title_hits}
    other_candidates = sorted(
        [item for item in positive_candidates if item.paper_id not in title_hit_ids],
        key=lambda item: item.pre_graph_score,
        reverse=True,
    )
    if len(title_hits) >= config.max_seed_papers:
        return title_hits[: config.max_seed_papers]
    remaining = config.max_seed_papers - len(title_hits)
    return title_hits + other_candidates[:remaining]


def _build_subgraph(
    seed_papers: list[CandidatePaper],
    seed_keyword_ids: list[str],
    repository: Neo4jSearchRepository,
    config: SearchConfig,
) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_seen: set[tuple[str, str, str, tuple[tuple[str, object], ...]]] = set()

    for candidate in seed_papers:
        nodes[candidate.paper_id] = GraphNode(
            node_id=candidate.paper_id,
            node_type="Paper",
            title=candidate.title,
            abstract=candidate.abstract,
            publication_year=candidate.publication_year,
            cited_by_count=candidate.cited_by_count,
        )
    for keyword_id in seed_keyword_ids:
        nodes.setdefault(keyword_id, GraphNode(node_id=keyword_id, node_type="Keyword"))

    frontier = {
        "Paper": [candidate.paper_id for candidate in seed_papers],
        "Keyword": list(seed_keyword_ids),
        "Author": [],
    }
    visited = {node_id: nodes[node_id].node_type for node_id in nodes}

    for _ in range(config.graph_hops):
        batch_nodes, batch_edges = repository.fetch_neighbors(
            paper_ids=frontier["Paper"],
            keyword_ids=frontier["Keyword"],
            author_ids=frontier["Author"],
            per_query_limit=config.graph_frontier_limit_per_type,
            after_year=config.after_year,
            before_year=config.before_year,
        )
        next_frontier = {"Paper": [], "Keyword": [], "Author": []}
        for node_id, node in batch_nodes.items():
            if node_id not in nodes:
                nodes[node_id] = node
            else:
                existing = nodes[node_id]
                if not existing.title and node.title:
                    existing.title = node.title
                if not existing.text and node.text:
                    existing.text = node.text
                if not existing.abstract and node.abstract:
                    existing.abstract = node.abstract
                if existing.publication_year is None and node.publication_year is not None:
                    existing.publication_year = node.publication_year
                existing.cited_by_count = max(existing.cited_by_count, node.cited_by_count)
            if node_id not in visited:
                visited[node_id] = node.node_type
                next_frontier[node.node_type].append(node_id)
        for edge in batch_edges:
            props_key = tuple(sorted((edge.properties or {}).items()))
            key = (edge.source_id, edge.target_id, edge.edge_type, props_key)
            if key not in edge_seen:
                edge_seen.add(key)
                edges.append(edge)
        frontier = {
            node_type: node_ids[: config.graph_frontier_limit_per_type]
            for node_type, node_ids in next_frontier.items()
        }
        if not any(frontier.values()):
            break

    return nodes, edges


def _build_seed_distribution(
    candidates: dict[str, CandidatePaper],
    seed_paper_ids: set[str],
    matched_keyword_scores: dict[str, float],
    importance_by_paper: dict[str, float],
    config: SearchConfig,
) -> dict[str, float]:
    seed_weights: dict[str, float] = {}
    for paper_id, candidate in candidates.items():
        if paper_id not in seed_paper_ids:
            continue
        if candidate.pre_graph_score <= 0:
            continue
        importance = importance_by_paper.get(paper_id, candidate.importance)
        seed_weights[paper_id] = candidate.pre_graph_score * (1.0 + config.seed_gamma * importance)
    for keyword_id, score in matched_keyword_scores.items():
        if score > 0:
            seed_weights[keyword_id] = max(seed_weights.get(keyword_id, 0.0), score)
    total = sum(seed_weights.values())
    if total <= 0:
        return {}
    return {node_id: weight / total for node_id, weight in seed_weights.items()}


def _build_transition_graph(
    graph_nodes: dict[str, GraphNode],
    graph_edges: list[GraphEdge],
    matched_keyword_scores: dict[str, float],
    config: SearchConfig,
) -> dict[str, list[tuple[str, float, str]]]:
    raw_adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)

    for edge in graph_edges:
        weight = _edge_weight(
            edge=edge,
            graph_nodes=graph_nodes,
            matched_keyword_scores=matched_keyword_scores,
            config=config,
        )
        if weight <= 0:
            continue
        raw_adjacency[edge.source_id].append((edge.target_id, weight, edge.edge_type))
        raw_adjacency[edge.target_id].append((edge.source_id, weight, edge.edge_type))

    adjacency: dict[str, list[tuple[str, float, str]]] = {}
    for node_id, items in raw_adjacency.items():
        total = sum(weight for _, weight, _ in items)
        if total <= 0:
            continue
        adjacency[node_id] = [(neighbor_id, weight / total, edge_type) for neighbor_id, weight, edge_type in items]
    return adjacency


def _edge_weight(
    edge: GraphEdge,
    graph_nodes: dict[str, GraphNode],
    matched_keyword_scores: dict[str, float],
    config: SearchConfig,
) -> float:
    props = edge.properties or {}
    if edge.edge_type == "HAS_KEYWORD":
        keyword_id = edge.source_id if graph_nodes[edge.source_id].node_type == "Keyword" else edge.target_id
        relevance = float(props.get("relevance_score") or 0.0)
        keyword_signal = matched_keyword_scores.get(keyword_id, config.graph_keyword_smoothing)
        return config.base_has_keyword * keyword_signal * relevance
    if edge.edge_type == "CITES":
        return config.base_cites
    if edge.edge_type == "RELATED_TO":
        return config.base_related
    if edge.edge_type == "AUTHORED":
        return config.base_authored
    if edge.edge_type == "COAUTHOR":
        count = min(config.graph_count_cap, math.log1p(max(0, int(props.get("count") or 0))))
        return config.base_coauthor * max(1.0, count)
    if edge.edge_type == "COOCCUR":
        count = min(config.graph_count_cap, math.log1p(max(0, int(props.get("count") or 0))))
        return config.base_cooccur * max(1.0, count)
    return 0.0


def _propagate_method_a(
    graph_nodes: dict[str, GraphNode],
    adjacency: dict[str, list[tuple[str, float, str]]],
    seed_distribution: dict[str, float],
    config: SearchConfig,
) -> dict[str, float]:
    current = dict(seed_distribution)
    node_scores: dict[str, float] = defaultdict(float)
    for hop in range(1, config.graph_hops + 1):
        next_scores: dict[str, float] = defaultdict(float)
        for node_id, mass in current.items():
            for neighbor_id, prob, _ in adjacency.get(node_id, []):
                contrib = mass * prob * (config.graph_hop_decay ** hop)
                next_scores[neighbor_id] += contrib
        for node_id, score in next_scores.items():
            if score > 0 and node_id in graph_nodes:
                node_scores[node_id] += score
        current = next_scores
        if not current:
            break
    return dict(node_scores)


def _propagate_method_b(
    graph_nodes: dict[str, GraphNode],
    adjacency: dict[str, list[tuple[str, float, str]]],
    seed_distribution: dict[str, float],
    config: SearchConfig,
) -> dict[str, float]:
    node_ids = list(graph_nodes)
    rank = {node_id: seed_distribution.get(node_id, 0.0) for node_id in node_ids}
    for _ in range(config.ppr_max_iter):
        next_rank = {node_id: config.ppr_alpha * seed_distribution.get(node_id, 0.0) for node_id in node_ids}
        for node_id, score in rank.items():
            if score <= 0:
                continue
            neighbors = adjacency.get(node_id, [])
            if not neighbors:
                next_rank[node_id] += (1.0 - config.ppr_alpha) * score
                continue
            for neighbor_id, prob, _ in neighbors:
                next_rank[neighbor_id] += (1.0 - config.ppr_alpha) * score * prob
        diff = sum(abs(next_rank[node_id] - rank.get(node_id, 0.0)) for node_id in node_ids)
        rank = next_rank
        if diff < config.ppr_tol:
            break
    return {
        node_id: score
        for node_id, score in rank.items()
        if node_id in graph_nodes and score > 0
    }


def _extract_positive_paper_scores(
    node_scores: dict[str, float],
    graph_nodes: dict[str, GraphNode],
) -> dict[str, float]:
    return {
        node_id: score
        for node_id, score in node_scores.items()
        if graph_nodes.get(node_id) is not None and graph_nodes[node_id].node_type == "Paper" and score > 0
    }


def _build_path_explanations(
    graph_nodes: dict[str, GraphNode],
    adjacency: dict[str, list[tuple[str, float, str]]],
    seed_distribution: dict[str, float],
    target_paper_ids: set[str],
    config: SearchConfig,
) -> dict[str, list[str]]:
    scored_paths: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for seed_id, seed_weight in sorted(seed_distribution.items(), key=lambda item: item[1], reverse=True):
        _dfs_explain(
            current_id=seed_id,
            current_weight=seed_weight,
            path_nodes=[seed_id],
            path_labels=[graph_nodes[seed_id].display_text],
            path_edges=[],
            remaining_hops=config.graph_hops,
            graph_nodes=graph_nodes,
            adjacency=adjacency,
            target_paper_ids=target_paper_ids,
            scored_paths=scored_paths,
            config=config,
        )

    explanations: dict[str, list[str]] = {}
    for paper_id, items in scored_paths.items():
        ranked = sorted(items, key=lambda item: item[0], reverse=True)[: config.explanation_max_paths_per_paper]
        explanations[paper_id] = [text for _, text in ranked]
    return explanations


def _dfs_explain(
    current_id: str,
    current_weight: float,
    path_nodes: list[str],
    path_labels: list[str],
    path_edges: list[str],
    remaining_hops: int,
    graph_nodes: dict[str, GraphNode],
    adjacency: dict[str, list[tuple[str, float, str]]],
    target_paper_ids: set[str],
    scored_paths: dict[str, list[tuple[float, str]]],
    config: SearchConfig,
) -> None:
    if remaining_hops <= 0:
        return
    neighbors = sorted(adjacency.get(current_id, []), key=lambda item: item[1], reverse=True)[
        : config.explanation_max_neighbors_per_hop
    ]
    for neighbor_id, prob, edge_type in neighbors:
        if neighbor_id in path_nodes:
            continue
        next_weight = current_weight * prob * config.graph_hop_decay
        next_labels = path_labels + [graph_nodes[neighbor_id].display_text]
        next_edges = path_edges + [edge_type]
        if neighbor_id in target_paper_ids and graph_nodes[neighbor_id].node_type == "Paper":
            summary = _format_path(next_labels, next_edges)
            scored_paths[neighbor_id].append((next_weight, summary))
        _dfs_explain(
            current_id=neighbor_id,
            current_weight=next_weight,
            path_nodes=path_nodes + [neighbor_id],
            path_labels=next_labels,
            path_edges=next_edges,
            remaining_hops=remaining_hops - 1,
            graph_nodes=graph_nodes,
            adjacency=adjacency,
            target_paper_ids=target_paper_ids,
            scored_paths=scored_paths,
            config=config,
        )


def _format_path(labels: list[str], edge_types: list[str]) -> str:
    parts = [labels[0]]
    for edge_type, label in zip(edge_types, labels[1:]):
        parts.append(f"-[{edge_type}]-> {label}")
    return " ".join(parts)
