#!/usr/bin/env python3
"""Explore missing literature-review regions in KG embedding space.

This is an offline experiment script. It consumes an existing lr_search
search_result.json, maps retrieved papers back to KG Paper nodes, builds a broad
topic universe from Neo4j vector indexes, and returns dense/high-novelty regions
that may correspond to missing method families.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN


LR_SEARCH_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = LR_SEARCH_DIR.parent
INNOEVAL_DIR = DOWNSTREAM_DIR.parent

for extra_path in (INNOEVAL_DIR, INNOEVAL_DIR / "search"):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from kg_search.config import SearchConfig  # noqa: E402
from kg_search.encoder import QueryEncoder  # noqa: E402
from kg_search.neo4j_repository import Neo4jSearchRepository  # noqa: E402
from kg_search.text_utils import normalize_title_exact, title_similarity  # noqa: E402


METHOD_RELEVANCE_TERMS = {
    "agent",
    "agents",
    "benchmark",
    "benchmarks",
    "chain",
    "cot",
    "deliberation",
    "inference",
    "logic",
    "logical",
    "math",
    "mathematical",
    "planning",
    "proof",
    "reason",
    "reasoning",
    "reflection",
    "refinement",
    "search",
    "solver",
    "symbolic",
    "system",
    "thought",
    "tool",
    "verification",
}


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def truncate_text(value: Any, max_chars: int) -> str:
    text = normalize_whitespace(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def stable_key_identifier(stable_key: str, prefix: str) -> str:
    text = normalize_whitespace(stable_key)
    if text.casefold().startswith(prefix):
        return text.split(":", 1)[1]
    return ""


def doi_candidates(card: dict[str, Any]) -> list[str]:
    values = [
        normalize_whitespace(card.get("doi")),
        stable_key_identifier(normalize_whitespace(card.get("stable_key")), "doi:"),
    ]
    cleaned = []
    for value in values:
        if not value:
            continue
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
        value = value.casefold()
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def openalex_candidates(card: dict[str, Any]) -> list[str]:
    values = [stable_key_identifier(normalize_whitespace(card.get("stable_key")), "openalex:")]
    url = normalize_whitespace(card.get("url"))
    if "openalex.org/" in url:
        values.append(url)
    cleaned = []
    for value in values:
        if not value:
            continue
        match = re.search(r"(?:https?://openalex\.org/)?(W\d+)", value, flags=re.IGNORECASE)
        if match:
            normalized = f"https://openalex.org/{match.group(1).upper()}"
            if normalized not in cleaned:
                cleaned.append(normalized)
    return cleaned


def arxiv_candidates(card: dict[str, Any]) -> list[str]:
    values = [
        stable_key_identifier(normalize_whitespace(card.get("stable_key")), "arxiv:"),
        normalize_whitespace(card.get("doi")),
        normalize_whitespace(card.get("url")),
    ]
    cleaned = []
    for value in values:
        match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", value)
        if match:
            arxiv_id = match.group(1)
            doi = f"10.48550/arxiv.{arxiv_id}"
            if doi not in cleaned:
                cleaned.append(doi)
    return cleaned


def best_title_row(
    repository: Neo4jSearchRepository,
    title: str,
    *,
    year: int | None,
    top_k: int,
    fuzzy_threshold: float,
) -> dict[str, Any] | None:
    title_text = normalize_whitespace(title)
    if not title_text:
        return None

    exact_rows = repository.match_papers_by_normalized_title(
        normalize_title_exact(title_text),
        top_k=top_k,
        after_year=year - 1 if year else None,
        before_year=year + 1 if year else None,
    )
    if not exact_rows and year:
        exact_rows = repository.match_papers_by_normalized_title(
            normalize_title_exact(title_text),
            top_k=top_k,
        )
    for row in exact_rows:
        matched_title = normalize_whitespace(row.get("title"))
        if normalize_title_exact(matched_title) == normalize_title_exact(title_text):
            return {**row, "match_type": "exact_normalized_title", "match_score": 1.0}

    ft_rows = repository.fulltext_search_papers(
        "paper_title_ft",
        title_text,
        top_k=top_k,
        after_year=year - 1 if year else None,
        before_year=year + 1 if year else None,
    )
    if not ft_rows and year:
        ft_rows = repository.fulltext_search_papers("paper_title_ft", title_text, top_k=top_k)

    best: dict[str, Any] | None = None
    best_score = 0.0
    for row in ft_rows:
        score = title_similarity(title_text, normalize_whitespace(row.get("title")))
        if score > best_score:
            best = row
            best_score = score
    if best is None or best_score < fuzzy_threshold:
        return None
    return {**best, "match_type": "fuzzy_title", "match_score": round(best_score, 4)}


def map_search_papers(
    repository: Neo4jSearchRepository,
    cards: list[dict[str, Any]],
    *,
    title_top_k: int,
    fuzzy_threshold: float,
    max_fuzzy_fallbacks: int,
    enable_doi_match: bool,
) -> list[dict[str, Any]]:
    direct_rows: dict[str, dict[str, Any]] = {}

    openalex_lookup: dict[str, list[str]] = {}
    doi_lookup: dict[str, list[str]] = {}
    title_lookup: dict[str, list[str]] = {}
    for card in cards:
        paper_id = normalize_whitespace(card.get("paper_id"))
        for value in openalex_candidates(card):
            openalex_lookup.setdefault(value, []).append(paper_id)
        if enable_doi_match:
            for value in sorted(set(doi_candidates(card) + arxiv_candidates(card))):
                doi_lookup.setdefault(value, []).append(paper_id)
        title_key = normalize_title_exact(normalize_whitespace(card.get("title")))
        if title_key:
            title_lookup.setdefault(title_key, []).append(paper_id)

    if openalex_lookup:
        rows = repository.run(
            """
            MATCH (p:Paper)
            WHERE p.id IN $ids
            RETURN p.id AS paper_id, p.title AS title, p.publication_year AS publication_year,
                   coalesce(p.cited_by_count, 0) AS cited_by_count, 1.0 AS score
            """,
            ids=list(openalex_lookup),
        )
        for row in rows:
            for paper_id in openalex_lookup.get(row["paper_id"], []):
                direct_rows.setdefault(
                    paper_id,
                    {**row, "match_type": "openalex_id", "match_score": 1.0},
                )

    if doi_lookup:
        rows = repository.run(
            """
            MATCH (p:Paper)
            WHERE toLower(p.doi) IN $dois
            RETURN toLower(p.doi) AS doi, p.id AS paper_id, p.title AS title,
                   p.publication_year AS publication_year, coalesce(p.cited_by_count, 0) AS cited_by_count,
                   1.0 AS score
            ORDER BY cited_by_count DESC
            """,
            dois=list(doi_lookup),
        )
        for row in rows:
            for paper_id in doi_lookup.get(row["doi"], []):
                direct_rows.setdefault(
                    paper_id,
                    {**row, "match_type": "doi", "match_score": 1.0},
                )

    if title_lookup:
        rows = repository.run(
            """
            MATCH (p:Paper)
            WHERE p.title_normalized IN $title_keys
            RETURN p.title_normalized AS title_key, p.id AS paper_id, p.title AS title,
                   p.publication_year AS publication_year, coalesce(p.cited_by_count, 0) AS cited_by_count,
                   1.0 AS score
            ORDER BY cited_by_count DESC
            """,
            title_keys=list(title_lookup),
        )
        seen_title_keys: set[str] = set()
        for row in rows:
            title_key = normalize_whitespace(row.get("title_key"))
            if title_key in seen_title_keys:
                continue
            seen_title_keys.add(title_key)
            for paper_id in title_lookup.get(title_key, []):
                direct_rows.setdefault(
                    paper_id,
                    {**row, "match_type": "exact_normalized_title", "match_score": 1.0},
                )

    mapped = []
    fuzzy_fallback_count = 0
    for card in cards:
        paper_id = normalize_whitespace(card.get("paper_id"))
        title = normalize_whitespace(card.get("title"))
        year = card.get("year") if isinstance(card.get("year"), int) else None

        row: dict[str, Any] | None = direct_rows.get(paper_id)
        match_type = normalize_whitespace(row.get("match_type") if row else "")
        match_score = float(row.get("match_score") or 0.0) if row else 0.0

        if row is None and fuzzy_fallback_count < max_fuzzy_fallbacks:
            fuzzy_fallback_count += 1
            title_row = best_title_row(
                repository,
                title,
                year=year,
                top_k=title_top_k,
                fuzzy_threshold=fuzzy_threshold,
            )
            if title_row is not None:
                row = title_row
                match_type = normalize_whitespace(title_row.get("match_type"))
                match_score = float(title_row.get("match_score") or 0.0)

        mapped.append(
            {
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "source": card.get("source", []),
                "stable_key": card.get("stable_key"),
                "kg_paper_id": row.get("paper_id") if row else None,
                "kg_title": row.get("title") if row else None,
                "kg_year": row.get("publication_year") if row else None,
                "match_type": match_type or "unmatched",
                "match_score": round(match_score, 4),
                "fuzzy_fallback_attempted": match_type == "fuzzy_title",
            }
        )
    return mapped


def merge_universe_rows(rows_by_source: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for source, rows in rows_by_source:
        for rank, row in enumerate(rows, start=1):
            paper_id = normalize_whitespace(row.get("paper_id"))
            if not paper_id:
                continue
            item = by_id.setdefault(
                paper_id,
                {
                    "kg_paper_id": paper_id,
                    "title": row.get("title"),
                    "abstract": row.get("abstract"),
                    "year": row.get("publication_year"),
                    "citation_count": row.get("cited_by_count") or 0,
                    "topic_scores": {},
                    "topic_ranks": {},
                },
            )
            score = float(row.get("score") or 0.0)
            item["topic_scores"][source] = max(score, item["topic_scores"].get(source, 0.0))
            item["topic_ranks"][source] = min(rank, item["topic_ranks"].get(source, rank))
            if not item.get("abstract") and row.get("abstract"):
                item["abstract"] = row.get("abstract")
    for item in by_id.values():
        item["topic_score"] = max(item["topic_scores"].values()) if item["topic_scores"] else 0.0
    return sorted(by_id.values(), key=lambda item: (-item["topic_score"], -(item.get("citation_count") or 0)))


def fetch_topic_keyword_metadata(repository: Neo4jSearchRepository, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not paper_ids:
        return {}
    rows = repository.run(
        """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(topic:Topic)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(keyword:Keyword)
        RETURN
          p.id AS paper_id,
          [item IN collect(DISTINCT coalesce(topic.display_name, topic.label)) WHERE item IS NOT NULL] AS topics,
          [item IN collect(DISTINCT coalesce(keyword.text, keyword.label)) WHERE item IS NOT NULL] AS keywords
        """,
        paper_ids=paper_ids,
    )
    return {
        row["paper_id"]: {
            "topics": sorted(row.get("topics") or []),
            "keywords": sorted(row.get("keywords") or []),
        }
        for row in rows
        if row.get("paper_id")
    }


def lexical_method_relevance(item: dict[str, Any], metadata: dict[str, Any] | None = None) -> float:
    metadata = metadata or {}
    text = " ".join(
        [
            normalize_whitespace(item.get("title")),
            normalize_whitespace(item.get("abstract")),
            " ".join(normalize_whitespace(value) for value in metadata.get("topics", [])),
            " ".join(normalize_whitespace(value) for value in metadata.get("keywords", [])),
        ]
    ).casefold()
    tokens = set(re.findall(r"[a-z][a-z0-9-]{2,}", text))
    hits = tokens & METHOD_RELEVANCE_TERMS
    score = min(1.0, len(hits) / 4.0)
    if "system 2" in text or "system two" in text:
        score = max(score, 1.0)
    if "scientific" in text and "reasoning" not in text and "inference" not in text:
        score *= 0.65
    return round(score, 4)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embedding_matrix(
    paper_ids: list[str],
    embeddings: dict[str, dict[str, Any]],
    *,
    field: str,
) -> tuple[list[str], np.ndarray]:
    ids = []
    vectors = []
    for paper_id in paper_ids:
        vector = embeddings.get(paper_id, {}).get(field)
        if isinstance(vector, list) and vector:
            ids.append(paper_id)
            vectors.append([float(value) for value in vector])
    if not vectors:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, l2_normalize(np.asarray(vectors, dtype=np.float32))


def min_max(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def cluster_centroids(
    clusters: list[dict[str, Any]],
    mapped_by_search_id: dict[str, dict[str, Any]],
    embeddings: dict[str, dict[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    results = []
    for cluster in clusters:
        kg_ids = []
        for search_paper_id in as_list(cluster.get("paper_ids")):
            mapped = mapped_by_search_id.get(normalize_whitespace(search_paper_id))
            kg_id = normalize_whitespace(mapped.get("kg_paper_id") if mapped else "")
            if kg_id and kg_id in embeddings:
                kg_ids.append(kg_id)
        kg_ids = sorted(set(kg_ids))
        ids, matrix = embedding_matrix(kg_ids, embeddings, field=field)
        if matrix.shape[0] == 0:
            continue
        centroid = l2_normalize(np.mean(matrix, axis=0, keepdims=True))[0]
        similarities = matrix @ centroid
        radius_sim = float(np.quantile(similarities, 0.20)) if similarities.size else 0.0
        results.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "name": cluster.get("name"),
                "definition": cluster.get("definition"),
                "mapped_paper_count": len(ids),
                "kg_paper_ids": ids,
                "centroid": centroid,
                "radius_similarity_p20": round(radius_sim, 4),
                "mean_member_similarity": round(float(np.mean(similarities)), 4),
            }
        )
    return results


def top_terms(texts: list[str], *, limit: int = 12) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "annual",
        "americas",
        "association",
        "are",
        "as",
        "by",
        "chapter",
        "computational",
        "conference",
        "for",
        "from",
        "human",
        "in",
        "is",
        "linguistics",
        "large",
        "language",
        "llm",
        "llms",
        "model",
        "models",
        "nations",
        "of",
        "on",
        "or",
        "proceedings",
        "paper",
        "papers",
        "technologies",
        "the",
        "shaun",
        "to",
        "using",
        "volume",
        "with",
        "yang",
    }
    counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", normalize_whitespace(text).casefold())
            if token not in stop
        ]
        counts.update(tokens)
        for n in (2, 3):
            for index in range(0, max(0, len(tokens) - n + 1)):
                phrase = " ".join(tokens[index : index + n])
                if not any(part in stop for part in phrase.split()):
                    phrase_counts[phrase] += 1
    terms = [term for term, count in phrase_counts.most_common(limit) if count >= 2]
    for term, _count in counts.most_common(limit):
        if term not in terms:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms[:limit]


def build_gap_regions(
    universe: list[dict[str, Any]],
    known_cluster_data: list[dict[str, Any]],
    mapped_kg_ids: set[str],
    embeddings: dict[str, dict[str, Any]],
    kg_metadata: dict[str, dict[str, Any]],
    *,
    field: str,
    min_region_size: int,
    max_regions: int,
    region_similarity_threshold: float,
    top_seed_pool: int,
    recall_paper_count: int,
    cluster_algorithm: str,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    mmr_lambda: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    universe_ids = [item["kg_paper_id"] for item in universe if item["kg_paper_id"] in embeddings]
    universe_ids, universe_matrix = embedding_matrix(universe_ids, embeddings, field=field)
    if universe_matrix.shape[0] == 0:
        return [], {"error": "No universe embeddings available."}

    universe_by_id = {item["kg_paper_id"]: item for item in universe}
    topic_scores = np.asarray([float(universe_by_id[pid].get("topic_score") or 0.0) for pid in universe_ids])
    topic_norm = min_max(topic_scores)

    if known_cluster_data:
        centroid_matrix = np.asarray([item["centroid"] for item in known_cluster_data], dtype=np.float32)
        cluster_sims = universe_matrix @ centroid_matrix.T
        nearest_known_sim = np.max(cluster_sims, axis=1)
        nearest_known_index = np.argmax(cluster_sims, axis=1)
    else:
        nearest_known_sim = np.zeros(universe_matrix.shape[0], dtype=np.float32)
        nearest_known_index = np.zeros(universe_matrix.shape[0], dtype=np.int64)

    sim_matrix = universe_matrix @ universe_matrix.T
    neighbor_k = min(16, max(2, universe_matrix.shape[0] - 1))
    sorted_sims = np.sort(sim_matrix, axis=1)
    density = np.mean(sorted_sims[:, -(neighbor_k + 1) : -1], axis=1)

    novelty = 1.0 - nearest_known_sim
    novelty_norm = min_max(novelty)
    density_norm = min_max(density)
    method_relevance = np.asarray(
        [
            lexical_method_relevance(universe_by_id[paper_id], kg_metadata.get(paper_id))
            for paper_id in universe_ids
        ],
        dtype=np.float32,
    )
    gap_score = (
        topic_norm
        * (0.15 + 0.85 * novelty_norm)
        * (0.25 + 0.75 * density_norm)
        * (0.55 + 0.45 * method_relevance)
    )

    residual_indexes = [
        index
        for index, paper_id in enumerate(universe_ids)
        if paper_id not in mapped_kg_ids
        and topic_norm[index] >= 0.20
        and novelty_norm[index] >= 0.30
        and density_norm[index] >= 0.20
        and method_relevance[index] >= 0.25
    ]
    residual_indexes.sort(key=lambda index: float(gap_score[index]), reverse=True)
    residual_indexes = residual_indexes[:top_seed_pool]

    region_member_sets: list[list[int]] = []
    effective_algorithm = cluster_algorithm
    if cluster_algorithm in {"hdbscan", "auto"} and len(residual_indexes) >= max(2, hdbscan_min_cluster_size):
        residual_matrix = universe_matrix[residual_indexes]
        try:
            labels = HDBSCAN(
                min_cluster_size=max(2, hdbscan_min_cluster_size),
                min_samples=max(1, hdbscan_min_samples),
                metric="euclidean",
            ).fit_predict(residual_matrix)
            by_label: dict[int, list[int]] = defaultdict(list)
            for local_index, label in enumerate(labels):
                if int(label) >= 0:
                    by_label[int(label)].append(residual_indexes[local_index])
            region_member_sets = [
                sorted(members, key=lambda index: float(gap_score[index]), reverse=True)
                for members in by_label.values()
                if len(members) >= min_region_size
            ]
            region_member_sets.sort(key=lambda members: float(np.max(gap_score[members])), reverse=True)
        except Exception:
            region_member_sets = []
    if not region_member_sets and cluster_algorithm in {"greedy", "auto", "hdbscan"}:
        effective_algorithm = "greedy_fallback" if cluster_algorithm != "greedy" else "greedy"
        residual_set = set(residual_indexes)
        assigned: set[int] = set()
        for seed_index in residual_indexes:
            if seed_index in assigned:
                continue
            sims_to_seed = sim_matrix[seed_index]
            members = [
                index
                for index in residual_indexes
                if index not in assigned and sims_to_seed[index] >= region_similarity_threshold
            ]
            if len(members) < min_region_size:
                nearest = [
                    index
                    for index in np.argsort(-sims_to_seed)
                    if index in residual_set and index not in assigned
                ][:min_region_size]
                members = nearest if len(nearest) >= min_region_size else members
            if len(members) < min_region_size:
                continue
            members = sorted(members, key=lambda index: float(gap_score[index]), reverse=True)
            assigned.update(members)
            region_member_sets.append(members)

    regions = []
    paper_region: dict[str, str] = {}
    all_region_paper_rows: dict[str, dict[str, Any]] = {}
    for members in region_member_sets:
        members = sorted(members, key=lambda index: float(gap_score[index]), reverse=True)

        member_ids = [universe_ids[index] for index in members]
        member_items = [universe_by_id[paper_id] for paper_id in member_ids]
        member_matrix = universe_matrix[members]
        centroid = l2_normalize(np.mean(member_matrix, axis=0, keepdims=True))[0]
        medoid_local = int(np.argmax(member_matrix @ centroid))
        medoid_index = members[medoid_local]
        nearest_cluster = known_cluster_data[int(nearest_known_index[medoid_index])] if known_cluster_data else {}
        titles = [normalize_whitespace(item.get("title")) for item in member_items]
        abstracts = [normalize_whitespace(item.get("abstract")) for item in member_items[:8]]

        paper_rows = []
        seen_region_titles: set[str] = set()
        for index in members[: max(20, recall_paper_count)]:
            item = universe_by_id[universe_ids[index]]
            title_key = normalize_title_exact(normalize_whitespace(item.get("title")))
            if title_key and title_key in seen_region_titles:
                continue
            if title_key:
                seen_region_titles.add(title_key)
            metadata = kg_metadata.get(universe_ids[index], {})
            row = {
                "kg_paper_id": universe_ids[index],
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "year": item.get("year"),
                "citation_count": item.get("citation_count"),
                "topic_score": round(float(topic_scores[index]), 4),
                "novelty": round(float(novelty[index]), 4),
                "density": round(float(density[index]), 4),
                "method_relevance": round(float(method_relevance[index]), 4),
                "gap_score": round(float(gap_score[index]), 4),
                "nearest_known_cluster_id": nearest_cluster.get("cluster_id"),
                "nearest_known_similarity": round(float(nearest_known_sim[index]), 4),
                "kg_topics": metadata.get("topics", [])[:8],
                "kg_keywords": metadata.get("keywords", [])[:12],
            }
            paper_rows.append(row)

        terms = top_terms(titles + abstracts)
        region_id = f"G{len(regions) + 1}"
        for row in paper_rows:
            paper_region[row["kg_paper_id"]] = region_id
            all_region_paper_rows[row["kg_paper_id"]] = row
        regions.append(
            {
                "region_id": region_id,
                "paper_count": len(member_ids),
                "medoid_paper_id": universe_ids[medoid_index],
                "query_seed": " ".join(terms[:6]),
                "top_terms": terms,
                "nearest_known_cluster": {
                    "cluster_id": nearest_cluster.get("cluster_id"),
                    "name": nearest_cluster.get("name"),
                    "medoid_similarity": round(float(nearest_known_sim[medoid_index]), 4),
                },
                "scores": {
                    "mean_topic_score": round(float(np.mean(topic_scores[members])), 4),
                    "mean_novelty": round(float(np.mean(novelty[members])), 4),
                    "mean_density": round(float(np.mean(density[members])), 4),
                    "mean_method_relevance": round(float(np.mean(method_relevance[members])), 4),
                    "mean_gap_score": round(float(np.mean(gap_score[members])), 4),
                    "max_gap_score": round(float(np.max(gap_score[members])), 4),
                },
                "papers": paper_rows,
            }
        )
        regions.sort(key=lambda item: item["scores"]["max_gap_score"], reverse=True)
        if len(regions) >= max_regions:
            break

    for index in residual_indexes:
        paper_id = universe_ids[index]
        if paper_id in all_region_paper_rows:
            continue
        item = universe_by_id[paper_id]
        nearest_cluster = known_cluster_data[int(nearest_known_index[index])] if known_cluster_data else {}
        metadata = kg_metadata.get(paper_id, {})
        all_region_paper_rows[paper_id] = {
            "kg_paper_id": paper_id,
            "title": item.get("title"),
            "abstract": item.get("abstract"),
            "year": item.get("year"),
            "citation_count": item.get("citation_count"),
            "topic_score": round(float(topic_scores[index]), 4),
            "novelty": round(float(novelty[index]), 4),
            "density": round(float(density[index]), 4),
            "method_relevance": round(float(method_relevance[index]), 4),
            "gap_score": round(float(gap_score[index]), 4),
            "nearest_known_cluster_id": nearest_cluster.get("cluster_id"),
            "nearest_known_similarity": round(float(nearest_known_sim[index]), 4),
            "kg_topics": metadata.get("topics", [])[:8],
            "kg_keywords": metadata.get("keywords", [])[:12],
        }
        paper_region[paper_id] = "UNCLUSTERED"

    recall_papers = select_recall_papers(
        regions=regions,
        extra_candidate_ids=list(all_region_paper_rows),
        paper_rows=all_region_paper_rows,
        universe_ids=universe_ids,
        universe_matrix=universe_matrix,
        gap_score=gap_score,
        recall_paper_count=recall_paper_count,
        mmr_lambda=mmr_lambda,
    )
    recall_rows = []
    for paper_id in recall_papers:
        row = dict(all_region_paper_rows.get(paper_id) or {})
        if not row:
            continue
        row["region_id"] = paper_region.get(paper_id)
        recall_rows.append(row)

    diagnostics = {
        "universe_embedding_count": len(universe_ids),
        "mapped_seed_kg_count": len(mapped_kg_ids),
        "residual_candidate_count": len(residual_indexes),
        "cluster_algorithm": effective_algorithm,
        "recall_paper_count": len(recall_rows),
        "topic_score_range": [round(float(np.min(topic_scores)), 4), round(float(np.max(topic_scores)), 4)],
        "nearest_known_similarity_range": [
            round(float(np.min(nearest_known_sim)), 4),
            round(float(np.max(nearest_known_sim)), 4),
        ],
        "density_range": [round(float(np.min(density)), 4), round(float(np.max(density)), 4)],
    }
    return regions, recall_rows, diagnostics


def select_recall_papers(
    *,
    regions: list[dict[str, Any]],
    extra_candidate_ids: list[str],
    paper_rows: dict[str, dict[str, Any]],
    universe_ids: list[str],
    universe_matrix: np.ndarray,
    gap_score: np.ndarray,
    recall_paper_count: int,
    mmr_lambda: float,
) -> list[str]:
    candidate_ids = []
    for region in regions:
        for row in region.get("papers", []):
            paper_id = normalize_whitespace(row.get("kg_paper_id"))
            if paper_id and paper_id not in candidate_ids:
                candidate_ids.append(paper_id)
    for paper_id in extra_candidate_ids:
        if paper_id and paper_id not in candidate_ids:
            candidate_ids.append(paper_id)
    if not candidate_ids:
        return []

    index_by_id = {paper_id: index for index, paper_id in enumerate(universe_ids)}
    selected: list[str] = []
    selected_titles: set[str] = set()
    remaining = [paper_id for paper_id in candidate_ids if paper_id in index_by_id]
    while remaining and len(selected) < recall_paper_count:
        best_id = ""
        best_value = -1e9
        for paper_id in remaining:
            title_key = normalize_title_exact(normalize_whitespace(paper_rows.get(paper_id, {}).get("title")))
            if title_key and title_key in selected_titles:
                continue
            index = index_by_id[paper_id]
            relevance = float(gap_score[index])
            if selected:
                selected_indexes = [index_by_id[selected_id] for selected_id in selected]
                redundancy = float(np.max(universe_matrix[index] @ universe_matrix[selected_indexes].T))
            else:
                redundancy = 0.0
            value = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            if value > best_value:
                best_id = paper_id
                best_value = value
        if not best_id:
            break
        selected.append(best_id)
        title_key = normalize_title_exact(normalize_whitespace(paper_rows.get(best_id, {}).get("title")))
        if title_key:
            selected_titles.add(title_key)
        remaining.remove(best_id)
    return selected


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Embedding Gap Exploration: {payload.get('topic')}",
        "",
        "## Diagnostics",
        "",
    ]
    diag = payload.get("diagnostics", {})
    mapping = diag.get("mapping", {})
    lines.extend(
        [
            f"- Search papers: {mapping.get('search_paper_count')}",
            f"- Mapped to KG: {mapping.get('mapped_count')} ({mapping.get('mapped_ratio')})",
            f"- Topic universe papers: {diag.get('gap_search', {}).get('universe_embedding_count')}",
            f"- Residual candidates: {diag.get('gap_search', {}).get('residual_candidate_count')}",
            f"- Cluster algorithm: {diag.get('gap_search', {}).get('cluster_algorithm')}",
            f"- Recalled papers: {diag.get('gap_search', {}).get('recall_paper_count')}",
            "",
            "## Recalled Papers",
            "",
            "| Rank | Region | Year | Citations | Gap | Method | Novelty | Title |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, paper in enumerate(payload.get("recall_papers", []), start=1):
        title = normalize_whitespace(paper.get("title")).replace("|", "\\|")
        lines.append(
            f"| {rank} | {paper.get('region_id') or ''} | {paper.get('year') or ''} | "
            f"{paper.get('citation_count') or 0} | {paper.get('gap_score')} | "
            f"{paper.get('method_relevance')} | {paper.get('novelty')} | {title} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Gap Regions",
            "",
        ]
    )
    for region in payload.get("gap_regions", []):
        scores = region.get("scores", {})
        nearest = region.get("nearest_known_cluster", {})
        lines.extend(
            [
                f"### {region.get('region_id')} · {region.get('query_seed') or 'Untitled region'}",
                "",
                f"- Papers: {region.get('paper_count')}",
                f"- Mean gap/topic/novelty/density: {scores.get('mean_gap_score')} / {scores.get('mean_topic_score')} / {scores.get('mean_novelty')} / {scores.get('mean_density')}",
                f"- Nearest known cluster: {nearest.get('cluster_id')} {nearest.get('name')} (medoid sim={nearest.get('medoid_similarity')})",
                f"- Top terms: {', '.join(region.get('top_terms', [])[:10])}",
                "",
                "| Rank | Year | Citations | Gap | Method | Novelty | Title |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for rank, paper in enumerate(region.get("papers", [])[:10], start=1):
            title = normalize_whitespace(paper.get("title")).replace("|", "\\|")
            lines.append(
                f"| {rank} | {paper.get('year') or ''} | {paper.get('citation_count') or 0} | "
                f"{paper.get('gap_score')} | {paper.get('method_relevance')} | {paper.get('novelty')} | {title} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    search_result_path = Path(args.search_result).expanduser().resolve()
    search_result = read_json(search_result_path)
    topic = normalize_whitespace(args.topic or search_result.get("topic"))
    if not topic:
        profile = search_result.get("topic_profile") if isinstance(search_result.get("topic_profile"), dict) else {}
        topic = normalize_whitespace(profile.get("normalized_topic"))
    if not topic:
        raise RuntimeError("Missing topic. Pass --topic or use a search_result.json with topic metadata.")

    cards = [item for item in search_result.get("paper_cards", []) if isinstance(item, dict)]
    clusters = [item for item in search_result.get("method_clusters", []) if isinstance(item, dict)]
    config = SearchConfig()
    config.embedding_device = args.embedding_device
    log(f"[1/5] Loading encoder and embedding topic: {topic}", quiet=args.quiet)
    encoder = QueryEncoder(config.embedding_model_path, device=args.embedding_device, paper_embed_dim=config.paper_embed_dim)
    topic_vector = encoder.paper_query_vector(topic)

    with Neo4jSearchRepository(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password or "",
        database=config.neo4j_database,
    ) as repository:
        log(f"[2/5] Mapping {len(cards)} search-result papers back to KG", quiet=args.quiet)
        mapped = map_search_papers(
            repository,
            cards,
            title_top_k=args.title_match_top_k,
            fuzzy_threshold=args.title_fuzzy_threshold,
            max_fuzzy_fallbacks=args.max_fuzzy_fallbacks,
            enable_doi_match=args.enable_doi_match,
        )
        mapped_kg_ids = {
            normalize_whitespace(item.get("kg_paper_id"))
            for item in mapped
            if normalize_whitespace(item.get("kg_paper_id"))
        }
        log(f"[3/5] Retrieving topic universe from KG vectors", quiet=args.quiet)
        title_rows = repository.vector_search_papers(
            "paper_title_embedding_idx",
            topic_vector,
            top_k=args.topic_title_top_k,
            after_year=args.after_year,
            before_year=args.before_year,
        )
        abstract_rows = repository.vector_search_papers(
            "paper_abstract_embedding_idx",
            topic_vector,
            top_k=args.topic_abstract_top_k,
            after_year=args.after_year,
            before_year=args.before_year,
        )
        universe = merge_universe_rows([("title_vector", title_rows), ("abstract_vector", abstract_rows)])
        all_embedding_ids = sorted(mapped_kg_ids | {item["kg_paper_id"] for item in universe})
        log(f"[4/5] Fetching embeddings for {len(all_embedding_ids)} KG papers", quiet=args.quiet)
        embeddings = repository.fetch_paper_embeddings(all_embedding_ids)
        kg_metadata = fetch_topic_keyword_metadata(repository, [item["kg_paper_id"] for item in universe])

    log("[5/5] Building known-cluster centroids and residual gap regions", quiet=args.quiet)
    mapped_by_search_id = {item["paper_id"]: item for item in mapped}
    known_clusters = cluster_centroids(
        clusters,
        mapped_by_search_id,
        embeddings,
        field=args.embedding_field,
    )
    gap_regions, recall_papers, gap_diagnostics = build_gap_regions(
        universe,
        known_clusters,
        mapped_kg_ids,
        embeddings,
        kg_metadata,
        field=args.embedding_field,
        min_region_size=args.min_region_size,
        max_regions=args.max_regions,
        region_similarity_threshold=args.region_similarity_threshold,
        top_seed_pool=args.top_seed_pool,
        recall_paper_count=args.recall_paper_count,
        cluster_algorithm=args.cluster_algorithm,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        mmr_lambda=args.mmr_lambda,
    )

    match_counts = Counter(item["match_type"] for item in mapped)
    mapped_count = sum(1 for item in mapped if item.get("kg_paper_id"))
    payload = {
        "topic": topic,
        "source_search_result": str(search_result_path),
        "parameters": {
            "embedding_field": args.embedding_field,
            "topic_title_top_k": args.topic_title_top_k,
            "topic_abstract_top_k": args.topic_abstract_top_k,
            "title_fuzzy_threshold": args.title_fuzzy_threshold,
            "region_similarity_threshold": args.region_similarity_threshold,
            "min_region_size": args.min_region_size,
            "max_regions": args.max_regions,
            "recall_paper_count": args.recall_paper_count,
            "cluster_algorithm": args.cluster_algorithm,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "mmr_lambda": args.mmr_lambda,
        },
        "diagnostics": {
            "mapping": {
                "search_paper_count": len(cards),
                "mapped_count": mapped_count,
                "mapped_ratio": round(mapped_count / max(1, len(cards)), 4),
                "match_type_counts": dict(match_counts),
            },
            "known_cluster_count": len(known_clusters),
            "topic_universe_count": len(universe),
            "gap_search": gap_diagnostics,
        },
        "known_clusters": [
            {key: value for key, value in item.items() if key != "centroid"}
            for item in known_clusters
        ],
        "gap_regions": gap_regions,
        "recall_papers": recall_papers,
        "mapped_search_papers": mapped,
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore missing method regions in KG embedding space.")
    parser.add_argument("--search-result", required=True, help="Path to lr_search search_result.json.")
    parser.add_argument("--topic", default=None, help="Topic text used to retrieve the broad KG universe.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-md", default=None, help="Output Markdown path.")
    parser.add_argument("--embedding-device", default=None, help="Torch device for BGE encoder.")
    parser.add_argument(
        "--embedding-field",
        choices=("abstract_embedding", "title_embedding"),
        default="abstract_embedding",
        help="Paper embedding field used for centroid and gap-space calculations.",
    )
    parser.add_argument("--topic-title-top-k", type=int, default=500, help="Topic universe title-vector top-k.")
    parser.add_argument("--topic-abstract-top-k", type=int, default=1000, help="Topic universe abstract-vector top-k.")
    parser.add_argument("--after-year", type=int, default=None, help="Optional lower publication year bound.")
    parser.add_argument("--before-year", type=int, default=None, help="Optional upper publication year bound.")
    parser.add_argument("--title-match-top-k", type=int, default=10, help="Title fallback candidate count.")
    parser.add_argument("--title-fuzzy-threshold", type=float, default=0.88, help="Title fallback fuzzy threshold.")
    parser.add_argument(
        "--max-fuzzy-fallbacks",
        type=int,
        default=20,
        help="Maximum unmatched papers that may use slower title fulltext fuzzy fallback.",
    )
    parser.add_argument(
        "--enable-doi-match",
        action="store_true",
        help="Also map papers by DOI. This may be slow if Paper.doi is not indexed.",
    )
    parser.add_argument("--top-seed-pool", type=int, default=300, help="Residual candidate cap before region growing.")
    parser.add_argument("--min-region-size", type=int, default=5, help="Minimum papers in a returned gap region.")
    parser.add_argument("--max-regions", type=int, default=8, help="Maximum returned gap regions.")
    parser.add_argument("--recall-paper-count", type=int, default=50, help="Maximum paper count returned for LLM follow-up.")
    parser.add_argument(
        "--cluster-algorithm",
        choices=("auto", "hdbscan", "greedy"),
        default="auto",
        help="Residual clustering algorithm. auto tries HDBSCAN and falls back to greedy.",
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--hdbscan-min-samples", type=int, default=2, help="HDBSCAN min_samples.")
    parser.add_argument("--mmr-lambda", type=float, default=0.75, help="MMR relevance/diversity tradeoff for recall papers.")
    parser.add_argument(
        "--region-similarity-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold used to grow a residual region around a seed.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print a short summary.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = run(args)
    search_result_path = Path(args.search_result).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else search_result_path.parent / "embedding_gap_candidates.json"
    )
    output_md = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md
        else search_result_path.parent / "embedding_gap_candidates.md"
    )
    write_json(output_json, payload)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    summary = {
        "output_json": str(output_json),
        "output_md": str(output_md),
        "mapped_count": payload["diagnostics"]["mapping"]["mapped_count"],
        "search_paper_count": payload["diagnostics"]["mapping"]["search_paper_count"],
        "gap_region_count": len(payload.get("gap_regions", [])),
    }
    print(json.dumps(summary if not args.pretty else payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
