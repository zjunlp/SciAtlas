from __future__ import annotations

from .config import SearchConfig
from .encoder import QueryEncoder
from .models import CandidatePaper
from .neo4j_repository import Neo4jSearchRepository
from .text_utils import combine_embedding_scores, min_max_normalize


def fuse_candidates(
    embedding_candidates: dict[str, CandidatePaper],
    title_candidates: dict[str, CandidatePaper],
) -> dict[str, CandidatePaper]:
    merged: dict[str, CandidatePaper] = {}

    def upsert(candidate: CandidatePaper) -> CandidatePaper:
        existing = merged.get(candidate.paper_id)
        if existing is None:
            merged[candidate.paper_id] = candidate
            return candidate
        existing.merge_metadata(candidate)
        existing.score_emb_path = max(existing.score_emb_path, candidate.score_emb_path)
        existing.score_title_path = max(existing.score_title_path, candidate.score_title_path)
        existing.hit_sources.update(candidate.hit_sources)
        existing.title_evidence.extend(candidate.title_evidence)
        existing.score_breakdown.update(candidate.score_breakdown)
        return existing

    for source in (embedding_candidates, title_candidates):
        for candidate in source.values():
            upsert(candidate)

    return merged


def refresh_pre_graph_scores(
    candidates: dict[str, CandidatePaper],
    config: SearchConfig,
    repository: Neo4jSearchRepository,
    query_vector: list[float],
    encoder: QueryEncoder,
) -> None:
    if not candidates:
        return
    embeddings_by_paper_id = repository.fetch_paper_embeddings(list(candidates))
    fallback_title_ids: list[str] = []
    fallback_abstract_ids: list[str] = []

    for paper_id, candidate in candidates.items():
        embedding_row = embeddings_by_paper_id.get(paper_id) or {}
        sim_title = _vector_similarity(query_vector, embedding_row.get("title_embedding"))
        sim_abstract = _vector_similarity(query_vector, embedding_row.get("abstract_embedding"))
        if sim_title is None and candidate.title:
            fallback_title_ids.append(paper_id)
        if sim_abstract is None and candidate.abstract:
            fallback_abstract_ids.append(paper_id)
        candidate.score_breakdown["sim_title"] = sim_title
        candidate.score_breakdown["sim_abstract"] = sim_abstract

    if fallback_title_ids:
        title_vectors = encoder.encode([candidates[paper_id].title for paper_id in fallback_title_ids])
        for paper_id, vector in zip(fallback_title_ids, title_vectors):
            candidates[paper_id].score_breakdown["sim_title"] = _vector_similarity(query_vector, vector)

    if fallback_abstract_ids:
        abstract_vectors = encoder.encode([candidates[paper_id].abstract or "" for paper_id in fallback_abstract_ids])
        for paper_id, vector in zip(fallback_abstract_ids, abstract_vectors):
            candidates[paper_id].score_breakdown["sim_abstract"] = _vector_similarity(query_vector, vector)

    emb_norm = min_max_normalize(
        {
            paper_id: combine_embedding_scores(
                _coerce_optional_float(candidate.score_breakdown.get("sim_title")),
                _coerce_optional_float(candidate.score_breakdown.get("sim_abstract")),
            )
            for paper_id, candidate in candidates.items()
        }
    )
    title_norm = min_max_normalize({paper_id: item.score_title_path for paper_id, item in candidates.items()})

    for paper_id, candidate in candidates.items():
        sim_title = _coerce_optional_float(candidate.score_breakdown.get("sim_title"))
        sim_abstract = _coerce_optional_float(candidate.score_breakdown.get("sim_abstract"))
        candidate.score_emb_path = combine_embedding_scores(sim_title, sim_abstract)
        title_bonus = (
            config.title_exact_pre_graph_bonus
            if any(item.match_type == "exact_normalized" for item in candidate.title_evidence)
            else (config.title_fuzzy_pre_graph_bonus if candidate.title_evidence else 0.0)
        )
        candidate.pre_graph_score = (
            config.weight_embedding_path * emb_norm.get(paper_id, 0.0)
            + config.weight_title_path * title_norm.get(paper_id, 0.0)
            + title_bonus
        )
        candidate.score_breakdown.update(
            {
                "sim_title": sim_title if sim_title is not None else 0.0,
                "sim_abstract": sim_abstract if sim_abstract is not None else 0.0,
                "norm_emb_path": emb_norm.get(paper_id, 0.0),
                "norm_title_path": title_norm.get(paper_id, 0.0),
                "title_pre_graph_bonus": title_bonus,
            }
        )


def _vector_similarity(query_vector: list[float], embedding: object) -> float | None:
    if not isinstance(embedding, list) or not embedding:
        return None
    if len(query_vector) != len(embedding):
        raise RuntimeError(
            f"embedding dim mismatch: query_dim={len(query_vector)} embedding_dim={len(embedding)}"
        )
    score = 0.0
    for idx in range(len(query_vector)):
        try:
            score += float(query_vector[idx]) * float(embedding[idx])
        except (TypeError, ValueError):
            return None
    return score


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
