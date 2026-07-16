from __future__ import annotations

from .config import SearchConfig
from .models import CandidatePaper
from .neo4j_repository import Neo4jSearchRepository
from .reranker import EmbeddingReranker
from .text_utils import combine_embedding_scores


def recall_from_embeddings(
    query_text: str,
    query_vector: list[float],
    config: SearchConfig,
    repository: Neo4jSearchRepository,
    reranker: EmbeddingReranker | None = None,
) -> dict[str, CandidatePaper]:
    title_rows = repository.vector_search_papers(
        index_name="paper_title_embedding_idx",
        query_vector=query_vector,
        top_k=config.topk_title_vector,
        after_year=config.after_year,
        before_year=config.before_year,
    )
    if config.enable_embedding_rerank and reranker is not None:
        title_rows = reranker.rerank(
            query_text,
            title_rows,
            text_field="title",
            top_k=config.topk_title_rerank,
        )
    abstract_rows = repository.vector_search_papers(
        index_name="paper_abstract_embedding_idx",
        query_vector=query_vector,
        top_k=config.topk_abstract_vector,
        after_year=config.after_year,
        before_year=config.before_year,
    )
    if config.enable_embedding_rerank and reranker is not None:
        abstract_rows = reranker.rerank(
            query_text,
            abstract_rows,
            text_field="abstract",
            top_k=config.topk_abstract_rerank,
        )

    merged: dict[str, dict[str, float | int | str | None]] = {}
    for row in title_rows:
        merged[row["paper_id"]] = {
            "paper_id": row["paper_id"],
            "title": row.get("title") or "",
            "abstract": row.get("abstract"),
            "publication_year": row.get("publication_year"),
            "cited_by_count": int(row.get("cited_by_count") or 0),
            "sim_title": float(row.get("score") or 0.0),
            "sim_abstract": None,
            "title_rerank_score": _coerce_optional_float(row.get("rerank_score")),
            "abstract_rerank_score": None,
        }
    for row in abstract_rows:
        item = merged.setdefault(
            row["paper_id"],
            {
                "paper_id": row["paper_id"],
                "title": row.get("title") or "",
                "abstract": row.get("abstract"),
                "publication_year": row.get("publication_year"),
                "cited_by_count": int(row.get("cited_by_count") or 0),
                "sim_title": None,
                "sim_abstract": None,
                "title_rerank_score": None,
                "abstract_rerank_score": None,
            },
        )
        item["title"] = item["title"] or row.get("title") or ""
        item["abstract"] = item["abstract"] or row.get("abstract")
        item["publication_year"] = item["publication_year"] or row.get("publication_year")
        item["cited_by_count"] = max(int(item["cited_by_count"] or 0), int(row.get("cited_by_count") or 0))
        item["sim_abstract"] = float(row.get("score") or 0.0)
        item["abstract_rerank_score"] = _coerce_optional_float(row.get("rerank_score"))

    results: dict[str, CandidatePaper] = {}
    for paper_id, item in merged.items():
        sim_title = item.get("sim_title")
        sim_abstract = item.get("sim_abstract")
        score = combine_embedding_scores(sim_title, sim_abstract)
        candidate = CandidatePaper(
            paper_id=paper_id,
            title=str(item["title"] or ""),
            abstract=item.get("abstract"),
            publication_year=item.get("publication_year"),
            cited_by_count=int(item.get("cited_by_count") or 0),
            score_emb_path=score,
            score_breakdown={
                "sim_title": sim_title,
                "sim_abstract": sim_abstract,
                "title_vector_rerank_score": item.get("title_rerank_score"),
                "abstract_vector_rerank_score": item.get("abstract_rerank_score"),
            },
        )
        if sim_title is not None:
            candidate.hit_sources.add("title_vector")
        if sim_abstract is not None:
            candidate.hit_sources.add("abstract_vector")
        results[paper_id] = candidate
    return results


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
