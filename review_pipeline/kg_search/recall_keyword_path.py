from __future__ import annotations

from .config import SearchConfig
from .encoder import QueryEncoder
from .models import QueryUnderstandingResult
from .neo4j_repository import Neo4jSearchRepository
from .text_utils import normalize_text


def recall_from_keywords(
    understanding: QueryUnderstandingResult,
    config: SearchConfig,
    repository: Neo4jSearchRepository,
    encoder: QueryEncoder,
) -> dict[str, float]:
    extracted_keywords = understanding.keywords[: config.max_keywords_from_llm]
    if not extracted_keywords:
        return {}

    normalized = [normalize_text(item.text) for item in extracted_keywords if normalize_text(item.text)]
    exact_rows = repository.match_keywords_exact(normalized)
    exact_by_normalized = {row["text_normalized"]: row for row in exact_rows}

    agg_keyword_match_score: dict[str, float] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for keyword in extracted_keywords:
        normalized_text = normalize_text(keyword.text)
        exact_row = exact_by_normalized.get(normalized_text)
        if exact_row is not None:
            pair = (keyword.text, exact_row["id"])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                agg_keyword_match_score[exact_row["id"]] = max(
                    agg_keyword_match_score.get(exact_row["id"], 0.0),
                    keyword.normalized_score,
                )

        vector = encoder.paper_query_vector(keyword.text)
        vector_rows = repository.vector_search_keywords(
            query_vector=vector,
            top_k=config.top_kg_keywords_per_kw,
        )
        for row in vector_rows:
            score = float(row.get("score") or 0.0)
            if score < config.kw_vector_threshold:
                continue
            pair = (keyword.text, row["id"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            agg_keyword_match_score[row["id"]] = max(
                agg_keyword_match_score.get(row["id"], 0.0),
                keyword.normalized_score * score,
            )

    return agg_keyword_match_score
