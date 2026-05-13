from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import DEFAULT_ENV_PATH, DEFAULT_RUN_ROOT


TASK_GROUNDED_REVIEW = "Idea Grounding and Evaluation"
TASK_IDEA_GENERATION = "Idea Generation"
TASK_TOPIC_TREND_REVIEW = "Research Trend Predicting"
TASK_RELATED_AUTHORS = "Related Author Retrieval"
TASK_AUTHOR_PROFILE = "Researcher Background Review"

SUPPORTED_TASK_TYPES = (
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_TOPIC_TREND_REVIEW,
    TASK_RELATED_AUTHORS,
    TASK_AUTHOR_PROFILE,
)


DEFAULT_TASK_PARAMS: dict[str, dict[str, Any]] = {
    TASK_GROUNDED_REVIEW: {
        "search_api_top_k": 20,
        "search_final_top_k": 20,
        "manifest_top_k": 20,
        "grounding_final_top_k": 8,
        "dense_candidate_k": 40,
        "max_paragraphs_per_paper": 2,
        "enable_grounding_refinement": True,
        "disable_experiment_grounding": True,
        "grobid_base_url": "http://127.0.0.1:8070",
        "grounding_device": None,
        "query_provider": None,
        "query_model": None,
        "query_api_url": None,
        "embedding_model": None,
        "reranker_model": None,
        "embedding_model_path": None,
        "reranker_model_path": None,
        "query_max_tokens": 1000,
        "rerank_batch_size": 4,
        "rerank_paper_coverage": 2,
        "rerank_max_parallel": 8,
        "rerank_max_tokens": 900,
        "rerank_temperature": 0.1,
        "max_titles_from_pdf_references": 10,
    },
    TASK_TOPIC_TREND_REVIEW: {
        "search_api_top_k": 40,
        "final_paper_count_for_summary": 25,
        "abstract_char_limit": 700,
        "target_field": None,
        "after": None,
        "before": None,
        "rerank_batch_size": 4,
        "rerank_paper_coverage": 2,
        "rerank_max_parallel": 8,
        "rerank_max_tokens": 900,
        "rerank_temperature": 0.1,
    },
    TASK_RELATED_AUTHORS: {
        "author_top_k": 10,
        "author_support_top_k": 3,
        "enrich_author_support_count": 5,
        "target_field": None,
        "after": None,
        "before": None,
        "fetch_author_stats": True,
        "author_search_fallback": "id_then_name",
        "grobid_base_url": "http://127.0.0.1:8070",
        "max_titles_from_pdf_references": 10,
    },
    TASK_AUTHOR_PROFILE: {
        "author_paper_sample_size": 40,
        "recent_paper_quota": 20,
        "top_cited_quota": 20,
        "representative_paper_top_k": 10,
        "merge_same_name_authors": True,
        "dedupe_papers": True,
        "include_abstract": True,
    },
    TASK_IDEA_GENERATION: {
        "search_api_top_k": 30,
        "final_paper_count_for_summary": 25,
        "abstract_char_limit": 600,
        "idea_count": 5,
        "target_field": None,
        "after": None,
        "before": None,
        "rerank_batch_size": 4,
        "rerank_paper_coverage": 2,
        "rerank_max_parallel": 8,
        "rerank_max_tokens": 900,
        "rerank_temperature": 0.1,
    },
}


@dataclass(slots=True)
class SciScholarRequest:
    task_type: str
    input_payload: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    output_root: Path = DEFAULT_RUN_ROOT
    env_path: Path = DEFAULT_ENV_PATH
    run_id: str | None = None


def default_task_params(task_type: str) -> dict[str, Any]:
    if task_type not in DEFAULT_TASK_PARAMS:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return dict(DEFAULT_TASK_PARAMS[task_type])


def merge_task_params(task_type: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = default_task_params(task_type)
    if overrides:
        params.update(overrides)
    return params
