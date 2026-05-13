from __future__ import annotations

from pathlib import Path

from ..core.api_client import SciScholarApiClient, load_scischolar_api_settings
from ..core.schemas import (
    TASK_AUTHOR_PROFILE,
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_RELATED_AUTHORS,
    TASK_TOPIC_TREND_REVIEW,
    SciScholarRequest,
)
from .author_profile import execute_author_profile
from .grounded_review import execute_grounded_review
from .idea_generation import execute_idea_generation
from .related_authors import execute_related_authors
from .topic_trend_review import execute_topic_trend_review


def execute_request(request: SciScholarRequest, run_dir: Path) -> dict[str, object]:
    settings = load_scischolar_api_settings(request.env_path, request.params)
    with SciScholarApiClient(settings) as client:
        if request.task_type == TASK_GROUNDED_REVIEW:
            response = execute_grounded_review(request, run_dir, client)
        elif request.task_type == TASK_TOPIC_TREND_REVIEW:
            response = execute_topic_trend_review(request, run_dir, client)
        elif request.task_type == TASK_RELATED_AUTHORS:
            response = execute_related_authors(request, run_dir, client)
        elif request.task_type == TASK_AUTHOR_PROFILE:
            response = execute_author_profile(request, run_dir, client)
        elif request.task_type == TASK_IDEA_GENERATION:
            response = execute_idea_generation(request, run_dir, client)
        else:
            raise ValueError(f"Unsupported task_type: {request.task_type}")

    response["task_type"] = request.task_type
    response["run_dir"] = str(run_dir.resolve())
    return response
