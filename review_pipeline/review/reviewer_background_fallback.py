from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ensure_directory, first_non_empty, load_env_values, normalize_whitespace, write_json
from .evaluation import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME, DEFAULT_LLM_TEMPERATURE, JsonLLMClient, ReviewGenerationConfig
from .persona import DEFAULT_PERSONA_JSON_PATH, DEFAULT_PERSONA_SUBJECT, select_random_persona


@dataclass(slots=True)
class ReviewerBackgroundFallbackConfig:
    idea_text: str
    env_path: Path | None = None
    llm_api_key: str | None = None
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    persona_subject: str = DEFAULT_PERSONA_SUBJECT
    persona_prompts_path: Path | None = None
    max_retries: int = 2


def _resolve_config(config: ReviewerBackgroundFallbackConfig) -> ReviewerBackgroundFallbackConfig:
    env_values = load_env_values(config.env_path)
    return ReviewerBackgroundFallbackConfig(
        idea_text=normalize_whitespace(config.idea_text),
        env_path=Path(config.env_path).expanduser().resolve() if config.env_path else None,
        llm_api_key=first_non_empty(
            config.llm_api_key,
            env_values.get("DMX-API-KEY"),
            env_values.get("DMX_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
        )
        or None,
        llm_base_url=first_non_empty(config.llm_base_url, env_values.get("OPENAI_BASE_URL"), DEFAULT_LLM_BASE_URL),
        llm_model_name=first_non_empty(config.llm_model_name, DEFAULT_LLM_MODEL_NAME),
        llm_temperature=float(config.llm_temperature),
        persona_subject=first_non_empty(
            config.persona_subject,
            env_values.get("REVIEWER_PERSONA_SUBJECT"),
            DEFAULT_PERSONA_SUBJECT,
        ),
        persona_prompts_path=(
            Path(config.persona_prompts_path).expanduser().resolve()
            if config.persona_prompts_path
            else DEFAULT_PERSONA_JSON_PATH
        ),
        max_retries=max(1, int(config.max_retries)),
    )


def _build_examples(successful_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for summary in successful_summaries:
        examples.append(
            {
                "author_name": normalize_whitespace(summary.get("author_name")),
                "overall_academic_profile": normalize_whitespace(summary.get("overall_academic_profile")),
                "relevant_research_trajectory": (
                    summary.get("relevant_research_trajectory")
                    if isinstance(summary.get("relevant_research_trajectory"), list)
                    else []
                ),
                "technical_arsenal": summary.get("technical_arsenal") if isinstance(summary.get("technical_arsenal"), list) else [],
            }
        )
    return examples


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reviewers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author_name": {"type": "string"},
                        "overall_academic_profile": {"type": "string"},
                        "relevant_research_trajectory": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "research_theme": {"type": "string"},
                                    "active_years": {"type": "string"},
                                },
                                "required": ["research_theme", "active_years"],
                                "additionalProperties": False,
                            },
                        },
                        "technical_arsenal": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "author_name",
                        "overall_academic_profile",
                        "relevant_research_trajectory",
                        "technical_arsenal",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviewers"],
        "additionalProperties": False,
    }


def _build_prompt(*, idea_text: str, successful_summaries: list[dict[str, Any]], missing_author_names: list[str]) -> str:
    grounded_examples = _build_examples(successful_summaries)
    return f"""
You are designing academic reviewer backgrounds for a simulated review board.

Research idea:
{idea_text}

Below are examples of reviewer academic backgrounds already available for this board:
{json.dumps(grounded_examples, ensure_ascii=False, indent=2)}

Now create academic backgrounds for the following reviewers:
{json.dumps(missing_author_names, ensure_ascii=False)}

Requirements:
- Output one reviewer background for each requested author name.
- Preserve diversity. The generated reviewer backgrounds must differ clearly from one another and from the provided examples.
- Keep each profile academically plausible and relevant to evaluating the research idea.
- Focus only on academic background, research trajectory, and technical arsenal.
- Do not include persona, personality, style notes, caveats, or provenance.
- Do not mention missing data, fallback generation, or uncertainty.
- Write concrete, specific academic summaries rather than generic machine-learning boilerplate.
- Ensure the set of generated reviewers covers different angles whenever possible, such as methods, applications, evaluation, systems, imaging, annotation workflow, optimization, or domain science.
""".strip()


def generate_fallback_reviewer_backgrounds(
    *,
    config: ReviewerBackgroundFallbackConfig,
    successful_summaries: list[dict[str, Any]],
    missing_author_names: list[str],
) -> dict[str, dict[str, Any]]:
    resolved = _resolve_config(config)
    if not missing_author_names:
        return {}
    if not resolved.llm_api_key:
        raise ValueError("LLM API key is required for reviewer background fallback generation")

    client = JsonLLMClient(
        ReviewGenerationConfig(
            env_path=resolved.env_path,
            llm_api_key=resolved.llm_api_key,
            llm_base_url=resolved.llm_base_url,
            llm_model_name=resolved.llm_model_name,
            llm_temperature=resolved.llm_temperature,
            max_retries=resolved.max_retries,
        )
    )
    payload = client.generate_json(
        system_prompt="You generate structured academic reviewer backgrounds as valid JSON.",
        user_prompt=_build_prompt(
            idea_text=resolved.idea_text,
            successful_summaries=successful_summaries,
            missing_author_names=missing_author_names,
        ),
        schema=_schema(),
    )
    raw_reviewers = payload.get("reviewers", [])
    if not isinstance(raw_reviewers, list):
        raw_reviewers = []

    generated_by_name: dict[str, dict[str, Any]] = {}
    for item in raw_reviewers:
        if not isinstance(item, dict):
            continue
        author_name = normalize_whitespace(item.get("author_name"))
        if not author_name:
            continue
        generated_by_name[author_name.casefold()] = {
            "author_name": author_name,
            "target_idea": resolved.idea_text,
            "overall_academic_profile": normalize_whitespace(item.get("overall_academic_profile")),
            "relevant_research_trajectory": (
                item.get("relevant_research_trajectory") if isinstance(item.get("relevant_research_trajectory"), list) else []
            ),
            "technical_arsenal": item.get("technical_arsenal") if isinstance(item.get("technical_arsenal"), list) else [],
            "persona": select_random_persona(
                persona_json_path=resolved.persona_prompts_path or DEFAULT_PERSONA_JSON_PATH,
                subject=resolved.persona_subject,
            ),
        }

    missing_keys = [name for name in missing_author_names if name.casefold() not in generated_by_name]
    if missing_keys:
        raise ValueError(f"Fallback generator did not return all requested reviewers: {missing_keys}")
    return generated_by_name


def write_fallback_summary(*, reviewer_dir: Path, author_name: str, summary_payload: dict[str, Any]) -> str:
    target_dir = ensure_directory(reviewer_dir / f"merged_{author_name.replace(' ', '_')}")
    summary_path = target_dir / "author_summary.json"
    write_json(summary_path, summary_payload)
    return str(summary_path.resolve())
