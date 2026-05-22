from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.common import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_NAME,
    first_non_empty,
    get_env_value,
    load_env_values,
    normalize_whitespace,
)


DEFAULT_LLM_PROVIDER = "openai_compatible"


@dataclass(frozen=True, slots=True)
class LLMSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0


def normalize_openai_base_url(value: Any) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    if text.endswith("/chat/completions"):
        text = text[: -len("/chat/completions")]
    return text.rstrip("/")


def _first_param_value(params: Mapping[str, Any], keys: Sequence[str]) -> str:
    return first_non_empty(*(params.get(key) for key in keys))


def resolve_llm_settings(
    env_path: Path,
    params: Mapping[str, Any] | None = None,
    *,
    require_api_key: bool = True,
    provider_keys: Sequence[str] = ("llm_provider",),
    api_key_keys: Sequence[str] = ("llm_api_key",),
    base_url_keys: Sequence[str] = ("llm_base_url", "llm_api_url"),
    model_keys: Sequence[str] = ("llm_model_name", "llm_model"),
    timeout_keys: Sequence[str] = ("llm_timeout", "llm_timeout_s"),
) -> LLMSettings:
    params = params or {}
    env_values = load_env_values(env_path)

    provider = (
        _first_param_value(params, provider_keys)
        or get_env_value(env_values, "LLM_PROVIDER")
        or DEFAULT_LLM_PROVIDER
    )
    if provider != DEFAULT_LLM_PROVIDER:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Currently implemented providers: {DEFAULT_LLM_PROVIDER!r}."
        )

    api_key = _first_param_value(params, api_key_keys) or get_env_value(
        env_values,
        "LLM_API_KEY",
        "OPENAI_API_KEY",
    )
    if require_api_key and not api_key:
        raise ValueError(
            f"Missing LLM API key in {env_path}. "
            "Set LLM_API_KEY (preferred) or OPENAI_API_KEY (legacy)."
        )

    raw_base_url = _first_param_value(params, base_url_keys) or get_env_value(
        env_values,
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
    )
    base_url = normalize_openai_base_url(raw_base_url or DEFAULT_LLM_BASE_URL)

    model = _first_param_value(params, model_keys) or get_env_value(
        env_values,
        "LLM_MODEL",
        "OPENAI_MODEL",
    )
    if not model:
        model = DEFAULT_LLM_MODEL_NAME

    timeout_value = _first_param_value(params, timeout_keys)
    if timeout_value:
        try:
            timeout = float(timeout_value)
        except ValueError as exc:
            raise ValueError(f"Invalid LLM timeout: {timeout_value!r}") from exc
    else:
        timeout = 60.0

    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
