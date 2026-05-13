from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LLMSettings, resolve_llm_settings
from .openai_compatible import OpenAICompatibleLLM


def build_llm_client(
    env_path: Path,
    params: dict[str, Any],
    *,
    provider_keys: tuple[str, ...] = ("llm_provider",),
    api_key_keys: tuple[str, ...] = ("llm_api_key",),
    base_url_keys: tuple[str, ...] = ("llm_base_url", "llm_api_url"),
    model_keys: tuple[str, ...] = ("llm_model_name", "llm_model"),
    timeout_keys: tuple[str, ...] = ("llm_timeout", "llm_timeout_s"),
    use_env_proxy: bool | None = None,
) -> OpenAICompatibleLLM:
    settings = resolve_llm_settings(
        env_path,
        params,
        provider_keys=provider_keys,
        api_key_keys=api_key_keys,
        base_url_keys=base_url_keys,
        model_keys=model_keys,
        timeout_keys=timeout_keys,
    )
    return build_llm_client_from_settings(settings, use_env_proxy=use_env_proxy)


def build_llm_client_from_settings(
    settings: LLMSettings,
    *,
    use_env_proxy: bool | None = None,
) -> OpenAICompatibleLLM:
    if settings.provider == "openai_compatible":
        return OpenAICompatibleLLM(settings, use_env_proxy=use_env_proxy)
    raise ValueError(f"Unsupported LLM provider: {settings.provider!r}")


def load_llm_client(env_path: Path, params: dict[str, Any]) -> tuple[OpenAICompatibleLLM, str]:
    client = build_llm_client(env_path, params)
    return client, client.settings.model


def call_llm_json(
    *,
    env_path: Path,
    params: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    artifact_path: Path | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    client = build_llm_client(env_path, params)
    return client.chat_json(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        artifact_path=artifact_path,
    )
