from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..core.common import extract_json_object, write_text
from .base import LLMSettings


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings, *, use_env_proxy: bool | None = None) -> None:
        from openai import OpenAI

        self.settings = settings
        self._http_client = httpx.Client(timeout=settings.timeout, trust_env=True if use_env_proxy is None else use_env_proxy)
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            http_client=self._http_client,
        )

    def chat_text(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        request_args: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_args["max_tokens"] = max_tokens
        if response_format is not None:
            request_args["response_format"] = response_format
        response = self._client.chat.completions.create(**request_args)
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        artifact_path: Path | None = None,
    ) -> dict[str, Any]:
        content = self.chat_text(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if artifact_path is not None:
            write_text(artifact_path, content)
        return extract_json_object(content)
