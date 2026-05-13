from __future__ import annotations

import os
from dataclasses import dataclass


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _is_placeholder(value: str | None) -> bool:
    normalized = _clean(value).lower()
    placeholder_tokens = (
        "your-openai-compatible-endpoint",
        "your-chat-completions-endpoint",
        "your-llm-provider.example",
        "your-provider-or-gateway.example",
    )
    return (not normalized) or normalized in {
        "replace-me",
        "your-token",
        "your-api-key",
        "your-personal-scischolar-token",
        "your-model-name",
    } or any(token in normalized for token in placeholder_tokens)


@dataclass(frozen=True)
class SciScholarConfig:
    base_url: str = "http://scinet.openkg.cn"
    api_key: str = ""
    timeout: int = 900

    llm_provider: str = "chat_completions"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_chat_completions_url: str = ""
    llm_model: str = ""
    llm_auth_header: str = ""
    llm_http_headers: str = ""
    llm_timeout: int = 30
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512

    openalex_api_key: str = ""
    openalex_mailto: str = ""
    grobid_base_url: str = ""

    @property
    def llm_configured(self) -> bool:
        endpoint = self.llm_chat_completions_url or self.llm_base_url
        return not (_is_placeholder(endpoint) or _is_placeholder(self.llm_model))

    @property
    def openalex_configured(self) -> bool:
        return bool(self.openalex_api_key.strip())

    def missing_required_config(self, *, require_grobid: bool = False) -> list[str]:
        missing: list[str] = []
        if _is_placeholder(self.api_key):
            missing.append("SCISCHOLAR_API_KEY")
        if require_grobid and _is_placeholder(self.grobid_base_url):
            missing.append("GROBID_BASE_URL")
        return missing


def load_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    llm_chat_completions_url: str | None = None,
    llm_model: str | None = None,
    llm_auth_header: str | None = None,
    llm_http_headers: str | None = None,
    openalex_api_key: str | None = None,
    openalex_mailto: str | None = None,
    grobid_base_url: str | None = None,
) -> SciScholarConfig:
    return SciScholarConfig(
        base_url=(base_url or _env_first("SCISCHOLAR_API_BASE_URL", "KG2API_BASE_URL") or "http://scinet.openkg.cn").rstrip("/"),
        api_key=_clean(api_key or _env_first("SCISCHOLAR_API_KEY", "KG2API_API_KEY")),
        timeout=int(timeout or _env_first("SCISCHOLAR_TIMEOUT", "SCISCHOLAR_API_TIMEOUT_DEFAULT") or 900),
        llm_provider=_clean(llm_provider or _env_first("LLM_PROVIDER", "SCISCHOLAR_LLM_PROVIDER") or "chat_completions"),
        llm_api_key=_clean(llm_api_key or _env_first("LLM_API_KEY", "SCISCHOLAR_LLM_API_KEY", "OPENAI_API_KEY")),
        llm_base_url=_clean(llm_base_url or _env_first("LLM_BASE_URL", "SCISCHOLAR_LLM_BASE_URL", "OPENAI_BASE_URL")),
        llm_chat_completions_url=_clean(llm_chat_completions_url or _env_first("LLM_CHAT_COMPLETIONS_URL", "SCISCHOLAR_LLM_CHAT_COMPLETIONS_URL", "OPENAI_CHAT_COMPLETIONS_URL")),
        llm_model=_clean(llm_model or _env_first("LLM_MODEL", "SCISCHOLAR_LLM_MODEL", "OPENAI_MODEL")),
        llm_auth_header=_clean(llm_auth_header or _env_first("LLM_AUTH_HEADER", "SCISCHOLAR_LLM_AUTH_HEADER")),
        llm_http_headers=_clean(llm_http_headers or _env_first("LLM_HTTP_HEADERS", "SCISCHOLAR_LLM_HTTP_HEADERS")),
        llm_timeout=int(_env_first("SCISCHOLAR_LLM_TIMEOUT", "LLM_TIMEOUT") or 30),
        llm_temperature=float(_env_first("SCISCHOLAR_LLM_TEMPERATURE", "LLM_TEMPERATURE") or 0),
        llm_max_tokens=int(_env_first("SCISCHOLAR_LLM_MAX_TOKENS", "LLM_MAX_TOKENS") or 512),
        openalex_api_key=_clean(openalex_api_key or _env_first("OA_API_KEY", "OPENALEX_API_KEY", "SCISCHOLAR_OPENALEX_API_KEY")),
        openalex_mailto=_clean(openalex_mailto or _env_first("OPENALEX_MAILTO", "OPENALEX_EMAIL", "SCISCHOLAR_OPENALEX_MAILTO")),
        grobid_base_url=_clean(grobid_base_url or _env_first("GROBID_BASE_URL", "SCISCHOLAR_GROBID_BASE_URL")),
    )
