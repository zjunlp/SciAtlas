from __future__ import annotations

import os
from dataclasses import dataclass


def _clean(value: str | None) -> str:
    return (value or "").strip()


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
        "your-personal-scinet-token",
        "your-model-name",
    } or any(token in normalized for token in placeholder_tokens)


@dataclass(frozen=True)
class SciNetConfig:
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
            missing.append("SCINET_API_KEY")
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
) -> SciNetConfig:
    return SciNetConfig(
        base_url=(base_url or os.getenv("SCINET_API_BASE_URL") or os.getenv("KG2API_BASE_URL") or "http://scinet.openkg.cn").rstrip("/"),
        api_key=_clean(api_key or os.getenv("SCINET_API_KEY") or os.getenv("KG2API_API_KEY") or ""),
        timeout=int(timeout or os.getenv("SCINET_TIMEOUT") or os.getenv("SCINET_API_TIMEOUT_DEFAULT") or 900),
        llm_provider=_clean(llm_provider or os.getenv("LLM_PROVIDER") or os.getenv("SCINET_LLM_PROVIDER") or "chat_completions"),
        llm_api_key=_clean(llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("SCINET_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""),
        llm_base_url=_clean(llm_base_url or os.getenv("LLM_BASE_URL") or os.getenv("SCINET_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""),
        llm_chat_completions_url=_clean(llm_chat_completions_url or os.getenv("LLM_CHAT_COMPLETIONS_URL") or os.getenv("SCINET_LLM_CHAT_COMPLETIONS_URL") or os.getenv("OPENAI_CHAT_COMPLETIONS_URL") or ""),
        llm_model=_clean(llm_model or os.getenv("LLM_MODEL") or os.getenv("SCINET_LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""),
        llm_auth_header=_clean(llm_auth_header or os.getenv("LLM_AUTH_HEADER") or os.getenv("SCINET_LLM_AUTH_HEADER") or ""),
        llm_http_headers=_clean(llm_http_headers or os.getenv("LLM_HTTP_HEADERS") or os.getenv("SCINET_LLM_HTTP_HEADERS") or ""),
        llm_timeout=int(os.getenv("SCINET_LLM_TIMEOUT") or os.getenv("LLM_TIMEOUT") or 30),
        llm_temperature=float(os.getenv("SCINET_LLM_TEMPERATURE") or os.getenv("LLM_TEMPERATURE") or 0),
        llm_max_tokens=int(os.getenv("SCINET_LLM_MAX_TOKENS") or os.getenv("LLM_MAX_TOKENS") or 512),
        openalex_api_key=_clean(openalex_api_key or os.getenv("OA_API_KEY") or os.getenv("OPENALEX_API_KEY") or os.getenv("SCINET_OPENALEX_API_KEY") or ""),
        openalex_mailto=_clean(openalex_mailto or os.getenv("OPENALEX_MAILTO") or os.getenv("OPENALEX_EMAIL") or os.getenv("SCINET_OPENALEX_MAILTO") or ""),
        grobid_base_url=_clean(grobid_base_url or os.getenv("GROBID_BASE_URL") or os.getenv("SCINET_GROBID_BASE_URL") or ""),
    )
