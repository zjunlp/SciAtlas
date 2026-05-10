from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from .common import (
    DEFAULT_SCINET_API_AUTHORS_PAPERS_TIMEOUT,
    DEFAULT_SCINET_API_AUTHORS_RELATED_TIMEOUT,
    DEFAULT_SCINET_API_CONNECT_TIMEOUT,
    DEFAULT_SCINET_API_POOL_TIMEOUT,
    DEFAULT_SCINET_API_SEARCH_TIMEOUT,
    DEFAULT_SCINET_API_SUPPORT_PAPERS_TIMEOUT,
    DEFAULT_SCINET_API_TIMEOUT,
    DEFAULT_SCINET_API_WRITE_TIMEOUT,
    get_env_value,
    load_env_values,
    normalize_whitespace,
)


class SciNetApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


@dataclass(frozen=True)
class SciNetApiSettings:
    base_url: str
    api_key: str
    timeout: float | None = None
    default_timeout: float = DEFAULT_SCINET_API_TIMEOUT
    search_timeout: float = DEFAULT_SCINET_API_SEARCH_TIMEOUT
    authors_related_timeout: float = DEFAULT_SCINET_API_AUTHORS_RELATED_TIMEOUT
    authors_papers_timeout: float = DEFAULT_SCINET_API_AUTHORS_PAPERS_TIMEOUT
    authors_support_papers_timeout: float = DEFAULT_SCINET_API_SUPPORT_PAPERS_TIMEOUT
    connect_timeout: float = DEFAULT_SCINET_API_CONNECT_TIMEOUT
    write_timeout: float = DEFAULT_SCINET_API_WRITE_TIMEOUT
    pool_timeout: float = DEFAULT_SCINET_API_POOL_TIMEOUT

    def __post_init__(self) -> None:
        if self.timeout is not None:
            object.__setattr__(self, "default_timeout", float(self.timeout))
        else:
            object.__setattr__(self, "timeout", self.default_timeout)


def _first_timeout_value(
    *,
    overrides: dict[str, Any],
    env_values: dict[str, str],
    param_keys: tuple[str, ...],
    env_keys: tuple[str, ...],
    default: float,
) -> float:
    raw_value = ""
    for key in param_keys:
        raw_value = normalize_whitespace(overrides.get(key))
        if raw_value:
            break
    if not raw_value:
        raw_value = get_env_value(env_values, *env_keys)
    if not raw_value:
        return float(default)
    try:
        value = float(raw_value)
    except ValueError as exc:
        keys = ", ".join((*param_keys, *env_keys))
        raise ValueError(f"Invalid SciScholar API timeout {raw_value!r} for one of: {keys}") from exc
    if value <= 0:
        keys = ", ".join((*param_keys, *env_keys))
        raise ValueError(f"SciScholar API timeout must be positive for one of: {keys}")
    return value


def load_scinet_api_settings(env_path: Path, params: dict[str, Any] | None = None) -> SciNetApiSettings:
    overrides = params or {}
    env_values = load_env_values(env_path)
    base_url = normalize_whitespace(
        overrides.get("scischolar_api_base_url")
        or overrides.get("scinet_api_base_url")
        or overrides.get("scimap_api_base_url")
        or overrides.get("kg2api_base_url")
        or get_env_value(env_values, "SCISCHOLAR_API_BASE_URL", "SCINET_API_BASE_URL", "SCIMAP_API_BASE_URL", "KG2API_BASE_URL")
    )
    api_key = normalize_whitespace(
        overrides.get("scischolar_api_key")
        or overrides.get("scinet_api_key")
        or overrides.get("scimap_api_key")
        or overrides.get("kg2api_api_key")
        or get_env_value(env_values, "SCISCHOLAR_API_KEY", "SCINET_API_KEY", "SCIMAP_API_KEY", "KG2API_API_KEY")
    )
    default_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=(
            "scischolar_api_timeout_default",
            "scinet_api_timeout_default",
            "api_timeout_default",
            "scischolar_api_timeout",
            "scinet_api_timeout",
            "scimap_api_timeout",
            "kg2api_timeout",
        ),
        env_keys=("SCISCHOLAR_API_TIMEOUT_DEFAULT", "SCINET_API_TIMEOUT_DEFAULT", "SCISCHOLAR_API_TIMEOUT", "SCINET_API_TIMEOUT", "SCIMAP_API_TIMEOUT", "KG2API_TIMEOUT"),
        default=DEFAULT_SCINET_API_TIMEOUT,
    )
    search_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=("scischolar_api_timeout_search", "scinet_api_timeout_search", "api_timeout_search", "search_timeout"),
        env_keys=("SCISCHOLAR_API_TIMEOUT_SEARCH", "SCINET_API_TIMEOUT_SEARCH", "SCIMAP_API_TIMEOUT_SEARCH", "KG2API_TIMEOUT_SEARCH"),
        default=DEFAULT_SCINET_API_SEARCH_TIMEOUT,
    )
    authors_related_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=("scischolar_api_timeout_authors_related", "scinet_api_timeout_authors_related", "api_timeout_authors_related", "authors_related_timeout"),
        env_keys=(
            "SCISCHOLAR_API_TIMEOUT_AUTHORS_RELATED",
            "SCINET_API_TIMEOUT_AUTHORS_RELATED",
            "SCIMAP_API_TIMEOUT_AUTHORS_RELATED",
            "KG2API_TIMEOUT_AUTHORS_RELATED",
        ),
        default=DEFAULT_SCINET_API_AUTHORS_RELATED_TIMEOUT,
    )
    authors_papers_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=("scischolar_api_timeout_authors_papers", "scinet_api_timeout_authors_papers", "api_timeout_authors_papers", "authors_papers_timeout"),
        env_keys=(
            "SCISCHOLAR_API_TIMEOUT_AUTHORS_PAPERS",
            "SCINET_API_TIMEOUT_AUTHORS_PAPERS",
            "SCIMAP_API_TIMEOUT_AUTHORS_PAPERS",
            "KG2API_TIMEOUT_AUTHORS_PAPERS",
        ),
        default=DEFAULT_SCINET_API_AUTHORS_PAPERS_TIMEOUT,
    )
    authors_support_papers_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=(
            "scischolar_api_timeout_support_papers",
            "scinet_api_timeout_support_papers",
            "api_timeout_support_papers",
            "support_papers_timeout",
            "authors_support_papers_timeout",
        ),
        env_keys=(
            "SCISCHOLAR_API_TIMEOUT_SUPPORT_PAPERS",
            "SCINET_API_TIMEOUT_SUPPORT_PAPERS",
            "SCIMAP_API_TIMEOUT_SUPPORT_PAPERS",
            "KG2API_TIMEOUT_SUPPORT_PAPERS",
        ),
        default=DEFAULT_SCINET_API_SUPPORT_PAPERS_TIMEOUT,
    )
    connect_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=("scischolar_api_connect_timeout", "scinet_api_connect_timeout", "api_connect_timeout"),
        env_keys=("SCISCHOLAR_API_CONNECT_TIMEOUT", "SCINET_API_CONNECT_TIMEOUT", "SCIMAP_API_CONNECT_TIMEOUT", "KG2API_CONNECT_TIMEOUT"),
        default=DEFAULT_SCINET_API_CONNECT_TIMEOUT,
    )
    write_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=("scischolar_api_write_timeout", "scinet_api_write_timeout", "api_write_timeout"),
        env_keys=("SCISCHOLAR_API_WRITE_TIMEOUT", "SCINET_API_WRITE_TIMEOUT", "SCIMAP_API_WRITE_TIMEOUT", "KG2API_WRITE_TIMEOUT"),
        default=DEFAULT_SCINET_API_WRITE_TIMEOUT,
    )
    pool_timeout = _first_timeout_value(
        overrides=overrides,
        env_values=env_values,
        param_keys=("scischolar_api_pool_timeout", "scinet_api_pool_timeout", "api_pool_timeout"),
        env_keys=("SCISCHOLAR_API_POOL_TIMEOUT", "SCINET_API_POOL_TIMEOUT", "SCIMAP_API_POOL_TIMEOUT", "KG2API_POOL_TIMEOUT"),
        default=DEFAULT_SCINET_API_POOL_TIMEOUT,
    )

    if not base_url:
        raise ValueError(f"Missing SCISCHOLAR_API_BASE_URL in {env_path}")
    if not api_key:
        raise ValueError(f"Missing SCISCHOLAR_API_KEY in {env_path}")
    return SciNetApiSettings(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        default_timeout=default_timeout,
        search_timeout=search_timeout,
        authors_related_timeout=authors_related_timeout,
        authors_papers_timeout=authors_papers_timeout,
        authors_support_papers_timeout=authors_support_papers_timeout,
        connect_timeout=connect_timeout,
        write_timeout=write_timeout,
        pool_timeout=pool_timeout,
    )


class SciNetApiClient:
    def __init__(self, settings: SciNetApiSettings) -> None:
        import httpx

        self.settings = settings
        self._httpx = httpx
        self._client = httpx.Client(
            base_url=settings.base_url,
            timeout=self._make_timeout(settings.default_timeout),
            trust_env=False,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": settings.api_key,
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SciNetApiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _make_timeout(self, read_timeout: float) -> Any:
        return self._httpx.Timeout(
            connect=self.settings.connect_timeout,
            write=self.settings.write_timeout,
            pool=self.settings.pool_timeout,
            read=read_timeout,
        )

    def _read_timeout_for_path(self, path: str) -> float:
        if path == "/v1/search":
            return self.settings.search_timeout
        if path == "/v1/authors/related":
            return self.settings.authors_related_timeout
        if path == "/v1/authors/papers":
            return self.settings.authors_papers_timeout
        if path == "/v1/authors/support-papers":
            return self.settings.authors_support_papers_timeout
        return self.settings.default_timeout

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        read_timeout = self._read_timeout_for_path(path)
        started_at = monotonic()
        try:
            response = self._client.post(path, json=payload, timeout=self._make_timeout(read_timeout))
        except self._httpx.TimeoutException as exc:
            elapsed = monotonic() - started_at
            raise SciNetApiError(
                f"SciScholar API {path} timed out after {elapsed:.1f}s "
                f"(read_timeout={read_timeout:.1f}s, base_url={self.settings.base_url})"
            ) from exc
        except self._httpx.HTTPError as exc:
            elapsed = monotonic() - started_at
            message = f"SciScholar API request failed for {path} after {elapsed:.1f}s: {exc}"
            if isinstance(exc, self._httpx.RemoteProtocolError):
                message += (
                    " This can happen when the backend interrupts a long-running request; "
                    "increase the endpoint timeout and check server logs."
                )
            raise SciNetApiError(message) from exc

        raw_body = response.text
        try:
            body = response.json()
        except ValueError:
            body = None

        if response.status_code >= 400:
            detail = None
            request_id = ""
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error") or body.get("message")
                request_id = normalize_whitespace(body.get("request_id"))
            message = normalize_whitespace(detail) or raw_body or f"HTTP {response.status_code}"
            if request_id:
                message = f"{message} (request_id={request_id})"
            raise SciNetApiError(
                f"SciScholar API {path} returned {response.status_code}: {message}",
                status_code=response.status_code,
                payload=body,
            )

        if not isinstance(body, dict):
            raise SciNetApiError(f"SciScholar API {path} returned a non-object response", payload=body)
        return body

    def search(self, *, plan: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan": plan,
        }
        if options:
            payload["options"] = options
        return self._request("/v1/search", payload)

    def authors_related(self, *, plan: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan": plan,
        }
        if options:
            payload["options"] = options
        return self._request("/v1/authors/related", payload)

    @staticmethod
    def _author_reference_payload(author: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        author_id = normalize_whitespace(author.get("author_id"))
        name = normalize_whitespace(author.get("name"))
        if author_id:
            payload["author_id"] = author_id
        if name:
            payload["name"] = name
        return payload

    def authors_support_papers(self, *, query_text: str, authors: list[dict[str, Any]], options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query_text": query_text,
            "authors": [self._author_reference_payload(author) for author in authors],
        }
        if options:
            payload["options"] = options
        return self._request("/v1/authors/support-papers", payload)

    def authors_papers(self, *, identifier: str, search_by: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "identifier": identifier,
            "search_by": search_by,
        }
        if options:
            payload["options"] = options
        return self._request("/v1/authors/papers", payload)
