"""Central configuration for the SciAtlas idea-generation pipeline.

Values are read from environment variables (optionally loaded from a local
``.env`` file). Secrets (API keys / tokens) must be supplied through the
environment and are never hard-coded here; copy ``.env.example`` to ``.env``
and fill in your own credentials. See the README for the full list of
supported variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SCIATLAS_BASE_URL = "http://sciatlas.openkg.cn"
LEGACY_SCIATLAS_BASE_URLS = {
    "http://scinet.openkg.cn": CANONICAL_SCIATLAS_BASE_URL,
    "https://scinet.openkg.cn": "https://sciatlas.openkg.cn",
}


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _normalize_sciatlas_base_url(value: str | None) -> str:
    normalized = (value or CANONICAL_SCIATLAS_BASE_URL).strip().rstrip("/")
    return LEGACY_SCIATLAS_BASE_URLS.get(normalized, normalized)


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


# --------------------------------------------------------------------------- #
# SciAtlas knowledge-graph API
# --------------------------------------------------------------------------- #
@dataclass
class SciAtlasConfig:
    base_url: str = _normalize_sciatlas_base_url(os.getenv("SCIATLAS_API_BASE_URL"))
    api_key: str = _env_first("SCIATLAS_API_KEY")
    # Hosted KG searches can take several minutes on broad graph queries.
    timeout: int = int(os.getenv("SCIATLAS_TIMEOUT", "900"))
    max_retries: int = int(os.getenv("SCIATLAS_MAX_RETRIES", "4"))
    retry_backoff: float = float(os.getenv("SCIATLAS_RETRY_BACKOFF", "5.0"))
    use_cache: bool = os.getenv("SCIATLAS_USE_CACHE", "1") != "0"
    cache_dir: Path = field(
        default_factory=lambda: _path_from_env("SCIATLAS_CACHE_DIR", REPO_ROOT / "runs" / "cache")
    )
    official_cli_root: Path = field(
        default_factory=lambda: _path_from_env("SCIATLAS_OFFICIAL_CLI_ROOT", REPO_ROOT)
    )
    # When set, the main graph search (`search`/`search_papers`) runs against the
    # *local* Neo4j+embedding KG engine bundled at `references/search`
    # (`innoeval_search`) instead of the hosted `/v1/search` REST gateway. The
    # engine loads bge embedding/reranker models in-process and queries a local
    # Neo4j; it reads NEO4J_*/OPENAI_* from `<local_search_root>/.env`.
    use_local_kg: bool = os.getenv("SCIATLAS_USE_LOCAL_KG", "0") != "0"
    local_search_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("SCIATLAS_LOCAL_SEARCH_ROOT", str(REPO_ROOT / "references" / "search"))
        )
    )


# --------------------------------------------------------------------------- #
# LLM provider (OpenAI-compatible endpoint, e.g. dmxapi)
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    api_key: str = _env_first("LLM_API_KEY", "SCIATLAS_LLM_API_KEY", "OPENAI_API_KEY")
    base_url: str = _env_first("LLM_BASE_URL", "SCIATLAS_LLM_BASE_URL", "OPENAI_BASE_URL") or "https://www.dmxapi.cn/v1"
    model: str = _env_first("LLM_MODEL", "SCIATLAS_LLM_MODEL", "OPENAI_MODEL") or "DeepSeek-V3.2"
    temperature: float = float(_env_first("SCIATLAS_LLM_TEMPERATURE", "LLM_TEMPERATURE") or "0.7")
    max_retries: int = int(_env_first("SCIATLAS_LLM_MAX_RETRIES", "LLM_MAX_RETRIES") or "4")
    request_timeout: int = int(_env_first("SCIATLAS_LLM_TIMEOUT", "LLM_TIMEOUT") or "180")


# --------------------------------------------------------------------------- #
# Pipeline hyper-parameters (defaults mirror sciatlas-ar-pipeline.md)
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    workflow_mode: str = "default"       # default, flash, full, or smoke
    # Step 1 — query refinement + seed retrieval
    anchor_top_k: int = 5
    keyword_min_papers: int = 2          # a keyword must connect >= 2 papers
    k_step1: int = 4                     # number of seed papers Step 1 returns
    # Multi-query seed retrieval: refine the topic into `seed_num_queries` diverse
    # queries, retrieve in parallel, keep `seed_per_query` seeds per query, then
    # LLM-select the final `k_step1` seeds from the pooled candidates.
    seed_num_queries: int = 8            # k: diverse queries generated from the topic
    seed_per_query: int = 3              # m: seeds kept per query before final selection
    seed_request_multiplier: int = 6     # candidate fetch width per kept seed
    seed_request_floor: int = 12         # minimum per-query retrieval top-k
    seed_recent_years: int = 7           # seeds must be published within the last N years
    seed_min_citations: int = 1000       # ...OR be highly cited (keeps foundational older papers)

    # Step 2 — citation-graph construction / merge_search-style expansion
    # The budget is the target total number of papers in the graph, including the seeds.
    graph_budget_ratio: float = 0.15
    graph_budget_min: int = 12
    graph_budget_max: int = 40           # raised for multi-seed graphs (k_step1=4)
    graph_max_predecessors_per_paper: int = 3
    graph_min_forward_papers: int = 3
    graph_candidate_fetch_limit: int = 40
    num_candidate_step2: int = 3         # predecessors selected per node
    num_expansion_step2: int = 1         # predecessors randomly expanded per node
    # External innoeval checkout providing the `s2api` / `search` modules used by
    # Step 1/2 retrieval. Set INNOEVAL_ROOT to point at your local checkout.
    innoeval_root: Path = field(
        default_factory=lambda: _path_from_env(
            "INNOEVAL_ROOT",
            REPO_ROOT / "references" / "search" / "src" / "innoeval_search",
        )
    )
    # GROBID server used to extract title/abstract from input PDFs (Step 1 --pdf).
    # The bundled S2 module defaults to :8070; override via env when the server
    # is mapped elsewhere (e.g. the docker container on host port 8090).
    grobid_base_url: str = os.getenv("GROBID_BASE_URL", "http://127.0.0.1:8070")

    # Step 5 — multi-radius inspiration probing
    num_cross_domains: int = 4           # R2: number of cross-domain queries
    inspiration_top_k_same_field: int = 6   # R1 same-field different sub-fields
    inspiration_top_k_per_domain: int = 3   # R2 per cross-domain
    max_novelty_feedback_rounds: int = 2

    # Step 6 — idea generation
    idea_count: int = 1

    # IO
    runs_dir: Path = field(
        default_factory=lambda: _path_from_env(
            "RUNS_DIR",
            _path_from_env("SCIATLAS_RUNS_DIR", REPO_ROOT / "runs"),
        )
    )


@dataclass
class Config:
    sciatlas: SciAtlasConfig = field(default_factory=SciAtlasConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config() -> Config:
    """Return a fully-populated :class:`Config` instance."""
    return Config()
