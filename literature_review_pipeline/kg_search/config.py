from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


DEFAULT_LLM_API_URL = "https://www.dmxapi.cn/v1/chat/completions"
DEFAULT_LLM_MODEL = "deepseek-v3.2"
DEFAULT_SEARCH_ENV_PATH = Path(os.getenv("SCIATLAS_WORKFLOW_ENV", ".env")).expanduser()
DEFAULT_EMBEDDING_MODEL_PATH = "/home/weiyunxiang/yunx/hf-models/BAAI--bge-large-en-v1.5"
DEFAULT_RERANKER_MODEL_PATH = "/home/weiyunxiang/yunx/hf-models/BAAI--bge-reranker-large"


def _load_search_env() -> None:
    env_path = DEFAULT_SEARCH_ENV_PATH
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_search_env()


def _default_database() -> str:
    return os.getenv("NEO4J_DB") or os.getenv("NEO4J_DATABASE") or "neo4j"


def parse_date_arg(value: str, arg_name: str) -> tuple[str, int]:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be in YYYY-MM-DD format, got: {value}") from exc
    return value, parsed.year


@dataclass(slots=True)
class SearchConfig:
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str | None = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD"))
    neo4j_database: str = field(default_factory=_default_database)

    embedding_model_path: str = field(default_factory=lambda: os.getenv("INNOEVAL_EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH))
    embedding_device: str | None = None
    reranker_model_path: str = field(default_factory=lambda: os.getenv("INNOEVAL_RERANKER_MODEL_PATH", DEFAULT_RERANKER_MODEL_PATH))
    reranker_device: str | None = None
    paper_embed_dim: int = 1024

    llm_api_url: str = field(
        default_factory=lambda: os.getenv("SEARCH_LLM_API_URL")
        or os.getenv("LLM_API_URL")
        or DEFAULT_LLM_API_URL
    )
    llm_model: str = field(default_factory=lambda: os.getenv("SEARCH_LLM_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL)
    llm_api_key: str | None = field(
        default_factory=lambda: os.getenv("SEARCH_LLM_API_KEY")
        or os.getenv("DMX-API-KEY")
        or os.getenv("DMX_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    llm_timeout_s: int = 60
    use_llm: bool = True
    enable_keyword_nodes: bool = True

    max_keywords_from_llm: int = 8
    max_titles_from_llm: int = 5
    max_titles_from_pdf_references: int = 10
    pdf_reference_selection_mode: str = "llm"
    pdf_debug_dir: str | None = None
    top_kg_keywords_per_kw: int = 3
    kw_vector_threshold: float = 0.70
    enable_embedding_rerank: bool = True
    topk_title_vector: int = 60
    topk_abstract_vector: int = 60
    topk_title_rerank: int = 15
    topk_abstract_rerank: int = 15
    topk_papers_per_title: int = 5
    paper_time_filter_oversample_factor: int = 10
    enable_title_ft: bool = True
    title_fuzzy_threshold: float = 0.88
    title_exact_boost: float = 1.0

    weight_embedding_path: float = 0.30
    weight_title_path: float = 0.8
    title_exact_pre_graph_bonus: float = 0.35
    title_fuzzy_pre_graph_bonus: float = 0.10
    uniform_importance: bool = False

    seed_gamma: float = 0.50
    graph_hops: int = 2
    max_seed_papers: int = 30
    max_seed_keywords: int = 30
    graph_method: str = "A"
    graph_hop_decay: float = 0.85
    ppr_alpha: float = 0.15
    ppr_max_iter: int = 50
    ppr_tol: float = 1e-6

    base_has_keyword: float = 1.20
    base_cites: float = 1.00
    base_related: float = 0.90
    base_authored: float = 0.80
    base_coauthor: float = 0.60
    base_cooccur: float = 0.60
    graph_count_cap: float = 2.0
    graph_keyword_smoothing: float = 0.25

    final_weight_pre_graph: float = 0.35
    final_weight_graph: float = 0.45
    final_weight_importance: float = 0.20
    final_title_bonus: float = 0.10
    final_pdf_source_title_filter_threshold: float = 0.90
    final_title_dedup_threshold: float = 0.80
    final_top_k: int = 20
    target_field: str | None = None

    explanation_max_paths_per_paper: int = 3
    explanation_max_neighbors_per_hop: int = 4
    graph_frontier_limit_per_type: int = 500
    after_year: int | None = None
    before_year: int | None = None

    def validate(self) -> None:
        if not self.neo4j_password:
            raise RuntimeError("NEO4J_PASSWORD is required.")
        if not self.use_llm and self.enable_keyword_nodes:
            raise RuntimeError("LLM must remain enabled for keyword extraction.")
        if not self.llm_api_url:
            raise RuntimeError("llm_api_url is required.")
        if not self.llm_model:
            raise RuntimeError("llm_model is required.")
        if not self.llm_api_key:
            raise RuntimeError("DMX-API-KEY is required. Set it in .env or SEARCH_LLM_API_KEY.")
        if self.graph_hops < 1:
            raise RuntimeError("graph_hops must be >= 1")
        if self.pdf_reference_selection_mode not in {"llm", "all"}:
            raise RuntimeError("pdf_reference_selection_mode must be one of: llm, all")
        if self.graph_method not in {"none", "A", "B"}:
            raise RuntimeError("graph_method must be one of: none, A, B")
        if self.paper_time_filter_oversample_factor < 1:
            raise RuntimeError("paper_time_filter_oversample_factor must be >= 1")
        if self.topk_title_vector < 1 or self.topk_abstract_vector < 1:
            raise RuntimeError("embedding vector top-k must be >= 1")
        if self.enable_embedding_rerank and (self.topk_title_rerank < 1 or self.topk_abstract_rerank < 1):
            raise RuntimeError("embedding rerank keep top-k must be >= 1 when rerank is enabled")
        if self.topk_title_rerank > self.topk_title_vector:
            raise RuntimeError("topk_title_rerank must be <= topk_title_vector")
        if self.topk_abstract_rerank > self.topk_abstract_vector:
            raise RuntimeError("topk_abstract_rerank must be <= topk_abstract_vector")
        if self.after_year is not None and self.before_year is not None and self.after_year > self.before_year:
            raise RuntimeError("after_year must be <= before_year")
