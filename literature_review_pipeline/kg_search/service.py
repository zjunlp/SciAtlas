from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr
from typing import Any

from cuda_devices import device_at, first_configured_cuda_devices

from .config import SearchConfig, parse_date_arg
from .neo4j_repository import Neo4jSearchRepository
from .pipeline import PaperSearchPipeline
from .text_utils import normalize_title_exact


def build_search_config(args: argparse.Namespace) -> SearchConfig:
    config = SearchConfig()
    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        config.final_top_k = int(top_k)
    config.target_field = getattr(args, "target_field", None)
    config.uniform_importance = not bool(getattr(args, "use_citation_importance", False))
    for arg_name, config_name, caster in (
        ("final_weight_pre_graph", "final_weight_pre_graph", float),
        ("final_weight_graph", "final_weight_graph", float),
        ("final_weight_importance", "final_weight_importance", float),
        ("weight_embedding_path", "weight_embedding_path", float),
        ("weight_title_path", "weight_title_path", float),
        ("seed_gamma", "seed_gamma", float),
        ("graph_hop_decay", "graph_hop_decay", float),
        ("ppr_alpha", "ppr_alpha", float),
        ("base_has_keyword", "base_has_keyword", float),
        ("base_cites", "base_cites", float),
        ("base_related", "base_related", float),
        ("topk_title_rerank", "topk_title_rerank", int),
        ("topk_abstract_rerank", "topk_abstract_rerank", int),
        ("topk_title_vector", "topk_title_vector", int),
        ("topk_abstract_vector", "topk_abstract_vector", int),
        ("title_exact_pre_graph_bonus", "title_exact_pre_graph_bonus", float),
        ("title_fuzzy_pre_graph_bonus", "title_fuzzy_pre_graph_bonus", float),
        ("final_title_bonus", "final_title_bonus", float),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, caster(value))
    max_pdf_refs = getattr(args, "max_titles_from_pdf_references", None)
    if max_pdf_refs is not None:
        config.max_titles_from_pdf_references = int(max_pdf_refs)
    pdf_ref_mode = getattr(args, "pdf_reference_selection_mode", None)
    if pdf_ref_mode is not None:
        config.pdf_reference_selection_mode = str(pdf_ref_mode)
    max_seed_papers = getattr(args, "max_seed_papers", None)
    if max_seed_papers is not None:
        config.max_seed_papers = int(max_seed_papers)
    graph_method = getattr(args, "graph_method", None)
    if graph_method is not None:
        config.graph_method = str(graph_method)
    graph_hops = getattr(args, "graph_hops", None)
    if graph_hops is not None:
        config.graph_hops = int(graph_hops)
    if bool(getattr(args, "disable_keywords", False)):
        config.enable_keyword_nodes = False
    config.pdf_debug_dir = getattr(args, "pdf_debug_dir", None)
    if bool(getattr(args, "unable_title_ft", False)):
        config.enable_title_ft = False
    cuda_devices = first_configured_cuda_devices()
    config.embedding_device = getattr(args, "embedding_device", None) or device_at(cuda_devices, 0)
    config.reranker_device = getattr(args, "reranker_device", None) or device_at(cuda_devices, 1) or config.embedding_device

    after = getattr(args, "after", None)
    before = getattr(args, "before", None)
    after_date = None
    before_date = None
    if after is not None:
        after_date, config.after_year = parse_date_arg(after, "--after")
    if before is not None:
        before_date, config.before_year = parse_date_arg(before, "--before")
    if after_date is not None and before_date is not None and after_date > before_date:
        raise ValueError(f"--after must be <= --before, got after={after_date}, before={before_date}")
    return config


def run_search_with_authors(args: argparse.Namespace) -> dict[str, object]:
    config = build_search_config(args)

    with redirect_stderr(io.StringIO()):
        pipeline = PaperSearchPipeline(config)
        if getattr(args, "pdf_path", None):
            result = pipeline.search_pdf(args.pdf_path)
        else:
            result = pipeline.search(args.idea_text)

    paper_ids = [item.paper_id for item in result.results if item.paper_id]
    with Neo4jSearchRepository(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password or "",
        database=config.neo4j_database,
    ) as repository:
        papers = repository.fetch_full_paper_records(paper_ids)

    return {
        "papers": papers,
        "authors": [
            {
                "author_id": item.author_id,
                "name": item.name,
                "score": item.score,
            }
            for item in result.authors
        ],
    }


def search_papers_by_title(
    title: str,
    *,
    top_k: int = 10,
    config: SearchConfig | None = None,
) -> list[dict[str, Any]]:
    return search_paper_title_candidates(title, top_k=top_k, config=config, include_fulltext=False)["exact"]


def search_paper_title_candidates(
    title: str,
    *,
    top_k: int = 10,
    config: SearchConfig | None = None,
    include_fulltext: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    title_text = " ".join(str(title or "").split()).strip()
    if not title_text:
        return {"exact": [], "fulltext": []}

    resolved_config = config or SearchConfig()
    with Neo4jSearchRepository(
        uri=resolved_config.neo4j_uri,
        user=resolved_config.neo4j_user,
        password=resolved_config.neo4j_password or "",
        database=resolved_config.neo4j_database,
    ) as repository:
        exact_rows = repository.match_papers_by_normalized_title(
            normalize_title_exact(title_text),
            top_k=top_k,
        )
        fulltext_rows = (
            repository.fulltext_search_papers("paper_title_ft", title_text, top_k=top_k)
            if include_fulltext
            else []
        )
    return {"exact": exact_rows, "fulltext": fulltext_rows}
