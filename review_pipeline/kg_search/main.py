from __future__ import annotations

import argparse
import json
import textwrap

from .config import SearchConfig, parse_date_arg
from .neo4j_repository import Neo4jSearchRepository
from .pipeline import PaperSearchPipeline
from .result_formatter import format_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-to-paper search pipeline over Neo4j.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-text", help="English idea text used for paper retrieval.")
    input_group.add_argument("--pdf-path", help="PDF path used for extraction-driven paper retrieval.")
    parser.add_argument(
        "--llm-api-url",
        default=None,
        help="Override the default OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the default LLM model name.",
    )
    parser.add_argument("--graph-method", choices=["none", "A", "B"], default="A")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--Target-Field",
        dest="target_field",
        default=None,
        help="Filter final ranked papers by Field before applying top-k.",
    )
    parser.add_argument("--after", default=None, help="Lower date bound in YYYY-MM-DD format.")
    parser.add_argument("--before", default=None, help="Upper date bound in YYYY-MM-DD format.")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-model-path", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--title-vector-topk", type=int, default=None)
    parser.add_argument("--abstract-vector-topk", type=int, default=None)
    parser.add_argument("--title-rerank-topk", type=int, default=None)
    parser.add_argument("--abstract-rerank-topk", type=int, default=None)
    parser.add_argument(
        "--disable-embedding-rerank",
        action="store_true",
        help="Disable cross-encoder reranking on the embedding recall path.",
    )
    parser.add_argument("--kw-threshold", type=float, default=None)
    parser.add_argument(
        "--unable-title-ft",
        action="store_true",
        help="Disable title fulltext fallback and use exact title matching only.",
    )
    parser.add_argument(
        "--max-titles-from-pdf-references",
        type=int,
        default=None,
        help="Override the maximum number of selected PDF reference titles used for title recall.",
    )
    parser.add_argument(
        "--pdf-debug-dir",
        default=None,
        help="Optional directory where PDF-search debug artifacts are written.",
    )
    parser.add_argument(
        "--uni-importance",
        action="store_true",
        help="Use a uniform importance=1.0 for all papers instead of citation-based normalization.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Print recalled papers in a readable demo list instead of JSON.",
    )
    return parser


def _format_demo_items(items: list[str] | None, indent: str = "     ") -> list[str]:
    values = items or []
    if not values:
        return [f"{indent}N/A"]
    return [f"{indent}- {item}" for item in values]


def fetch_demo_metadata(config: SearchConfig, results: list[dict]) -> dict[str, dict]:
    paper_ids = [item["paper_id"] for item in results if item.get("paper_id")]
    if not paper_ids:
        return {}
    with Neo4jSearchRepository(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password or "",
        database=config.neo4j_database,
    ) as repository:
        return repository.fetch_demo_metadata(paper_ids)


def format_demo_output(results: list[dict], metadata_by_paper_id: dict[str, dict] | None = None) -> str:
    if not results:
        return "No papers retrieved."

    metadata_by_paper_id = metadata_by_paper_id or {}
    lines: list[str] = [f"Retrieved Papers: {len(results)}", ""]
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or "(Untitled)"
        abstract = item.get("abstract") or "N/A"
        year = item.get("publication_year") or "N/A"
        cited = item.get("cited_by_count")
        final_score = item.get("final_score")
        metadata = metadata_by_paper_id.get(item.get("paper_id"), {})
        cited_text = str(cited) if cited is not None else "N/A"
        score_text = f"{final_score:.4f}" if isinstance(final_score, (int, float)) else "N/A"
        type_text = metadata.get("type") or "N/A"

        lines.append(f"{idx}. {title}")
        lines.append(f"   Year: {year} | Cited: {cited_text} | Final Score: {score_text}")
        lines.append(f"   Type: {type_text}")
        lines.append("   Topic:")
        lines.extend(_format_demo_items(metadata.get("topics")))
        lines.append("   Domain:")
        lines.extend(_format_demo_items(metadata.get("domains")))
        lines.append("   Field:")
        lines.extend(_format_demo_items(metadata.get("fields")))
        lines.append("   Subfield:")
        lines.extend(_format_demo_items(metadata.get("subfields")))
        lines.append("   Abstract:")
        wrapped_abstract = textwrap.wrap(
            abstract,
            width=88,
            initial_indent="     ",
            subsequent_indent="     ",
        )
        lines.extend(wrapped_abstract or ["     N/A"])
        if idx < len(results):
            lines.append("")

    return "\n".join(lines)


def _build_effective_params(args: argparse.Namespace, config: SearchConfig) -> dict[str, object]:
    input_type = "pdf" if args.pdf_path else "idea_text"
    return {
        "input_type": input_type,
        "idea_text": args.idea_text,
        "pdf_path": args.pdf_path,
        "neo4j_uri": config.neo4j_uri,
        "neo4j_user": config.neo4j_user,
        "neo4j_database": config.neo4j_database,
        "llm_api_url": config.llm_api_url,
        "llm_model": config.llm_model,
        "graph_method": config.graph_method,
        "top_k": config.final_top_k,
        "target_field": config.target_field,
        "after": args.after,
        "before": args.before,
        "after_year": config.after_year,
        "before_year": config.before_year,
        "embedding_model_path": config.embedding_model_path,
        "embedding_device": config.embedding_device,
        "enable_embedding_rerank": config.enable_embedding_rerank,
        "reranker_model_path": config.reranker_model_path if config.enable_embedding_rerank else None,
        "reranker_device": config.reranker_device if config.enable_embedding_rerank else None,
        "title_vector_topk": config.topk_title_vector,
        "abstract_vector_topk": config.topk_abstract_vector,
        "title_rerank_topk": config.topk_title_rerank if config.enable_embedding_rerank else None,
        "abstract_rerank_topk": config.topk_abstract_rerank if config.enable_embedding_rerank else None,
        "kw_threshold": config.kw_vector_threshold,
        "enable_title_ft": config.enable_title_ft,
        "uniform_importance": config.uniform_importance,
        "pretty": args.pretty,
        "demo": args.demo,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = SearchConfig()
    config.graph_method = args.graph_method
    config.final_top_k = args.top_k
    config.target_field = args.target_field
    if args.llm_api_url is not None:
        config.llm_api_url = args.llm_api_url
    if args.llm_model is not None:
        config.llm_model = args.llm_model
    if args.embedding_model_path is not None:
        config.embedding_model_path = args.embedding_model_path
    if args.embedding_device is not None:
        config.embedding_device = args.embedding_device
    if args.reranker_model_path is not None:
        config.reranker_model_path = args.reranker_model_path
    if args.reranker_device is not None:
        config.reranker_device = args.reranker_device
    if args.title_vector_topk is not None:
        config.topk_title_vector = args.title_vector_topk
    if args.abstract_vector_topk is not None:
        config.topk_abstract_vector = args.abstract_vector_topk
    if args.title_rerank_topk is not None:
        config.topk_title_rerank = args.title_rerank_topk
    if args.abstract_rerank_topk is not None:
        config.topk_abstract_rerank = args.abstract_rerank_topk
    if args.disable_embedding_rerank:
        config.enable_embedding_rerank = False
    if args.kw_threshold is not None:
        config.kw_vector_threshold = args.kw_threshold
    if args.unable_title_ft:
        config.enable_title_ft = False
    if args.max_titles_from_pdf_references is not None:
        config.max_titles_from_pdf_references = args.max_titles_from_pdf_references
    config.pdf_debug_dir = args.pdf_debug_dir
    if args.uni_importance:
        config.uniform_importance = True
    after_date = None
    before_date = None
    if args.after is not None:
        after_date, config.after_year = parse_date_arg(args.after, "--after")
    if args.before is not None:
        before_date, config.before_year = parse_date_arg(args.before, "--before")
    if after_date is not None and before_date is not None and after_date > before_date:
        raise ValueError(f"--after must be <= --before, got after={after_date}, before={before_date}")

    effective_params = _build_effective_params(args, config)
    print(
        f"[params] {json.dumps(effective_params, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )

    pipeline = PaperSearchPipeline(config)
    if args.pdf_path:
        result = pipeline.search_pdf(args.pdf_path)
    else:
        result = pipeline.search(args.idea_text)
    formatted_results = format_results(result.results)
    payload = {
        "query": result.query,
        "input_context": {
            "input_type": result.understanding.source_type,
            "pdf_path": args.pdf_path,
            "pdf_title": result.understanding.source_title,
            "pdf_reference_count": len(result.understanding.reference_titles),
        },
        "keyword_source": result.understanding.keyword_source,
        "title_source": result.understanding.title_source,
        "target_field": config.target_field,
        "keywords": [{"text": item.text, "score": item.score} for item in result.understanding.keywords],
        "titles": [{"title": item.title, "confidence": item.confidence} for item in result.understanding.titles],
        "authors": [
            {
                "author_id": item.author_id,
                "name": item.name,
                "score": item.score,
            }
            for item in result.authors
        ],
        "results": formatted_results,
    }
    if args.demo:
        demo_metadata = fetch_demo_metadata(config, formatted_results)
        print(f"query: {result.query}")
        if args.pdf_path:
            print(f"pdf: {args.pdf_path}")
            print(f"pdf_title: {result.understanding.source_title or 'N/A'}")
            print(f"pdf_reference_count: {len(result.understanding.reference_titles)}")
        print(format_demo_output(formatted_results, demo_metadata))
    elif args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
