#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_search.service import run_search_with_authors  # noqa: E402
from pipeline import build_reviewer_selection  # noqa: E402
from review.common import first_non_empty, load_env_values, normalize_whitespace  # noqa: E402

METHOD_FIELD_KEYS = {
    "pipeline": "pipeline_reviewers",
    "baseline": "baseline_reviewers",
    "our_wo_keywords": "our_wo_keywords_reviewers",
    "our_wo_graphwalk": "our_wo_graphwalk_reviewers",
}

METHOD_ALIASES = {
    "pipeline": "pipeline",
    "baseline": "baseline",
    "our_wo_keywords": "our_wo_keywords",
    "ours_wo_keywords": "our_wo_keywords",
    "our-wo-keywords": "our_wo_keywords",
    "our_wo_graphwalk": "our_wo_graphwalk",
    "ours_wo_graphwalk": "our_wo_graphwalk",
    "our-wo-graphwalk": "our_wo_graphwalk",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select reviewer lists for the KG-author pipeline strategy and the "
            "top-K-paper first-author baseline."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-text", help="Idea text used to run KG search.")
    input_group.add_argument("--pdf-path", help="PDF path. In default mode, title/abstract are extracted like pipeline.py.")
    input_group.add_argument(
        "--search-result",
        help="Existing search result JSON. Accepts either this script output or pipeline search/result.json.",
    )

    parser.add_argument("--output-dir", default=None, help="Directory for reviewer selection artifacts.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with Neo4j settings.")
    parser.add_argument(
        "--methods",
        default="pipeline,baseline",
        help="Comma-separated methods to output: pipeline,baseline,our_wo_keywords,our_wo_graphwalk.",
    )
    parser.add_argument("--reviewer-count", type=int, default=10, help="Number of reviewers to select.")
    parser.add_argument(
        "--reviewer-min-works-count",
        type=int,
        default=10,
        help="Minimum works_count sum used by the existing pipeline reviewer selector.",
    )
    parser.add_argument(
        "--reviewer-selection-top-k",
        type=int,
        default=30,
        help="Restrict pipeline reviewer sampling to the top-K ranked merged candidates before even spacing.",
    )
    parser.add_argument("--kg-top-k", type=int, default=50, help="KG paper top-k when running search.")
    parser.add_argument(
        "--baseline-scan-limit",
        type=int,
        default=50,
        help="How many ranked KG papers to scan when unique first authors are needed.",
    )
    parser.add_argument("--target-field", default=None, help="Optional KG target field filter.")
    parser.add_argument("--after", default=None, help="Optional KG lower date bound, YYYY-MM-DD.")
    parser.add_argument("--before", default=None, help="Optional KG upper date bound, YYYY-MM-DD.")
    parser.add_argument("--kg-embedding-device", default=None, help="Torch device for KG embedding model.")
    parser.add_argument("--kg-reranker-device", default=None, help="Torch device for KG reranker model.")
    parser.add_argument(
        "--pdf-input-mode",
        choices=("pipeline", "kg-pdf"),
        default="pipeline",
        help=(
            "For --pdf-path, `pipeline` extracts title/abstract and searches with that retrieval text, "
            "matching pipeline.py. `kg-pdf` uses kg_search.search_pdf directly."
        ),
    )
    parser.add_argument("--grobid-base-url", default="http://127.0.0.1:8070", help="GROBID base URL for PDF mode.")
    parser.add_argument("--grobid-start-page", type=int, default=None, help="First PDF page sent to GROBID.")
    parser.add_argument(
        "--disable-baseline-author-dedupe",
        action="store_true",
        help="Keep duplicate first authors in the baseline instead of scanning for unique authors.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the final JSON to stdout.")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"reviewer_selection_{stamp}"


def parse_methods(value: str) -> list[str]:
    methods: list[str] = []
    for raw_part in value.split(","):
        part = normalize_whitespace(raw_part)
        if not part:
            continue
        method = METHOD_ALIASES.get(part, part)
        if method not in METHOD_FIELD_KEYS:
            raise ValueError(f"Unsupported method: {part}")
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("--methods must include at least one method")
    return methods


def extract_pipeline_pdf_context(args: argparse.Namespace) -> dict[str, Any]:
    from s2api.search_s2 import extract_pdf_payload

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    pdf_payload = extract_pdf_payload(
        pdf_path,
        grobid_base_url=args.grobid_base_url,
        grobid_start_page=args.grobid_start_page,
    )
    title = normalize_whitespace(pdf_payload.get("title"))
    abstract = normalize_whitespace(pdf_payload.get("abstract"))
    body = normalize_whitespace(pdf_payload.get("body"))
    retrieval_text = f"Title: {title}\nAbstract: {abstract}" if title else abstract
    if not retrieval_text:
        retrieval_text = body[:2000]
    source_text = "\n\n".join(
        part
        for part in [
            f"Title: {title}" if title else "",
            f"Abstract: {abstract}" if abstract else "",
            body,
        ]
        if part
    )
    return {
        "status": "ok",
        "input": {
            "input_type": "pdf",
            "pdf_path": str(pdf_path),
        },
        "retrieval_seed": {
            "source": "pdf.abstract",
            "title": title,
            "abstract": abstract,
            "text": retrieval_text,
            "pdf": {
                "title": title,
                "abstract": abstract,
                "body_chars": len(body),
            },
        },
        "source_full_text": {
            "source": "grobid.pdf.full_text",
            "title": title,
            "abstract": abstract,
            "body_chars": len(body),
            "text": source_text,
        },
    }


def prepare_search_seed(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    pdf_context = None
    idea_text = normalize_whitespace(args.idea_text)
    pdf_path = None
    input_type = "idea_text"
    if args.pdf_path:
        input_type = "pdf"
        if args.pdf_input_mode == "pipeline":
            pdf_context = extract_pipeline_pdf_context(args)
            idea_text = normalize_whitespace(pdf_context.get("retrieval_seed", {}).get("text"))
            if not idea_text:
                raise ValueError(f"Unable to build pipeline retrieval text from PDF: {args.pdf_path}")
        else:
            pdf_path = str(Path(args.pdf_path).expanduser().resolve())
    return {
        "input_type": input_type,
        "idea_text": idea_text,
        "pdf_path": pdf_path,
        "original_pdf_path": str(Path(args.pdf_path).expanduser().resolve()) if args.pdf_path else None,
    }, pdf_context


def run_kg_search(
    args: argparse.Namespace,
    seed: dict[str, Any],
    *,
    disable_keywords: bool = False,
    graph_method: str | None = None,
) -> dict[str, Any]:
    kg_args = argparse.Namespace(
        idea_text=seed["idea_text"],
        pdf_path=seed["pdf_path"],
        top_k=args.kg_top_k,
        target_field=args.target_field,
        after=args.after,
        before=args.before,
        embedding_device=args.kg_embedding_device,
        reranker_device=args.kg_reranker_device,
        unable_title_ft=False,
        max_titles_from_pdf_references=None,
        pdf_reference_selection_mode=None,
        max_seed_papers=None,
        pdf_debug_dir=None,
        graph_method=graph_method,
        disable_keywords=disable_keywords,
        use_citation_importance=False,
        pretty=False,
    )
    kg_payload = run_search_with_authors(kg_args)
    papers = kg_payload.get("papers") if isinstance(kg_payload, dict) else []
    authors = kg_payload.get("authors") if isinstance(kg_payload, dict) else []
    if not isinstance(papers, list):
        papers = []
    if not isinstance(authors, list):
        authors = []
    return {
        "status": "ok",
        "input_type": seed["input_type"],
        "idea_text": seed["idea_text"],
        "pdf_path": seed["original_pdf_path"],
        "pdf_input_mode": args.pdf_input_mode if args.pdf_path else None,
        "sources": {
            "kg": {
                "source": "kg",
                "status": "ok",
                "paper_count": len(papers),
                "papers": papers,
                "author_count": len(authors),
                "authors": authors,
            }
        },
        "ranking": {
            "status": "kg_only",
            "strategy": "kg_final_score",
            "papers": [
                {
                    "rank": index,
                    "source": "kg",
                    "source_rank": index,
                    "title": normalize_whitespace(paper.get("title")) if isinstance(paper, dict) else "",
                    "paper": paper,
                }
                for index, paper in enumerate(papers, start=1)
                if isinstance(paper, dict)
            ],
        },
    }


def kg_paper_candidates(search_payload: dict[str, Any], scan_limit: int) -> list[dict[str, Any]]:
    kg_papers = search_payload.get("sources", {}).get("kg", {}).get("papers", [])
    if not isinstance(kg_papers, list):
        return []

    candidates: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()
    for paper_rank, paper in enumerate(kg_papers, start=1):
        if not isinstance(paper, dict):
            continue
        paper_id = normalize_whitespace(paper.get("id") or paper.get("paper_id"))
        if not paper_id or paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(paper_id)
        candidates.append(
            {
                "paper_rank": paper_rank,
                "paper_id": paper_id,
                "paper_title": normalize_whitespace(paper.get("title")),
            }
        )
        if len(candidates) >= scan_limit:
            break
    return candidates


def neo4j_connection(env_path: str | None) -> tuple[str, str, str, str]:
    from review.author_background import DEFAULT_NEO4J_PASSWORD, DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER

    env_values = load_env_values(Path(env_path).expanduser().resolve() if env_path else None)
    uri = first_non_empty(env_values.get("NEO4J_URI"), DEFAULT_NEO4J_URI)
    user = first_non_empty(env_values.get("NEO4J_USER"), DEFAULT_NEO4J_USER)
    password = first_non_empty(env_values.get("NEO4J_PASSWORD"), DEFAULT_NEO4J_PASSWORD)
    database = first_non_empty(env_values.get("NEO4J_DB"), env_values.get("NEO4J_DATABASE"), "neo4j")
    return uri, user, password, database


def fetch_first_authors_for_papers(
    paper_candidates: list[dict[str, Any]],
    *,
    env_path: str | None,
) -> dict[str, dict[str, Any]]:
    if not paper_candidates:
        return {}

    from neo4j import GraphDatabase

    uri, user, password, database = neo4j_connection(env_path)
    query = """
    UNWIND $papers AS paper
    MATCH (p:Paper {id: paper.paper_id})
    CALL (p) {
      OPTIONAL MATCH (a:Author)-[r:AUTHORED]->(p)
      WITH a, r
      ORDER BY
        CASE WHEN r.position IS NULL THEN 1 ELSE 0 END,
        r.position ASC,
        coalesce(a.display_name, a.label, r.raw_name, a.id) ASC
      WITH collect(
        CASE
          WHEN a IS NULL THEN NULL
          ELSE {
            author_id: a.id,
            display_name: a.display_name,
            label: a.label,
            works_count: a.works_count,
            raw_name: r.raw_name,
            position: r.position
          }
        END
      ) AS authors
      RETURN [item IN authors WHERE item IS NOT NULL][0] AS first_author
    }
    RETURN
      paper.paper_id AS paper_id,
      p.title AS paper_title,
      first_author AS first_author
    """

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            rows = session.run(query, papers=paper_candidates)
            result: dict[str, dict[str, Any]] = {}
            for row in rows:
                paper_id = normalize_whitespace(row["paper_id"])
                first_author = row.get("first_author")
                result[paper_id] = {
                    "paper_id": paper_id,
                    "paper_title": normalize_whitespace(row.get("paper_title")),
                    "first_author": dict(first_author) if first_author else None,
                }
            return result
    finally:
        driver.close()


def build_baseline_selection(
    search_payload: dict[str, Any],
    *,
    reviewer_count: int,
    scan_limit: int,
    env_path: str | None,
    dedupe_authors: bool,
) -> dict[str, Any]:
    candidates = kg_paper_candidates(search_payload, max(scan_limit, reviewer_count))
    authors_by_paper = fetch_first_authors_for_papers(candidates, env_path=env_path)

    reviewers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_author_ids: set[str] = set()

    for paper in candidates:
        paper_id = paper["paper_id"]
        row = authors_by_paper.get(paper_id)
        first_author = row.get("first_author") if isinstance(row, dict) else None
        if not isinstance(first_author, dict):
            skipped.append({**paper, "reason": "missing_first_author"})
            continue

        author_id = normalize_whitespace(first_author.get("author_id"))
        author_name = (
            normalize_whitespace(first_author.get("display_name"))
            or normalize_whitespace(first_author.get("label"))
            or normalize_whitespace(first_author.get("raw_name"))
            or author_id
        )
        if not author_id and not author_name:
            skipped.append({**paper, "reason": "empty_author_identity"})
            continue
        if dedupe_authors and author_id and author_id in seen_author_ids:
            skipped.append({**paper, "author_id": author_id, "author_name": author_name, "reason": "duplicate_author"})
            continue
        if author_id:
            seen_author_ids.add(author_id)

        reviewers.append(
            {
                "selection_rank": len(reviewers) + 1,
                "author_id": author_id,
                "author_name": author_name,
                "display_name": normalize_whitespace(first_author.get("display_name")),
                "works_count": first_author.get("works_count"),
                "source": "kg_top_paper_first_author",
                "paper_rank": paper["paper_rank"],
                "paper_id": paper_id,
                "paper_title": row.get("paper_title") or paper["paper_title"],
                "author_position": first_author.get("position"),
            }
        )
        if len(reviewers) >= reviewer_count:
            break

    status = "ok" if len(reviewers) >= reviewer_count else ("partial_error" if reviewers else "error")
    return {
        "status": status,
        "selection_strategy": "kg-top-paper-first-author",
        "requested_reviewer_count": reviewer_count,
        "baseline_scan_limit": scan_limit,
        "dedupe_authors": dedupe_authors,
        "candidate_paper_count": len(candidates),
        "selected_reviewer_count": len(reviewers),
        "reviewers": reviewers,
        "skipped": skipped,
    }


def simplify_pipeline_reviewers(selection_payload: dict[str, Any]) -> list[dict[str, Any]]:
    reviewers = selection_payload.get("sampled_reviewers", [])
    if not isinstance(reviewers, list):
        return []

    simplified: list[dict[str, Any]] = []
    for index, reviewer in enumerate(reviewers, start=1):
        if not isinstance(reviewer, dict):
            continue
        author_id = normalize_whitespace(reviewer.get("author_id") or reviewer.get("representative_author_id"))
        author_name = normalize_whitespace(reviewer.get("author_name") or reviewer.get("name")) or author_id
        simplified.append(
            {
                "selection_rank": index,
                "author_id": author_id,
                "author_name": author_name,
                "display_name": author_name,
                "works_count_sum": reviewer.get("works_count_sum"),
                "score": reviewer.get("score"),
                "first_rank": reviewer.get("first_rank"),
                "source": "pipeline_author_ranking",
                "reviewer_key": reviewer.get("reviewer_key"),
                "member_author_ids": reviewer.get("member_author_ids") or [],
            }
        )
    return simplified


def build_method_payload(
    method: str,
    search_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if method == "baseline":
        selection_payload = build_baseline_selection(
            search_payload,
            reviewer_count=args.reviewer_count,
            scan_limit=args.baseline_scan_limit,
            env_path=str(Path(args.env).expanduser().resolve()) if args.env else None,
            dedupe_authors=not args.disable_baseline_author_dedupe,
        )
        return selection_payload, selection_payload["reviewers"], "baseline_first_author_selection.json"

    selection_payload = build_reviewer_selection(
        search_payload,
        max_reviewers=args.reviewer_count,
        min_works_count=args.reviewer_min_works_count,
        selection_top_k=args.reviewer_selection_top_k,
        env_path=str(Path(args.env).expanduser().resolve()) if args.env else None,
        use_metadata=True,
    )
    return selection_payload, simplify_pipeline_reviewers(selection_payload), f"{method}_selection.json"


def run_selection(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_methods(args.methods)
    pdf_context = None
    existing_result_path = output_dir / "reviewer_lists.json"
    existing_result: dict[str, Any] = {}
    if existing_result_path.exists():
        try:
            existing_payload = read_json(existing_result_path)
        except Exception:
            existing_payload = {}
        if isinstance(existing_payload, dict):
            existing_result = existing_payload

    base_search_payload: dict[str, Any] | None = None
    base_search_result_path: Path | None = None
    seed: dict[str, Any] | None = None
    if args.search_result:
        base_search_result_path = Path(args.search_result).expanduser().resolve()
        base_search_payload = read_json(base_search_result_path)
        if any(method.startswith("our_wo_") for method in methods):
            seed_text = normalize_whitespace(base_search_payload.get("idea_text"))
            if not seed_text:
                raise ValueError(
                    "--search-result can only be used with our ablation methods when the JSON includes top-level `idea_text`."
                )
            seed = {
                "input_type": normalize_whitespace(base_search_payload.get("input_type")) or "idea_text",
                "idea_text": seed_text,
                "pdf_path": None,
                "original_pdf_path": normalize_whitespace(base_search_payload.get("pdf_path")) or None,
            }
    else:
        seed, pdf_context = prepare_search_seed(args)
        if pdf_context is not None:
            write_json(output_dir / "idea_context.json", pdf_context)

    selection_paths: dict[str, str] = {}
    search_result_paths: dict[str, str] = {}
    selection_payloads: dict[str, dict[str, Any]] = {}
    selected_reviewers: dict[str, list[dict[str, Any]]] = {}

    for method in methods:
        if method in {"pipeline", "baseline"}:
            if base_search_payload is None:
                if seed is None:
                    raise ValueError("Missing search seed for pipeline/baseline selection")
                base_search_payload = run_kg_search(args, seed)
                base_search_result_path = output_dir / "search_result.json"
                write_json(base_search_result_path, base_search_payload)
            search_payload = base_search_payload
            search_path = base_search_result_path
        elif method == "our_wo_keywords":
            if seed is None:
                raise ValueError("Missing search seed for our_wo_keywords selection")
            search_payload = run_kg_search(args, seed, disable_keywords=True)
            search_path = output_dir / "search_result_our_wo_keywords.json"
            write_json(search_path, search_payload)
        elif method == "our_wo_graphwalk":
            if seed is None:
                raise ValueError("Missing search seed for our_wo_graphwalk selection")
            search_payload = run_kg_search(args, seed, graph_method="none")
            search_path = output_dir / "search_result_our_wo_graphwalk.json"
            write_json(search_path, search_payload)
        else:
            raise ValueError(f"Unsupported method: {method}")

        selection_payload, reviewers, selection_filename = build_method_payload(method, search_payload, args)
        selection_path = output_dir / selection_filename
        write_json(selection_path, selection_payload)
        selection_payloads[method] = selection_payload
        selected_reviewers[method] = reviewers
        selection_paths[method] = str(selection_path)
        if search_path is not None:
            search_result_paths[method] = str(search_path)

    ok_methods = [method for method in methods if selected_reviewers.get(method)]
    merged_paths = existing_result.get("paths", {}) if isinstance(existing_result.get("paths"), dict) else {}
    if search_result_paths:
        prior_search_result_paths = (
            merged_paths.get("search_result_paths", {})
            if isinstance(merged_paths.get("search_result_paths"), dict)
            else {}
        )
        merged_paths["search_result_paths"] = {**prior_search_result_paths, **search_result_paths}
    if selection_paths:
        prior_selection_paths = (
            merged_paths.get("selection_paths", {})
            if isinstance(merged_paths.get("selection_paths"), dict)
            else {}
        )
        merged_paths["selection_paths"] = {**prior_selection_paths, **selection_paths}
    merged_paths.update(
        {
            "output_dir": str(output_dir),
            "idea_context_path": (
                str(output_dir / "idea_context.json")
                if pdf_context is not None
                else merged_paths.get("idea_context_path")
            ),
            "search_result_path": (
                str(base_search_result_path)
                if base_search_result_path is not None
                else merged_paths.get("search_result_path")
            ),
            "pipeline_selection_path": selection_paths.get(
                "pipeline",
                merged_paths.get("pipeline_selection_path"),
            ),
            "baseline_selection_path": selection_paths.get(
                "baseline",
                merged_paths.get("baseline_selection_path"),
            ),
            "result_path": str(output_dir / "reviewer_lists.json"),
        }
    )

    merged_input = existing_result.get("input", {}) if isinstance(existing_result.get("input"), dict) else {}
    merged_input.update(
        {
            "idea_text": args.idea_text if args.idea_text is not None else merged_input.get("idea_text"),
            "pdf_path": (
                str(Path(args.pdf_path).expanduser().resolve())
                if args.pdf_path
                else merged_input.get("pdf_path")
            ),
            "pdf_input_mode": args.pdf_input_mode if args.pdf_path else merged_input.get("pdf_input_mode"),
            "search_result_path": (
                str(base_search_result_path)
                if base_search_result_path is not None
                else merged_input.get("search_result_path")
            ),
            "reviewer_count": args.reviewer_count,
            "reviewer_min_works_count": args.reviewer_min_works_count,
            "reviewer_selection_top_k": args.reviewer_selection_top_k,
        }
    )
    prior_methods = merged_input.get("methods", [])
    if not isinstance(prior_methods, list):
        prior_methods = []
    merged_input["methods"] = list(dict.fromkeys([*prior_methods, *methods]))

    result: dict[str, Any] = dict(existing_result)
    result.update(
        {
            "status": "ok" if len(ok_methods) == len(methods) else "partial_error",
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "input": merged_input,
            "paths": merged_paths,
        }
    )
    for method, field_key in METHOD_FIELD_KEYS.items():
        if method in selected_reviewers:
            result[field_key] = selected_reviewers[method]
    write_json(output_dir / "reviewer_lists.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir().resolve()
    result = run_selection(args, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
