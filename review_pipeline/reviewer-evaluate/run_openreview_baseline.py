#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "dataset_runs" / "pairs_v2_final" / "summary_reviewers_20260529_130404.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import ensure_reviewer_key, fetch_author_metadata_by_id  # noqa: E402
from review.common import first_non_empty, load_env_values, normalize_whitespace, write_json  # noqa: E402
from review.author_background import DEFAULT_NEO4J_PASSWORD, DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER  # noqa: E402


SCINCL_MODEL_NAME = "malteos/scincl"
SPECTER2_MODEL_NAME = "allenai/specter2_aug2023refresh_base"

LOCAL_MODEL_CANDIDATES = {
    "specter2": [
        Path("/data2/yunx/hf-models/allenai--specter2_base"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--specter2_base",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--specter2",
    ],
    "scincl": [
        Path("/data2/yunx/hf-models/malteos--scincl"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--malteos--scincl",
    ],
}


@dataclass(slots=True)
class CandidateReviewer:
    author_id: str
    author_name: str
    score: float
    rank: int
    papers: list[dict[str, Any]]
    matched_paper: dict[str, Any] | None
    matched_publication: dict[str, Any] | None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal OpenReview-style reviewer baseline on top of existing reviewer-evaluate outputs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reviewer-lists", help="Existing reviewer_lists.json path for one paper.")
    source.add_argument("--summary-json", help="Existing summary_reviewers_*.json path.")
    parser.add_argument("--paper-index", type=int, default=0, help="0-based index into --summary-json results.")
    parser.add_argument("--output-dir", default=None, help="Override output dir. Default: same folder as reviewer_lists.json.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with Neo4j settings.")
    parser.add_argument(
        "--model",
        choices=("scincl", "specter2"),
        default="scincl",
        help="Scientific encoder used for reviewer-publication reranking.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, default cpu.")
    parser.add_argument("--candidate-limit", type=int, default=200, help="How many KG candidate authors to rerank.")
    parser.add_argument(
        "--papers-per-author",
        type=int,
        default=20,
        help="How many publications per candidate author to use as expertise archive.",
    )
    parser.add_argument("--reviewer-count", type=int, default=10, help="How many final reviewers to keep.")
    parser.add_argument(
        "--aggregate",
        choices=("max", "mean_top3"),
        default="max",
        help="How to aggregate target-paper vs reviewer-publication similarities.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print final payload.")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def resolve_local_hf_snapshot(path: Path) -> str | None:
    if not path.exists():
        return None
    snapshots_dir = path / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted(item for item in snapshots_dir.iterdir() if item.is_dir())
        if snapshots:
            return str(snapshots[-1].resolve())
    if (path / "config.json").exists():
        return str(path.resolve())
    return None


def resolve_model_source(model_key: str) -> str:
    for candidate in LOCAL_MODEL_CANDIDATES.get(model_key, []):
        resolved = resolve_local_hf_snapshot(candidate)
        if resolved:
            return resolved
    if model_key == "specter2":
        return SPECTER2_MODEL_NAME
    return SCINCL_MODEL_NAME


def neo4j_connection(env_path: str | None) -> tuple[str, str, str, str]:
    env_values = load_env_values(Path(env_path).expanduser().resolve() if env_path else None)
    uri = first_non_empty(env_values.get("NEO4J_URI"), DEFAULT_NEO4J_URI)
    user = first_non_empty(env_values.get("NEO4J_USER"), DEFAULT_NEO4J_USER)
    password = first_non_empty(env_values.get("NEO4J_PASSWORD"), DEFAULT_NEO4J_PASSWORD)
    database = first_non_empty(env_values.get("NEO4J_DB"), env_values.get("NEO4J_DATABASE"), "neo4j")
    return uri, user, password, database


def load_run_context(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if args.reviewer_lists:
        reviewer_lists_path = Path(args.reviewer_lists).expanduser().resolve()
        reviewer_lists = read_json(reviewer_lists_path)
        output_dir = reviewer_lists_path.parent
        search_result = read_json(output_dir / "search_result.json")
        idea_context_path = output_dir / "idea_context.json"
        idea_context = read_json(idea_context_path) if idea_context_path.exists() else {}
        return output_dir, reviewer_lists, search_result, idea_context

    summary_path = Path(args.summary_json).expanduser().resolve()
    summary_payload = read_json(summary_path)
    rows = summary_payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No results found in {summary_path}")
    if args.paper_index < 0 or args.paper_index >= len(rows):
        raise IndexError(f"--paper-index out of range: {args.paper_index}, total={len(rows)}")
    row = rows[args.paper_index]
    if not isinstance(row, dict):
        raise ValueError("Selected summary row is not an object")
    reviewer_lists_path = Path(row["reviewer_lists_path"]).expanduser().resolve()
    output_dir = reviewer_lists_path.parent
    reviewer_lists = read_json(reviewer_lists_path)
    search_result = read_json(output_dir / "search_result.json")
    idea_context_path = output_dir / "idea_context.json"
    idea_context = read_json(idea_context_path) if idea_context_path.exists() else {}
    paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
    if paper:
        idea_context.setdefault("paper", paper)
    return output_dir, reviewer_lists, search_result, idea_context


def normalized_title(text: Any) -> str:
    return "".join(ch.lower() for ch in normalize_whitespace(text) if ch.isalnum())


def target_submission(idea_context: dict[str, Any], reviewer_lists: dict[str, Any]) -> dict[str, Any]:
    paper = idea_context.get("paper") if isinstance(idea_context.get("paper"), dict) else {}
    title = normalize_whitespace(
        first_non_empty(
            paper.get("title"),
            idea_context.get("retrieval_seed", {}).get("title") if isinstance(idea_context.get("retrieval_seed"), dict) else "",
        )
    )
    abstract = normalize_whitespace(
        first_non_empty(
            paper.get("abstract"),
            idea_context.get("retrieval_seed", {}).get("abstract") if isinstance(idea_context.get("retrieval_seed"), dict) else "",
            reviewer_lists.get("input", {}).get("idea_text") if isinstance(reviewer_lists.get("input"), dict) else "",
        )
    )
    if not title and not abstract:
        raise ValueError("Unable to infer target paper title/abstract from existing run artifacts")
    paper_id = normalize_whitespace(first_non_empty(paper.get("source_key"), paper.get("pdf_path"), title, "target-paper"))
    return {
        "id": paper_id,
        "content": {
            "title": title,
            "abstract": abstract,
        },
    }


def target_paper_identifiers(idea_context: dict[str, Any]) -> tuple[str, str]:
    paper = idea_context.get("paper") if isinstance(idea_context.get("paper"), dict) else {}
    return (
        normalize_whitespace(first_non_empty(paper.get("title"))),
        normalize_whitespace(first_non_empty(paper.get("doi"))).lower(),
    )


def collect_candidate_author_ids(search_result: dict[str, Any], candidate_limit: int) -> list[str]:
    kg_authors = search_result.get("sources", {}).get("kg", {}).get("authors", [])
    if not isinstance(kg_authors, list):
        kg_authors = []
    author_ids: list[str] = []
    seen: set[str] = set()
    for item in kg_authors:
        if not isinstance(item, dict):
            continue
        author_id = normalize_whitespace(item.get("author_id"))
        if not author_id or author_id in seen:
            continue
        seen.add(author_id)
        author_ids.append(author_id)
        if len(author_ids) >= candidate_limit:
            break
    return author_ids


def fetch_candidate_archives(
    author_ids: list[str],
    *,
    env_path: str | None,
    papers_per_author: int,
    target_title_key: str,
    target_doi: str,
) -> dict[str, list[dict[str, Any]]]:
    if not author_ids:
        return {}

    uri, user, password, database = neo4j_connection(env_path)
    query = """
    UNWIND $author_ids AS aid
    MATCH (a:Author {id: aid})-[r:AUTHORED]->(p:Paper)
    WHERE p.has_abstract = true
      AND p.abstract IS NOT NULL
      AND trim(p.abstract) <> ''
    WITH a, p, r
    ORDER BY a.id, p.publication_year DESC, coalesce(p.cited_by_count, 0) DESC
    WITH a, collect({
      id: p.id,
      title: p.title,
      abstract: p.abstract,
      doi: p.doi,
      year: p.publication_year,
      cited_by_count: coalesce(p.cited_by_count, 0),
      position: r.position,
      is_corresponding: r.is_corresponding
    })[..$papers_per_author] AS papers
    RETURN
      a.id AS author_id,
      a.display_name AS author_name,
      a.works_count AS works_count,
      papers AS papers
    """

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            rows = session.run(
                query,
                author_ids=author_ids,
                papers_per_author=max(papers_per_author, 1),
            )
            archives: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                author_id = normalize_whitespace(row["author_id"])
                papers = row.get("papers")
                if not author_id or not isinstance(papers, list):
                    continue
                cleaned: list[dict[str, Any]] = []
                for paper in papers:
                    if not isinstance(paper, dict):
                        continue
                    title = normalize_whitespace(paper.get("title"))
                    abstract = normalize_whitespace(paper.get("abstract"))
                    doi = normalize_whitespace(paper.get("doi")).lower()
                    if not title and not abstract:
                        continue
                    if title and normalized_title(title) == target_title_key:
                        continue
                    if target_doi and doi and doi == target_doi:
                        continue
                    cleaned.append(
                        {
                            "id": normalize_whitespace(paper.get("id")) or title,
                            "content": {
                                "title": title,
                                "abstract": abstract,
                            },
                            "doi": doi,
                            "year": paper.get("year"),
                            "cited_by_count": paper.get("cited_by_count"),
                            "author_position": paper.get("position"),
                            "is_corresponding": paper.get("is_corresponding"),
                        }
                    )
                if cleaned:
                    archives[author_id] = cleaned
            return archives
    finally:
        driver.close()


def fetch_target_author_ids(
    *,
    env_path: str | None,
    target_title: str,
    target_doi: str,
) -> set[str]:
    if not target_title and not target_doi:
        return set()

    uri, user, password, database = neo4j_connection(env_path)
    query = """
    MATCH (a:Author)-[:AUTHORED]->(p:Paper)
    WHERE ($target_doi <> '' AND toLower(coalesce(p.doi, '')) = $target_doi)
       OR ($target_title_key <> '' AND toLower(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(
            coalesce(p.title, ''),
            ' ', ''), '-', ''), ':', ''), ';', ''), ',', ''), '.', ''), '(', ''), ')', ''), '/', ''), '†', '')) = $target_title_key)
    RETURN DISTINCT a.id AS author_id
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            rows = session.run(
                query,
                target_doi=target_doi,
                target_title_key=normalized_title(target_title),
            )
            return {
                normalize_whitespace(row["author_id"])
                for row in rows
                if normalize_whitespace(row["author_id"])
            }
    finally:
        driver.close()


class TextEncoder:
    def __init__(self, model_name: str, *, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        embeddings: list[torch.Tensor] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            if total_batches > 1:
                log(f"[embed] batch {batch_index}/{total_batches} size={len(batch)}")
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            pooled = output.last_hidden_state[:, 0, :]
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)


def publication_text(publication: dict[str, Any]) -> str:
    content = publication.get("content") if isinstance(publication.get("content"), dict) else {}
    title = normalize_whitespace(content.get("title"))
    abstract = normalize_whitespace(content.get("abstract"))
    if title and abstract:
        return f"{title} [SEP] {abstract}"
    return title or abstract


def build_candidates(
    *,
    author_ids: list[str],
    metadata_by_id: dict[str, dict[str, Any]],
    archives: dict[str, list[dict[str, Any]]],
    encoder: TextEncoder,
    submission: dict[str, Any],
    aggregate: str,
) -> list[CandidateReviewer]:
    submission_text = publication_text(submission)
    log("[score] encoding target submission")
    query_emb = encoder.encode([submission_text])[0]

    flat_papers: list[dict[str, Any]] = []
    paper_owner_ids: list[str] = []
    owner_to_slice: dict[str, tuple[int, int]] = {}
    offset = 0
    for author_id in author_ids:
        papers = archives.get(author_id) or []
        if not papers:
            continue
        start = offset
        flat_papers.extend(papers)
        paper_owner_ids.extend([author_id] * len(papers))
        offset += len(papers)
        owner_to_slice[author_id] = (start, offset)

    log(f"[score] encoding {len(flat_papers)} reviewer publications from {len(owner_to_slice)} candidate reviewers")
    flat_texts = [publication_text(paper) for paper in flat_papers]
    flat_embs = encoder.encode(flat_texts)
    all_scores = torch.matmul(flat_embs, query_emb)

    candidates: list[CandidateReviewer] = []
    for rank, author_id in enumerate(author_ids, start=1):
        papers = archives.get(author_id) or []
        if not papers:
            continue
        start_end = owner_to_slice.get(author_id)
        if start_end is None:
            continue
        start, end = start_end
        scores = all_scores[start:end]
        if scores.numel() == 0:
            continue
        if aggregate == "mean_top3":
            topk = min(3, scores.shape[0])
            reviewer_score = float(torch.topk(scores, k=topk).values.mean().item())
        else:
            reviewer_score = float(scores.max().item())
        best_index = int(scores.argmax().item())
        meta = metadata_by_id.get(author_id, {})
        author_name = normalize_whitespace(first_non_empty(meta.get("display_name"), author_id))
        matched_publication = papers[best_index]
        candidates.append(
            CandidateReviewer(
                author_id=author_id,
                author_name=author_name,
                score=reviewer_score,
                rank=rank,
                papers=papers,
                matched_paper=matched_publication,
                matched_publication=matched_publication,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.rank, item.author_name.casefold()))
    return candidates


def serialize_reviewers(candidates: list[CandidateReviewer], reviewer_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:reviewer_count], start=1):
        matched = candidate.matched_publication or {}
        matched_content = matched.get("content") if isinstance(matched.get("content"), dict) else {}
        reviewer = {
            "selection_rank": index,
            "author_id": candidate.author_id,
            "author_name": candidate.author_name,
            "display_name": candidate.author_name,
            "score": round(candidate.score, 6),
            "source": "openreview_style_dense_affinity",
            "reviewer_key": ensure_reviewer_key(
                {
                    "author_id": candidate.author_id,
                    "author_name": candidate.author_name,
                }
            ),
            "works_count": len(candidate.papers),
            "matched_publication_id": normalize_whitespace(matched.get("id")),
            "matched_publication_title": normalize_whitespace(matched_content.get("title")),
            "matched_publication_year": matched.get("year"),
        }
        rows.append(reviewer)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir, reviewer_lists, search_result, idea_context = load_run_context(args)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    submission = target_submission(idea_context, reviewer_lists)
    log(f"[start] paper={normalize_whitespace(submission.get('content', {}).get('title'))[:140]}")
    target_title, target_doi = target_paper_identifiers(idea_context)
    author_ids = collect_candidate_author_ids(search_result, args.candidate_limit)
    if not author_ids:
        raise ValueError("No KG candidate authors found in search_result.json")
    log(f"[candidates] initial author candidates={len(author_ids)}")

    target_title_key = normalized_title(submission.get("content", {}).get("title"))
    log("[neo4j] fetching target authors for COI filtering")
    target_author_ids = fetch_target_author_ids(
        env_path=args.env,
        target_title=target_title or normalize_whitespace(submission.get("content", {}).get("title")),
        target_doi=target_doi,
    )
    author_ids = [author_id for author_id in author_ids if author_id not in target_author_ids]
    log(f"[candidates] after target-author exclusion={len(author_ids)} removed={len(target_author_ids)}")
    log("[neo4j] fetching candidate author archives")
    archives = fetch_candidate_archives(
        author_ids,
        env_path=args.env,
        papers_per_author=args.papers_per_author,
        target_title_key=target_title_key,
        target_doi=target_doi,
    )
    kept_author_ids = [author_id for author_id in author_ids if author_id in archives]
    log(f"[archives] reviewers with usable archives={len(kept_author_ids)}")
    log("[neo4j] fetching author metadata")
    metadata_by_id = fetch_author_metadata_by_id(kept_author_ids, args.env)

    model_name = resolve_model_source(args.model)
    log(f"[model] loading model source={model_name} device={args.device} batch_size={args.batch_size}")
    encoder = TextEncoder(model_name, device=args.device, batch_size=args.batch_size)
    candidates = build_candidates(
        author_ids=kept_author_ids,
        metadata_by_id=metadata_by_id,
        archives=archives,
        encoder=encoder,
        submission=submission,
        aggregate=args.aggregate,
    )
    reviewers = serialize_reviewers(candidates, args.reviewer_count)
    log(f"[done] selected reviewers={len(reviewers)} elapsed_s={round(time.perf_counter() - started_at, 1)}")

    result = {
        "status": "ok" if reviewers else "error",
        "method": "openreview",
        "selection_strategy": "kg-candidate-recall-plus-openreview-style-dense-affinity",
        "model": {
            "name": args.model,
            "hf_model": model_name,
            "aggregate": args.aggregate,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "submission": submission,
        "candidate_author_count": len(author_ids),
        "excluded_target_author_count": len(target_author_ids),
        "candidate_author_with_archives_count": len(kept_author_ids),
        "reviewer_count": len(reviewers),
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "openreview_reviewers": reviewers,
    }

    openreview_result_path = output_dir / "openreview_baseline.json"
    write_json(openreview_result_path, result)

    merged_reviewer_lists = dict(reviewer_lists)
    merged_reviewer_lists["openreview_reviewers"] = reviewers
    paths = merged_reviewer_lists.get("paths")
    if isinstance(paths, dict):
        paths["openreview_result_path"] = str(openreview_result_path.resolve())
    write_json(output_dir / "reviewer_lists.json", merged_reviewer_lists)

    return {
        **result,
        "output_dir": str(output_dir.resolve()),
        "reviewer_lists_path": str((output_dir / "reviewer_lists.json").resolve()),
        "openreview_result_path": str(openreview_result_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
