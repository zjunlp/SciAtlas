#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = REPO_ROOT / "dataset" / "pairs_v2_final.json"
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "dataset_runs"


def normalize(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def slugify(text: Any, *, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalize(text)).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        return "paper"
    return slug[:limit].rstrip("-") or "paper"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reviewer-list selection for papers in pairs_v2_final-style datasets."
    )
    parser.add_argument("--pair-json", default=str(DEFAULT_DATASET), help="Input pair dataset JSON.")
    parser.add_argument(
        "--paper-key",
        choices=("all", "paper_nc", "paper2"),
        default="all",
        help="Which paper field(s) to run.",
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="Root output dir. Default: reviewer-evaluate/dataset_runs/<pair-json-stem>.",
    )
    parser.add_argument("--run-prefix", default="reviewers", help="Prefix added to the timestamped run id.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many unique papers after filtering.")
    parser.add_argument("--max-papers", type=int, default=0, help="Run at most N unique papers. 0 means all.")
    parser.add_argument("--only-pattern", default="", help="Only keep papers whose title/path/domain contains this text.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with Neo4j settings.")
    parser.add_argument(
        "--methods",
        default="pipeline,baseline",
        help="Comma-separated methods to output: pipeline,baseline,our_wo_keywords,our_wo_graphwalk.",
    )
    parser.add_argument("--reviewer-count", type=int, default=10, help="Number of reviewers selected per method.")
    parser.add_argument("--reviewer-min-works-count", type=int, default=10)
    parser.add_argument("--reviewer-selection-top-k", type=int, default=30)
    parser.add_argument("--kg-top-k", type=int, default=50)
    parser.add_argument("--baseline-scan-limit", type=int, default=50)
    parser.add_argument("--target-field", default=None)
    parser.add_argument("--after", default=None)
    parser.add_argument("--before", default=None)
    parser.add_argument("--kg-embedding-device", default=None)
    parser.add_argument("--kg-reranker-device", default=None)
    parser.add_argument("--pdf-input-mode", choices=("pipeline", "kg-pdf"), default="pipeline")
    parser.add_argument("--grobid-base-url", default="http://127.0.0.1:8070")
    parser.add_argument("--grobid-start-page", type=int, default=None)
    parser.add_argument("--disable-baseline-author-dedupe", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue after a paper fails. Enabled by default.",
    )
    parser.add_argument("--stop-on-error", action="store_false", dest="continue_on_error")
    return parser


def iter_dataset_pairs(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload.get("pairs"), list):
        return [("pairs", item) for item in payload["pairs"] if isinstance(item, dict)]

    pairs: list[tuple[str, dict[str, Any]]] = []
    for group_name, items in payload.items():
        if isinstance(items, list):
            pairs.extend((str(group_name), item) for item in items if isinstance(item, dict))
    return pairs


def collect_papers(args: argparse.Namespace) -> list[dict[str, Any]]:
    pair_json = Path(args.pair_json).expanduser().resolve()
    payload = read_json(pair_json)
    paper_keys = ["paper_nc", "paper2"] if args.paper_key == "all" else [args.paper_key]
    only_pattern = normalize(args.only_pattern).casefold()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group_name, item in iter_dataset_pairs(payload):
        pair_id = normalize(item.get("id"))
        for paper_key in paper_keys:
            paper = item.get(paper_key)
            if not isinstance(paper, dict):
                continue
            raw_path = normalize(paper.get("path"))
            title = normalize(paper.get("title")) or Path(raw_path).stem
            domain = normalize(paper.get("domain")) or normalize(paper.get("source_domain")) or group_name
            unique_key = raw_path or title
            if not unique_key or unique_key in seen:
                continue
            match_text = f"{title}\n{raw_path}\n{domain}\n{pair_id}\n{paper_key}".casefold()
            if only_pattern and only_pattern not in match_text:
                continue
            seen.add(unique_key)
            rows.append(
                {
                    "pair_id": pair_id,
                    "paper_key": paper_key,
                    "title": title,
                    "domain": domain or "unknown",
                    "pdf_path": str(Path(raw_path).expanduser().resolve()) if raw_path else "",
                    "paper_slug": slugify(title, limit=80),
                    "domain_slug": slugify(domain or "unknown", limit=40),
                    "source_key": unique_key,
                    "doi": normalize(paper.get("doi")),
                    "citations": paper.get("citations"),
                    "abstract": normalize(paper.get("abstract")),
                    "source_pair_id": normalize(paper.get("source_pair_id")),
                    "source_dataset": normalize(paper.get("source_dataset") or item.get("source_dataset")),
                }
            )

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_papers < 0:
        raise ValueError("--max-papers must be non-negative")
    rows = rows[args.start_index :]
    if args.max_papers:
        rows = rows[: args.max_papers]
    return rows


def build_select_command(args: argparse.Namespace, paper: dict[str, Any], output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "select_reviewers.py"),
        "--pdf-path",
        paper["pdf_path"],
        "--output-dir",
        str(output_dir),
        "--env",
        str(Path(args.env).expanduser().resolve()),
        "--methods",
        args.methods,
        "--reviewer-count",
        str(args.reviewer_count),
        "--reviewer-min-works-count",
        str(args.reviewer_min_works_count),
        "--reviewer-selection-top-k",
        str(args.reviewer_selection_top_k),
        "--kg-top-k",
        str(args.kg_top_k),
        "--baseline-scan-limit",
        str(args.baseline_scan_limit),
        "--pdf-input-mode",
        args.pdf_input_mode,
        "--grobid-base-url",
        args.grobid_base_url,
    ]
    if args.grobid_start_page is not None:
        cmd.extend(["--grobid-start-page", str(args.grobid_start_page)])
    if args.target_field:
        cmd.extend(["--target-field", args.target_field])
    if args.after:
        cmd.extend(["--after", args.after])
    if args.before:
        cmd.extend(["--before", args.before])
    if args.kg_embedding_device:
        cmd.extend(["--kg-embedding-device", args.kg_embedding_device])
    if args.kg_reranker_device:
        cmd.extend(["--kg-reranker-device", args.kg_reranker_device])
    if args.disable_baseline_author_dedupe:
        cmd.append("--disable-baseline-author-dedupe")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair_json = Path(args.pair_json).expanduser().resolve()
    if not pair_json.exists():
        raise FileNotFoundError(f"Pair JSON not found: {pair_json}")

    result_root = (
        Path(args.result_root).expanduser().resolve()
        if args.result_root
        else (DEFAULT_RESULT_ROOT / pair_json.stem).resolve()
    )
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{slugify(args.run_prefix, limit=40)}_{run_ts}" if args.run_prefix else run_ts
    papers = collect_papers(args)

    summary_path = result_root / f"summary_{run_id}.json"
    batch_status_dir = result_root / "_batch_status" / run_id
    batch_log_dir = result_root / "_batch_logs" / run_id
    if not args.dry_run:
        batch_status_dir.mkdir(parents=True, exist_ok=True)
        batch_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"pair_json={pair_json}")
    print(f"paper_key={args.paper_key}")
    print(f"result_root={result_root}")
    print(f"run_id={run_id}")
    print(f"matched_papers={len(papers)}")
    print()

    results: list[dict[str, Any]] = []
    success_count = 0
    skipped_count = 0
    failure_count = 0
    started_at = time.perf_counter()

    for index, paper in enumerate(papers, start=1):
        paper_root = result_root / paper["domain_slug"] / paper["paper_slug"]
        output_dir = paper_root / run_id
        log_path = batch_log_dir / f"{index:04d}_{paper['paper_slug']}.log"
        status_path = batch_status_dir / f"{index:04d}_{paper['paper_slug']}.status.json"
        cmd = build_select_command(args, paper, output_dir)

        print(f"[{index}/{len(papers)}] {paper['title']}")
        print(f"  domain={paper['domain']}")
        print(f"  pdf={paper['pdf_path']}")
        print(f"  output_dir={output_dir}")

        if not paper["pdf_path"] or not Path(paper["pdf_path"]).exists():
            item = {
                "status": "skip",
                "reason": "missing_pdf",
                "paper": paper,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
            skipped_count += 1
            results.append(item)
            if not args.dry_run:
                write_json(status_path, item)
            print("  skipped=missing_pdf")
            continue

        if args.dry_run:
            item = {
                "status": "dry_run",
                "paper": paper,
                "output_dir": str(output_dir),
                "command": cmd,
            }
            results.append(item)
            print("  dry-run:", " ".join(cmd))
            continue

        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        log_path.write_text(
            "COMMAND\n"
            + " ".join(cmd)
            + "\n\nSTDOUT\n"
            + (completed.stdout or "")
            + "\n\nSTDERR\n"
            + (completed.stderr or ""),
            encoding="utf-8",
        )

        reviewer_lists_path = output_dir / "reviewer_lists.json"
        if completed.returncode == 0 and reviewer_lists_path.exists():
            try:
                reviewer_payload = read_json(reviewer_lists_path)
                pipeline_count = len(reviewer_payload.get("pipeline_reviewers", []))
                baseline_count = len(reviewer_payload.get("baseline_reviewers", []))
                wo_keywords_count = len(reviewer_payload.get("our_wo_keywords_reviewers", []))
                wo_graphwalk_count = len(reviewer_payload.get("our_wo_graphwalk_reviewers", []))
            except Exception:
                pipeline_count = 0
                baseline_count = 0
                wo_keywords_count = 0
                wo_graphwalk_count = 0
            item = {
                "status": "ok",
                "paper": paper,
                "output_dir": str(output_dir),
                "reviewer_lists_path": str(reviewer_lists_path),
                "pipeline_reviewer_count": pipeline_count,
                "baseline_reviewer_count": baseline_count,
                "our_wo_keywords_reviewer_count": wo_keywords_count,
                "our_wo_graphwalk_reviewer_count": wo_graphwalk_count,
                "log_path": str(log_path),
            }
            success_count += 1
            print(
                "  ok"
                f" pipeline_reviewers={pipeline_count}"
                f" baseline_reviewers={baseline_count}"
                f" our_wo_keywords_reviewers={wo_keywords_count}"
                f" our_wo_graphwalk_reviewers={wo_graphwalk_count}"
            )
        else:
            item = {
                "status": "fail",
                "returncode": completed.returncode,
                "paper": paper,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
            failure_count += 1
            print(f"  failed returncode={completed.returncode}")
            if not args.continue_on_error:
                results.append(item)
                write_json(status_path, item)
                break

        results.append(item)
        write_json(status_path, item)

    summary = {
        "status": "ok" if failure_count == 0 else "partial_error",
        "pair_json": str(pair_json),
        "paper_key": args.paper_key,
        "result_root": str(result_root),
        "run_id": run_id,
        "matched_paper_count": len(papers),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "results": results,
    }
    if not args.dry_run:
        write_json(summary_path, summary)
    print()
    print(f"summary_path={summary_path}")
    print(f"success_count={success_count} skipped_count={skipped_count} failure_count={failure_count}")
    return 0 if failure_count == 0 or args.continue_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
