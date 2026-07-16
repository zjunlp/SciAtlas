#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUBSET_JSON = SCRIPT_DIR / "dataset_runs" / "pairs_v2_final" / "openreview_subset_70.json"
DEFAULT_ENV_PATH = SCRIPT_DIR.parent.parent / ".env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenReview-style baseline over a prepared subset manifest."
    )
    parser.add_argument("--subset-json", default=str(DEFAULT_SUBSET_JSON), help="Subset manifest from build_openreview_subset.py.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with Neo4j settings.")
    parser.add_argument("--model", choices=("specter2", "scincl"), default="specter2")
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--papers-per-author", type=int, default=10)
    parser.add_argument("--reviewer-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aggregate", choices=("max", "mean_top3"), default="max")
    parser.add_argument("--start-index", type=int, default=0, help="0-based offset into subset list.")
    parser.add_argument("--max-papers", type=int, default=0, help="0 means all remaining papers.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subset_path = Path(args.subset_json).expanduser().resolve()
    subset = read_json(subset_path)
    rows = [row for row in subset.get("results", []) if isinstance(row, dict)]
    rows = rows[args.start_index :]
    if args.max_papers > 0:
        rows = rows[: args.max_papers]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    summary_path = subset_path.parent / f"openreview_subset_run_{run_id}.json"
    results: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    print(f"subset_json={subset_path}", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"paper_count={len(rows)}", flush=True)
    print(
        f"model={args.model} candidate_limit={args.candidate_limit} papers_per_author={args.papers_per_author} "
        f"reviewer_count={args.reviewer_count} batch_size={args.batch_size} device={args.device}",
        flush=True,
    )
    print(f"summary_path={summary_path}", flush=True)

    for index, row in enumerate(rows, start=1):
        reviewer_lists_path = Path(row["reviewer_lists_path"]).expanduser().resolve()
        title = (row.get("paper") or {}).get("title")
        print(f"[{index}/{len(rows)}] start title={title}", flush=True)
        cmd = [
            sys.executable,
            str((SCRIPT_DIR / "run_openreview_baseline.py").resolve()),
            "--reviewer-lists",
            str(reviewer_lists_path),
            "--env",
            str(Path(args.env).expanduser().resolve()),
            "--model",
            args.model,
            "--candidate-limit",
            str(args.candidate_limit),
            "--papers-per-author",
            str(args.papers_per_author),
            "--reviewer-count",
            str(args.reviewer_count),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--aggregate",
            args.aggregate,
        ]

        started_at = time.perf_counter()
        completed = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent), capture_output=True, text=True, check=False)
        item: dict[str, Any] = {
            "index": index,
            "paper": row.get("paper"),
            "reviewer_lists_path": str(reviewer_lists_path),
            "returncode": completed.returncode,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "status": "ok" if completed.returncode == 0 else "error",
        }
        results.append(item)
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
        print(
            json.dumps(
                {
                    "index": index,
                    "title": (row.get("paper") or {}).get("title"),
                    "status": item["status"],
                    "elapsed_ms": item["elapsed_ms"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        payload = {
            "status": "running",
            "subset_json": str(subset_path),
            "requested_paper_count": len(rows),
            "success_count": sum(1 for result in results if result["status"] == "ok"),
            "failure_count": sum(1 for result in results if result["status"] != "ok"),
            "finished_count": len(results),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "results": results,
        }
        write_json(summary_path, payload)
        if completed.returncode != 0 and args.stop_on_error:
            break

    success_count = sum(1 for item in results if item["status"] == "ok")
    failure_count = sum(1 for item in results if item["status"] != "ok")
    payload = {
        "status": "ok" if failure_count == 0 else "partial_error",
        "subset_json": str(subset_path),
        "requested_paper_count": len(rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "results": results,
    }
    write_json(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
