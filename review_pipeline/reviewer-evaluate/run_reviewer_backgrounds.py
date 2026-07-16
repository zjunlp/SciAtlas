#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKER_PATH = REPO_ROOT / "workers" / "reviewer_background_worker.py"
DEFAULT_SUMMARY = (
    SCRIPT_DIR
    / "dataset_runs"
    / "pairs_v2_final"
    / "summary_reviewers_20260529_130404.json"
)
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"


def normalize(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def slugify(text: Any, *, limit: int = 80) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalize(text)).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:limit].rstrip("-") if slug else "") or "item"


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
        description="Generate reviewer academic backgrounds from reviewer-evaluate reviewer_lists.json files."
    )
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY), help="run_pairs_dataset summary JSON.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with Neo4j and LLM settings.")
    parser.add_argument(
        "--methods",
        default="pipeline,baseline",
        help="Comma-separated methods to run: pipeline,baseline,our_wo_keywords,our_wo_graphwalk,openreview.",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum parallel reviewer worker processes.")
    parser.add_argument(
        "--device",
        default="",
        help=(
            "Torch device passed to each reviewer background worker, e.g. cpu or cuda:0. "
            "If unset, the worker chooses its default device."
        ),
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated torch CUDA devices to round-robin, e.g. 1, cuda:1, or cuda:0,cuda:1. "
            "The runner passes these as the worker input device and does not set CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many tasks after filtering.")
    parser.add_argument("--max-tasks", type=int, default=0, help="Run at most N tasks. 0 means all.")
    parser.add_argument("--only-paper-pattern", default="", help="Only run papers whose title/path/domain contains text.")
    parser.add_argument("--only-reviewer-pattern", default="", help="Only run reviewers whose name/id contains text.")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun even if the worker result already has a usable author_summary.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned task count without launching workers.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Batch run id for logs/status. Defaults to backgrounds_<timestamp>.",
    )
    parser.add_argument(
        "--persona-subject",
        default="computer science",
        help="Reviewer persona subject passed to the worker.",
    )
    return parser


def method_key(method: str) -> str:
    if method == "pipeline":
        return "pipeline_reviewers"
    if method == "baseline":
        return "baseline_reviewers"
    if method == "our_wo_keywords":
        return "our_wo_keywords_reviewers"
    if method == "our_wo_graphwalk":
        return "our_wo_graphwalk_reviewers"
    if method == "openreview":
        return "openreview_reviewers"
    raise ValueError(f"Unknown method: {method}")


def reviewer_identity(reviewer: dict[str, Any]) -> tuple[str, str, str]:
    author_id = normalize(reviewer.get("author_id"))
    author_name = normalize(reviewer.get("author_name")) or normalize(reviewer.get("display_name"))
    key = normalize(reviewer.get("reviewer_key")) or author_id or author_name
    return key, author_id, author_name


def summary_has_profile(summary_path: Path) -> bool:
    try:
        payload = read_json(summary_path)
    except Exception:
        return False
    return bool(normalize(payload.get("overall_academic_profile")))


def cached_ok(work_dir: Path) -> bool:
    result_path = work_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        payload = read_json(result_path)
    except Exception:
        return False
    if normalize(payload.get("status")) != "ok":
        return False
    summary_path = normalize(payload.get("summary_path"))
    return bool(summary_path and summary_has_profile(Path(summary_path)))


def load_idea_text(output_dir: Path) -> str:
    idea_context_path = output_dir / "idea_context.json"
    payload = read_json(idea_context_path)
    retrieval_seed = payload.get("retrieval_seed") if isinstance(payload.get("retrieval_seed"), dict) else {}
    source_full_text = payload.get("source_full_text") if isinstance(payload.get("source_full_text"), dict) else {}
    return (
        normalize(retrieval_seed.get("text"))
        or normalize(source_full_text.get("text"))
        or normalize(source_full_text.get("abstract"))
    )


def collect_tasks(args: argparse.Namespace, run_id: str) -> list[dict[str, Any]]:
    summary_path = Path(args.summary_json).expanduser().resolve()
    summary = read_json(summary_path)
    methods = [normalize(item) for item in args.methods.split(",") if normalize(item)]
    paper_pattern = normalize(args.only_paper_pattern).casefold()
    reviewer_pattern = normalize(args.only_reviewer_pattern).casefold()

    tasks: list[dict[str, Any]] = []
    for result_index, result in enumerate(summary.get("results", []), start=1):
        if not isinstance(result, dict) or normalize(result.get("status")) != "ok":
            continue
        paper = result.get("paper") if isinstance(result.get("paper"), dict) else {}
        output_dir = Path(normalize(result.get("output_dir"))).expanduser().resolve()
        reviewer_lists_path = Path(normalize(result.get("reviewer_lists_path"))).expanduser().resolve()
        paper_match_text = "\n".join(
            [
                normalize(paper.get("pair_id")),
                normalize(paper.get("paper_key")),
                normalize(paper.get("title")),
                normalize(paper.get("domain")),
                normalize(paper.get("pdf_path")),
            ]
        ).casefold()
        if paper_pattern and paper_pattern not in paper_match_text:
            continue
        if not reviewer_lists_path.exists():
            continue
        try:
            idea_text = load_idea_text(output_dir)
            reviewer_payload = read_json(reviewer_lists_path)
        except Exception:
            continue
        if not idea_text:
            continue

        for method in methods:
            reviewers = reviewer_payload.get(method_key(method), [])
            if not isinstance(reviewers, list):
                continue
            for reviewer_index, reviewer in enumerate(reviewers, start=1):
                if not isinstance(reviewer, dict):
                    continue
                reviewer_key, author_id, author_name = reviewer_identity(reviewer)
                reviewer_match_text = f"{reviewer_key}\n{author_id}\n{author_name}".casefold()
                if reviewer_pattern and reviewer_pattern not in reviewer_match_text:
                    continue
                if not author_id and not author_name:
                    continue
                reviewer_slug = slugify(reviewer_key or author_name or author_id, limit=100)
                work_dir = output_dir / "reviewer_backgrounds" / method / f"{reviewer_index:02d}_{reviewer_slug}"
                tasks.append(
                    {
                        "task_id": len(tasks) + 1,
                        "summary_result_index": result_index,
                        "method": method,
                        "reviewer_index": reviewer_index,
                        "reviewer_key": reviewer_key,
                        "author_id": author_id,
                        "author_name": author_name,
                        "paper": paper,
                        "output_dir": str(output_dir),
                        "reviewer_lists_path": str(reviewer_lists_path),
                        "idea_text": idea_text,
                        "work_dir": str(work_dir),
                        "input_path": str((work_dir / "input.json").resolve()),
                        "run_id": run_id,
                    }
                )

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_tasks < 0:
        raise ValueError("--max-tasks must be non-negative")
    tasks = tasks[args.start_index :]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    return tasks


def prepare_task_input(task: dict[str, Any], args: argparse.Namespace) -> None:
    work_dir = Path(task["work_dir"])
    payload = {
        "reviewer_key": task["reviewer_key"],
        "author_id": task["author_id"],
        "author_name": task["author_name"],
        "idea_text": task["idea_text"],
        "env_path": str(Path(args.env).expanduser().resolve()) if args.env else None,
        "search_method": "id" if task["author_id"] else "name",
        "persona_subject": args.persona_subject,
        "device": task.get("device") or normalize(args.device),
        "source": {
            "method": task["method"],
            "reviewer_index": task["reviewer_index"],
            "reviewer_lists_path": task["reviewer_lists_path"],
            "paper": task["paper"],
        },
    }
    write_json(work_dir / "input.json", payload)


def launch_task(task: dict[str, Any], args: argparse.Namespace, torch_device: str | None) -> subprocess.Popen[str]:
    if torch_device:
        task["device"] = torch_device
    prepare_task_input(task, args)
    work_dir = Path(task["work_dir"])
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    command = [
        sys.executable,
        str(WORKER_PATH.resolve()),
        "--input-file",
        str((work_dir / "input.json").resolve()),
        "--work-dir",
        str(work_dir.resolve()),
    ]
    (work_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=stdout,
        stderr=stderr,
        text=True,
        env=env,
    )
    process._innoeval_stdout = stdout  # type: ignore[attr-defined]
    process._innoeval_stderr = stderr  # type: ignore[attr-defined]
    return process


def close_process_files(process: subprocess.Popen[str]) -> None:
    for attr in ("_innoeval_stdout", "_innoeval_stderr"):
        handle = getattr(process, attr, None)
        if handle:
            handle.close()


def task_status(task: dict[str, Any], returncode: int | None) -> dict[str, Any]:
    work_dir = Path(task["work_dir"])
    result_path = work_dir / "result.json"
    payload: dict[str, Any] = {
        "task_id": task["task_id"],
        "method": task["method"],
        "reviewer_index": task["reviewer_index"],
        "reviewer_key": task["reviewer_key"],
        "author_id": task["author_id"],
        "author_name": task["author_name"],
        "paper": task["paper"],
        "work_dir": str(work_dir),
        "result_path": str(result_path),
        "returncode": returncode,
    }
    if result_path.exists():
        try:
            result = read_json(result_path)
            payload["status"] = normalize(result.get("status")) or ("ok" if returncode == 0 else "error")
            payload["summary_path"] = result.get("summary_path")
            payload["relevant_papers_path"] = result.get("relevant_papers_path")
            payload["error"] = result.get("error")
            payload["error_type"] = result.get("error_type")
        except Exception as exc:
            payload["status"] = "error"
            payload["error_type"] = exc.__class__.__name__
            payload["error"] = str(exc)
    else:
        payload["status"] = "error" if returncode else "unknown"
        payload["error"] = f"Missing worker result: {result_path}"
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    run_id = normalize(args.run_id) or f"backgrounds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tasks = collect_tasks(args, run_id)
    status_root = Path(args.summary_json).expanduser().resolve().parent / "_background_status" / run_id
    status_root.mkdir(parents=True, exist_ok=True)
    summary_path = status_root / "summary.json"

    pending: deque[dict[str, Any]] = deque()
    skipped: list[dict[str, Any]] = []
    for task in tasks:
        if not args.rerun and cached_ok(Path(task["work_dir"])):
            skipped.append({**task_status(task, 0), "status": "cached"})
        else:
            pending.append(task)

    print(f"summary_json={Path(args.summary_json).expanduser().resolve()}")
    print(f"run_id={run_id}")
    print(f"methods={args.methods}")
    print(f"planned_tasks={len(tasks)} pending={len(pending)} cached={len(skipped)}")
    print(f"max_workers={args.max_workers} device={normalize(args.device)} cuda_devices={args.cuda_devices or ''}")
    print(f"status_summary={summary_path}")
    if args.dry_run:
        write_json(
            summary_path,
            {
                "status": "dry_run",
                "run_id": run_id,
                "planned_task_count": len(tasks),
                "pending_task_count": len(pending),
                "cached_count": len(skipped),
                "tasks_preview": list(tasks[:20]),
            },
        )
        return 0

    cuda_devices = []
    for item in (args.cuda_devices or "").split(","):
        text = normalize(item)
        if not text:
            continue
        cuda_devices.append(text if text.startswith("cuda:") else f"cuda:{text}")
    active: dict[subprocess.Popen[str], tuple[dict[str, Any], str | None, float]] = {}
    results: list[dict[str, Any]] = list(skipped)
    launched = 0
    success_count = sum(1 for item in skipped if item.get("status") == "cached")
    failure_count = 0
    started_at = time.perf_counter()

    while pending or active:
        while pending and len(active) < args.max_workers:
            task = pending.popleft()
            torch_device = cuda_devices[launched % len(cuda_devices)] if cuda_devices else None
            process = launch_task(task, args, torch_device)
            active[process] = (task, torch_device, time.perf_counter())
            launched += 1
            print(
                f"launched {launched}/{len(tasks) - len(skipped)} "
                f"task_id={task['task_id']} method={task['method']} "
                f"reviewer={task['reviewer_index']} device={torch_device or ''} "
                f"paper={normalize(task['paper'].get('title'))[:90]}"
            )

        finished = [process for process in active if process.poll() is not None]
        if not finished:
            time.sleep(2)
            continue

        for process in finished:
            task, torch_device, task_started_at = active.pop(process)
            returncode = process.returncode
            close_process_files(process)
            item = task_status(task, returncode)
            item["device"] = torch_device
            item["elapsed_ms"] = round((time.perf_counter() - task_started_at) * 1000, 1)
            results.append(item)
            if item.get("status") == "ok":
                success_count += 1
            else:
                failure_count += 1
            print(
                f"finished task_id={task['task_id']} status={item.get('status')} "
                f"returncode={returncode} elapsed_ms={item['elapsed_ms']}"
            )
            if len(results) % 20 == 0 or not pending and not active:
                write_json(
                    summary_path,
                    {
                        "status": "running" if pending or active else ("ok" if failure_count == 0 else "partial_error"),
                        "run_id": run_id,
                        "planned_task_count": len(tasks),
                        "success_count": success_count,
                        "cached_count": len(skipped),
                        "failure_count": failure_count,
                        "pending_count": len(pending),
                        "active_count": len(active),
                        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
                        "results": results,
                    },
                )

    write_json(
        summary_path,
        {
            "status": "ok" if failure_count == 0 else "partial_error",
            "run_id": run_id,
            "planned_task_count": len(tasks),
            "success_count": success_count,
            "cached_count": len(skipped),
            "failure_count": failure_count,
            "pending_count": 0,
            "active_count": 0,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "results": results,
        },
    )
    print(f"done success_count={success_count} cached_count={len(skipped)} failure_count={failure_count}")
    print(f"summary_path={summary_path}")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
