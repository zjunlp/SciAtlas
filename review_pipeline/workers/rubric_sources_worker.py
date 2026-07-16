#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker_common import build_file_logger, load_json, normalize_whitespace, write_json
from cuda_devices import default_torch_device, device_at, first_configured_cuda_devices, format_cuda_devices

RUBRIC_SCHEMA_VERSION = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rubric sources worker.")
    parser.add_argument("--input-file", required=True, help="JSON payload for the worker.")
    parser.add_argument("--work-dir", required=True, help="Directory for artifacts and result.json.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--smoke", action="store_true", help="Write stub artifacts without running the real flow.")
    return parser


def run_smoke(payload: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    papers_path = work_dir / "paper_contexts.json"
    write_json(papers_path, {"papers": []})
    write_json(work_dir / "retrieved_papers.json", {"papers": []})
    write_json(work_dir / "processing_index.json", {"papers": []})
    return {
        "status": "ok",
        "mode": "smoke",
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "result_path": str(papers_path),
        "paper_contexts_path": str(papers_path),
        "retrieved_papers_path": str((work_dir / "retrieved_papers.json").resolve()),
        "processing_index_path": str((work_dir / "processing_index.json").resolve()),
        "work_dir": str(work_dir),
    }


def visible_device_count(cuda_visible_devices: str | None) -> int:
    visible = normalize_whitespace(cuda_visible_devices) or normalize_whitespace(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not visible:
        return 0
    return len([item.strip() for item in visible.split(",") if item.strip()])


def run_real(payload: dict[str, Any], work_dir: Path, cuda_visible_devices: str | None) -> dict[str, Any]:
    from review.idea_rubric import DEFAULT_MAX_WORKERS, IdeaRubricConfig, run_rubric_sources

    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    visible_count = visible_device_count(cuda_visible_devices)
    configured_devices = first_configured_cuda_devices()
    default_embed_device = device_at(configured_devices, 0) or default_torch_device(index=0)
    default_rerank_device = device_at(configured_devices, 1) or default_torch_device(
        index=0 if visible_count == 1 else 1
    )

    config = IdeaRubricConfig(
        target_idea=normalize_whitespace(payload.get("idea_text")),
        artifact_root=work_dir,
        sources_root=work_dir,
        output_path=(work_dir / "unused.idea_rubric.json"),
        env_path=Path(payload["env_path"]).expanduser().resolve() if payload.get("env_path") else None,
        llm_api_key=payload.get("llm_api_key"),
        search_top_k=int(payload["search_top_k"]) if payload.get("search_top_k") is not None else 50,
        search_final_k=int(payload["search_final_k"]) if payload.get("search_final_k") is not None else 15,
        max_workers=int(payload["max_workers"]) if payload.get("max_workers") is not None else DEFAULT_MAX_WORKERS,
        embed_device=normalize_whitespace(payload.get("embed_device")) or default_embed_device,
        rerank_device=normalize_whitespace(payload.get("rerank_device")) or default_rerank_device,
    )

    result = run_rubric_sources(config)
    return {
        "status": result["status"],
        "mode": "real",
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "result_path": result["result_path"],
        "paper_contexts_path": result["paper_contexts_path"],
        "retrieved_papers_path": result["retrieved_papers_path"],
        "processing_index_path": result["processing_index_path"],
        "timing_log_path": result.get("timing_log_path"),
        "work_dir": str(work_dir),
        "cuda_devices": format_cuda_devices(configured_devices),
    }


def main() -> int:
    args = build_parser().parse_args()
    started_at = time.perf_counter()
    input_path = Path(args.input_file).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = build_file_logger(work_dir / "worker.log", Path(__file__).name)
    payload = load_json(input_path)
    logger.info(
        "Worker started mode=%s input_file=%s work_dir=%s cuda_visible_devices=%s",
        "smoke" if args.smoke else "real",
        input_path,
        work_dir,
        args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    result_path = work_dir / "result.json"
    try:
        if args.smoke:
            result = run_smoke(payload, work_dir)
        else:
            result = run_real(payload, work_dir, args.cuda_visible_devices)
        result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
        write_json(result_path, result)
        logger.info(
            "Worker finished status=%s elapsed_ms=%s result_path=%s",
            result.get("status"),
            result.get("elapsed_ms"),
            result_path,
        )
        return 0
    except Exception as exc:
        error_payload = {
            "status": "error",
            "mode": "smoke" if args.smoke else "real",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }
        write_json(result_path, error_payload)
        logger.exception("Worker failed error=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
