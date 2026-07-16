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

RUBRIC_SCHEMA_VERSION = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rubric llm worker.")
    parser.add_argument("--input-file", required=True, help="JSON payload for the worker.")
    parser.add_argument("--work-dir", required=True, help="Directory for artifacts and result.json.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--smoke", action="store_true", help="Write stub artifacts without running the real flow.")
    return parser


def run_smoke(payload: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    output_path = work_dir / "idea_rubric.json"
    idea_text = normalize_whitespace(payload.get("idea_text"))
    smoke_standard = {
        "source_tag": "General",
        "core_philosophy": "Smoke rubric standard used to validate the four-section review pipeline.",
        "required_evidence": "A concrete, section-specific claim and supporting evidence in the idea or source text.",
    }
    write_json(
        output_path,
        {
            "schema_version": RUBRIC_SCHEMA_VERSION,
            "Idea_Summary": idea_text[:200],
            "Motivation_Standards": [
                {
                    **smoke_standard,
                    "dimension_name": "Smoke Motivation Framing",
                    "target_section": "Motivation",
                }
            ],
            "Method_Standards": [
                {
                    **smoke_standard,
                    "dimension_name": "Smoke Method Mechanism",
                    "target_section": "Method",
                }
            ],
            "Result_Standards": [
                {
                    **smoke_standard,
                    "dimension_name": "Smoke Result Evidence",
                    "target_section": "Result",
                }
            ],
            "Discussion_Standards": [
                {
                    **smoke_standard,
                    "dimension_name": "Smoke Discussion Boundary",
                    "target_section": "Discussion",
                }
            ],
            "Idea_Breakdown": None,
            "source": "smoke",
        },
    )
    write_json(work_dir / "processing_index.json", {"papers": []})
    (work_dir / "historical_context.txt").write_text("", encoding="utf-8")
    return {
        "status": "ok",
        "mode": "smoke",
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "result_path": str(output_path),
        "processing_index_path": str((work_dir / "processing_index.json").resolve()),
        "historical_context_path": str((work_dir / "historical_context.txt").resolve()),
        "work_dir": str(work_dir),
    }


def run_real(payload: dict[str, Any], work_dir: Path, cuda_visible_devices: str | None) -> dict[str, Any]:
    from review.idea_rubric import DEFAULT_MAX_WORKERS, IdeaRubricConfig, run_rubric_llm

    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    config = IdeaRubricConfig(
        target_idea=normalize_whitespace(payload.get("idea_text")),
        artifact_root=work_dir,
        sources_root=Path(payload["sources_root"]).expanduser().resolve(),
        output_path=(work_dir / "idea_rubric.json"),
        env_path=Path(payload["env_path"]).expanduser().resolve() if payload.get("env_path") else None,
        llm_api_key=payload.get("llm_api_key"),
        llm_timeout_seconds=int(payload["llm_timeout_seconds"])
        if payload.get("llm_timeout_seconds") is not None
        else 120,
        max_workers=int(payload["max_workers"]) if payload.get("max_workers") is not None else DEFAULT_MAX_WORKERS,
    )

    result = run_rubric_llm(config)
    output_path = Path(result["result_path"]).resolve()
    return {
        "status": result["status"],
        "mode": "real",
        "rubric_schema_version": result.get("rubric_schema_version"),
        "result_path": str(output_path),
        "processing_index_path": result["processing_index_path"],
        "historical_context_path": result["historical_context_path"],
        "general_rubric_path": result.get("general_rubric_path"),
        "detailed_rubric_path": result.get("detailed_rubric_path"),
        "synthesis_rubric_path": result.get("synthesis_rubric_path"),
        "timing_log_path": result.get("timing_log_path"),
        "work_dir": str(work_dir),
        "sources_root": str(Path(payload["sources_root"]).expanduser().resolve()),
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
