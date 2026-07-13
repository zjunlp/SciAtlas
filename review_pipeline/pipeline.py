#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from cuda_devices import device_at, format_cuda_devices, parse_cuda_devices, unique_cuda_devices


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_RUN_ROOT = REPO_ROOT / "pipeline_runs"
WORKERS_DIR = REPO_ROOT / "workers"
REVIEWER_WORKER_PATH = WORKERS_DIR / "reviewer_background_worker.py"
RUBRIC_SOURCES_WORKER_PATH = WORKERS_DIR / "rubric_sources_worker.py"
RUBRIC_LLM_WORKER_PATH = WORKERS_DIR / "rubric_llm_worker.py"
LOG_FORMAT = "%(asctime)s %(source_file)s %(levelname)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def slugify(text: str, *, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalize_whitespace(text)).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        return "run"
    return slug[:limit].rstrip("-") or "run"


def preserve_text_block(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def reset_stage_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rewrite_text_paths(path: Path, replacements: list[tuple[str, str]]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    updated = text
    for old, new in replacements:
        if old:
            updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def materialize_cache_directory(
    source_dir: Path,
    target_dir: Path,
    *,
    cache_source_dir: Path,
    run_dir: Path,
) -> None:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    if source_dir == target_dir:
        return
    if not source_dir.exists():
        return

    # Cache hits are copied into the current run so every run remains self-contained.
    # Downstream stages must receive paths under target_dir, not references to source_dir.
    reset_stage_directory(target_dir)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    replacements: list[tuple[str, str]] = []
    if source_dir.name == target_dir.name:
        replacements.append((str(source_dir.parent), str(target_dir.parent)))
    if source_dir.parent.name == target_dir.parent.name:
        replacements.append((str(source_dir.parent.parent), str(target_dir.parent.parent)))
    replacements.extend([
        (str(cache_source_dir.expanduser().resolve()), str(run_dir.expanduser().resolve())),
        (str(source_dir), str(target_dir)),
    ])
    for path in target_dir.rglob("*"):
        if path.is_file():
            rewrite_text_paths(path, replacements)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    payload = read_json(path)
    payload["config"] = str(path)
    return payload


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-level InnoEval orchestrator.")
    parser.add_argument("--config", default=None, help="Optional JSON config file for pipeline defaults.")
    parser.add_argument("--idea-text", default=None, help="Idea text used as the top-level input.")
    parser.add_argument("--pdf-path", default=None, help="PDF path used as the top-level input.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Root directory for pipeline runs.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier. Defaults to timestamped slug.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to shared .env file.")
    parser.add_argument("--smoke", action="store_true", help="Run stubbed stages without the real heavy pipeline.")
    parser.add_argument(
        "--disable-run-cache",
        action="store_true",
        help="Ignore existing artifacts in the run directory and rerun all enabled stages.",
    )
    parser.add_argument(
        "--force-stages",
        default="",
        help="Comma-separated stage names to rerun even when cached artifacts exist.",
    )
    parser.add_argument(
        "--cache-source-run-id",
        default=None,
        help="Optional run id under --run-root to use as the explicit cache source.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Comma-separated physical CUDA device ids used by every stage that loads CUDA models, e.g. 3 or 3,5.",
    )

    parser.add_argument("--kg-top-k", type=int, default=20, help="KG search top-k.")
    parser.add_argument("--s2-top-k", type=int, default=20, help="S2 search top-k.")
    parser.add_argument("--manifest-top-k", type=int, default=20, help="Paper count sent into pdf/xml stage.")
    parser.add_argument("--manifest-max-workers", type=int, default=4, help="Maximum parallel manifest paper workers.")
    parser.add_argument(
        "--manifest-grobid-max-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent GROBID requests during manifest generation.",
    )
    parser.add_argument("--grounding-final-top-k", type=int, default=8, help="Final grounding top-k.")
    parser.add_argument(
        "--disable-grounding-refinement",
        action="store_true",
        help="Skip LLM refinement for final paragraph grounding matches.",
    )
    parser.add_argument(
        "--grounding-device",
        default=None,
        help="Deprecated. Use --cuda-devices; kept as a compatibility alias.",
    )
    parser.add_argument("--disable-llm-ranking", action="store_true", help="Skip search-stage LLM reranking.")

    parser.add_argument("--disable-reviewers", action="store_true", help="Skip reviewer background branch.")
    parser.add_argument("--disable-rubric", action="store_true", help="Skip idea rubric branch.")
    parser.add_argument("--max-reviewers", type=int, default=3, help="Maximum reviewer candidates to fan out.")
    parser.add_argument(
        "--reviewer-min-works-count",
        type=int,
        default=10,
        help="Minimum summed works_count required for a merged reviewer name candidate.",
    )
    parser.add_argument(
        "--reviewer-selection-top-k",
        type=int,
        default=30,
        help="Restrict reviewer sampling to the top-K ranked merged candidates before even spacing.",
    )
    parser.add_argument(
        "--reviewer-max-workers",
        type=int,
        default=6,
        help="Maximum parallel reviewer subprocesses.",
    )
    parser.add_argument(
        "--reviewer-gpu-devices",
        default=None,
        help="Deprecated. Use --cuda-devices; kept as a compatibility alias.",
    )
    parser.add_argument(
        "--reviewer-persona-subject",
        default="computer science",
        help="Subject persona group merged with general for reviewer persona sampling.",
    )
    parser.add_argument(
        "--rubric-gpu-devices",
        default=None,
        help="Deprecated. Use --cuda-devices; kept as a compatibility alias.",
    )
    parser.add_argument(
        "--rubric-search-top-k",
        type=int,
        default=None,
        help="Optional override for rubric retrieval SEARCH_TOP_K.",
    )
    parser.add_argument(
        "--rubric-search-final-k",
        type=int,
        default=None,
        help="Optional override for rubric retrieval SEARCH_FINAL_K.",
    )
    parser.add_argument(
        "--rubric-max-workers",
        type=int,
        default=None,
        help="Optional override for rubric processing MAX_WORKERS.",
    )
    parser.add_argument(
        "--rubric-llm-timeout-seconds",
        type=int,
        default=300,
        help="Per-call timeout for rubric LLM requests.",
    )
    parser.add_argument("--disable-review-stage", action="store_true", help="Skip reviewer evaluation stage.")
    parser.add_argument("--disable-report-stage", action="store_true", help="Skip report synthesis stage.")
    parser.add_argument(
        "--short-report",
        action="store_true",
        help="Generate the compact meta-review report variant instead of the full reviewer-report document.",
    )
    parser.add_argument("--disable-review-llm", action="store_true", help="Use deterministic review/report fallback.")
    parser.add_argument("--review-max-workers", type=int, default=5, help="Maximum parallel reviewer evaluation tasks.")
    parser.add_argument("--review-llm-api-key", default=None, help="Optional review/report LLM API key override.")
    parser.add_argument("--review-llm-base-url", default="https://www.dmxapi.cn/v1", help="Review/report LLM base URL.")
    parser.add_argument("--review-llm-model-name", default="DeepSeek-V3.2", help="Review/report LLM model name.")
    parser.add_argument("--review-temperature", type=float, default=0.2, help="Review/report LLM temperature.")
    parser.add_argument(
        "--review-llm-timeout-seconds",
        type=int,
        default=600,
        help="Per-call timeout for review/report LLM requests.",
    )
    parser.add_argument(
        "--review-llm-max-retries",
        type=int,
        default=3,
        help="Logical retries for review/report LLM requests.",
    )
    parser.add_argument(
        "--review-max-evidence-cards-per-section",
        type=int,
        default=40,
        help="Maximum paragraph evidence cards supplied to one section review.",
    )
    parser.add_argument(
        "--review-max-evidence-cards-per-query",
        type=int,
        default=4,
        help="Maximum paragraph evidence cards from one grounding query in a section review.",
    )
    parser.add_argument(
        "--review-max-experiment-cards-per-section",
        type=int,
        default=20,
        help="Maximum experiment grounding cards supplied to Method/Result/Discussion reviews.",
    )
    parser.add_argument(
        "--review-max-total-evidence-chars-per-section",
        type=int,
        default=60000,
        help="Approximate character budget for evidence supplied to one section review.",
    )
    parser.add_argument(
        "--enable-figure-cards",
        action="store_true",
        help="Enable multimodal figure-card extraction and feed figure evidence into review prompts.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Directory containing one figure PDF per manuscript figure.",
    )
    parser.add_argument(
        "--review-max-figure-cards-per-section",
        type=int,
        default=4,
        help="Maximum figure cards supplied to one section review when figure cards are enabled.",
    )

    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    defaults = load_config_defaults(bootstrap_args.config)
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    if bool(args.idea_text) == bool(args.pdf_path):
        parser.error("Exactly one of --idea-text or --pdf-path must be provided.")
    return args


def make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return slugify(args.run_id, limit=120)
    source = normalize_whitespace(args.idea_text) if args.idea_text else Path(args.pdf_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{slugify(source, limit=60)}"


def stage_dir(run_dir: Path, stage_name: str) -> Path:
    path = run_dir / stage_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_file_logger(log_path: Path, source_file: str) -> logging.LoggerAdapter:
    resolved_path = log_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"innoeval.{source_file}.{resolved_path}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(resolved_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)

    return logging.LoggerAdapter(logger, {"source_file": source_file})


def log_to_many(loggers: list[logging.LoggerAdapter], level: str, message: str) -> None:
    for logger in loggers:
        getattr(logger, level)(message)


def log_multiline(
    loggers: list[logging.LoggerAdapter],
    level: str,
    source_label: str,
    text: str,
) -> None:
    for line in text.splitlines():
        cleaned = line.rstrip()
        if cleaned:
            log_to_many(loggers, level, f"[{source_label}] {cleaned}")


class LogCaptureStream(io.TextIOBase):
    def __init__(self, loggers: list[logging.LoggerAdapter], *, source_label: str, level: str) -> None:
        self.loggers = loggers
        self.source_label = source_label
        self.level = level
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                cleaned = line.rstrip("\r")
                if cleaned:
                    log_to_many(self.loggers, self.level, f"[{self.source_label}] {cleaned}")
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                cleaned = self._buffer.rstrip("\r")
                if cleaned:
                    log_to_many(self.loggers, self.level, f"[{self.source_label}] {cleaned}")
                self._buffer = ""


@contextlib.contextmanager
def capture_stage_output(loggers: list[logging.LoggerAdapter], *, stage_name: str):
    stdout_stream = LogCaptureStream(loggers, source_label=f"{stage_name}.stdout", level="info")
    stderr_stream = LogCaptureStream(loggers, source_label=f"{stage_name}.stderr", level="warning")
    with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
        try:
            yield
        finally:
            stdout_stream.flush()
            stderr_stream.flush()


def parse_device_pool(raw_value: str | None) -> list[str]:
    return parse_cuda_devices(raw_value)


def resolve_pipeline_cuda_devices(args: argparse.Namespace) -> list[str]:
    return unique_cuda_devices(
        [
            args.cuda_devices,
            args.grounding_device,
            args.reviewer_gpu_devices,
            args.rubric_gpu_devices,
            os.environ.get("INNOEVAL_CUDA_DEVICES"),
            os.environ.get("CUDA_DEVICES"),
        ]
    )


def combine_status(stage_statuses: list[str]) -> str:
    normalized = [status for status in stage_statuses if status and status != "skipped"]
    if not normalized:
        return "ok"
    if normalized and all(status == "ok" for status in normalized):
        return "ok"
    if any(status == "ok" for status in normalized):
        return "partial_error"
    return "error"


CACHE_OK_STATUSES = {"ok", "partial_error"}
STAGE_DEPENDENCIES: dict[str, set[str]] = {
    "idea_context": set(),
    "search": {"idea_context"},
    "reviewers": {"search"},
    "rubric_sources": {"idea_context"},
    "rubric_llm": {"idea_context", "rubric_sources"},
    "manifest": {"search"},
    "grounding": {"manifest"},
    "review": {"idea_context", "rubric_llm", "reviewers", "grounding"},
    "report": {"review"},
}


@dataclass(frozen=True)
class CacheCandidate:
    run_dir: Path
    final_payload: dict[str, Any] | None
    config: dict[str, Any] | None
    input_payload: dict[str, Any] | None


def parse_force_stages(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {item.strip().lower() for item in raw_value.split(",") if item.strip()}


def expand_force_stages(force_stages: set[str]) -> set[str]:
    if not force_stages or "all" in force_stages:
        return set(force_stages)

    dependents: dict[str, set[str]] = {stage: set() for stage in STAGE_DEPENDENCIES}
    for stage, dependencies in STAGE_DEPENDENCIES.items():
        for dependency in dependencies:
            dependents.setdefault(dependency, set()).add(stage)

    expanded = set(force_stages)
    pending = list(force_stages)
    while pending:
        stage = pending.pop()
        for dependent in dependents.get(stage, set()):
            if dependent not in expanded:
                expanded.add(dependent)
                pending.append(dependent)
    return expanded


def find_latest_cache_run_dir(
    run_root: Path,
    current_run_id: str,
    current_input_payload: dict[str, Any] | None = None,
) -> Path | None:
    # run_root is expected to contain runs for one paper/input. The input filter is
    # still required because ad-hoc test roots may contain multiple papers.
    if not run_root.exists():
        return None
    candidates = [
        path
        for path in run_root.iterdir()
        if path.is_dir() and path.name != current_run_id and (path / "input.json").exists()
    ]
    if current_input_payload is not None:
        candidates = [
            path
            for path in candidates
            if input_payload_matches_for_cache(
                load_json_if_valid(path / "input.json", lambda payload: True),
                current_input_payload,
            )
        ]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[str, str]:
        timestamp_match = re.search(r"\d{8}_\d{6}", path.name)
        timestamp = timestamp_match.group(0) if timestamp_match else ""
        return timestamp, path.name

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def cache_run_sort_key(path: Path) -> tuple[str, str]:
    timestamp_match = re.search(r"\d{8}_\d{6}", path.name)
    timestamp = timestamp_match.group(0) if timestamp_match else ""
    return timestamp, path.name


def list_cache_candidates(
    run_root: Path,
    current_run_id: str,
    current_input_payload: dict[str, Any],
    *,
    explicit_cache_source: Path | None = None,
) -> list[CacheCandidate]:
    if explicit_cache_source is not None:
        raw_candidates = [explicit_cache_source]
    elif run_root.exists():
        raw_candidates = [
            path
            for path in run_root.iterdir()
            if path.is_dir() and path.name != current_run_id and (path / "input.json").exists()
        ]
        raw_candidates.sort(key=cache_run_sort_key, reverse=True)
    else:
        raw_candidates = []

    candidates: list[CacheCandidate] = []
    for candidate_dir in raw_candidates:
        input_payload = load_json_if_valid(candidate_dir / "input.json", lambda payload: True)
        if not input_payload_matches_for_cache(input_payload, current_input_payload):
            continue
        candidates.append(
            CacheCandidate(
                run_dir=candidate_dir,
                final_payload=load_json_if_valid(candidate_dir / "result.json", lambda payload: True),
                config=load_json_if_valid(candidate_dir / "config.resolved.json", lambda payload: True),
                input_payload=input_payload,
            )
        )
    return candidates


def stage_cache_dir_from_result(
    final_payload: dict[str, Any] | None,
    stage_name: str,
    fallback_dir: Path,
) -> Path:
    if not isinstance(final_payload, dict):
        return fallback_dir
    stages = final_payload.get("stages")
    stage_payload = stages.get(stage_name) if isinstance(stages, dict) else None
    if not isinstance(stage_payload, dict):
        return fallback_dir

    if stage_name == "idea_context":
        path_value = stage_payload.get("result_path")
    elif stage_name == "search":
        path_value = stage_payload.get("result_path")
    elif stage_name == "manifest":
        path_value = stage_payload.get("manifest_path")
    elif stage_name == "grounding":
        path_value = stage_payload.get("result_path")
    elif stage_name == "reviewers":
        path_value = stage_payload.get("index_path")
    elif stage_name == "rubric_sources":
        path_value = stage_payload.get("worker_result_path") or stage_payload.get("result_path")
    elif stage_name == "rubric_llm":
        path_value = stage_payload.get("worker_result_path") or stage_payload.get("result_path")
    elif stage_name == "review":
        path_value = stage_payload.get("reviewer_reviews_path")
    elif stage_name == "report":
        path_value = stage_payload.get("final_report_json_path") or stage_payload.get("final_report_path")
    else:
        path_value = None

    if not path_value:
        return fallback_dir
    try:
        artifact_path = Path(str(path_value)).expanduser()
    except (OSError, ValueError):
        return fallback_dir

    if stage_name == "manifest" and artifact_path.name == "manifest.json" and artifact_path.parent.name == "run":
        return artifact_path.parent.parent
    return artifact_path.parent


def status_allows_cache(payload: dict[str, Any]) -> bool:
    return normalize_whitespace(payload.get("status")).lower() in CACHE_OK_STATUSES


def path_exists(value: Any) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).expanduser().exists()
    except (OSError, ValueError):
        return False


def load_json_if_valid(
    path: Path,
    validator: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not validator(payload):
        return None
    return payload


def cache_allowed(
    *,
    args: argparse.Namespace,
    stage_name: str,
    force_stages: set[str],
    input_matches_existing_run: bool,
) -> bool:
    if args.disable_run_cache:
        return False
    if not input_matches_existing_run:
        return False
    normalized_stage = stage_name.lower()
    return "all" not in force_stages and normalized_stage not in force_stages


def values_match_for_cache(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    return str(left) == str(right)


def paths_match_for_cache(left: Any, right: Any) -> bool:
    if not left or not right:
        return left == right
    if str(left) == str(right):
        return True
    try:
        return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()
    except (OSError, ValueError):
        return False


def input_payload_matches_for_cache(
    previous_input: dict[str, Any] | None,
    current_input: dict[str, Any],
) -> bool:
    if previous_input is None:
        return True
    if previous_input.get("input_type") != current_input.get("input_type"):
        return False
    if current_input.get("input_type") == "pdf":
        return paths_match_for_cache(previous_input.get("pdf_path"), current_input.get("pdf_path"))
    return normalize_whitespace(previous_input.get("idea_text")) == normalize_whitespace(current_input.get("idea_text"))


def config_matches_for_cache(
    previous_config: dict[str, Any] | None,
    args: argparse.Namespace,
    keys: list[str],
) -> bool:
    if not previous_config:
        return True
    for key in keys:
        current_value = getattr(args, key)
        if key not in previous_config:
            if current_value not in (None, "", False):
                return False
            continue
        if not values_match_for_cache(previous_config.get(key), current_value):
            return False
    return True


def cache_key_union(*groups: list[str]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for key in group:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def compatible_cache_candidates(
    *,
    candidates: list[CacheCandidate],
    args: argparse.Namespace,
    config_keys: list[str],
) -> list[CacheCandidate]:
    return [
        candidate
        for candidate in candidates
        if config_matches_for_cache(candidate.config, args, config_keys)
    ]


def select_stage_cache_source(
    *,
    candidates: list[CacheCandidate],
    stage_name: str,
    fallback_run_dir: Path,
    args: argparse.Namespace,
    config_keys: list[str],
    loggers: list[logging.LoggerAdapter],
) -> tuple[Path, Path, dict[str, Any] | None, bool]:
    for candidate in candidates:
        if not config_matches_for_cache(candidate.config, args, config_keys):
            continue
        cache_dir = stage_cache_dir_from_result(
            candidate.final_payload,
            stage_name,
            candidate.run_dir / stage_name,
        )
        log_to_many(
            loggers,
            "info",
            f"Selected cache candidate stage={stage_name} run_dir={candidate.run_dir.resolve()} cache_dir={cache_dir.resolve()}",
        )
        return cache_dir, candidate.run_dir, candidate.config, True
    return fallback_run_dir / stage_name, fallback_run_dir, None, False


def cached_stage_from_candidates(
    *,
    candidates: list[CacheCandidate],
    stage_name: str,
    materialized_output_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    config_keys: list[str],
    loader: Callable[[Path, Path, Path], dict[str, Any] | None],
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    for candidate in candidates:
        if not config_matches_for_cache(candidate.config, args, config_keys):
            continue
        cache_dir = stage_cache_dir_from_result(
            candidate.final_payload,
            stage_name,
            candidate.run_dir / stage_name,
        )
        cached_stage = loader(cache_dir, materialized_output_dir, candidate.run_dir)
        if cached_stage is None:
            log_to_many(
                loggers,
                "info",
                f"Skipping incompatible cache artifact stage={stage_name} run_dir={candidate.run_dir.resolve()}",
            )
            continue
        cached_stage["cache_source_run_dir"] = str(candidate.run_dir.resolve())
        return cached_stage
    return None


def cache_source_path(stage_payload: dict[str, Any] | None) -> Path | None:
    if not isinstance(stage_payload, dict):
        return None
    source = normalize_whitespace(stage_payload.get("cache_source_run_dir"))
    return Path(source) if source else None


def candidates_for_dependencies(
    candidates: list[CacheCandidate],
    *stage_payloads: dict[str, Any] | None,
) -> list[CacheCandidate]:
    sources = [cache_source_path(payload) for payload in stage_payloads]
    if any(source is None for source in sources):
        return []
    normalized_sources = {str(source.expanduser().resolve()) for source in sources if source is not None}
    if len(normalized_sources) != 1:
        return []
    required_source = normalized_sources.pop()
    return [candidate for candidate in candidates if str(candidate.run_dir.expanduser().resolve()) == required_source]


def log_cache_hit(loggers: list[logging.LoggerAdapter], stage_name: str, path: Path) -> None:
    log_to_many(loggers, "info", f"Using cached {stage_name} artifact path={path.resolve()}")


def validate_search_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    ranking_papers = payload.get("ranking", {}).get("papers") if isinstance(payload.get("ranking"), dict) else None
    combined_papers = payload.get("combined", {}).get("papers") if isinstance(payload.get("combined"), dict) else None
    return bool(ranking_papers or combined_papers or isinstance(payload.get("sources"), dict))


def validate_idea_context_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    retrieval_seed = payload.get("retrieval_seed")
    evaluation_target = payload.get("evaluation_target")
    source_full_text = payload.get("source_full_text")
    if not isinstance(retrieval_seed, dict) or not isinstance(evaluation_target, dict):
        return False
    if not isinstance(source_full_text, dict):
        return False
    if not isinstance(evaluation_target.get("structured"), dict):
        return False
    return bool(normalize_whitespace(retrieval_seed.get("text"))) and bool(
        normalize_whitespace(evaluation_target.get("text"))
    ) and bool(
        normalize_whitespace(source_full_text.get("text"))
    )


def validate_manifest_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    papers = payload.get("papers")
    success_count = parse_optional_int(payload.get("success_count"))
    paper_count = parse_optional_int(payload.get("paper_count"))
    return bool((isinstance(papers, list) and papers) or success_count or paper_count)


def validate_grounding_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    input_payload = payload.get("input")
    if isinstance(input_payload, dict) and normalize_whitespace(input_payload.get("type")) != "idea_context":
        return False
    return any(isinstance(payload.get(key), dict) and payload.get(key) for key in ["retrieval", "corpus", "experiment_grounding"])


def validate_reviewer_selection_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    reviewers = payload.get("sampled_reviewers") or payload.get("reviewers")
    return isinstance(reviewers, list)


def validate_reviewer_worker_payload(payload: dict[str, Any]) -> bool:
    if normalize_whitespace(payload.get("status")).lower() != "ok":
        return False
    summary_path = normalize_whitespace(payload.get("summary_path"))
    if not summary_path or not path_exists(summary_path):
        return False
    try:
        summary_payload = read_json(Path(summary_path))
    except Exception:
        return False
    return bool(
        normalize_whitespace(summary_payload.get("overall_academic_profile"))
        or normalize_whitespace(summary_payload.get("source")) == "smoke"
        or isinstance(summary_payload.get("summary"), list)
    )


def validate_reviewers_index_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    if not path_exists(payload.get("idea_context_path")):
        return False
    results = payload.get("results")
    return isinstance(results, list) and all(
        isinstance(item, dict) and validate_reviewer_worker_payload(item)
        for item in results
    )


def validate_rubric_sources_payload(payload: dict[str, Any]) -> bool:
    if normalize_whitespace(payload.get("status")).lower() != "ok":
        return False
    schema_version = parse_optional_int(payload.get("rubric_schema_version"))
    if schema_version is None or schema_version < 5:
        return False
    return path_exists(payload.get("result_path")) and path_exists(payload.get("paper_contexts_path"))


def validate_rubric_llm_payload(payload: dict[str, Any]) -> bool:
    if normalize_whitespace(payload.get("status")).lower() != "ok":
        return False
    schema_version = parse_optional_int(payload.get("rubric_schema_version"))
    if schema_version is None or schema_version < 5:
        return False
    return path_exists(payload.get("result_path")) and path_exists(payload.get("processing_index_path"))


def validate_review_index_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    reviewer_reviews = payload.get("reviewer_reviews")
    if not bool(payload.get("successful_count")) or not isinstance(reviewer_reviews, list):
        return False
    for review in reviewer_reviews:
        if not isinstance(review, dict):
            return False
        overall = review.get("overall") if isinstance(review.get("overall"), dict) else {}
        if review.get("review_prompt_version") != 5:
            return False
        if "score" in overall or "recommendation" in overall:
            return False
        for section in review.get("section_reviews", []):
            if not isinstance(section, dict):
                return False
            if "section_score" in section:
                return False
            if "dimension_reviews" in section:
                return False
            if not isinstance(section.get("strengths"), list) or not isinstance(section.get("weaknesses"), list):
                return False
    return True


def validate_report_payload(payload: dict[str, Any]) -> bool:
    if not status_allows_cache(payload):
        return False
    meta_review = payload.get("meta_review") if isinstance(payload.get("meta_review"), dict) else {}
    if "score" in meta_review:
        return False
    return path_exists(payload.get("final_report_path"))


def validate_report_payload_for_args(payload: dict[str, Any], args: argparse.Namespace) -> bool:
    if not validate_report_payload(payload):
        return False
    if payload.get("report_prompt_version") != 5:
        return False
    return bool(payload.get("short_report")) == bool(args.short_report)


def find_manifest_cache_path(manifest_output_dir: Path) -> Path:
    tagged_path = manifest_output_dir / "run" / "manifest.json"
    if tagged_path.exists():
        return tagged_path
    return manifest_output_dir / "manifest.json"


def cached_search_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    expected_retrieval_text: str = "",
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    result_path = output_dir / "result.json"
    payload = load_json_if_valid(result_path, validate_search_payload)
    if payload is None:
        return None
    if expected_retrieval_text and normalize_whitespace(payload.get("idea_text")) != normalize_whitespace(
        expected_retrieval_text
    ):
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_search_stage(
                materialized_output_dir,
                expected_retrieval_text=expected_retrieval_text,
                loggers=loggers,
            )
    log_cache_hit(loggers, "search", result_path)
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": 0.0,
        "result_path": str(result_path.resolve()),
        "payload": payload,
        "cache": "hit",
    }


def cached_idea_context_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    result_path = output_dir / "idea_context.json"
    payload = load_json_if_valid(result_path, validate_idea_context_payload)
    if payload is None:
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_idea_context_stage(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "idea_context", result_path)
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": 0.0,
        "result_path": str(result_path.resolve()),
        "payload": payload,
        "cache": "hit",
    }


def cached_manifest_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    manifest_path = find_manifest_cache_path(output_dir)
    payload = load_json_if_valid(manifest_path, validate_manifest_payload)
    if payload is None:
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_manifest_stage(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "manifest", manifest_path)
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": 0.0,
        "manifest_path": str(manifest_path.resolve()),
        "payload": payload,
        "cache": "hit",
    }


def cached_grounding_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    result_path = output_dir / "result.json"
    payload = load_json_if_valid(result_path, validate_grounding_payload)
    if payload is None:
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_grounding_stage(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "grounding", result_path)
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": 0.0,
        "result_path": str(result_path.resolve()),
        "payload": payload,
        "cache": "hit",
    }


def cached_rubric_sources_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    worker_result_path = output_dir / "result.json"
    payload = load_json_if_valid(worker_result_path, validate_rubric_sources_payload)
    if payload is None:
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_rubric_sources_stage(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "rubric_sources", worker_result_path)
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "result_path": normalize_whitespace(payload.get("result_path")) or str(worker_result_path.resolve()),
        "worker_result_path": str(worker_result_path.resolve()),
        "rubric_schema_version": payload.get("rubric_schema_version"),
        "paper_contexts_path": normalize_whitespace(payload.get("paper_contexts_path")) or None,
        "sources_root": str(output_dir.resolve()),
        "cache": "hit",
    }


def cached_rubric_llm_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    worker_result_path = output_dir / "result.json"
    payload = load_json_if_valid(worker_result_path, validate_rubric_llm_payload)
    if payload is None:
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_rubric_llm_stage(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "rubric_llm", worker_result_path)
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "result_path": normalize_whitespace(payload.get("result_path")) or str(worker_result_path.resolve()),
        "worker_result_path": str(worker_result_path.resolve()),
        "rubric_schema_version": payload.get("rubric_schema_version"),
        "processing_index_path": normalize_whitespace(payload.get("processing_index_path")) or None,
        "historical_context_path": normalize_whitespace(payload.get("historical_context_path")) or None,
        "cache": "hit",
    }


def cached_review_stage(
    output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    review_index_path = output_dir / "reviewer_reviews.index.json"
    review_index = load_json_if_valid(review_index_path, validate_review_index_payload)
    if review_index is None:
        return None
    if not isinstance(review_index.get("source_full_text_chars"), int):
        return None
    normalized_paths = {
        "normalized_rubric_path": output_dir / "rubric.normalized.json",
        "normalized_reviewers_path": output_dir / "reviewers.normalized.json",
        "searched_papers_path": output_dir / "searched_papers.json",
        "evidence_bank_path": output_dir / "evidence_bank.json",
    }
    if not all(path.exists() for path in normalized_paths.values()):
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_review_stage(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "review", review_index_path)
    return {
        "status": review_index.get("status"),
        "elapsed_ms": 0.0,
        "reviewer_count": review_index.get("reviewer_count", 0),
        "successful_count": review_index.get("successful_count", 0),
        "failed_count": review_index.get("failed_count", 0),
        "reviewer_reviews_path": str(review_index_path.resolve()),
        **{key: str(path.resolve()) for key, path in normalized_paths.items()},
        "cache": "hit",
    }


def cached_report_stage(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    report_json_path = output_dir / "final_report.json"
    report_payload = load_json_if_valid(report_json_path, lambda payload: validate_report_payload_for_args(payload, args))
    if report_payload is None:
        return None
    artifact_paths = report_payload.get("artifact_paths")
    if not isinstance(artifact_paths, dict) or not path_exists(artifact_paths.get("reviewer_reviews_path")):
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(output_dir, materialized_output_dir, cache_source_dir=cache_source_dir, run_dir=run_dir)
        if output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_report_stage(materialized_output_dir, args=args, loggers=loggers)
    log_cache_hit(loggers, "report", report_json_path)
    return {
        "status": report_payload.get("status"),
        "elapsed_ms": 0.0,
        "final_report_path": normalize_whitespace(report_payload.get("final_report_path"))
        or str((output_dir / "final_report.md").resolve()),
        "final_report_json_path": str(report_json_path.resolve()),
        "md_tree_path": normalize_whitespace(report_payload.get("md_tree_path"))
        or str((output_dir / "md_tree.json").resolve()),
        "meta_review": report_payload.get("meta_review"),
        "short_report": bool(report_payload.get("short_report")),
        "cache": "hit",
    }


def cached_reviewers_summary(
    reviewers_output_dir: Path,
    *,
    materialized_output_dir: Path | None = None,
    cache_source_dir: Path | None = None,
    run_dir: Path | None = None,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    index_path = reviewers_output_dir / "index.json"
    payload = load_json_if_valid(index_path, validate_reviewers_index_payload)
    if payload is None:
        return None
    if materialized_output_dir is not None and cache_source_dir is not None and run_dir is not None:
        materialize_cache_directory(
            reviewers_output_dir,
            materialized_output_dir,
            cache_source_dir=cache_source_dir,
            run_dir=run_dir,
        )
        if reviewers_output_dir.expanduser().resolve() != materialized_output_dir.expanduser().resolve():
            return cached_reviewers_summary(materialized_output_dir, loggers=loggers)
    log_cache_hit(loggers, "reviewers", index_path)
    return payload


def build_smoke_search_payload(args: argparse.Namespace, idea_context: dict[str, Any] | None = None) -> dict[str, Any]:
    retrieval_seed = idea_context.get("retrieval_seed") if isinstance(idea_context, dict) else {}
    idea_text = normalize_whitespace(
        retrieval_seed.get("text") if isinstance(retrieval_seed, dict) else ""
    ) or normalize_whitespace(args.idea_text)
    if args.pdf_path:
        pdf_name = Path(args.pdf_path).stem or "smoke-pdf"
        pdf_payload = {
            "title": f"{pdf_name} title",
            "abstract": f"Abstract extracted from {pdf_name} for smoke pipeline validation.",
            "body": "Smoke body text used only for structural validation.",
        }
    else:
        pdf_payload = None

    papers = [
        {
            "id": "W_smoke_1",
            "title": "Smoke Paper One",
            "abstract": "A structural test paper for the smoke pipeline.",
            "year": 2024,
            "doi": "10.0000/smoke.1",
            "pdf_url": "https://example.org/smoke1.pdf",
        },
        {
            "id": "W_smoke_2",
            "title": "Smoke Paper Two",
            "abstract": "Another structural test paper for the smoke pipeline.",
            "year": 2023,
            "doi": "10.0000/smoke.2",
            "pdf_url": "https://example.org/smoke2.pdf",
        },
    ]
    ranking_papers = [
        {
            "title": paper["title"],
            "abstract": paper["abstract"],
            "year": paper["year"],
            "source": "kg" if index == 0 else "s2",
            "source_rank": index + 1,
            "paper": paper,
            "pdf_url": paper["pdf_url"],
        }
        for index, paper in enumerate(papers)
    ]
    payload: dict[str, Any] = {
        "status": "ok",
        "input_type": "pdf" if args.pdf_path else "idea_text",
        "successful_source_count": 2,
        "failed_source_count": 0,
        "idea_text": idea_text or None,
        "pdf_path": str(Path(args.pdf_path).resolve()) if args.pdf_path else None,
        "sources": {
            "kg": {
                "source": "kg",
                "status": "ok",
                "elapsed_ms": 1.0,
                "paper_count": len(papers),
                "papers": papers,
                "author_count": 3,
                "authors": [
                    {"author_id": "A_smoke_1", "name": "Alice Example", "score": 0.97},
                    {"author_id": "A_smoke_2", "name": "Bob Example", "score": 0.91},
                    {"author_id": "A_smoke_3", "name": "Carol Example", "score": 0.87},
                    {"author_id": "A_smoke_4", "name": "David Example", "score": 0.83},
                    {"author_id": "A_smoke_5", "name": "Eve Example", "score": 0.79},
                ],
            },
            "s2": {
                "source": "s2",
                "status": "ok",
                "elapsed_ms": 1.2,
                "paper_count": len(papers),
                "papers": papers,
            },
        },
        "combined": {
            "paper_count": len(ranking_papers),
            "papers": ranking_papers,
        },
        "filter": {
            "unique_paper_count": len(ranking_papers),
            "unique_papers": ranking_papers,
        },
        "ranking": {
            "status": "ok",
            "papers": ranking_papers,
        },
    }
    if pdf_payload is not None:
        payload["sources"]["s2"]["pdf"] = pdf_payload
    return payload


def format_structured_idea_text(structured: dict[str, Any]) -> str:
    sections = [
        ("Basic idea", structured.get("basic_idea")),
        ("Motivation", structured.get("motivation")),
        ("Method", structured.get("method")),
        ("Experimental focus", structured.get("experimental_focus")),
    ]
    blocks: list[str] = []
    for heading, raw_items in sections:
        if not isinstance(raw_items, list):
            continue
        items = [normalize_whitespace(item) for item in raw_items if normalize_whitespace(item)]
        if items:
            blocks.append(heading + ":\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks)


def build_smoke_idea_context(args: argparse.Namespace) -> dict[str, Any]:
    if args.pdf_path:
        title = f"{Path(args.pdf_path).stem or 'smoke-pdf'} title"
        abstract = f"Abstract extracted from {Path(args.pdf_path).name} for smoke pipeline validation."
        source_text = f"Title: {title}\nAbstract: {abstract}\nBody: Smoke full text with methods and experiments."
        retrieval_seed = {
            "source": "smoke.pdf.abstract",
            "title": title,
            "abstract": abstract,
            "text": f"Title: {title}\nAbstract: {abstract}",
        }
    else:
        source_text = normalize_whitespace(args.idea_text)
        retrieval_seed = {
            "source": "input.idea_text",
            "title": "",
            "abstract": "",
            "text": source_text,
        }
    source_full_text = {
        "source": "smoke.pdf.full_text" if args.pdf_path else "input.idea_text",
        "text": source_text,
    }

    structured = {
        "basic_idea": [source_text[:300] or "Smoke idea."],
        "motivation": ["Smoke motivation for structural validation."],
        "method": ["Smoke method for structural validation."],
        "experimental_focus": ["Smoke experiment focus for structural validation."],
    }
    return {
        "status": "ok",
        "input": {
            "input_type": "pdf" if args.pdf_path else "idea_text",
            "idea_text": normalize_whitespace(args.idea_text) if args.idea_text else None,
            "pdf_path": str(Path(args.pdf_path).expanduser().resolve()) if args.pdf_path else None,
        },
        "retrieval_seed": retrieval_seed,
        "evaluation_target": {
            "source": "smoke.structured_idea",
            "format": "structured_idea",
            "text": format_structured_idea_text(structured),
            "structured": structured,
        },
        "source_full_text": source_full_text,
        "extraction": {
            "status": "ok",
            "model": "smoke",
            "source_text_chars": len(source_text),
            "error": None,
        },
    }


def run_idea_context_stage(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any]:
    result_path = output_dir / "idea_context.json"
    started_at = time.perf_counter()
    log_to_many(
        loggers,
        "info",
        (
            f"Starting idea_context stage output_dir={output_dir} "
            f"input_type={'pdf' if args.pdf_path else 'idea_text'} smoke={args.smoke}"
        ),
    )
    with capture_stage_output(loggers, stage_name="idea_context"):
        if args.smoke:
            payload = build_smoke_idea_context(args)
        else:
            from grounding.grounding_pipeline import extract_structured_idea

            input_payload = {
                "input_type": "pdf" if args.pdf_path else "idea_text",
                "idea_text": normalize_whitespace(args.idea_text) if args.idea_text else None,
                "pdf_path": str(Path(args.pdf_path).expanduser().resolve()) if args.pdf_path else None,
            }
            if args.idea_text:
                source_text = normalize_whitespace(args.idea_text)
                source_full_text = {
                    "source": "input.idea_text",
                    "title": "",
                    "abstract": "",
                    "body_chars": 0,
                    "text": source_text,
                }
                retrieval_seed = {
                    "source": "input.idea_text",
                    "title": "",
                    "abstract": "",
                    "text": source_text,
                }
                evaluation_source = "input.idea_text"
            else:
                from s2api.search_s2 import extract_pdf_payload

                pdf_path = Path(args.pdf_path).expanduser().resolve()
                pdf_payload = extract_pdf_payload(
                    pdf_path,
                    grobid_base_url="http://127.0.0.1:8070",
                    grobid_start_page=None,
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
                source_full_text = {
                    "source": "grobid.pdf.full_text",
                    "title": title,
                    "abstract": abstract,
                    "body_chars": len(body),
                    "text": source_text,
                }
                retrieval_seed = {
                    "source": "pdf.abstract",
                    "title": title,
                    "abstract": abstract,
                    "text": retrieval_text,
                    "pdf": {
                        "title": title,
                        "abstract": abstract,
                        "body_chars": len(body),
                    },
                }
                evaluation_source = "pdf.full_text"

            extraction = extract_structured_idea(
                source_text,
                env_path=Path(args.env).expanduser().resolve(),
                debug_dir=output_dir / "debug_llm",
                logger=loggers[0] if loggers else None,
            )
            payload = {
                "status": "ok",
                "input": input_payload,
                "retrieval_seed": retrieval_seed,
                "evaluation_target": {
                    "source": evaluation_source,
                    "format": "structured_idea",
                    "text": extraction["text"],
                    "structured": extraction["structured"],
                },
                "source_full_text": source_full_text,
                "extraction": {
                    "status": extraction["status"],
                    "model": extraction["model"],
                    "source_text_chars": extraction["source_text_chars"],
                    "error": extraction["error"],
                },
            }

    write_json(result_path, payload)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    log_to_many(
        loggers,
        "info",
        (
            f"Finished idea_context stage status={normalize_whitespace(payload.get('status')) or 'ok'} "
            f"retrieval_chars={len(normalize_whitespace(payload.get('retrieval_seed', {}).get('text')))} "
            f"evaluation_chars={len(normalize_whitespace(payload.get('evaluation_target', {}).get('text')))} "
            f"source_full_text_chars={len(str(payload.get('source_full_text', {}).get('text') or ''))} "
            f"elapsed_ms={elapsed_ms} result_path={result_path.resolve()}"
        ),
    )
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": elapsed_ms,
        "result_path": str(result_path.resolve()),
        "payload": payload,
    }


def build_smoke_manifest(search_payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    papers_root = output_dir / "papers"
    papers_root.mkdir(parents=True, exist_ok=True)
    ranking_papers = list(search_payload.get("ranking", {}).get("papers", []))
    manifest_entries: list[dict[str, Any]] = []
    for index, paper in enumerate(ranking_papers, start=1):
        paper_dir = papers_root / f"paper_{index:02d}"
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / "paper.pdf"
        xml_path = paper_dir / "paper.tei.xml"
        parsed_json_path = paper_dir / "parsed_document.json"
        pdf_path.write_text("smoke-pdf-placeholder\n", encoding="utf-8")
        xml_path.write_text("<TEI>smoke</TEI>\n", encoding="utf-8")
        write_json(
            parsed_json_path,
            {
                "title": normalize_whitespace(paper.get("title")),
                "abstract": normalize_whitespace(paper.get("abstract")),
                "sections": [],
            },
        )
        manifest_entries.append(
            {
                "paper_index": index,
                "title": normalize_whitespace(paper.get("title")),
                "paper_dir": str(paper_dir.resolve()),
                "pdf_path": str(pdf_path.resolve()),
                "tei_xml_path": str(xml_path.resolve()),
                "parsed_json_path": str(parsed_json_path.resolve()),
                "status": "ok",
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "ok",
        "requested_top_k": len(manifest_entries),
        "candidate_pool_count": len(manifest_entries),
        "attempted_count": len(manifest_entries),
        "processed_count": len(manifest_entries),
        "success_count": len(manifest_entries),
        "failed_count": 0,
        "selected_count": len(manifest_entries),
        "paper_count": len(manifest_entries),
        "papers": manifest_entries,
        "selected_papers": manifest_entries,
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json(manifest_path, manifest)
    return manifest


def build_smoke_grounding_payload(
    args: argparse.Namespace,
    *,
    idea_text: str,
    idea_context_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "input": {
            "type": "idea_context",
            "idea_context_path": str(idea_context_path.resolve()),
            "pdf_path": str(Path(args.pdf_path).resolve()) if args.pdf_path else None,
        },
        "idea_text": idea_text,
        "pdf_path": str(Path(args.pdf_path).resolve()) if args.pdf_path else None,
        "manifest_path": str(manifest_path.resolve()),
        "retrieval": {
            "results": [],
            "status": "smoke",
        },
        "experiment_grounding": {
            "status": "skipped",
            "reason": "smoke",
        },
    }


def run_search_stage(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    idea_context: dict[str, Any],
    cuda_devices: list[str],
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    started_at = time.perf_counter()
    log_to_many(
        loggers,
        "info",
        (
            f"Starting search stage output_dir={output_dir} "
            f"input_type={'pdf' if args.pdf_path else 'idea_text'} smoke={args.smoke}"
        ),
    )
    with capture_stage_output(loggers, stage_name="search"):
        if args.smoke:
            payload = build_smoke_search_payload(args, idea_context)
        else:
            from search import merge_search

            cli_args: list[str] = [
                "--env",
                str(Path(args.env).expanduser().resolve()),
                "--cache-path",
                str((output_dir / "cache.json").resolve()),
                "--kg-top-k",
                str(args.kg_top_k),
                "--s2-top-k",
                str(args.s2_top_k),
                "--disable-result-log",
            ]
            kg_embedding_device = device_at(cuda_devices, 0)
            kg_reranker_device = device_at(cuda_devices, 1) or kg_embedding_device
            if kg_embedding_device:
                cli_args.extend(["--kg-embedding-device", kg_embedding_device])
            if kg_reranker_device:
                cli_args.extend(["--kg-reranker-device", kg_reranker_device])
            if args.disable_llm_ranking:
                cli_args.append("--disable-llm-ranking")
            retrieval_seed = idea_context.get("retrieval_seed") if isinstance(idea_context, dict) else {}
            retrieval_text = normalize_whitespace(retrieval_seed.get("text") if isinstance(retrieval_seed, dict) else "")
            if not retrieval_text:
                raise ValueError("idea_context.retrieval_seed.text is required for search")
            cli_args.extend(["--idea-text", retrieval_text])
            search_args = merge_search.build_parser().parse_args(cli_args)
            payload = merge_search.run_combined_search(search_args)

    write_json(result_path, payload)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    log_to_many(
        loggers,
        "info",
        (
            f"Finished search stage status={normalize_whitespace(payload.get('status')) or 'ok'} "
            f"paper_count={payload.get('combined', {}).get('paper_count')} elapsed_ms={elapsed_ms} "
            f"result_path={result_path.resolve()}"
        ),
    )
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": elapsed_ms,
        "result_path": str(result_path.resolve()),
        "payload": payload,
    }


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sample_evenly(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    if len(items) <= max_items:
        return list(items)
    if max_items == 1:
        return [items[0]]

    last_index = len(items) - 1
    indexes = [round(index * last_index / (max_items - 1)) for index in range(max_items)]
    sampled: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for index in indexes:
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        sampled.append(items[index])
    return sampled


def reviewer_source_author_ids(reviewer: dict[str, Any]) -> list[str]:
    raw_values = reviewer.get("source_author_ids")
    if raw_values is None:
        raw_values = reviewer.get("source_author_id")
    if isinstance(raw_values, list):
        values = raw_values
    else:
        values = [raw_values]
    return [
        normalize_whitespace(value)
        for value in values
        if normalize_whitespace(value)
    ]


def reviewer_key_for(
    *,
    author_id: Any,
    author_name: Any,
    source_author_ids: Any = None,
) -> str:
    normalized_author_id = normalize_whitespace(author_id)
    if normalized_author_id:
        return f"author-id-{slugify(normalized_author_id, limit=96)}"

    raw_source_ids = source_author_ids if isinstance(source_author_ids, list) else [source_author_ids]
    for source_author_id in raw_source_ids:
        normalized_source_author_id = normalize_whitespace(source_author_id)
        if normalized_source_author_id:
            return f"source-author-id-{slugify(normalized_source_author_id, limit=96)}"

    normalized_name = normalize_whitespace(author_name)
    if normalized_name:
        return f"name-{slugify(normalized_name, limit=96)}"
    return "unknown-reviewer"


def ensure_reviewer_key(reviewer: dict[str, Any]) -> str:
    reviewer_key = normalize_whitespace(reviewer.get("reviewer_key"))
    if not reviewer_key:
        reviewer_key = reviewer_key_for(
            author_id=reviewer.get("author_id") or reviewer.get("representative_author_id"),
            author_name=reviewer.get("author_name") or reviewer.get("name"),
            source_author_ids=reviewer_source_author_ids(reviewer),
        )
        reviewer["reviewer_key"] = reviewer_key
    return reviewer_key


def reviewer_profile_dir(reviewers_dir: Path, reviewer: dict[str, Any]) -> Path:
    return reviewers_dir / "profiles" / ensure_reviewer_key(reviewer)


def selected_reviewer_keys(reviewers: list[Any]) -> list[str]:
    keys: list[str] = []
    for reviewer in reviewers:
        if isinstance(reviewer, dict):
            keys.append(ensure_reviewer_key(reviewer))
    return keys


def legacy_reviewer_dir(reviewers_dir: Path, reviewer: dict[str, Any], index: int) -> Path:
    display_name = (
        normalize_whitespace(reviewer.get("author_name"))
        or normalize_whitespace(reviewer.get("name"))
        or normalize_whitespace(reviewer.get("author_id"))
        or normalize_whitespace(reviewer.get("reviewer_key"))
        or f"reviewer-{index + 1:02d}"
    )
    return reviewers_dir / f"{index + 1:02d}_{slugify(display_name)}"


def reviewer_identity_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_key = normalize_whitespace(left.get("reviewer_key"))
    right_key = normalize_whitespace(right.get("reviewer_key"))
    if left_key and right_key and left_key == right_key:
        return True

    left_author_id = normalize_whitespace(left.get("author_id") or left.get("representative_author_id"))
    right_author_id = normalize_whitespace(right.get("author_id") or right.get("representative_author_id"))
    if left_author_id and right_author_id and left_author_id == right_author_id:
        return True

    left_source_ids = set(reviewer_source_author_ids(left))
    right_source_ids = set(reviewer_source_author_ids(right))
    if left_source_ids and right_source_ids and not left_source_ids.isdisjoint(right_source_ids):
        return True

    left_name = normalize_whitespace(left.get("author_name") or left.get("name")).casefold()
    right_name = normalize_whitespace(right.get("author_name") or right.get("name")).casefold()
    return bool(left_name and right_name and left_name == right_name)


def find_legacy_reviewer_result_dir(
    reviewers_dir: Path,
    reviewer: dict[str, Any],
    index: int,
) -> Path | None:
    indexed_dir = legacy_reviewer_dir(reviewers_dir, reviewer, index)
    if (indexed_dir / "result.json").exists():
        return indexed_dir

    index_payload = load_json_if_valid(reviewers_dir / "index.json", lambda payload: True)
    results = index_payload.get("results") if isinstance(index_payload, dict) else None
    if not isinstance(results, list):
        return None

    for item in results:
        if not isinstance(item, dict) or not reviewer_identity_matches(reviewer, item):
            continue
        work_dir = normalize_whitespace(item.get("work_dir"))
        if not work_dir:
            continue
        try:
            candidate_dir = Path(work_dir).expanduser()
        except (OSError, ValueError):
            continue
        if not candidate_dir.is_absolute():
            candidate_dir = reviewers_dir / candidate_dir
        if (candidate_dir / "result.json").exists():
            return candidate_dir
    return None


def materialize_reviewer_cache(
    *,
    source_dir: Path,
    target_dir: Path,
    cache_source_dir: Path,
    run_dir: Path,
) -> dict[str, Any] | None:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    cache_source_dir = cache_source_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    materialize_cache_directory(
        source_dir,
        target_dir,
        cache_source_dir=cache_source_dir,
        run_dir=run_dir,
    )
    residual_current_dir = Path(str(source_dir).replace(str(cache_source_dir), str(run_dir), 1))
    if residual_current_dir != target_dir:
        for path in target_dir.rglob("*"):
            if path.is_file():
                rewrite_text_paths(
                    path,
                    [
                        (str(residual_current_dir), str(target_dir)),
                        (str(source_dir), str(target_dir)),
                    ],
                )
    return load_json_if_valid(target_dir / "result.json", validate_reviewer_worker_payload)


def cached_reviewers_summary_for_selection(
    *,
    candidates: list[CacheCandidate],
    selected_keys: list[str],
    materialized_output_dir: Path,
    run_dir: Path,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any] | None:
    for candidate in candidates:
        reviewers_dir = stage_cache_dir_from_result(
            candidate.final_payload,
            "reviewers",
            candidate.run_dir / "reviewers",
        )
        index_payload = load_json_if_valid(reviewers_dir / "index.json", validate_reviewers_index_payload)
        if index_payload is None:
            continue

        selection_payload = None
        selection_path = normalize_whitespace(index_payload.get("selection_path"))
        if selection_path:
            try:
                selection_payload = load_json_if_valid(Path(selection_path), validate_reviewer_selection_payload)
            except (OSError, ValueError):
                selection_payload = None
        if selection_payload is None:
            selection_payload = load_json_if_valid(reviewers_dir / "selection.json", validate_reviewer_selection_payload)
        cached_selection = selection_payload.get("sampled_reviewers", []) if selection_payload else []
        if selected_reviewer_keys(cached_selection) != selected_keys:
            log_to_many(
                loggers,
                "info",
                f"Skipping reviewers/index.json cache with different reviewer keys run_dir={candidate.run_dir.resolve()}",
            )
            continue

        materialize_cache_directory(
            reviewers_dir,
            materialized_output_dir,
            cache_source_dir=candidate.run_dir,
            run_dir=run_dir,
        )
        payload = cached_reviewers_summary(materialized_output_dir, loggers=loggers)
        if payload is None:
            continue
        payload["cache_source_run_dir"] = str(candidate.run_dir.resolve())
        return payload
    return None


def find_reviewer_profile_cache(
    *,
    reviewer: dict[str, Any],
    index: int,
    cache_candidates: list[CacheCandidate],
) -> tuple[Path, Path] | None:
    for candidate in cache_candidates:
        reviewers_dir = stage_cache_dir_from_result(
            candidate.final_payload,
            "reviewers",
            candidate.run_dir / "reviewers",
        )
        profile_dir = reviewer_profile_dir(reviewers_dir, reviewer)
        if (profile_dir / "result.json").exists():
            return profile_dir, candidate.run_dir

        legacy_dir = find_legacy_reviewer_result_dir(reviewers_dir, reviewer, index)
        if legacy_dir is not None:
            return legacy_dir, candidate.run_dir
    return None


def collect_reviewer_author_candidates(search_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    kg_authors = search_payload.get("sources", {}).get("kg", {}).get("authors", [])
    if isinstance(kg_authors, list) and kg_authors:
        return [item for item in kg_authors if isinstance(item, dict)], "kg"

    s2_papers = search_payload.get("sources", {}).get("s2", {}).get("papers", [])
    if not isinstance(s2_papers, list) or not s2_papers:
        combined_papers = search_payload.get("combined", {}).get("papers", [])
        if isinstance(combined_papers, list):
            s2_papers = [
                item.get("paper", {})
                for item in combined_papers
                if isinstance(item, dict) and normalize_whitespace(item.get("source")) == "s2"
            ]
        else:
            s2_papers = []

    candidates: list[dict[str, Any]] = []
    for paper_rank, paper in enumerate(s2_papers, start=1):
        if not isinstance(paper, dict):
            continue
        authors = paper.get("authors", [])
        if not isinstance(authors, list):
            continue
        paper_score = 1.0 / max(paper_rank, 1)
        for author_rank, author in enumerate(authors, start=1):
            if not isinstance(author, dict):
                continue
            author_name = normalize_whitespace(author.get("name"))
            if not author_name:
                continue
            candidates.append(
                {
                    "author_id": "",
                    "source_author_id": normalize_whitespace(author.get("authorId") or author.get("author_id")),
                    "name": author_name,
                    "score": round(paper_score / max(author_rank, 1), 6),
                    "source": "s2",
                    "paper_rank": paper_rank,
                    "author_rank": author_rank,
                }
            )
    return candidates, "s2" if candidates else "none"


def fetch_author_metadata_by_id(author_ids: list[str], env_path: str | None) -> dict[str, dict[str, Any]]:
    if not author_ids:
        return {}

    from neo4j import GraphDatabase
    from review.author_background import DEFAULT_NEO4J_PASSWORD, DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER
    from review.common import first_non_empty, load_env_values

    env_values = load_env_values(Path(env_path).expanduser().resolve() if env_path else None)
    uri = first_non_empty(env_values.get("NEO4J_URI"), DEFAULT_NEO4J_URI)
    user = first_non_empty(env_values.get("NEO4J_USER"), DEFAULT_NEO4J_USER)
    password = first_non_empty(env_values.get("NEO4J_PASSWORD"), DEFAULT_NEO4J_PASSWORD)

    query = """
    UNWIND $author_ids AS aid
    MATCH (a:Author {id: aid})
    RETURN
        a.id AS author_id,
        a.display_name AS display_name,
        a.works_count AS works_count
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            rows = session.run(query, author_ids=author_ids)
            return {
                normalize_whitespace(row["author_id"]): {
                    "author_id": normalize_whitespace(row["author_id"]),
                    "display_name": normalize_whitespace(row["display_name"]),
                    "works_count": parse_optional_int(row["works_count"]),
                }
                for row in rows
                if normalize_whitespace(row["author_id"])
            }
    finally:
        driver.close()


def build_reviewer_selection(
    search_payload: dict[str, Any],
    *,
    max_reviewers: int,
    min_works_count: int,
    selection_top_k: int,
    env_path: str | None,
    use_metadata: bool = True,
) -> dict[str, Any]:
    raw_authors, author_source = collect_reviewer_author_candidates(search_payload)

    normalized_authors: list[dict[str, Any]] = []
    seen: set[str] = set()
    author_ids: list[str] = []
    for rank, item in enumerate(raw_authors, start=1):
        if not isinstance(item, dict):
            continue
        author_id = normalize_whitespace(item.get("author_id"))
        author_name = normalize_whitespace(item.get("name"))
        if not author_name and not author_id:
            continue
        normalized_authors.append(
            {
                "rank": rank,
                "author_id": author_id,
                "source_author_id": normalize_whitespace(item.get("source_author_id")),
                "name": author_name,
                "score": item.get("score"),
                "source": normalize_whitespace(item.get("source")) or author_source,
                "paper_rank": item.get("paper_rank"),
                "author_rank": item.get("author_rank"),
            }
        )
        if author_id and author_id not in seen:
            seen.add(author_id)
            author_ids.append(author_id)

    metadata_status = "skipped" if not author_ids or not use_metadata else "ok"
    metadata_error = ""
    metadata_by_id: dict[str, dict[str, Any]] = {}
    if author_ids and use_metadata:
        try:
            metadata_by_id = fetch_author_metadata_by_id(author_ids, env_path)
        except Exception as exc:
            metadata_status = "error"
            metadata_error = str(exc)

    grouped_by_name: dict[str, dict[str, Any]] = {}
    for author in normalized_authors:
        author_id = author["author_id"]
        metadata = metadata_by_id.get(author_id, {})
        display_name = normalize_whitespace(metadata.get("display_name")) or author["name"]
        if not display_name:
            display_name = author_id
        if not display_name:
            continue

        works_count = parse_optional_int(metadata.get("works_count"))
        if works_count is None:
            works_count = parse_optional_int(author.get("works_count"))

        if display_name not in grouped_by_name:
            grouped_by_name[display_name] = {
                "name": display_name,
                "score": author.get("score"),
                "first_rank": author["rank"],
                "source": author.get("source"),
                "works_count_sum": 0,
                "works_count_known_count": 0,
                "works_count_missing_count": 0,
                "representative_author_id": author_id,
                "member_author_ids": [],
                "source_author_ids": [],
            }

        group = grouped_by_name[display_name]
        if author_id and author_id not in group["member_author_ids"]:
            group["member_author_ids"].append(author_id)
        source_author_id = normalize_whitespace(author.get("source_author_id"))
        if source_author_id and source_author_id not in group["source_author_ids"]:
            group["source_author_ids"].append(source_author_id)
        if not group["representative_author_id"] and author_id:
            group["representative_author_id"] = author_id
        if works_count is None:
            group["works_count_missing_count"] += 1
        else:
            group["works_count_sum"] += works_count
            group["works_count_known_count"] += 1

    merged_candidates = sorted(grouped_by_name.values(), key=lambda item: item["first_rank"])
    enforce_works_filter = metadata_status == "ok"
    if enforce_works_filter:
        filtered_candidates = [
            candidate for candidate in merged_candidates if candidate["works_count_sum"] >= min_works_count
        ]
    else:
        filtered_candidates = [dict(candidate, filter_reason="metadata_unavailable") for candidate in merged_candidates]

    ordered_reviewer_pool: list[dict[str, Any]] = []
    seen_reviewer_keys: set[str] = set()
    for candidate in filtered_candidates:
        reviewer = dict(candidate)
        reviewer["author_id"] = reviewer.get("representative_author_id")
        reviewer["author_name"] = reviewer.get("name")
        reviewer_key = ensure_reviewer_key(reviewer)
        if reviewer_key in seen_reviewer_keys:
            reviewer_key = f"{reviewer_key}-{parse_optional_int(reviewer.get('first_rank')) or len(seen_reviewer_keys) + 1}"
            reviewer["reviewer_key"] = reviewer_key
        seen_reviewer_keys.add(reviewer_key)
        ordered_reviewer_pool.append(reviewer)

    effective_selection_top_k = max(selection_top_k, 0)
    truncated_reviewer_pool = (
        ordered_reviewer_pool[:effective_selection_top_k]
        if effective_selection_top_k > 0
        else list(ordered_reviewer_pool)
    )
    sampled_reviewers = sample_evenly(truncated_reviewer_pool, max(max_reviewers, 0))

    if not normalized_authors:
        status = "skipped"
    elif metadata_status == "error":
        status = "partial_error"
    else:
        status = "ok"

    return {
        "status": status,
        "metadata_status": metadata_status,
        "metadata_error": metadata_error or None,
        "max_reviewers": max_reviewers,
        "min_works_count": min_works_count,
        "selection_top_k": selection_top_k,
        "author_source": author_source,
        "raw_author_count": len(normalized_authors),
        "merged_name_count": len(merged_candidates),
        "filtered_name_count": len(filtered_candidates),
        "truncated_pool_count": len(truncated_reviewer_pool),
        "sampled_reviewer_count": len(sampled_reviewers),
        "selection_strategy": "top-k-then-evenly-spaced-first-rank",
        "selected_reviewer_keys": [reviewer.get("reviewer_key") for reviewer in sampled_reviewers],
        "works_count_filter_enforced": enforce_works_filter,
        "raw_authors": normalized_authors,
        "merged_name_candidates": merged_candidates,
        "filtered_name_candidates": filtered_candidates,
        "ordered_reviewer_pool": ordered_reviewer_pool,
        "truncated_reviewer_pool": truncated_reviewer_pool,
        "sampled_reviewers": sampled_reviewers,
        "reviewers": sampled_reviewers,
    }


def launch_worker(
    worker_script: Path,
    *,
    input_payload: dict[str, Any],
    work_dir: Path,
    smoke: bool,
    loggers: list[logging.LoggerAdapter],
    stage_name: str,
    cuda_visible_devices: str | None = None,
    cuda_devices: list[str] | None = None,
) -> dict[str, Any]:
    input_path = work_dir / "input.json"
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    result_path = work_dir / "result.json"

    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(input_path, input_payload)

    command = [
        sys.executable,
        str(worker_script.resolve()),
        "--input-file",
        str(input_path.resolve()),
        "--work-dir",
        str(work_dir.resolve()),
    ]
    if smoke:
        command.append("--smoke")

    started_at = time.perf_counter()
    subprocess_env = os.environ.copy()
    if cuda_devices:
        subprocess_env["INNOEVAL_CUDA_DEVICES"] = format_cuda_devices(cuda_devices)
    if cuda_visible_devices:
        subprocess_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    log_to_many(
        loggers,
        "info",
        (
            f"Launching {stage_name} worker script={worker_script.name} work_dir={work_dir} "
            f"smoke={smoke} cuda_devices={format_cuda_devices(cuda_devices or [])} "
            f"cuda_visible_devices={cuda_visible_devices or ''}"
        ),
    )
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=subprocess_env,
    )
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.stdout:
        log_multiline(loggers, "info", f"{stage_name}.stdout", completed.stdout)
    if completed.stderr:
        log_multiline(loggers, "warning", f"{stage_name}.stderr", completed.stderr)

    payload: dict[str, Any]
    if result_path.exists():
        payload = read_json(result_path)
    else:
        payload = {
            "status": "error",
            "error_type": "MissingWorkerResult",
            "error": f"Worker did not write {result_path}",
        }
    payload.update(
        {
            "returncode": completed.returncode,
            "cuda_devices": format_cuda_devices(cuda_devices or []),
            "cuda_visible_devices": cuda_visible_devices or subprocess_env.get("CUDA_VISIBLE_DEVICES", ""),
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }
    )
    if input_payload.get("idea_context_path"):
        payload["idea_context_path"] = input_payload.get("idea_context_path")
    write_json(result_path, payload)
    log_to_many(
        loggers,
        "info",
        (
            f"Finished {stage_name} worker status={normalize_whitespace(payload.get('status')) or 'error'} "
            f"returncode={completed.returncode} elapsed_ms={payload['elapsed_ms']} "
            f"result_path={result_path.resolve()}"
        ),
    )
    return payload


def launch_reviewer_inline(
    *,
    input_payload: dict[str, Any],
    work_dir: Path,
    smoke: bool,
    loggers: list[logging.LoggerAdapter],
    stage_name: str,
    device: str | None = None,
    cuda_devices: list[str] | None = None,
) -> dict[str, Any]:
    from review.author_background import AuthorBackgroundConfig, run_author_background
    from review.persona import DEFAULT_PERSONA_SUBJECT, select_random_persona

    input_path = work_dir / "input.json"
    result_path = work_dir / "result.json"
    worker_log_path = work_dir / "worker.log"
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(input_path, input_payload)
    worker_logger = build_file_logger(worker_log_path, "reviewer_background_inline")

    started_at = time.perf_counter()
    if cuda_devices:
        os.environ["INNOEVAL_CUDA_DEVICES"] = format_cuda_devices(cuda_devices)
    worker_logger.info(
        "Worker started mode=%s input_file=%s work_dir=%s device=%s cuda_devices=%s",
        "smoke" if smoke else "real",
        input_path,
        work_dir,
        device or "",
        format_cuda_devices(cuda_devices or []),
    )
    log_to_many(
        loggers,
        "info",
        (
            f"Launching {stage_name} inline worker work_dir={work_dir} smoke={smoke} "
            f"device={device or ''} cuda_devices={format_cuda_devices(cuda_devices or [])}"
        ),
    )

    try:
        if smoke:
            author_name = normalize_whitespace(input_payload.get("author_name")) or normalize_whitespace(
                input_payload.get("author_id")
            )
            persona_subject = normalize_whitespace(input_payload.get("persona_subject")) or DEFAULT_PERSONA_SUBJECT
            persona = select_random_persona(subject=persona_subject)
            profile_dir = work_dir / f"merged_{slugify(author_name)}"
            profile_dir.mkdir(parents=True, exist_ok=True)
            summary_path = profile_dir / "author_summary.json"
            write_json(
                summary_path,
                {
                    "author_name": author_name,
                    "author_id": normalize_whitespace(input_payload.get("author_id")),
                    "summary": [
                        "Smoke reviewer background generated by inline reviewer worker.",
                        f"Idea anchor: {normalize_whitespace(input_payload.get('idea_text'))[:200]}",
                    ],
                    "persona": persona,
                    "source": "smoke",
                },
            )
            payload = {
                "status": "ok",
                "mode": "smoke",
                "reviewer_key": normalize_whitespace(input_payload.get("reviewer_key")),
                "author_name": author_name,
                "author_id": normalize_whitespace(input_payload.get("author_id")),
                "persona_subject": persona_subject,
                "summary_path": str(summary_path.resolve()),
                "work_dir": str(work_dir.resolve()),
            }
        else:
            author_name = normalize_whitespace(input_payload.get("author_name"))
            author_id = normalize_whitespace(input_payload.get("author_id"))
            search_method = normalize_whitespace(input_payload.get("search_method")) or ("id" if author_id else "name")
            target_author = author_id if search_method == "id" and author_id else (author_name or author_id)
            if not target_author:
                raise ValueError("Worker payload must contain author_name or author_id")

            config = AuthorBackgroundConfig(
                target_author=target_author,
                target_idea=normalize_whitespace(input_payload.get("idea_text")),
                output_root=work_dir,
                search_method=search_method,
                env_path=Path(input_payload["env_path"]).expanduser().resolve() if input_payload.get("env_path") else None,
                llm_api_key=input_payload.get("llm_api_key"),
                device=normalize_whitespace(device) or device_at(cuda_devices or [], 0) or "cpu",
                persona_subject=normalize_whitespace(input_payload.get("persona_subject")) or DEFAULT_PERSONA_SUBJECT,
                persona_prompts_path=(
                    Path(input_payload["persona_prompts_path"]).expanduser().resolve()
                    if input_payload.get("persona_prompts_path")
                    else None
                ),
            )
            result = run_author_background(config)
            payload = {
                "status": result["status"],
                "mode": "real",
                "reviewer_key": normalize_whitespace(input_payload.get("reviewer_key")),
                "author_name": author_name,
                "author_id": author_id,
                "summary_path": result["summary_path"],
                "relevant_papers_path": result["relevant_papers_path"],
                "target_folder": result["target_folder"],
                "persona_subject": result.get("persona_subject"),
                "persona_prompts_path": result.get("persona_prompts_path"),
                "work_dir": str(work_dir.resolve()),
            }
        payload.update(
            {
                "returncode": 0,
                "device": device or "",
                "cuda_devices": format_cuda_devices(cuda_devices or []),
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            }
        )
        write_json(result_path, payload)
        worker_logger.info(
            "Worker finished status=%s elapsed_ms=%s result_path=%s",
            payload.get("status"),
            payload.get("elapsed_ms"),
            result_path,
        )
        log_to_many(
            loggers,
            "info",
            (
                f"Finished {stage_name} inline worker status={normalize_whitespace(payload.get('status')) or 'error'} "
                f"elapsed_ms={payload['elapsed_ms']} result_path={result_path.resolve()}"
            ),
        )
        return payload
    except Exception as exc:
        payload = {
            "status": "error",
            "mode": "smoke" if smoke else "real",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "returncode": 1,
            "device": device or "",
            "cuda_devices": format_cuda_devices(cuda_devices or []),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }
        write_json(result_path, payload)
        worker_logger.exception("Worker failed error=%s", exc)
        log_to_many(
            loggers,
            "info",
            (
                f"Finished {stage_name} inline worker status=error "
                f"elapsed_ms={payload['elapsed_ms']} result_path={result_path.resolve()}"
            ),
        )
        return payload


def run_manifest_stage(
    args: argparse.Namespace,
    search_result_path: Path,
    output_dir: Path,
    *,
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    log_to_many(
        loggers,
        "info",
        f"Starting manifest stage input={search_result_path.resolve()} output_dir={output_dir} smoke={args.smoke}",
    )
    with capture_stage_output(loggers, stage_name="manifest"):
        if args.smoke:
            search_payload = read_json(search_result_path)
            manifest = build_smoke_manifest(search_payload, output_dir)
        else:
            from search import pdf_xml_pipeline

            cli_args = [
                "--input",
                str(search_result_path.resolve()),
                "--top-k",
                str(args.manifest_top_k),
                "--max-workers",
                str(args.manifest_max_workers),
                "--grobid-max-concurrency",
                str(args.manifest_grobid_max_concurrency),
                "--env",
                str(Path(args.env).expanduser().resolve()),
                "--output-root",
                str(output_dir.resolve()),
                "--result-tag",
                "run",
            ]
            manifest_args = pdf_xml_pipeline.build_parser().parse_args(cli_args)
            manifest = pdf_xml_pipeline.run_pipeline(manifest_args)

    manifest_path = Path(manifest["manifest_path"]).resolve()
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    log_to_many(
        loggers,
        "info",
        (
            f"Finished manifest stage status={normalize_whitespace(manifest.get('status')) or 'ok'} "
            f"paper_count={manifest.get('paper_count')} elapsed_ms={elapsed_ms} "
            f"manifest_path={manifest_path}"
        ),
    )
    return {
        "status": normalize_whitespace(manifest.get("status")) or "ok",
        "elapsed_ms": elapsed_ms,
        "manifest_path": str(manifest_path),
        "payload": manifest,
    }


def run_grounding_stage(
    args: argparse.Namespace,
    *,
    idea_context: dict[str, Any],
    idea_context_path: Path,
    manifest_path: Path,
    output_dir: Path,
    cuda_devices: list[str],
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    result_path = output_dir / "result.json"
    grounding_device = device_at(cuda_devices, 0)
    log_to_many(
        loggers,
        "info",
        (
            f"Starting grounding stage manifest={manifest_path.resolve()} output_dir={output_dir} "
            f"device={grounding_device or ''} smoke={args.smoke}"
        ),
    )
    with capture_stage_output(loggers, stage_name="grounding"):
        evaluation_target = idea_context.get("evaluation_target") if isinstance(idea_context, dict) else {}
        evaluation_text = normalize_whitespace(
            evaluation_target.get("text") if isinstance(evaluation_target, dict) else ""
        )
        if args.smoke:
            payload = build_smoke_grounding_payload(
                args,
                idea_text=evaluation_text,
                idea_context_path=idea_context_path,
                manifest_path=manifest_path,
            )
        else:
            from grounding import grounding_pipeline

            cli_args = [
                "--manifest",
                str(manifest_path.resolve()),
                "--env",
                str(Path(args.env).expanduser().resolve()),
                "--final-top-k",
                str(args.grounding_final_top_k),
                "--target-dir",
                str((output_dir / "target").resolve()),
                "--output",
                str(result_path.resolve()),
            ]
            if grounding_device:
                cli_args.extend(["--device", grounding_device])
            if not args.disable_grounding_refinement:
                cli_args.append("--enable-grounding-refinement")
            cli_args.extend(["--idea-context", str(idea_context_path.resolve())])
            grounding_args = grounding_pipeline.build_parser().parse_args(cli_args)
            payload = grounding_pipeline.run_grounding(grounding_args)

    write_json(result_path, payload)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    log_to_many(
        loggers,
        "info",
        (
            f"Finished grounding stage status={normalize_whitespace(payload.get('status')) or 'ok'} "
            f"elapsed_ms={elapsed_ms} result_path={result_path.resolve()}"
        ),
    )
    return {
        "status": normalize_whitespace(payload.get("status")) or "ok",
        "elapsed_ms": elapsed_ms,
        "result_path": str(result_path.resolve()),
        "payload": payload,
    }


def run_review_stage(
    args: argparse.Namespace,
    *,
    idea_text: str,
    source_full_text: str,
    output_dir: Path,
    rubric_path: str | None,
    reviewers_index_path: str | None,
    grounding_result_path: str | None,
    artifact_paths: dict[str, Any],
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    log_to_many(
        loggers,
        "info",
        (
            f"Starting review stage output_dir={output_dir} smoke={args.smoke} "
            f"llm_enabled={not args.disable_review_llm and not args.smoke}"
        ),
    )

    if args.disable_review_stage:
        payload = {
            "status": "skipped",
            "elapsed_ms": 0.0,
        }
        log_to_many(loggers, "info", "Skipping review stage because --disable-review-stage is set")
        return payload

    from review.adapters import normalize_artifacts
    from review.evaluation import ReviewGenerationConfig, run_reviewer_evaluations

    normalized = normalize_artifacts(
        idea_text=idea_text,
        source_full_text=source_full_text,
        rubric_path=rubric_path,
        reviewers_index_path=reviewers_index_path,
        grounding_result_path=grounding_result_path,
        search_result_path=artifact_paths.get("search_result_path"),
        enable_figure_cards=bool(args.enable_figure_cards),
        figure_dir=args.figure_dir,
        output_dir=output_dir,
    )
    log_to_many(
        loggers,
        "info",
        (
            "Normalized review artifacts "
            f"rubric_status={normalized['rubric'].get('status')} "
            f"reviewer_count={normalized['reviewers'].get('reviewer_count')} "
            f"evidence_count={normalized['evidence_bank'].get('stats', {}).get('evidence_card_count')} "
            f"experiment_count={normalized['evidence_bank'].get('stats', {}).get('experiment_card_count')} "
            f"figure_count={normalized['evidence_bank'].get('stats', {}).get('figure_card_count')}"
        ),
    )

    review_config = ReviewGenerationConfig(
        env_path=Path(args.env).expanduser().resolve() if args.env else None,
        llm_api_key=args.review_llm_api_key,
        llm_base_url=args.review_llm_base_url,
        llm_model_name=args.review_llm_model_name,
        llm_temperature=args.review_temperature,
        enable_llm=not args.disable_review_llm,
        smoke=args.smoke,
        max_workers=args.review_max_workers,
        max_evidence_cards_per_section=args.review_max_evidence_cards_per_section,
        max_evidence_cards_per_query=args.review_max_evidence_cards_per_query,
        max_experiment_cards_per_section=args.review_max_experiment_cards_per_section,
        max_figure_cards_per_section=args.review_max_figure_cards_per_section,
        max_total_evidence_chars_per_section=args.review_max_total_evidence_chars_per_section,
        llm_timeout_seconds=args.review_llm_timeout_seconds,
        max_retries=args.review_llm_max_retries,
        loggers=tuple(loggers),
    )
    review_index = run_reviewer_evaluations(
        idea_text=idea_text,
        source_full_text=source_full_text,
        normalized_rubric=normalized["rubric"],
        normalized_reviewers=normalized["reviewers"],
        evidence_bank=normalized["evidence_bank"],
        output_dir=output_dir,
        config=review_config,
    )
    log_to_many(
        loggers,
        "info",
        (
            f"Finished reviewer evaluations status={review_index.get('status')} "
            f"successful_count={review_index.get('successful_count')} "
            f"index_path={review_index.get('index_path')}"
        ),
    )

    status = combine_status(
        [
            normalize_whitespace(normalized["rubric"].get("status")),
            normalize_whitespace(normalized["reviewers"].get("status")),
            normalize_whitespace(normalized["evidence_bank"].get("status")),
            normalize_whitespace(review_index.get("status")),
        ]
    )

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    payload = {
        "status": status,
        "elapsed_ms": elapsed_ms,
        "reviewer_count": review_index.get("reviewer_count", 0),
        "successful_count": review_index.get("successful_count", 0),
        "failed_count": review_index.get("failed_count", 0),
        "reviewer_reviews_path": review_index.get("index_path"),
        "normalized_rubric_path": normalized["paths"]["rubric"],
        "normalized_reviewers_path": normalized["paths"]["reviewers"],
        "searched_papers_path": normalized["paths"]["searched_papers"],
        "evidence_bank_path": normalized["paths"]["evidence_bank"],
        "figure_cards_path": normalized["paths"].get("figure_cards"),
        "normalized": {
            "rubric_status": normalized["rubric"].get("status"),
            "reviewers_status": normalized["reviewers"].get("status"),
            "evidence_bank_status": normalized["evidence_bank"].get("status"),
        },
    }
    log_to_many(
        loggers,
        "info",
        (
            f"Finished review stage status={status} elapsed_ms={elapsed_ms} "
            f"reviewer_reviews_path={review_index.get('index_path')}"
        ),
    )
    return payload


def run_report_stage(
    args: argparse.Namespace,
    *,
    idea_text: str,
    output_dir: Path,
    review_stage: dict[str, Any],
    artifact_paths: dict[str, Any],
    loggers: list[logging.LoggerAdapter],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    log_to_many(
        loggers,
        "info",
        (
            f"Starting report stage output_dir={output_dir} smoke={args.smoke} "
            f"llm_enabled={not args.disable_review_llm and not args.smoke} short_report={args.short_report}"
        ),
    )

    if args.disable_report_stage:
        log_to_many(loggers, "info", "Skipping report stage because --disable-report-stage is set")
        return {"status": "skipped", "elapsed_ms": 0.0}

    reviewer_reviews_path = normalize_whitespace(review_stage.get("reviewer_reviews_path"))
    if not reviewer_reviews_path or not Path(reviewer_reviews_path).exists():
        log_to_many(loggers, "info", "Skipping report stage because reviewer reviews are unavailable")
        return {"status": "skipped", "elapsed_ms": 0.0}

    from review.report import ReportSynthesisConfig, synthesize_final_report

    review_index = read_json(Path(reviewer_reviews_path))
    normalized_rubric = read_json(Path(review_stage["normalized_rubric_path"]))
    normalized_reviewers = read_json(Path(review_stage["normalized_reviewers_path"]))
    evidence_bank = read_json(Path(review_stage["evidence_bank_path"]))
    report_artifact_paths = {
        **artifact_paths,
        "normalized_rubric_path": review_stage.get("normalized_rubric_path"),
        "normalized_reviewers_path": review_stage.get("normalized_reviewers_path"),
        "searched_papers_path": review_stage.get("searched_papers_path"),
        "evidence_bank_path": review_stage.get("evidence_bank_path"),
        "reviewer_reviews_path": reviewer_reviews_path,
    }
    report_config = ReportSynthesisConfig(
        env_path=Path(args.env).expanduser().resolve() if args.env else None,
        llm_api_key=args.review_llm_api_key,
        llm_base_url=args.review_llm_base_url,
        llm_model_name=args.review_llm_model_name,
        llm_temperature=args.review_temperature,
        enable_llm=not args.disable_review_llm,
        smoke=args.smoke,
        llm_timeout_seconds=args.review_llm_timeout_seconds,
        max_retries=args.review_llm_max_retries,
        short_report=args.short_report,
        loggers=tuple(loggers),
    )
    report_result = synthesize_final_report(
        idea_text=idea_text,
        normalized_rubric=normalized_rubric,
        normalized_reviewers=normalized_reviewers,
        evidence_bank=evidence_bank,
        reviewer_reviews=review_index.get("reviewer_reviews", []),
        output_dir=output_dir,
        artifact_paths=report_artifact_paths,
        config=report_config,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    payload = {
        "status": report_result.get("status"),
        "elapsed_ms": elapsed_ms,
        "final_report_path": report_result.get("final_report_path"),
        "final_report_json_path": report_result.get("final_report_json_path"),
        "md_tree_path": report_result.get("md_tree_path"),
        "meta_review": report_result.get("meta_review"),
        "short_report": report_result.get("short_report", args.short_report),
    }
    log_to_many(
        loggers,
        "info",
        (
            f"Finished report stage status={payload['status']} elapsed_ms={elapsed_ms} "
            f"final_report_path={payload.get('final_report_path')}"
        ),
    )
    return payload


def summarize_worker_group(
    results: list[dict[str, Any]],
    *,
    index_path: Path,
    selection_path: Path,
    idea_context_path: Path,
    selection_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    statuses = [normalize_whitespace(item.get("status")) for item in results]
    selection_status = normalize_whitespace((selection_payload or {}).get("status"))
    summary = {
        "status": combine_status([selection_status, *statuses]) if statuses else (selection_status or "skipped"),
        "count": len(results),
        "results": results,
        "selection": {
            "status": selection_status or "skipped",
            "author_source": (selection_payload or {}).get("author_source"),
            "raw_author_count": (selection_payload or {}).get("raw_author_count", 0),
            "merged_name_count": (selection_payload or {}).get("merged_name_count", 0),
            "filtered_name_count": (selection_payload or {}).get("filtered_name_count", 0),
            "sampled_reviewer_count": (selection_payload or {}).get("sampled_reviewer_count", 0),
            "works_count_filter_enforced": (selection_payload or {}).get("works_count_filter_enforced", False),
        },
        "selection_path": str(selection_path.resolve()),
        "idea_context_path": str(idea_context_path.resolve()),
        "index_path": str(index_path.resolve()),
    }
    write_json(index_path, summary)
    return summary


def apply_reviewer_background_fallbacks(
    *,
    args: argparse.Namespace,
    reviewer_results: list[dict[str, Any]],
    reviewer_selection: list[dict[str, Any]],
    reviewers_output_dir: Path,
    evaluation_target_text: str,
    loggers: list[logging.LoggerAdapter],
) -> list[dict[str, Any]]:
    if args.smoke or not reviewer_results or not evaluation_target_text:
        return reviewer_results

    successful_summaries: list[dict[str, Any]] = []
    failed_items: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, result in enumerate(reviewer_results):
        reviewer = reviewer_selection[index] if index < len(reviewer_selection) and isinstance(reviewer_selection[index], dict) else {}
        if normalize_whitespace(result.get("status")) == "ok":
            summary_path = normalize_whitespace(result.get("summary_path"))
            if summary_path and Path(summary_path).exists():
                try:
                    summary_payload = read_json(Path(summary_path))
                    if normalize_whitespace(summary_payload.get("overall_academic_profile")):
                        successful_summaries.append(summary_payload)
                        continue
                except Exception:
                    pass
        failed_items.append((index, result, reviewer))

    if not failed_items:
        return reviewer_results

    missing_author_names = [
        normalize_whitespace(reviewer.get("author_name")) or normalize_whitespace(reviewer.get("author_id")) or f"Reviewer {index + 1}"
        for index, _, reviewer in failed_items
    ]
    if not all(missing_author_names):
        return reviewer_results

    from review.reviewer_background_fallback import (
        ReviewerBackgroundFallbackConfig,
        generate_fallback_reviewer_backgrounds,
        write_fallback_summary,
    )

    try:
        generated = generate_fallback_reviewer_backgrounds(
            config=ReviewerBackgroundFallbackConfig(
                idea_text=evaluation_target_text,
                env_path=Path(args.env).expanduser().resolve() if args.env else None,
                llm_api_key=args.review_llm_api_key,
                llm_base_url=args.review_llm_base_url,
                llm_model_name=args.review_llm_model_name,
                llm_temperature=args.review_temperature,
                persona_subject=args.reviewer_persona_subject,
            ),
            successful_summaries=successful_summaries,
            missing_author_names=missing_author_names,
        )
    except Exception as exc:
        log_to_many(
            loggers,
            "warning",
            f"Reviewer background fallback failed; keeping original reviewer failures error={exc.__class__.__name__}: {exc}",
        )
        return reviewer_results

    updated_results = list(reviewer_results)
    for (index, _, reviewer), author_name in zip(failed_items, missing_author_names):
        summary_payload = generated[author_name.casefold()]
        reviewer_dir = reviewer_profile_dir(reviewers_output_dir, reviewer)
        summary_path = write_fallback_summary(
            reviewer_dir=reviewer_dir,
            author_name=author_name,
            summary_payload=summary_payload,
        )
        updated_results[index] = {
            "status": "ok",
            "mode": "real",
            "reviewer_key": ensure_reviewer_key(reviewer),
            "author_name": author_name,
            "author_id": normalize_whitespace(reviewer.get("author_id")) or None,
            "summary_path": summary_path,
            "relevant_papers_path": None,
            "target_folder": str((reviewer_dir / f"merged_{author_name.replace(' ', '_')}").resolve()),
            "persona_subject": args.reviewer_persona_subject,
            "persona_prompts_path": None,
            "work_dir": str(reviewer_dir.resolve()),
        }
        write_json(reviewer_dir / "result.json", updated_results[index])

    log_to_many(
        loggers,
        "info",
        f"Applied fallback academic backgrounds for {len(failed_items)} reviewers",
    )
    return updated_results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cuda_devices = resolve_pipeline_cuda_devices(args)
    args.cuda_devices = format_cuda_devices(cuda_devices)
    run_id = make_run_id(args)
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_payload = {
        "input_type": "pdf" if args.pdf_path else "idea_text",
        "idea_text": normalize_whitespace(args.idea_text) if args.idea_text else None,
        "pdf_path": str(Path(args.pdf_path).expanduser().resolve()) if args.pdf_path else None,
    }
    requested_force_stages = parse_force_stages(args.force_stages)
    force_stages = expand_force_stages(requested_force_stages)
    # Historical cache directories are only sources. On cache hit, stage artifacts
    # are materialized into run_dir before their paths are passed downstream.
    explicit_cache_source = (
        run_root / normalize_whitespace(args.cache_source_run_id)
        if normalize_whitespace(args.cache_source_run_id)
        else None
    )
    if explicit_cache_source is not None and not explicit_cache_source.exists():
        raise FileNotFoundError(f"--cache-source-run-id does not exist under run root: {explicit_cache_source}")
    cache_candidates = list_cache_candidates(
        run_root,
        run_id,
        input_payload,
        explicit_cache_source=explicit_cache_source,
    )
    cache_run_dir = cache_candidates[0].run_dir if cache_candidates else None
    cache_source_dir = cache_run_dir or run_dir
    previous_input_payload = cache_candidates[0].input_payload if cache_candidates else None

    idea_context_output_dir = stage_dir(run_dir, "idea_context")
    search_output_dir = stage_dir(run_dir, "search")
    manifest_output_dir = stage_dir(run_dir, "manifest")
    grounding_output_dir = stage_dir(run_dir, "grounding")
    reviewers_output_dir = stage_dir(run_dir, "reviewers")
    rubric_sources_output_dir = stage_dir(run_dir, "rubric_sources")
    rubric_llm_output_dir = stage_dir(run_dir, "rubric_llm")
    review_output_dir = stage_dir(run_dir, "review")
    report_output_dir = stage_dir(run_dir, "report")
    logs_output_dir = stage_dir(run_dir, "logs")
    pipeline_logger = build_file_logger(logs_output_dir / "pipeline.log", Path(__file__).name)
    idea_context_logger = build_file_logger(logs_output_dir / "idea_context.log", "idea_context")
    search_logger = build_file_logger(logs_output_dir / "search.log", "search")
    manifest_logger = build_file_logger(logs_output_dir / "manifest.log", "manifest")
    grounding_logger = build_file_logger(logs_output_dir / "grounding.log", "grounding")
    reviewers_logger = build_file_logger(logs_output_dir / "reviewers.log", "reviewers")
    rubric_sources_logger = build_file_logger(logs_output_dir / "rubric_sources.log", "rubric_sources")
    rubric_llm_logger = build_file_logger(logs_output_dir / "rubric_llm.log", "rubric_llm")
    review_logger = build_file_logger(logs_output_dir / "review.log", "review")
    report_logger = build_file_logger(logs_output_dir / "report.log", "report")
    log_to_many(
        [pipeline_logger],
        "info",
        (
            f"Pipeline started run_id={run_id} input_type={'pdf' if args.pdf_path else 'idea_text'} "
            f"run_dir={run_dir.resolve()} run_cache_enabled={not args.disable_run_cache} "
            f"requested_force_stages={','.join(sorted(requested_force_stages))} "
            f"force_stages={','.join(sorted(force_stages))} "
            f"cache_policy=latest-compatible-stage cache_candidate_count={len(cache_candidates)} "
            f"latest_cache_source_dir={cache_source_dir.resolve()} "
            f"cuda_devices={args.cuda_devices}"
        ),
    )

    input_matches_existing_run = input_payload_matches_for_cache(previous_input_payload, input_payload)
    if not input_matches_existing_run:
        log_to_many(
            [pipeline_logger],
            "warning",
            "Cache source input.json does not match current input; run-cache will be ignored for this run",
        )

    write_json(
        run_dir / "config.resolved.json",
        {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    )
    write_json(run_dir / "input.json", input_payload)

    stage_payloads: dict[str, dict[str, Any]] = {}
    final_output_path = run_dir / "result.json"

    try:
        idea_context_cache_keys = ["smoke"]
        idea_context_cache_allowed = cache_allowed(
            args=args,
            stage_name="idea_context",
            force_stages=force_stages,
            input_matches_existing_run=input_matches_existing_run,
        )
        idea_context_stage = (
            cached_stage_from_candidates(
                candidates=cache_candidates,
                stage_name="idea_context",
                materialized_output_dir=idea_context_output_dir,
                run_dir=run_dir,
                args=args,
                config_keys=idea_context_cache_keys,
                loader=lambda cache_dir, output_dir, source_dir: cached_idea_context_stage(
                    cache_dir,
                    materialized_output_dir=output_dir,
                    cache_source_dir=source_dir,
                    run_dir=run_dir,
                    loggers=[pipeline_logger, idea_context_logger],
                ),
                loggers=[pipeline_logger, idea_context_logger],
            )
            if idea_context_cache_allowed
            else None
        )
        if idea_context_stage is None:
            idea_context_stage = run_idea_context_stage(
                args,
                idea_context_output_dir,
                loggers=[pipeline_logger, idea_context_logger],
            )
        idea_context_payload = idea_context_stage["payload"]
        idea_context_path = Path(idea_context_stage["result_path"]).resolve()
        evaluation_target = idea_context_payload.get("evaluation_target", {})
        evaluation_target_text = preserve_text_block(
            evaluation_target.get("text") if isinstance(evaluation_target, dict) else ""
        )
        source_full_text_payload = idea_context_payload.get("source_full_text")
        source_full_text = (
            str(source_full_text_payload.get("text")).strip()
            if isinstance(source_full_text_payload, dict) and source_full_text_payload.get("text") is not None
            else ""
        )
        stage_payloads["idea_context"] = {
            "status": idea_context_stage["status"],
            "result_path": idea_context_stage["result_path"],
            "elapsed_ms": idea_context_stage["elapsed_ms"],
        }
        if idea_context_stage.get("cache"):
            stage_payloads["idea_context"]["cache"] = idea_context_stage["cache"]
        if idea_context_stage.get("cache_source_run_dir"):
            stage_payloads["idea_context"]["cache_source_run_dir"] = idea_context_stage["cache_source_run_dir"]

        search_cache_keys = ["smoke", "kg_top_k", "s2_top_k", "disable_llm_ranking"]
        search_cache_allowed = cache_allowed(
            args=args,
            stage_name="search",
            force_stages=force_stages,
            input_matches_existing_run=input_matches_existing_run,
        )
        search_stage = (
            cached_stage_from_candidates(
                candidates=cache_candidates,
                stage_name="search",
                materialized_output_dir=search_output_dir,
                run_dir=run_dir,
                args=args,
                config_keys=search_cache_keys,
                loader=lambda cache_dir, output_dir, source_dir: cached_search_stage(
                    cache_dir,
                    materialized_output_dir=output_dir,
                    cache_source_dir=source_dir,
                    run_dir=run_dir,
                    expected_retrieval_text=normalize_whitespace(
                        idea_context_payload.get("retrieval_seed", {}).get("text")
                        if isinstance(idea_context_payload.get("retrieval_seed"), dict)
                        else ""
                    ),
                    loggers=[pipeline_logger, search_logger],
                ),
                loggers=[pipeline_logger, search_logger],
            )
            if search_cache_allowed
            else None
        )
        if search_stage is None:
            search_stage = run_search_stage(
                args,
                search_output_dir,
                idea_context=idea_context_payload,
                cuda_devices=cuda_devices,
                loggers=[pipeline_logger, search_logger],
            )
        stage_payloads["search"] = {
            "status": search_stage["status"],
            "result_path": search_stage["result_path"],
            "elapsed_ms": search_stage["elapsed_ms"],
        }
        if search_stage.get("cache"):
            stage_payloads["search"]["cache"] = search_stage["cache"]
        if search_stage.get("cache_source_run_dir"):
            stage_payloads["search"]["cache_source_run_dir"] = search_stage["cache_source_run_dir"]

        selection_path = reviewers_output_dir / "selection.json"
        reviewer_own_cache_keys = [
            "smoke",
            "reviewer_min_works_count",
            "reviewer_selection_top_k",
            "reviewer_persona_subject",
        ]
        reviewer_cache_keys = cache_key_union(search_cache_keys, reviewer_own_cache_keys)
        reviewer_profile_cache_candidates = compatible_cache_candidates(
            candidates=cache_candidates,
            args=args,
            config_keys=reviewer_cache_keys,
        )
        reviewer_item_cache_allowed = cache_allowed(
            args=args,
            stage_name="reviewers",
            force_stages=force_stages,
            input_matches_existing_run=input_matches_existing_run,
        )
        reviewer_selection_payload = build_reviewer_selection(
            search_stage["payload"],
            max_reviewers=args.max_reviewers,
            min_works_count=args.reviewer_min_works_count,
            selection_top_k=args.reviewer_selection_top_k,
            env_path=str(Path(args.env).expanduser().resolve()) if args.env else None,
            use_metadata=not args.smoke,
        )
        write_json(selection_path, reviewer_selection_payload)
        reviewer_selection = reviewer_selection_payload.get("sampled_reviewers", [])
        selected_keys = selected_reviewer_keys(reviewer_selection)
        reviewers_summary = (
            cached_reviewers_summary_for_selection(
                candidates=reviewer_profile_cache_candidates,
                selected_keys=selected_keys,
                materialized_output_dir=reviewers_output_dir,
                run_dir=run_dir,
                loggers=[pipeline_logger, reviewers_logger],
            )
            if reviewer_item_cache_allowed
            else None
        )
        if reviewers_summary is not None:
            reviewers_summary["selection_path"] = str(selection_path.resolve())
            stage_payloads["reviewers"] = {**reviewers_summary, "cache": "hit"}

        log_to_many(
            [pipeline_logger, reviewers_logger],
            "info",
            (
                f"Selected {len(reviewer_selection)} reviewers status={reviewer_selection_payload['status']} "
                f"merged={reviewer_selection_payload.get('merged_name_count', 0)} "
                f"filtered={reviewer_selection_payload.get('filtered_name_count', 0)} "
                f"top_k={reviewer_selection_payload.get('selection_top_k')} "
                f"selection_path={selection_path.resolve()} "
                f"profile_cache_candidates={len(reviewer_profile_cache_candidates)}"
            ),
        )

        rubric_sources_future = None
        rubric_llm_future = None
        reviewer_futures: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        reviewer_results_by_index: list[dict[str, Any] | None] = [None] * len(reviewer_selection)
        reviewer_results: list[dict[str, Any]] = []
        device_pool = cuda_devices
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(args.reviewer_max_workers, 1))
        try:
            if not args.disable_rubric and evaluation_target_text:
                rubric_sources_own_cache_keys = [
                    "smoke",
                    "disable_rubric",
                    "rubric_search_top_k",
                    "rubric_search_final_k",
                ]
                rubric_sources_cache_keys = cache_key_union(idea_context_cache_keys, rubric_sources_own_cache_keys)
                rubric_sources_cache_allowed = cache_allowed(
                    args=args,
                    stage_name="rubric_sources",
                    force_stages=force_stages,
                    input_matches_existing_run=input_matches_existing_run,
                )
                cached_rubric_sources = (
                    cached_stage_from_candidates(
                        candidates=candidates_for_dependencies(
                            cache_candidates,
                            stage_payloads.get("idea_context"),
                        ),
                        stage_name="rubric_sources",
                        materialized_output_dir=rubric_sources_output_dir,
                        run_dir=run_dir,
                        args=args,
                        config_keys=rubric_sources_cache_keys,
                        loader=lambda cache_dir, output_dir, source_dir: cached_rubric_sources_stage(
                            cache_dir,
                            materialized_output_dir=output_dir,
                            cache_source_dir=source_dir,
                            run_dir=run_dir,
                            loggers=[pipeline_logger, rubric_sources_logger],
                        ),
                        loggers=[pipeline_logger, rubric_sources_logger],
                    )
                    if rubric_sources_cache_allowed
                    else None
                )
                if cached_rubric_sources is not None:
                    stage_payloads["rubric_sources"] = cached_rubric_sources
                else:
                    reset_stage_directory(rubric_sources_output_dir)
                    rubric_sources_input = {
                        "idea_text": evaluation_target_text,
                        "idea_context_path": str(idea_context_path),
                        "env_path": str(Path(args.env).expanduser().resolve()),
                        "search_top_k": args.rubric_search_top_k,
                        "search_final_k": args.rubric_search_final_k,
                        "max_workers": args.rubric_max_workers,
                        "llm_timeout_seconds": args.rubric_llm_timeout_seconds,
                        "embed_device": device_at(cuda_devices, 0),
                        "rerank_device": device_at(cuda_devices, 1) or device_at(cuda_devices, 0),
                    }
                    rubric_sources_future = executor.submit(
                        launch_worker,
                        RUBRIC_SOURCES_WORKER_PATH,
                        input_payload=rubric_sources_input,
                        work_dir=rubric_sources_output_dir,
                        smoke=args.smoke,
                        loggers=[pipeline_logger, rubric_sources_logger],
                        stage_name="rubric_sources",
                        cuda_devices=cuda_devices,
                    )
            else:
                log_to_many(
                    [pipeline_logger, rubric_sources_logger, rubric_llm_logger],
                    "info",
                    (
                        f"Skipping rubric branch disable_rubric={args.disable_rubric} "
                        f"has_evaluation_target={bool(evaluation_target_text)}"
                    ),
                )

            if reviewers_summary is not None:
                log_to_many(
                    [pipeline_logger, reviewers_logger],
                    "info",
                    "Skipping reviewer workers because reviewers/index.json matched selected reviewer keys",
                )
            elif not args.disable_reviewers and reviewer_selection and evaluation_target_text:
                for index, reviewer in enumerate(reviewer_selection):
                    reviewer_key = ensure_reviewer_key(reviewer)
                    reviewer_dir = reviewer_profile_dir(reviewers_output_dir, reviewer)
                    reviewer_input = {
                        "reviewer_key": reviewer_key,
                        "author_id": reviewer.get("author_id"),
                        "author_name": reviewer.get("author_name"),
                        "score": reviewer.get("score"),
                        "works_count_sum": reviewer.get("works_count_sum"),
                        "member_author_ids": reviewer.get("member_author_ids"),
                        "source_author_ids": reviewer.get("source_author_ids"),
                        "idea_text": evaluation_target_text,
                        "idea_context_path": str(idea_context_path),
                        "env_path": str(Path(args.env).expanduser().resolve()),
                        "search_method": "id" if reviewer.get("author_id") else "name",
                        "persona_subject": args.reviewer_persona_subject,
                    }
                    assigned_torch_device = device_at(device_pool, index)
                    cached_reviewer = None
                    cache_hit_source_dir: Path | None = None
                    if reviewer_item_cache_allowed:
                        cache_hit = find_reviewer_profile_cache(
                            reviewer=reviewer,
                            index=index,
                            cache_candidates=reviewer_profile_cache_candidates,
                        )
                        if cache_hit is not None:
                            cache_hit_source_dir, cache_source_run_dir = cache_hit
                            cached_reviewer = materialize_reviewer_cache(
                                source_dir=cache_hit_source_dir,
                                target_dir=reviewer_dir,
                                cache_source_dir=cache_source_run_dir,
                                run_dir=run_dir,
                            )
                    if cached_reviewer is not None:
                        cached_reviewer["reviewer_key"] = reviewer_key
                        cached_reviewer["work_dir"] = str(reviewer_dir.resolve())
                        write_json(reviewer_dir / "result.json", cached_reviewer)
                        log_cache_hit(
                            [pipeline_logger, reviewers_logger],
                            f"reviewer.{index + 1:02d}.{reviewer_key}",
                            reviewer_dir / "result.json",
                        )
                        reviewer_results_by_index[index] = cached_reviewer
                    else:
                        future = executor.submit(
                            launch_reviewer_inline,
                            input_payload=reviewer_input,
                            work_dir=reviewer_dir,
                            smoke=args.smoke,
                            loggers=[pipeline_logger, reviewers_logger],
                            stage_name=f"reviewer.{index + 1:02d}",
                            device=assigned_torch_device,
                            cuda_devices=cuda_devices,
                        )
                        reviewer_futures[future] = index
            else:
                log_to_many(
                    [pipeline_logger, reviewers_logger],
                    "info",
                    (
                        f"Skipping reviewer branch disable_reviewers={args.disable_reviewers} "
                        f"selected_count={len(reviewer_selection)} has_evaluation_target={bool(evaluation_target_text)}"
                    ),
                )

            manifest_own_cache_keys = ["smoke", "manifest_top_k"]
            manifest_cache_keys = cache_key_union(search_cache_keys, manifest_own_cache_keys)
            manifest_cache_allowed = cache_allowed(
                args=args,
                stage_name="manifest",
                force_stages=force_stages,
                input_matches_existing_run=input_matches_existing_run,
            )
            manifest_stage = (
                cached_stage_from_candidates(
                    candidates=candidates_for_dependencies(
                        cache_candidates,
                        stage_payloads.get("search"),
                    ),
                    stage_name="manifest",
                    materialized_output_dir=manifest_output_dir,
                    run_dir=run_dir,
                    args=args,
                    config_keys=manifest_cache_keys,
                    loader=lambda cache_dir, output_dir, source_dir: cached_manifest_stage(
                        cache_dir,
                        materialized_output_dir=output_dir,
                        cache_source_dir=source_dir,
                        run_dir=run_dir,
                        loggers=[pipeline_logger, manifest_logger],
                    ),
                    loggers=[pipeline_logger, manifest_logger],
                )
                if manifest_cache_allowed
                else None
            )
            if manifest_stage is None:
                manifest_stage = run_manifest_stage(
                    args,
                    Path(search_stage["result_path"]),
                    manifest_output_dir,
                    loggers=[pipeline_logger, manifest_logger],
                )
            stage_payloads["manifest"] = {
                "status": manifest_stage["status"],
                "manifest_path": manifest_stage["manifest_path"],
                "elapsed_ms": manifest_stage["elapsed_ms"],
            }
            if manifest_stage.get("cache"):
                stage_payloads["manifest"]["cache"] = manifest_stage["cache"]
            if manifest_stage.get("cache_source_run_dir"):
                stage_payloads["manifest"]["cache_source_run_dir"] = manifest_stage["cache_source_run_dir"]

            grounding_own_cache_keys = ["smoke", "grounding_final_top_k", "disable_grounding_refinement"]
            grounding_cache_keys = cache_key_union(manifest_cache_keys, grounding_own_cache_keys)
            grounding_cache_allowed = cache_allowed(
                args=args,
                stage_name="grounding",
                force_stages=force_stages,
                input_matches_existing_run=input_matches_existing_run,
            )
            grounding_stage = (
                cached_stage_from_candidates(
                    candidates=candidates_for_dependencies(
                        cache_candidates,
                        stage_payloads.get("manifest"),
                    ),
                    stage_name="grounding",
                    materialized_output_dir=grounding_output_dir,
                    run_dir=run_dir,
                    args=args,
                    config_keys=grounding_cache_keys,
                    loader=lambda cache_dir, output_dir, source_dir: cached_grounding_stage(
                        cache_dir,
                        materialized_output_dir=output_dir,
                        cache_source_dir=source_dir,
                        run_dir=run_dir,
                        loggers=[pipeline_logger, grounding_logger],
                    ),
                    loggers=[pipeline_logger, grounding_logger],
                )
                if grounding_cache_allowed
                else None
            )
            if grounding_stage is None:
                grounding_stage = run_grounding_stage(
                    args,
                    idea_context=idea_context_payload,
                    idea_context_path=idea_context_path,
                    manifest_path=Path(manifest_stage["manifest_path"]),
                    output_dir=grounding_output_dir,
                    cuda_devices=cuda_devices,
                    loggers=[pipeline_logger, grounding_logger],
                )
            stage_payloads["grounding"] = {
                "status": grounding_stage["status"],
                "result_path": grounding_stage["result_path"],
                "elapsed_ms": grounding_stage["elapsed_ms"],
            }
            if grounding_stage.get("cache"):
                stage_payloads["grounding"]["cache"] = grounding_stage["cache"]
            if grounding_stage.get("cache_source_run_dir"):
                stage_payloads["grounding"]["cache_source_run_dir"] = grounding_stage["cache_source_run_dir"]

            for future, index in reviewer_futures.items():
                reviewer_results_by_index[index] = future.result()
            reviewer_results = [result for result in reviewer_results_by_index if result is not None]
            reviewer_results = apply_reviewer_background_fallbacks(
                args=args,
                reviewer_results=reviewer_results,
                reviewer_selection=reviewer_selection,
                reviewers_output_dir=reviewers_output_dir,
                evaluation_target_text=evaluation_target_text,
                loggers=[pipeline_logger, reviewers_logger],
            )
            if reviewers_summary is None:
                reviewers_summary = summarize_worker_group(
                    reviewer_results,
                    index_path=reviewers_output_dir / "index.json",
                    selection_path=selection_path,
                    idea_context_path=idea_context_path,
                    selection_payload=reviewer_selection_payload,
                )
                stage_payloads["reviewers"] = reviewers_summary

            if "rubric_sources" not in stage_payloads:
                if rubric_sources_future is not None:
                    rubric_sources_result = rubric_sources_future.result()
                    rubric_sources_status = normalize_whitespace(rubric_sources_result.get("status")) or "error"
                    rubric_sources_content_path = (
                        normalize_whitespace(rubric_sources_result.get("result_path"))
                        if rubric_sources_status == "ok"
                        else ""
                    )
                    stage_payloads["rubric_sources"] = {
                        "status": rubric_sources_status,
                        "rubric_schema_version": rubric_sources_result.get("rubric_schema_version"),
                        "result_path": rubric_sources_content_path or None,
                        "paper_contexts_path": rubric_sources_result.get("paper_contexts_path"),
                        "sources_root": str(rubric_sources_output_dir.resolve()),
                        "worker_result_path": str((rubric_sources_output_dir / "result.json").resolve()),
                    }
                else:
                    stage_payloads["rubric_sources"] = {"status": "skipped"}

            if not args.disable_rubric and evaluation_target_text and stage_payloads["rubric_sources"].get("status") == "ok":
                rubric_llm_own_cache_keys = [
                    "smoke",
                    "disable_rubric",
                ]
                rubric_llm_cache_keys = cache_key_union(rubric_sources_cache_keys, rubric_llm_own_cache_keys)
                rubric_llm_cache_allowed = cache_allowed(
                    args=args,
                    stage_name="rubric_llm",
                    force_stages=force_stages,
                    input_matches_existing_run=input_matches_existing_run,
                )
                cached_rubric_llm = (
                    cached_stage_from_candidates(
                        candidates=candidates_for_dependencies(
                            cache_candidates,
                            stage_payloads.get("rubric_sources"),
                        ),
                        stage_name="rubric_llm",
                        materialized_output_dir=rubric_llm_output_dir,
                        run_dir=run_dir,
                        args=args,
                        config_keys=rubric_llm_cache_keys,
                        loader=lambda cache_dir, output_dir, source_dir: cached_rubric_llm_stage(
                            cache_dir,
                            materialized_output_dir=output_dir,
                            cache_source_dir=source_dir,
                            run_dir=run_dir,
                            loggers=[pipeline_logger, rubric_llm_logger],
                        ),
                        loggers=[pipeline_logger, rubric_llm_logger],
                    )
                    if rubric_llm_cache_allowed
                    else None
                )
                if cached_rubric_llm is not None:
                    stage_payloads["rubric_llm"] = cached_rubric_llm
                else:
                    reset_stage_directory(rubric_llm_output_dir)
                    rubric_llm_input = {
                        "idea_text": evaluation_target_text,
                        "idea_context_path": str(idea_context_path),
                        "sources_root": stage_payloads["rubric_sources"].get("sources_root")
                        or str(rubric_sources_output_dir.resolve()),
                        "env_path": str(Path(args.env).expanduser().resolve()),
                        "max_workers": args.rubric_max_workers,
                        "llm_timeout_seconds": args.rubric_llm_timeout_seconds,
                    }
                    rubric_llm_future = executor.submit(
                        launch_worker,
                        RUBRIC_LLM_WORKER_PATH,
                        input_payload=rubric_llm_input,
                        work_dir=rubric_llm_output_dir,
                        smoke=args.smoke,
                        loggers=[pipeline_logger, rubric_llm_logger],
                        stage_name="rubric_llm",
                        cuda_devices=cuda_devices,
                    )
            else:
                stage_payloads["rubric_llm"] = {"status": "skipped"}

            if stage_payloads.get("rubric_llm", {}).get("status") == "skipped" and rubric_llm_future is not None:
                pass
            elif rubric_llm_future is not None and stage_payloads.get("rubric_llm", {}).get("status") != "ok":
                rubric_llm_result = rubric_llm_future.result()
                rubric_llm_status = normalize_whitespace(rubric_llm_result.get("status")) or "error"
                rubric_llm_content_path = (
                    normalize_whitespace(rubric_llm_result.get("result_path"))
                    if rubric_llm_status == "ok"
                    else ""
                )
                stage_payloads["rubric_llm"] = {
                    "status": rubric_llm_status,
                    "rubric_schema_version": rubric_llm_result.get("rubric_schema_version"),
                    "result_path": rubric_llm_content_path or None,
                    "processing_index_path": rubric_llm_result.get("processing_index_path"),
                    "historical_context_path": rubric_llm_result.get("historical_context_path"),
                    "worker_result_path": str((rubric_llm_output_dir / "result.json").resolve()),
                }
            else:
                stage_payloads.setdefault("rubric_llm", {"status": "skipped"})
        finally:
            executor.shutdown(wait=True)

        artifact_index = {
            "idea_context_path": str(idea_context_path),
            "search_result_path": stage_payloads.get("search", {}).get("result_path"),
            "manifest_path": stage_payloads.get("manifest", {}).get("manifest_path"),
            "grounding_result_path": stage_payloads.get("grounding", {}).get("result_path"),
            "reviewers_index_path": reviewers_summary.get("index_path"),
            "reviewer_selection_path": str(selection_path.resolve()),
            "rubric_path": stage_payloads.get("rubric_llm", {}).get("result_path"),
            "idea_context_dir": str(idea_context_output_dir.resolve()),
            "search_dir": str(search_output_dir.resolve()),
            "manifest_dir": str(manifest_output_dir.resolve()),
            "grounding_dir": str(grounding_output_dir.resolve()),
            "reviewers_dir": str(reviewers_output_dir.resolve()),
            "rubric_sources_dir": str(rubric_sources_output_dir.resolve()),
            "rubric_llm_dir": str(rubric_llm_output_dir.resolve()),
            "review_dir": str(review_output_dir.resolve()),
            "report_dir": str(report_output_dir.resolve()),
        }

        review_own_cache_keys = [
            "smoke",
            "disable_review_stage",
            "disable_review_llm",
            "review_llm_base_url",
            "review_llm_model_name",
            "review_temperature",
            "enable_figure_cards",
            "figure_dir",
            "review_max_figure_cards_per_section",
            "review_max_evidence_cards_per_section",
            "review_max_evidence_cards_per_query",
            "review_max_experiment_cards_per_section",
            "review_max_total_evidence_chars_per_section",
        ]
        review_cache_keys = cache_key_union(
            idea_context_cache_keys,
            reviewer_cache_keys,
            grounding_cache_keys,
            rubric_llm_cache_keys if "rubric_llm_cache_keys" in locals() else [],
            review_own_cache_keys,
        )
        review_cache_allowed = cache_allowed(
            args=args,
            stage_name="review",
            force_stages=force_stages,
            input_matches_existing_run=input_matches_existing_run,
        )
        review_stage = (
            cached_stage_from_candidates(
                candidates=candidates_for_dependencies(
                    cache_candidates,
                    stage_payloads.get("idea_context"),
                    stage_payloads.get("reviewers"),
                    stage_payloads.get("grounding"),
                    stage_payloads.get("rubric_llm"),
                ),
                stage_name="review",
                materialized_output_dir=review_output_dir,
                run_dir=run_dir,
                args=args,
                config_keys=review_cache_keys,
                loader=lambda cache_dir, output_dir, source_dir: cached_review_stage(
                    cache_dir,
                    materialized_output_dir=output_dir,
                    cache_source_dir=source_dir,
                    run_dir=run_dir,
                    loggers=[pipeline_logger, review_logger],
                ),
                loggers=[pipeline_logger, review_logger],
            )
            if review_cache_allowed
            else None
        )
        if review_stage is None:
            review_stage = run_review_stage(
                args,
                idea_text=evaluation_target_text,
                source_full_text=source_full_text,
                output_dir=review_output_dir,
                rubric_path=stage_payloads.get("rubric_llm", {}).get("result_path"),
                reviewers_index_path=reviewers_summary.get("index_path"),
                grounding_result_path=stage_payloads.get("grounding", {}).get("result_path"),
                artifact_paths=artifact_index,
                loggers=[pipeline_logger, review_logger],
            )
        stage_payloads["review"] = review_stage

        artifact_index = {
            **artifact_index,
            "reviewer_reviews_path": stage_payloads.get("review", {}).get("reviewer_reviews_path"),
            "normalized_rubric_path": stage_payloads.get("review", {}).get("normalized_rubric_path"),
            "normalized_reviewers_path": stage_payloads.get("review", {}).get("normalized_reviewers_path"),
            "searched_papers_path": stage_payloads.get("review", {}).get("searched_papers_path"),
            "evidence_bank_path": stage_payloads.get("review", {}).get("evidence_bank_path"),
        }

        report_own_cache_keys = [
            "smoke",
            "disable_report_stage",
            "short_report",
            "disable_review_llm",
            "review_llm_base_url",
            "review_llm_model_name",
            "review_temperature",
        ]
        report_cache_keys = cache_key_union(review_cache_keys, report_own_cache_keys)
        report_cache_allowed = cache_allowed(
            args=args,
            stage_name="report",
            force_stages=force_stages,
            input_matches_existing_run=input_matches_existing_run,
        )
        report_stage = (
            cached_stage_from_candidates(
                candidates=candidates_for_dependencies(
                    cache_candidates,
                    stage_payloads.get("review"),
                ),
                stage_name="report",
                materialized_output_dir=report_output_dir,
                run_dir=run_dir,
                args=args,
                config_keys=report_cache_keys,
                loader=lambda cache_dir, output_dir, source_dir: cached_report_stage(
                    cache_dir,
                    args=args,
                    materialized_output_dir=output_dir,
                    cache_source_dir=source_dir,
                    run_dir=run_dir,
                    loggers=[pipeline_logger, report_logger],
                ),
                loggers=[pipeline_logger, report_logger],
            )
            if report_cache_allowed
            else None
        )
        if report_stage is None:
            report_stage = run_report_stage(
                args,
                idea_text=evaluation_target_text,
                output_dir=report_output_dir,
                review_stage=stage_payloads["review"],
                artifact_paths=artifact_index,
                loggers=[pipeline_logger, report_logger],
            )
        stage_payloads["report"] = report_stage

        final_status = combine_status([stage.get("status", "") for stage in stage_payloads.values()])
        if normalize_whitespace(stage_payloads.get("report", {}).get("status")) == "error":
            final_status = "error"

        final_payload = {
            "status": final_status,
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "cache_policy": "latest-compatible-stage",
            "latest_cache_source_run_dir": str(cache_run_dir.resolve()) if cache_run_dir else None,
            "cache_source_run_dir": str(cache_run_dir.resolve()) if cache_run_dir else None,
            "requested_force_stages": sorted(requested_force_stages),
            "effective_force_stages": sorted(force_stages),
            "input": input_payload,
            "idea_context_path": str(idea_context_path),
            "final_report_path": stage_payloads.get("report", {}).get("final_report_path"),
            "final_report_json_path": stage_payloads.get("report", {}).get("final_report_json_path"),
            "short_report": stage_payloads.get("report", {}).get("short_report", args.short_report),
            "meta_review": stage_payloads.get("report", {}).get("meta_review"),
            "review": {
                "status": stage_payloads.get("review", {}).get("status"),
                "reviewer_count": stage_payloads.get("review", {}).get("reviewer_count", 0),
                "successful_count": stage_payloads.get("review", {}).get("successful_count", 0),
                "reviewer_reviews_path": stage_payloads.get("review", {}).get("reviewer_reviews_path"),
            },
            "stages": stage_payloads,
            "artifacts": {
                **artifact_index,
                "idea_context_dir": str(idea_context_output_dir.resolve()),
                "search_dir": str(search_output_dir.resolve()),
                "manifest_dir": str(manifest_output_dir.resolve()),
                "grounding_dir": str(grounding_output_dir.resolve()),
                "reviewers_dir": str(reviewers_output_dir.resolve()),
                "rubric_sources_dir": str(rubric_sources_output_dir.resolve()),
                "rubric_llm_dir": str(rubric_llm_output_dir.resolve()),
                "review_dir": str(review_output_dir.resolve()),
                "report_dir": str(report_output_dir.resolve()),
                "logs_dir": str(logs_output_dir.resolve()),
                "log_paths": {
                    "pipeline": str((logs_output_dir / "pipeline.log").resolve()),
                    "idea_context": str((logs_output_dir / "idea_context.log").resolve()),
                    "search": str((logs_output_dir / "search.log").resolve()),
                    "manifest": str((logs_output_dir / "manifest.log").resolve()),
                    "grounding": str((logs_output_dir / "grounding.log").resolve()),
                    "reviewers": str((logs_output_dir / "reviewers.log").resolve()),
                    "rubric_sources": str((logs_output_dir / "rubric_sources.log").resolve()),
                    "rubric_llm": str((logs_output_dir / "rubric_llm.log").resolve()),
                    "review": str((logs_output_dir / "review.log").resolve()),
                    "report": str((logs_output_dir / "report.log").resolve()),
                },
            },
        }
        write_json(final_output_path, final_payload)
        log_to_many(
            [pipeline_logger],
            "info",
            f"Pipeline finished status={final_payload['status']} result_path={final_output_path.resolve()}",
        )
        print(json.dumps(final_payload, ensure_ascii=False, indent=2))
        return 0 if final_payload["status"] in {"ok", "partial_error"} else 1

    except Exception as exc:
        error_payload = {
            "status": "error",
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "cache_policy": "latest-compatible-stage",
            "latest_cache_source_run_dir": str(cache_run_dir.resolve()) if cache_run_dir else None,
            "cache_source_run_dir": str(cache_run_dir.resolve()) if cache_run_dir else None,
            "requested_force_stages": sorted(requested_force_stages),
            "effective_force_stages": sorted(force_stages),
            "input": input_payload,
            "stages": stage_payloads,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(final_output_path, error_payload)
        log_to_many(
            [pipeline_logger],
            "error",
            f"Pipeline failed error_type={exc.__class__.__name__} error={exc}",
        )
        log_multiline([pipeline_logger], "error", "pipeline.traceback", error_payload["traceback"])
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
