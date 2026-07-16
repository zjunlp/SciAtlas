#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from review.common import first_non_empty, load_env_values, normalize_whitespace
from review.evaluation import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL_NAME


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"


@dataclass(slots=True)
class AttemptRecord:
    attempt: int
    status: str
    elapsed_ms: float
    error_type: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ProbeResult:
    request_id: int
    attempt_count: int
    status: str
    elapsed_ms: float
    attempts: list[AttemptRecord]
    content: str | None = None
    error_type: str | None = None
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe LLM network behavior under concurrent chat completion requests."
    )
    parser.add_argument(
        "--env",
        default=str(DEFAULT_ENV_PATH),
        help="Env file with DMX_API_KEY/OPENAI_API_KEY.",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key override.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--model", default=None, help="Optional model name.")
    parser.add_argument("--requests", type=int, default=20, help="Total logical requests to send.")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent requests.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Per-call timeout.")
    parser.add_argument(
        "--sdk-max-retries",
        type=int,
        default=0,
        help="OpenAI SDK internal retries. Use 0 to expose raw connection failures.",
    )
    parser.add_argument(
        "--app-retries",
        type=int,
        default=1,
        help="Logical retries around each request. Use 5 to mimic review/report retry count.",
    )
    parser.add_argument(
        "--client-mode",
        choices=("shared", "per-request"),
        default="shared",
        help="Use one shared client or create one client per logical request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Small response budget for the probe request.",
    )
    parser.add_argument(
        "--prompt",
        default="Return JSON with keys ok=true and request_id copied from the user message.",
        help="Small user prompt for each request.",
    )
    parser.add_argument(
        "--prompt-chars",
        type=int,
        default=0,
        help="Append deterministic filler text until the user prompt is about this many characters.",
    )
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    requested_env_path = Path(args.env).expanduser().resolve() if args.env else None
    env_path = requested_env_path
    if env_path is not None and not env_path.exists() and DEFAULT_ENV_PATH.exists():
        env_path = DEFAULT_ENV_PATH.resolve()
    env_values = load_env_values(env_path)
    api_key = first_non_empty(
        args.api_key,
        env_values.get("DMX-API-KEY"),
        env_values.get("DMX_API_KEY"),
        env_values.get("OPENAI_API_KEY"),
        os.environ.get("DMX-API-KEY"),
        os.environ.get("DMX_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    )
    if not api_key:
        env_hint = str(env_path) if env_path else "<none>"
        raise ValueError(
            "Missing API key. "
            f"Checked env file {env_hint} and process environment. "
            "Pass --api-key or set DMX_API_KEY/OPENAI_API_KEY."
        )
    return {
        "api_key": api_key,
        "requested_env_path": str(requested_env_path) if requested_env_path else None,
        "env_path": str(env_path) if env_path else None,
        "base_url": first_non_empty(args.base_url, env_values.get("OPENAI_BASE_URL"), DEFAULT_LLM_BASE_URL),
        "model": first_non_empty(args.model, DEFAULT_LLM_MODEL_NAME),
        "requests": max(1, int(args.requests)),
        "concurrency": max(1, int(args.concurrency)),
        "timeout_seconds": float(args.timeout_seconds),
        "sdk_max_retries": max(0, int(args.sdk_max_retries)),
        "app_retries": max(1, int(args.app_retries)),
        "client_mode": args.client_mode,
        "max_tokens": max(1, int(args.max_tokens)),
        "prompt": args.prompt,
        "prompt_chars": max(0, int(args.prompt_chars)),
    }


def _make_client(settings: dict[str, Any]) -> OpenAI:
    return OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=settings["sdk_max_retries"],
    )


def _probe_one(
    *,
    request_id: int,
    settings: dict[str, Any],
    shared_client: OpenAI | None,
) -> ProbeResult:
    started_at = time.perf_counter()
    last_error: Exception | None = None
    attempts: list[AttemptRecord] = []
    user_prompt = f"{settings['prompt']}\nrequest_id={request_id}"
    if settings["prompt_chars"] > len(user_prompt):
        filler_unit = (
            "\nThis is deterministic probe filler text for testing large LLM request payloads. "
            "It carries no task-specific meaning."
        )
        while len(user_prompt) < settings["prompt_chars"]:
            user_prompt += filler_unit
    for attempt in range(1, settings["app_retries"] + 1):
        attempt_started_at = time.perf_counter()
        try:
            client = shared_client if shared_client is not None else _make_client(settings)
            response = client.chat.completions.create(
                model=settings["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "Return only a compact JSON object. Do not include markdown.",
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=settings["max_tokens"],
                timeout=settings["timeout_seconds"],
            )
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    status="ok",
                    elapsed_ms=round((time.perf_counter() - attempt_started_at) * 1000, 1),
                )
            )
            content = normalize_whitespace(response.choices[0].message.content)
            return ProbeResult(
                request_id=request_id,
                attempt_count=attempt,
                status="ok",
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1),
                attempts=attempts,
                content=content[:500],
            )
        except Exception as exc:
            last_error = exc
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    status="error",
                    elapsed_ms=round((time.perf_counter() - attempt_started_at) * 1000, 1),
                    error_type=exc.__class__.__name__,
                    error=normalize_whitespace(exc)[:1000],
                )
            )
            if attempt < settings["app_retries"]:
                time.sleep(2 ** (attempt - 1))

    return ProbeResult(
        request_id=request_id,
        attempt_count=settings["app_retries"],
        status="error",
        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1),
        attempts=attempts,
        error_type=last_error.__class__.__name__ if last_error is not None else "UnknownError",
        error=normalize_whitespace(last_error)[:1000] if last_error is not None else "",
    )


def _summarize(results: list[ProbeResult], settings: dict[str, Any], total_elapsed_ms: float) -> dict[str, Any]:
    ok_results = [result for result in results if result.status == "ok"]
    error_results = [result for result in results if result.status != "ok"]
    elapsed_values = [result.elapsed_ms for result in ok_results]
    errors_by_type: dict[str, int] = {}
    for result in error_results:
        key = result.error_type or "UnknownError"
        errors_by_type[key] = errors_by_type.get(key, 0) + 1
    attempt_errors_by_type: dict[str, int] = {}
    attempt_error_count = 0
    for result in results:
        for attempt in result.attempts:
            if attempt.status != "error":
                continue
            attempt_error_count += 1
            key = attempt.error_type or "UnknownError"
            attempt_errors_by_type[key] = attempt_errors_by_type.get(key, 0) + 1

    return {
        "status": "ok" if not error_results else "partial_error" if ok_results else "error",
        "requested_env_path": settings["requested_env_path"],
        "env_path": settings["env_path"],
        "base_url": settings["base_url"],
        "model": settings["model"],
        "requests": settings["requests"],
        "concurrency": settings["concurrency"],
        "client_mode": settings["client_mode"],
        "sdk_max_retries": settings["sdk_max_retries"],
        "app_retries": settings["app_retries"],
        "prompt_chars": settings["prompt_chars"],
        "max_tokens": settings["max_tokens"],
        "timeout_seconds": settings["timeout_seconds"],
        "total_elapsed_ms": round(total_elapsed_ms, 1),
        "success_count": len(ok_results),
        "error_count": len(error_results),
        "errors_by_type": errors_by_type,
        "attempt_error_count": attempt_error_count,
        "attempt_errors_by_type": attempt_errors_by_type,
        "retry_success_count": sum(1 for result in ok_results if result.attempt_count > 1),
        "max_attempt_count": max((result.attempt_count for result in results), default=0),
        "success_latency_ms": {
            "min": round(min(elapsed_values), 1) if elapsed_values else None,
            "median": round(statistics.median(elapsed_values), 1) if elapsed_values else None,
            "max": round(max(elapsed_values), 1) if elapsed_values else None,
        },
        "results": [asdict(result) for result in sorted(results, key=lambda item: item.request_id)],
    }


def main() -> int:
    args = _parse_args()
    settings = _resolve_settings(args)
    shared_client = _make_client(settings) if settings["client_mode"] == "shared" else None
    started_at = time.perf_counter()
    results: list[ProbeResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=settings["concurrency"]) as executor:
        futures = [
            executor.submit(
                _probe_one,
                request_id=request_id,
                settings=settings,
                shared_client=shared_client,
            )
            for request_id in range(1, settings["requests"] + 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)

    summary = _summarize(results, settings, (time.perf_counter() - started_at) * 1000)
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
