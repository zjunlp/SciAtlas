from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from re import sub
from typing import Any, Callable

from pydantic import BaseModel


NON_RETRYABLE_ERROR_MARKERS = (
    "insufficient_user_quota",
    "permissiondeniederror",
    "invalid_api_key",
    "authenticationerror",
)


def schema_json(response_model_schema: Any) -> str:
    if isinstance(response_model_schema, dict):
        schema = response_model_schema
    elif isinstance(response_model_schema, type) and issubclass(response_model_schema, BaseModel):
        if hasattr(response_model_schema, "model_json_schema"):
            schema = response_model_schema.model_json_schema()
        else:
            schema = response_model_schema.schema()
    else:
        raise TypeError("response_model_schema must be a pydantic model class or dict")
    return json.dumps(schema, ensure_ascii=False, indent=2)


def call_llm_json_with_retry(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    response_model_schema: Any,
    timeout_seconds: int,
    max_retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    helper_path: Path | None = None,
    label: str | None = None,
    debug_dir: Path | None = None,
    log_attempt: Callable[..., None] | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any] | None:
    worker_path = helper_path or Path(__file__).with_name("llm_json_call_worker.py")
    schema_str = schema_json(response_model_schema)
    debug_label = sub(r"[^A-Za-z0-9_.-]+", "_", label or "llm").strip("_") or "llm"

    last_error_text = ""
    for attempt in range(max_retries):
        attempt_started_at = time.perf_counter()
        effective_system_prompt = system_prompt
        if attempt > 0:
            effective_system_prompt += (
                "\n\nPrevious attempts returned invalid JSON. Output only a valid JSON object. "
                "Inside JSON strings, every backslash must be escaped as a double backslash. "
                "For example, write \\\\alpha, \\\\hat{x}, and \\\\text{...}; never write a single "
                "backslash before a letter."
            )
        raw_response_path = None
        repaired_response_path = None
        if debug_dir is not None:
            raw_response_path = debug_dir / f"{debug_label}_attempt_{attempt + 1}_raw.txt"
            repaired_response_path = debug_dir / f"{debug_label}_attempt_{attempt + 1}_repaired_input.txt"
        request_payload = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "system_prompt": effective_system_prompt,
            "user_content": user_content,
            "schema_str": schema_str,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
            "raw_response_path": str(raw_response_path) if raw_response_path is not None else None,
            "repaired_response_path": str(repaired_response_path) if repaired_response_path is not None else None,
        }
        error_text = ""
        try:
            completed = subprocess.run(
                [sys.executable, str(worker_path)],
                input=json.dumps(request_payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds + 10,
            )
            if completed.returncode == 0:
                result = json.loads(completed.stdout)
            else:
                result = {
                    "status": "error",
                    "error": completed.stderr.strip() or f"LLM worker exited with code {completed.returncode}",
                }
        except subprocess.TimeoutExpired:
            result = {"status": "error", "error": "LLM request timed out"}
        except Exception as exc:
            result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}

        elapsed_ms = round((time.perf_counter() - attempt_started_at) * 1000, 1)
        if isinstance(result, dict) and result.get("status") == "ok":
            payload = result.get("payload")
            if log_attempt is not None:
                log_attempt(
                    label=label,
                    attempt=attempt + 1,
                    status="ok",
                    elapsed_ms=elapsed_ms,
                    prompt_chars=len(user_content),
                    repaired_json=bool(result.get("repaired_json")),
                )
            return payload if isinstance(payload, dict) else None

        if log_attempt is not None:
            if isinstance(result, dict):
                error_text = " ".join(str(result.get("error", "")).split())[:300]
                last_error_text = error_text
            log_attempt(
                label=label,
                attempt=attempt + 1,
                status="error",
                elapsed_ms=elapsed_ms,
                prompt_chars=len(user_content),
                error=error_text,
            )
        elif isinstance(result, dict):
            error_text = " ".join(str(result.get("error", "")).split())[:300]
            if error_text:
                last_error_text = error_text
        if error_text and any(marker in error_text.casefold() for marker in NON_RETRYABLE_ERROR_MARKERS):
            break
        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    if debug_dir is not None:
        debug_label = sub(r"[^A-Za-z0-9_.-]+", "_", label or "llm").strip("_") or "llm"
        failure_path = debug_dir / f"{debug_label}_final_error.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(last_error_text or "LLM JSON generation failed", encoding="utf-8")
    if raise_on_error:
        raise RuntimeError(last_error_text or "LLM JSON generation failed")
    return None
