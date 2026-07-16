"""Small shared helpers: logging, retries, JSON extraction and run IO."""
from __future__ import annotations

import json
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger("sciatlas.utils")


def retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    backoff: float = 5.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    description: str = "call",
) -> T:
    """Run ``fn`` with exponential backoff. Re-raises the last error on failure."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203 - explicit retry loop
            last_exc = exc
            if attempt == max_retries:
                break
            wait = backoff * (2 ** (attempt - 1))
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description, attempt, max_retries, exc, wait,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles ```json fenced blocks and leading/trailing prose. Raises
    ``ValueError`` if nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty LLM response")

    # 1) fenced code block
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2) direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 3) first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"could not parse JSON from LLM response: {text[:200]!r}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
