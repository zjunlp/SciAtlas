from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
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
        return "item"
    return slug[:limit].rstrip("-") or "item"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_file_logger(log_path: Path, source_file: str) -> logging.LoggerAdapter:
    resolved_path = log_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"innoeval.worker.{source_file}.{resolved_path}"
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
