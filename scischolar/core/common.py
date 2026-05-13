from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs"
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_SCISCHOLAR_API_TIMEOUT = 120
DEFAULT_SCISCHOLAR_API_SEARCH_TIMEOUT = 1800
DEFAULT_SCISCHOLAR_API_AUTHORS_RELATED_TIMEOUT = 120
DEFAULT_SCISCHOLAR_API_AUTHORS_PAPERS_TIMEOUT = 300
DEFAULT_SCISCHOLAR_API_SUPPORT_PAPERS_TIMEOUT = 600
DEFAULT_SCISCHOLAR_API_CONNECT_TIMEOUT = 10
DEFAULT_SCISCHOLAR_API_WRITE_TIMEOUT = 60
DEFAULT_SCISCHOLAR_API_POOL_TIMEOUT = 10


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def truncate_text(text: Any, *, max_chars: int = 240) -> str:
    normalized = normalize_whitespace(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_env_value(env_values: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = normalize_whitespace(env_values.get(key))
        if value:
            return value
    return ""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
        else:
            cleaned = "\n".join(lines[1:]).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not locate a JSON object in the LLM response.") from None
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw_item in items:
        item = normalize_whitespace(raw_item)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(output_root: Path, task_type: str, run_id: str | None, input_summary: str) -> Path:
    output_root = ensure_dir(output_root.expanduser().resolve())
    safe_run_id = slugify(run_id or f"{make_timestamp()}_{task_type}_{input_summary}", limit=120)
    return ensure_dir(output_root / safe_run_id)


def relative_path(path: Path, *, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except Exception:
        return str(path.resolve())
