from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from .common import normalize_whitespace, write_json


DEFAULT_PERSONA_SUBJECT = "computer science"
DEFAULT_PROMPTS_MD_PATH = Path(__file__).with_name("prompts.md")
DEFAULT_PERSONA_JSON_PATH = Path(__file__).with_name("persona_prompts.json")
SUBJECT_ALIASES = {
    "material": "material science",
    "materials": "material science",
    "materials science": "material science",
}


def normalize_subject(subject: Any) -> str:
    normalized = normalize_whitespace(subject).casefold()
    return SUBJECT_ALIASES.get(normalized, normalized)


def _persona_id(subject: str, index: int) -> str:
    prefix = re.sub(r"[^a-z0-9]+", "_", normalize_subject(subject)).strip("_")
    return f"{prefix or 'persona'}_{index:02d}"


def parse_persona_markdown(markdown_text: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = {}
    current_subject = ""
    current_lines: list[str] = []

    def flush_item() -> None:
        if not current_subject or not current_lines:
            return
        text = normalize_whitespace(" ".join(current_lines))
        if not text:
            return
        subject_items = groups.setdefault(current_subject, [])
        subject_items.append(
            {
                "id": _persona_id(current_subject, len(subject_items) + 1),
                "text": text,
            }
        )

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading_match = re.match(r"^#{2,6}\s+(.+?)\s*$", line)
        if heading_match:
            flush_item()
            current_lines = []
            current_subject = normalize_subject(heading_match.group(1))
            continue

        item_match = re.match(r"^\d+\.\s+(.+?)\s*$", line)
        if item_match:
            flush_item()
            current_lines = [item_match.group(1)]
            continue

        if current_lines:
            current_lines.append(line)

    flush_item()
    return {
        "schema_version": 1,
        "groups": groups,
    }


def convert_persona_markdown_to_json(
    md_path: Path = DEFAULT_PROMPTS_MD_PATH,
    output_path: Path = DEFAULT_PERSONA_JSON_PATH,
) -> dict[str, Any]:
    md_path = Path(md_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    payload = parse_persona_markdown(md_path.read_text(encoding="utf-8"))
    payload["source_path"] = str(md_path)
    write_json(output_path, payload)
    return payload


def load_persona_payload(path: Path = DEFAULT_PERSONA_JSON_PATH) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected persona JSON object in {resolved_path}")
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"Expected persona JSON to contain a groups object: {resolved_path}")
    return payload


def select_random_persona(
    *,
    persona_json_path: Path = DEFAULT_PERSONA_JSON_PATH,
    subject: str = DEFAULT_PERSONA_SUBJECT,
    rng: random.Random | None = None,
) -> str:
    payload = load_persona_payload(persona_json_path)
    groups = payload["groups"]
    subject_key = normalize_subject(subject) or DEFAULT_PERSONA_SUBJECT

    missing = [key for key in ("general", subject_key) if key not in groups]
    if missing:
        available = ", ".join(sorted(str(key) for key in groups))
        raise ValueError(
            f"Missing persona group(s): {', '.join(missing)}. Available groups: {available}"
        )

    candidates: list[str] = []
    seen: set[str] = set()
    for group_key in ("general", subject_key):
        items = groups[group_key]
        if not isinstance(items, list):
            raise ValueError(f"Persona group {group_key!r} must be a list")
        for item in items:
            if not isinstance(item, dict):
                continue
            text = normalize_whitespace(item.get("text"))
            if text and text not in seen:
                seen.add(text)
                candidates.append(text)

    if not candidates:
        raise ValueError(f"No persona candidates found for subject {subject_key!r}")

    chooser = rng or random
    return chooser.choice(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert reviewer persona prompts.md into JSON.")
    parser.add_argument("--md-path", default=str(DEFAULT_PROMPTS_MD_PATH), help="Input markdown prompt file.")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_PERSONA_JSON_PATH),
        help="Output persona JSON file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = convert_persona_markdown_to_json(
        md_path=Path(args.md_path),
        output_path=Path(args.output_path),
    )
    total = sum(len(items) for items in payload["groups"].values())
    print(f"Wrote {total} persona prompts across {len(payload['groups'])} groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
