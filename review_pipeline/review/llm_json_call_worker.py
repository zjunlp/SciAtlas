#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


def clean_json_response(result_str: str) -> str:
    stripped = result_str.strip()
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if match:
        return match.group(0)
    return stripped


def repair_invalid_json_escapes(json_text: str) -> str:
    """Escape lone backslashes inside JSON strings without touching valid escapes."""
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    repaired: list[str] = []
    in_string = False
    i = 0

    while i < len(json_text):
        char = json_text[i]
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            i += 1
            continue

        if char != "\\":
            repaired.append(char)
            i += 1
            continue

        if i + 1 >= len(json_text):
            repaired.append("\\\\")
            i += 1
            continue

        next_char = json_text[i + 1]
        if next_char in valid_escapes:
            repaired.append(char)
            repaired.append(next_char)
            i += 2
            continue

        if next_char == "u" and i + 5 < len(json_text):
            hex_digits = json_text[i + 2 : i + 6]
            if all(c in "0123456789abcdefABCDEF" for c in hex_digits):
                repaired.append(json_text[i : i + 6])
                i += 6
                continue

        repaired.append("\\\\")
        i += 1

    return "".join(repaired)


def parse_json_response(content: str) -> tuple[dict[str, Any], bool]:
    cleaned = clean_json_response(content)
    try:
        payload = json.loads(cleaned)
        return payload, False
    except json.JSONDecodeError:
        repaired = repair_invalid_json_escapes(cleaned)
        payload = json.loads(repaired)
        return payload, True


def write_debug_file(path_value: Any, text: str) -> None:
    if not path_value:
        return
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        timeout_seconds = int(payload["timeout_seconds"])
        client = OpenAI(
            api_key=payload["api_key"],
            base_url=payload["base_url"],
            timeout=timeout_seconds,
        )
        response = client.chat.completions.create(
            model=payload["model_name"],
            messages=[
                {
                    "role": "system",
                    "content": payload["system_prompt"] + f"\n\nJSON Schema to follow:\n{payload['schema_str']}",
                },
                {"role": "user", "content": payload["user_content"]},
            ],
            response_format={"type": "json_object"},
            temperature=float(payload["temperature"]),
            max_tokens=int(payload["max_tokens"]),
            timeout=timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        try:
            parsed_payload, repaired = parse_json_response(content)
            result = {"status": "ok", "payload": parsed_payload}
            if repaired:
                result["repaired_json"] = True
                write_debug_file(payload.get("repaired_response_path"), clean_json_response(content))
        except Exception:
            write_debug_file(payload.get("raw_response_path"), content)
            raise
    except Exception as exc:
        result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
