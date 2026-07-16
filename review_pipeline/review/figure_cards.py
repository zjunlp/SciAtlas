from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .common import first_non_empty, normalize_whitespace, slugify, write_json


def _codex_default_paths() -> tuple[Path, Path]:
    return Path("~/.codex/config.toml").expanduser(), Path("~/.codex/auth.json").expanduser()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_between(text: str, start: str, end_tokens: tuple[str, ...]) -> str:
    anchor = text.find(start)
    if anchor < 0:
        return ""
    rest = text[anchor + len(start) :]
    end_positions = [pos for token in end_tokens if (pos := rest.find(token)) >= 0]
    if not end_positions:
        return rest.strip()
    return rest[: min(end_positions)].strip()


def _load_codex_provider_defaults() -> dict[str, str]:
    config_path, auth_path = _codex_default_paths()
    values: dict[str, str] = {}
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        provider = _extract_between(text, 'model_provider = "', ('"', "\n")) or _extract_between(
            text, 'model_provider="', ('"', "\n")
        )
        model = _extract_between(text, 'model = "', ('"', "\n")) or _extract_between(text, 'model="', ('"', "\n"))
        if provider:
            block_key = f"[model_providers.{provider}]"
            block_start = text.find(block_key)
            if block_start >= 0:
                block = text[block_start:]
                next_block = block.find("\n[", len(block_key))
                if next_block >= 0:
                    block = block[:next_block]
                base_url = _extract_between(block, 'base_url = "', ('"', "\n")) or _extract_between(
                    block, 'base_url = "', ('"', "\n")
                )
                values["provider_name"] = provider
                values["base_url"] = base_url
        if model:
            values["model"] = model
    if auth_path.exists():
        payload = _load_json(auth_path)
        api_key = first_non_empty(payload.get("OPENAI_API_KEY"))
        if api_key:
            values["api_key"] = api_key
    return values


def _normalize_base_url(base_url: str) -> str:
    trimmed = normalize_whitespace(base_url).rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def _figure_card_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "figure_label": {"type": "string"},
            "paper_sections": {"type": "array", "items": {"type": "string"}},
            "section_path_text": {"type": "string"},
            "idea_section": {"type": "string"},
            "related_claim": {"type": "string"},
            "figure_type": {"type": "string"},
            "evidence_text": {"type": "string"},
            "visual_observation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "axes": {"type": "array", "items": {"type": "string"}},
                    "series": {"type": "array", "items": {"type": "string"}},
                    "readable_values": {"type": "array", "items": {"type": "string"}},
                    "trends": {"type": "array", "items": {"type": "string"}},
                    "anomalies": {"type": "array", "items": {"type": "string"}},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["axes", "series", "readable_values", "trends", "anomalies", "uncertainties"],
            },
            "paper_alignment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "author_claim": {"type": "string"},
                    "claimed_takeaway": {"type": "string"},
                    "supports_claim_well": {"type": "boolean"},
                    "support_rationale": {"type": "string"},
                },
                "required": ["author_claim", "claimed_takeaway", "supports_claim_well", "support_rationale"],
            },
            "reviewer_caution": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "cannot_conclude": {"type": "array", "items": {"type": "string"}},
                    "missing_controls": {"type": "array", "items": {"type": "string"}},
                    "overclaim_risk": {"type": "string"},
                },
                "required": ["cannot_conclude", "missing_controls", "overclaim_risk"],
            },
            "relevance": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "Motivation": {"type": "number"},
                    "Method": {"type": "number"},
                    "Result": {"type": "number"},
                    "Discussion": {"type": "number"},
                },
                "required": ["Motivation", "Method", "Result", "Discussion"],
            },
            "support_strength": {"type": "number"},
        },
        "required": [
            "figure_label",
            "paper_sections",
            "section_path_text",
            "idea_section",
            "related_claim",
            "figure_type",
            "evidence_text",
            "visual_observation",
            "paper_alignment",
            "reviewer_caution",
            "relevance",
            "support_strength",
        ],
    }


def _card_prompt(source_full_text: str, file_name: str) -> str:
    truncated = source_full_text[:50000]
    return f"""
You are creating a structured figure card for a scientific peer-review pipeline.

The attached file is one figure PDF from the target manuscript.
Use both the attached figure and the manuscript full text below.

Manuscript full text:
{truncated}

Instructions:
- Be faithful to what is visually present in the figure.
- Also align the figure to the manuscript's likely section and claim.
- Distinguish visible evidence from author interpretation.
- `paper_sections` must use only items from: Motivation, Method, Result, Discussion.
- `idea_section` should be one short lowercase tag like motivation, method, experiment, evaluation, discussion.
- `support_strength` and section relevance scores must be between 0 and 1.
- `evidence_text` should be compact but reviewer-ready: 3 to 6 sentences.
- If uncertain, say so explicitly in `uncertainties` or `overclaim_risk`.
- Use the file name `{file_name}` only as a weak hint; do not hallucinate if the text/figure do not support it.
""".strip()


def _build_input_message(prompt: str, file_name: str, file_bytes: bytes) -> list[dict[str, Any]]:
    import base64
    import mimetypes

    mime_type, _ = mimetypes.guess_type(file_name)
    mime_type = mime_type or "application/pdf"
    file_data = base64.b64encode(file_bytes).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_file",
                    "filename": file_name,
                    "file_data": f"data:{mime_type};base64,{file_data}",
                    "detail": "high",
                },
            ],
        }
    ]


def _responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    dumped = response.model_dump()
    chunks: list[str] = []
    for item in dumped.get("output", []):
        for content in item.get("content", []):
            text = normalize_whitespace(content.get("text"))
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end >= start else text


def _sanitize_card(raw: dict[str, Any], *, figure_id: str, file_path: Path, model_name: str) -> dict[str, Any]:
    def list_str(key: str, parent: dict[str, Any] | None = None) -> list[str]:
        source = parent if parent is not None else raw
        value = source.get(key)
        if not isinstance(value, list):
            value = [value] if value is not None else []
        seen: set[str] = set()
        items: list[str] = []
        for item in value:
            text = normalize_whitespace(item)
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                items.append(text)
        return items

    visual = raw.get("visual_observation") if isinstance(raw.get("visual_observation"), dict) else {}
    align = raw.get("paper_alignment") if isinstance(raw.get("paper_alignment"), dict) else {}
    caution = raw.get("reviewer_caution") if isinstance(raw.get("reviewer_caution"), dict) else {}
    relevance_raw = raw.get("relevance") if isinstance(raw.get("relevance"), dict) else {}

    def score(name: str, default: float) -> float:
        try:
            value = float(relevance_raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(0.0, min(1.0, value))

    try:
        support_strength = float(raw.get("support_strength", 0.5))
    except (TypeError, ValueError):
        support_strength = 0.5
    support_strength = max(0.0, min(1.0, support_strength))

    paper_sections = [section for section in list_str("paper_sections") if section in {"Motivation", "Method", "Result", "Discussion"}]
    if not paper_sections:
        best_section = max(
            ("Motivation", "Method", "Result", "Discussion"),
            key=lambda name: score(name, 0.0),
        )
        paper_sections = [best_section]

    return {
        "figure_id": figure_id,
        "figure_label": normalize_whitespace(raw.get("figure_label")) or file_path.stem,
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "paper_sections": paper_sections,
        "section_path_text": normalize_whitespace(raw.get("section_path_text")) or "Result",
        "idea_section": normalize_whitespace(raw.get("idea_section")) or "experiment",
        "related_claim": normalize_whitespace(raw.get("related_claim")),
        "figure_type": normalize_whitespace(raw.get("figure_type")) or "unknown",
        "evidence_text": normalize_whitespace(raw.get("evidence_text")),
        "visual_observation": {
            "axes": list_str("axes", visual),
            "series": list_str("series", visual),
            "readable_values": list_str("readable_values", visual),
            "trends": list_str("trends", visual),
            "anomalies": list_str("anomalies", visual),
            "uncertainties": list_str("uncertainties", visual),
        },
        "paper_alignment": {
            "author_claim": normalize_whitespace(align.get("author_claim")),
            "claimed_takeaway": normalize_whitespace(align.get("claimed_takeaway")),
            "supports_claim_well": bool(align.get("supports_claim_well")),
            "support_rationale": normalize_whitespace(align.get("support_rationale")),
        },
        "reviewer_caution": {
            "cannot_conclude": list_str("cannot_conclude", caution),
            "missing_controls": list_str("missing_controls", caution),
            "overclaim_risk": normalize_whitespace(caution.get("overclaim_risk")),
        },
        "relevance": {
            "Motivation": score("Motivation", 0.1),
            "Method": score("Method", 0.25),
            "Result": score("Result", 0.8),
            "Discussion": score("Discussion", 0.4),
        },
        "support_strength": support_strength,
        "generation": {
            "model": model_name,
            "status": "ok",
        },
    }


def generate_figure_cards(
    *,
    figure_dir: Path | str | None,
    source_full_text: str,
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    figure_root = Path(str(figure_dir)).expanduser().resolve() if figure_dir else None
    defaults = _load_codex_provider_defaults()
    resolved_base_url = _normalize_base_url(first_non_empty(base_url, defaults.get("base_url")))
    resolved_model = first_non_empty(model_name, defaults.get("model"))
    resolved_api_key = first_non_empty(api_key, defaults.get("api_key"))

    errors: list[dict[str, str]] = []
    cards: list[dict[str, Any]] = []

    if figure_root is None or not figure_root.exists() or not figure_root.is_dir():
        payload = {
            "status": "missing",
            "figure_dir": str(figure_root) if figure_root else None,
            "figure_card_count": 0,
            "figure_cards": [],
            "errors": [{"path": str(figure_root) if figure_root else "", "error": "figure directory missing"}],
        }
        if output_path is not None:
            write_json(Path(output_path), payload)
        return payload

    if not resolved_base_url or not resolved_model or not resolved_api_key:
        payload = {
            "status": "error",
            "figure_dir": str(figure_root),
            "figure_card_count": 0,
            "figure_cards": [],
            "errors": [{"path": str(figure_root), "error": "review multimodal provider configuration is incomplete"}],
        }
        if output_path is not None:
            write_json(Path(output_path), payload)
        return payload

    client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
    files = sorted(path for path in figure_root.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    for index, file_path in enumerate(files, start=1):
        figure_id = f"F{index:03d}"
        try:
            response = client.responses.create(
                model=resolved_model,
                input=_build_input_message(_card_prompt(source_full_text, file_path.name), file_path.name, file_path.read_bytes()),
                max_output_tokens=2200,
                store=False,
                text={"format": {"type": "json_schema", "name": "figure_card", "schema": _figure_card_schema(), "strict": True}},
            )
            raw = json.loads(_clean_json_text(_responses_text(response)))
            cards.append(_sanitize_card(raw, figure_id=figure_id, file_path=file_path, model_name=resolved_model))
        except Exception as exc:
            errors.append({"path": str(file_path), "error": str(exc)})

    payload = {
        "status": "ok" if cards else ("partial_error" if errors else "missing"),
        "figure_dir": str(figure_root),
        "figure_card_count": len(cards),
        "figure_cards": cards,
        "errors": errors,
    }
    if cards and errors:
        payload["status"] = "partial_error"
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def summarize_figure_card(card: dict[str, Any]) -> str:
    label = normalize_whitespace(card.get("figure_label")) or normalize_whitespace(card.get("file_name")) or slugify(
        normalize_whitespace(card.get("figure_id"))
    )
    match = re.search(r"fig(?:ure)?\.?\s*([0-9]+(?:\([a-z]\)|[a-z])?)", label, flags=re.IGNORECASE)
    recommended_citation = f"Fig. {match.group(1)}" if match else label
    section_path = normalize_whitespace(card.get("section_path_text")) or "Result"
    evidence = normalize_whitespace(card.get("evidence_text"))
    claim = normalize_whitespace(card.get("related_claim"))
    caution = normalize_whitespace(
        (card.get("reviewer_caution") or {}).get("overclaim_risk") if isinstance(card.get("reviewer_caution"), dict) else ""
    )
    return "\n".join(
        [
            f"Figure: {label}",
            f"Recommended manuscript citation: {recommended_citation}",
            f"Likely manuscript section: {section_path}",
            f"Paper sections: {json.dumps(card.get('paper_sections') or [], ensure_ascii=False)}",
            f"Related claim: {claim or 'Not specified.'}",
            f"Figure-grounded summary: {evidence or 'Not available.'}",
            f"Reviewer caution: {caution or 'Not specified.'}",
        ]
    )
