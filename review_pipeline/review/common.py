from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REVIEW_SECTIONS = ("Motivation", "Method", "Result", "Discussion")

RUBRIC_STANDARD_KEYS = {
    "Motivation": "Motivation_Standards",
    "Method": "Method_Standards",
    "Result": "Result_Standards",
    "Discussion": "Discussion_Standards",
}

LEGACY_SECTION_ALIASES = {
    "Introduction": "Motivation",
    "Methods": "Method",
    "Results": "Result",
    "Discussion": "Discussion",
    "Idea-Specific": "",
}


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


def load_env_values(env_path: Path | None) -> dict[str, str]:
    if env_path is None:
        return {}
    path = Path(env_path).expanduser().resolve()
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def citation_year(*values: Any) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if not text:
            continue
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return match.group(0)
    return "n.d."


def author_last_name(name: Any) -> str:
    text = normalize_whitespace(name)
    if not text:
        return ""
    tokens = [token for token in re.split(r"\s+", text) if token]
    return tokens[-1].strip(",.;") if tokens else ""


def citation_author_label(authors: Any) -> str:
    if not isinstance(authors, list):
        authors = []
    last_names = [author_last_name(author.get("name") if isinstance(author, dict) else author) for author in authors]
    last_names = [name for name in last_names if name]
    if not last_names:
        return "Unknown Author"
    if len(last_names) == 1:
        return last_names[0]
    if len(last_names) == 2:
        return f"{last_names[0]} and {last_names[1]}"
    return f"{last_names[0]} et al."


def citation_label(authors: Any, *, year: Any = None, publication_date: Any = None) -> str:
    return f"{citation_author_label(authors)}, {citation_year(publication_date, year)}"


def citation_markdown(
    authors: Any,
    *,
    year: Any = None,
    publication_date: Any = None,
    url: Any = None,
) -> str:
    label = citation_label(authors, year=year, publication_date=publication_date)
    link = normalize_whitespace(url)
    return f"[{label}]({link})" if link else label


_MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", flags=re.DOTALL)
_NUMERIC_PROSE_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def _apply_outside_math_spans(text: str, transform: Any) -> str:
    parts: list[str] = []
    cursor = 0
    for match in _MATH_SPAN_RE.finditer(text):
        if match.start() > cursor:
            parts.append(transform(text[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    if cursor < len(text):
        parts.append(transform(text[cursor:]))
    return "".join(parts)


def normalize_citation_spacing(text: Any) -> str:
    value = str(text or "")

    def transform(segment: str) -> str:
        segment = re.sub(r"(\]\([^)]+\))(?=[A-Za-z0-9])", r"\1 ", segment)
        segment = re.sub(
            r"(?<=[A-Za-z0-9),.;:!?])(?=\[[A-Za-z][^\]\n]{0,120},\s*(?:19|20)\d{2}\]\()",
            " ",
            segment,
        )
        segment = re.sub(r"[ \t]+([,.;:])", r"\1", segment)
        segment = re.sub(r"[ \t]{2,}", " ", segment)
        return segment

    return _apply_outside_math_spans(value, transform).strip()


def link_author_year_citation_labels(text: Any, citation_lookup: dict[str, str] | None = None) -> str:
    value = str(text or "")
    if not citation_lookup:
        return normalize_citation_spacing(value)

    normalized_lookup: dict[str, str] = {}
    for key, citation in citation_lookup.items():
        key_text = normalize_whitespace(key)
        citation_text = normalize_whitespace(citation)
        if not key_text or not citation_text:
            continue
        normalized_lookup[key_text] = citation_text
        normalized_lookup[key_text.casefold()] = citation_text
        match = re.fullmatch(r"\[([^\]\n]+)\]\([^)]+\)", citation_text)
        if match:
            label = normalize_whitespace(match.group(1))
            normalized_lookup[label] = citation_text
            normalized_lookup[label.casefold()] = citation_text

    author_year_group_re = re.compile(
        r"\[([^\]\n]{2,120},\s*(?:19|20)\d{2}(?:\s*;\s*[^\]\n]{2,120},\s*(?:19|20)\d{2})*)\](?!\()"
    )

    def transform(segment: str) -> str:
        def replace(match: re.Match[str]) -> str:
            labels = [normalize_whitespace(label) for label in match.group(1).split(";")]
            labels = [label for label in labels if label]
            linked: list[str] = []
            changed = False
            for label in labels:
                citation = normalized_lookup.get(label) or normalized_lookup.get(label.casefold())
                if citation:
                    linked.append(citation)
                    changed = True
                else:
                    linked.append(f"[{label}]")
            return "; ".join(linked) if changed else match.group(0)

        return author_year_group_re.sub(replace, segment)

    return normalize_citation_spacing(_apply_outside_math_spans(value, transform))


def strip_numeric_prose_citations(text: Any) -> str:
    value = str(text or "")

    def transform(segment: str) -> str:
        segment = re.sub(
            r"\(\s*(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*,?\s*)+\)",
            " ",
            segment,
        )
        segment = _NUMERIC_PROSE_CITATION_RE.sub(" ", segment)
        segment = re.sub(r"[ \t]+([,.;:])", r"\1", segment)
        segment = re.sub(r"[ \t]{2,}", " ", segment)
        segment = re.sub(r"[ \t]+\n", "\n", segment)
        segment = re.sub(r"\n[ \t]+", "\n", segment)
        return segment

    return normalize_citation_spacing(_apply_outside_math_spans(value, transform))


def contains_numeric_prose_citation(text: Any) -> bool:
    value = str(text or "")
    found = False

    def transform(segment: str) -> str:
        nonlocal found
        if _NUMERIC_PROSE_CITATION_RE.search(segment):
            found = True
        return segment

    _apply_outside_math_spans(value, transform)
    return found


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


CITATION_WRITING_REQUIREMENTS = """Citation Requirements

- Use one citation style in all natural-language prose: linked author-year Markdown, e.g. `[Smith et al., 2024](URL)`.
- Do not use numeric prose citations such as `[1]`, `[1, 2]`, or adjacent numeric brackets in generated reviewer text.
- When multiple prior papers support one claim, cite them as linked author-year citations separated by semicolons.
- Only cite papers from the provided allowed paper list.
- Do not invent authors, years, URLs, titles, or paper claims.
- Every substantive claim about prior work, feasibility evidence, experiment coverage, literature overlap, or benchmark precedent must be citation-backed."""
