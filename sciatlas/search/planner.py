from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..core.common import ensure_dir, get_env_value, load_env_values, normalize_whitespace, write_json
from ..llm.client import call_llm_json


KEYWORD_SYSTEM_PROMPT = "You extract high-level academic keywords and return strict JSON only."
KEYWORD_PROMPT = """You are an expert assistant that extracts high-level academic keywords for knowledge graph search.

Goal:
From the following research text, extract a small set of canonical, high-level keywords that represent the main research topics, tasks, methods, or application areas.

Requirements:
- Extract only 3-8 keywords.
- Prefer reusable academic concepts suitable as knowledge graph entities.
- Use concise English noun phrases, usually 1-4 words.
- Avoid system names, long descriptive fragments, marketing wording, and paper-specific phrases.
- Also score each keyword's relevance to the text on a 1-10 integer scale.

Research text:
{input_text}

Return ONLY a JSON object:
{{"keywords": ["keyword1", "keyword2"], "scores": [8, 7]}}
"""

TITLE_SYSTEM_PROMPT = "You extract explicit academic paper titles and return strict JSON only."
TITLE_PROMPT = """You are an expert assistant for academic paper retrieval.

Goal:
Extract only explicit or highly certain English paper titles mentioned in the idea text.

Requirements:
- Do not paraphrase.
- Do not guess missing titles.
- At most 5 titles.
- confidence must be a float from 0 to 1.

Idea text:
{idea_text}

Return ONLY a JSON object:
{{"titles": [{{"title": "Attention Is All You Need", "confidence": 0.96}}]}}
"""

REFERENCE_SELECTION_SYSTEM_PROMPT = """You are an expert research assistant for academic paper analysis.

You will receive:
1. The paper title.
2. The paper abstract.
3. The paper body as JSON. Each item is a section with:
- "heading": section title
- "paragraphs": paragraph strings
- "subsections": nested sections
4. A references list where each line has the format:
ref_id="b..." title="..."
5. A hard cap named "max_total_ref_ids". The total number of unique reference ids across your full output must not exceed this cap.

The paragraph text may contain inline bibliography references such as:
<ref type="bibr" target="#b31">Reference Title</ref>

Task:
1. Read the paper body and identify the key context sentences or paragraphs that discuss related works most similar to the paper's main work.
2. The goal is not to keep every important citation. The goal is to recover citations that should behave like "related works" for this paper: papers that are directly related and similar to the paper's main research work.
3. Treat a citation as a strong candidate if it has one or more of these properties:
   - it addresses the same or a very similar core problem
   - it uses a similar methodology, modeling idea, framework, or technical approach
   - it is presented as a baseline, direct comparison target, or closely competing method
   - it belongs to the same research area, task, or problem setting as the paper
4. Do not keep citations that are only loosely connected or merely supportive, such as:
   - references from unrelated fields or side topics
   - citations that do not focus on the same central problem
   - example systems, datasets, benchmarks, tools, software libraries, APIs, platforms, or generic resources
   - backbone model papers, implementation dependencies, technical manuals, or model documentation unless the paper is actually about that same model family/problem
   - incidental references used only to justify a component, preprocessing step, or borrowed utility
5. Use both:
   - the paper title and abstract, to infer the paper's main work
   - the surrounding body context, to judge whether a citation is being discussed as a genuinely similar prior work or baseline
   - the referenced paper titles from the <ref ...> tags and the references list
   to decide which citations are true related works.
6. For each selected key context sentence or paragraph, extract only the most relevant bibliography reference targets mentioned in that context.
7. For each selected context, write a concise "reason" that explains:
   - why this context is important for identifying similar related works
   - why the chosen references are directly related or similar to the paper's main work
   - if the context cites more references but you keep only part of them, explain why the omitted references are less central as related works
   - keep each reason short: at most 2 sentences, plain text only
8. Return grouped results. Each group corresponds to one key context and contains:
   - "ref_ids": the selected bibliography ids from that context
   - "reason": the explanation for that context and those references
9. Order the output groups from highest importance to lowest importance.

Rules:
- Focus on references that would be good related-work candidates for this paper.
- Preserve only ids that correspond to bibliography references, such as b31.
- Do not invent ids.
- Do not exceed max_total_ref_ids unique ids across the entire response.
- If no good references qualify, return an empty selections list.
- Output valid JSON only.

Return ONLY a JSON object in this format:
{"selections": [{"ref_ids": ["b31", "b52"], "reason": "..."}, {"ref_ids": ["b11"], "reason": "..."}]}"""

_REF_TAG_PATTERN = re.compile(r"<ref\b([^>]*)>(.*?)</ref>", re.IGNORECASE | re.DOTALL)
_TARGET_PATTERN = re.compile(r'target\s*=\s*["\']#(b\d+)["\']', re.IGNORECASE)


def _normalize_ref_id(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    if not text:
        return ""
    text = text.lstrip("#")
    return text if re.fullmatch(r"b\d+", text) else ""


def _build_keywords(input_text: str, *, env_path: Path, params: dict[str, Any], artifact_dir: Path | None) -> list[dict[str, Any]]:
    payload = call_llm_json(
        env_path=env_path,
        params=params,
        system_prompt=KEYWORD_SYSTEM_PROMPT,
        user_prompt=KEYWORD_PROMPT.format(input_text=input_text),
        artifact_path=artifact_dir / "keywords.raw.json" if artifact_dir is not None else None,
        temperature=0.1,
        max_tokens=400,
    )
    raw_keywords = payload.get("keywords")
    raw_scores = payload.get("scores")
    if not isinstance(raw_keywords, list) or not isinstance(raw_scores, list):
        raise ValueError("Keyword planner must return list fields: keywords, scores")
    keywords: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keyword, score in zip(raw_keywords, raw_scores):
        text = normalize_whitespace(keyword)
        if not text:
            continue
        try:
            score_value = int(score)
        except (TypeError, ValueError):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        keywords.append({"text": text, "score": max(1, min(10, score_value))})
    if not keywords:
        raise ValueError("Keyword planner returned no valid keywords")
    return keywords


def _build_titles(input_text: str, *, env_path: Path, params: dict[str, Any], artifact_dir: Path | None) -> list[dict[str, Any]]:
    payload = call_llm_json(
        env_path=env_path,
        params=params,
        system_prompt=TITLE_SYSTEM_PROMPT,
        user_prompt=TITLE_PROMPT.format(idea_text=input_text),
        artifact_path=artifact_dir / "titles.raw.json" if artifact_dir is not None else None,
        temperature=0.0,
        max_tokens=400,
    )
    raw_titles = payload.get("titles")
    if not isinstance(raw_titles, list):
        raise ValueError("Title planner must return a list field: titles")
    titles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_titles:
        if not isinstance(item, dict):
            continue
        title = normalize_whitespace(item.get("title"))
        if not title:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        titles.append({"title": title, "confidence": max(0.0, min(1.0, confidence))})
    return titles


def _reference_title_map(references: list[dict[str, Any]]) -> dict[str, str]:
    return {
        ref_id: title
        for ref in references
        if isinstance(ref, dict)
        for ref_id in [_normalize_ref_id(ref.get("ref_id"))]
        for title in [normalize_whitespace(ref.get("title"))]
        if ref_id and title
    }


def _preprocess_paragraph(paragraph: str, reference_title_map: dict[str, str]) -> str:
    def replace_ref(match: re.Match[str]) -> str:
        attrs = match.group(1) or ""
        target_match = _TARGET_PATTERN.search(attrs)
        if target_match is None:
            return ""
        ref_id = _normalize_ref_id(target_match.group(1))
        if not ref_id:
            return ""
        title = reference_title_map.get(ref_id)
        if not title:
            return ""
        return f'<ref type="bibr" target="#{ref_id}">{title}</ref>'

    return _REF_TAG_PATTERN.sub(replace_ref, paragraph)


def _preprocess_body_sections(body_sections: list[dict[str, Any]], reference_title_map: dict[str, str]) -> list[dict[str, Any]]:
    processed_sections: list[dict[str, Any]] = []
    for section in body_sections:
        if not isinstance(section, dict):
            continue
        paragraphs = section.get("paragraphs")
        subsections = section.get("subsections")
        processed_sections.append(
            {
                "heading": section.get("heading", ""),
                "paragraphs": [
                    _preprocess_paragraph(str(paragraph), reference_title_map)
                    for paragraph in (paragraphs if isinstance(paragraphs, list) else [])
                ],
                "subsections": _preprocess_body_sections(
                    subsections if isinstance(subsections, list) else [],
                    reference_title_map,
                ),
            }
        )
    return processed_sections


def _select_reference_titles(
    *,
    title: str,
    abstract: str,
    body_sections: list[dict[str, Any]],
    references: list[dict[str, Any]],
    env_path: Path,
    params: dict[str, Any],
    artifact_dir: Path | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    title_map = _reference_title_map(references)
    payload = call_llm_json(
        env_path=env_path,
        params=params,
        system_prompt=REFERENCE_SELECTION_SYSTEM_PROMPT,
        user_prompt=json.dumps(
            {
                "title": title,
                "abstract": abstract,
                "body": _preprocess_body_sections(body_sections, title_map),
                "references": "\n".join(
                    f'ref_id="{ref_id}" title="{ref_title}"'
                    for ref_id, ref_title in sorted(title_map.items())
                ),
                "max_total_ref_ids": max(1, int(params.get("max_titles_from_pdf_references") or 10)),
            },
            ensure_ascii=False,
        ),
        artifact_path=artifact_dir / "reference_selection.raw.json" if artifact_dir is not None else None,
        temperature=0.0,
        max_tokens=1200,
    )
    raw_selections = payload.get("selections")
    if not isinstance(raw_selections, list):
        raise ValueError("Reference selector must return a list field: selections")

    selected_titles: list[str] = []
    selection_groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_selections:
        if not isinstance(item, dict):
            continue
        raw_ref_ids = item.get("ref_ids")
        if not isinstance(raw_ref_ids, list):
            continue
        group_ref_ids: list[str] = []
        for raw_ref_id in raw_ref_ids:
            ref_id = _normalize_ref_id(raw_ref_id)
            if not ref_id or ref_id in seen or ref_id not in title_map:
                continue
            seen.add(ref_id)
            group_ref_ids.append(ref_id)
            selected_titles.append(title_map[ref_id])
        if group_ref_ids:
            selection_groups.append(
                {
                    "ref_ids": group_ref_ids,
                    "reason": normalize_whitespace(item.get("reason")),
                }
            )
    return selected_titles, selection_groups


def build_search_plan(
    *,
    text: str | None = None,
    pdf_path: str | None = None,
    params: dict[str, Any],
    env_path: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    if bool(normalize_whitespace(text)) == bool(normalize_whitespace(pdf_path)):
        raise ValueError("Exactly one of text or pdf_path is required for search planning")

    artifact_dir = ensure_dir(artifact_dir) if artifact_dir is not None else None
    if text:
        normalized_text = normalize_whitespace(text)
        keywords = _build_keywords(normalized_text, env_path=env_path, params=params, artifact_dir=artifact_dir)
        titles = _build_titles(normalized_text, env_path=env_path, params=params, artifact_dir=artifact_dir)
        plan = {
            "query_text": normalized_text,
            "source_type": "idea_text",
            "source_title": None,
            "keywords": keywords,
            "titles": titles,
            "reference_titles": [],
        }
        if artifact_dir is not None:
            write_json(artifact_dir / "plan.json", plan)
        return plan

    from ..evidence.vendor.pdf_extraction.extractor import extract_pdf

    env_values = load_env_values(env_path)
    grobid_base_url = normalize_whitespace(
        params.get("grobid_base_url")
        or get_env_value(env_values, "GROBID_BASE_URL")
        or "http://127.0.0.1:8070"
    )
    document = extract_pdf(
        pdf_path,
        base_url=grobid_base_url,
        preserve_bibr_refs=True,
    )
    extracted_pdf = document.to_dict()
    if artifact_dir is not None:
        write_json(artifact_dir / "pdf_extraction.json", extracted_pdf)

    source_title = normalize_whitespace(extracted_pdf.get("title"))
    abstract = normalize_whitespace(extracted_pdf.get("abstract"))
    if not abstract:
        raise ValueError("PDF planner requires a non-empty abstract extracted from the PDF")
    body_sections = extracted_pdf.get("body") or []
    references = extracted_pdf.get("references") or []
    reference_titles, selection_groups = _select_reference_titles(
        title=source_title,
        abstract=abstract,
        body_sections=body_sections if isinstance(body_sections, list) else [],
        references=references if isinstance(references, list) else [],
        env_path=env_path,
        params=params,
        artifact_dir=artifact_dir,
    )
    if artifact_dir is not None:
        write_json(
            artifact_dir / "reference_selection.json",
            {
                "reference_titles": reference_titles,
                "selection_groups": selection_groups,
            },
        )
    keywords = _build_keywords(abstract, env_path=env_path, params=params, artifact_dir=artifact_dir)
    titles = [{"title": title, "confidence": 1.0} for title in reference_titles]
    plan = {
        "query_text": abstract,
        "source_type": "pdf",
        "source_title": source_title,
        "keywords": keywords,
        "titles": titles,
        "reference_titles": reference_titles,
    }
    if artifact_dir is not None:
        write_json(artifact_dir / "plan.json", plan)
    return plan
