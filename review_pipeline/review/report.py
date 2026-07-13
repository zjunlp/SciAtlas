from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    CITATION_WRITING_REQUIREMENTS,
    REVIEW_SECTIONS,
    contains_numeric_prose_citation,
    first_non_empty,
    link_author_year_citation_labels,
    load_env_values,
    normalize_citation_spacing,
    normalize_whitespace,
    strip_numeric_prose_citations,
    write_json,
)
from .common import citation_author_label, citation_label as format_citation_label
from .common import citation_markdown as format_citation_markdown
from .common import citation_year as format_citation_year
from .evaluation import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_LLM_TEMPERATURE,
    JsonLLMClient,
    ReviewGenerationConfig,
    _normalize_list,
)
from .figure_cards import summarize_figure_card

REPORT_SECTION_ORDER = REVIEW_SECTIONS
REPORT_PROMPT_VERSION = 6

MARKDOWN_WRITING_REQUIREMENTS = """Markdown Writing Requirements

Write all natural-language content in polished Markdown-ready form rather than plain text.

Formatting goals:
- Make the output easy to read inside a Markdown report.
- Use clear structure and visual hierarchy.
- Prefer concise, well-organized writing over dense plain paragraphs.

Formatting rules:
- Use short section-like lead-ins or clearly separable paragraphs when helpful.
- Use bullet lists or numbered lists whenever the content is naturally list-like.
- When writing list-like content inside a JSON string field, put each item on its own line beginning with `- `; never write inline list fragments such as `: * **Label:**`.
- Use **bold** to highlight core claims, key findings, important limitations, or section labels.
- Use *italics* sparingly for emphasis on assumptions, caveats, or nuanced terms.
- When several related points are presented, format them as separate list items rather than merging them into one long sentence.
- Keep paragraphs reasonably short.
- Prefer one idea per paragraph or one claim per bullet.

Math and notation rules:
- When referring to formulas, variables, metrics, or symbolic mechanisms, use Markdown math notation.
- Use inline math like `$R = J_{sc} V_{oc} FF / P_{in}$`, `$\\eta_V$`, `$t_L$`, `$E_{HOMO}$` for short expressions.
- Use display math `$$ ... $$` only when a standalone equation materially improves readability.
- Never write formulas only in plain text if a mathematical expression would be clearer.
- Never wrap math expressions in backticks or code formatting.
- Write `$...$` or `$$...$$`, not `` `$...$` ``, `` `$$...$$` ``, or other code-styled variants.

Style constraints:
- Keep the tone professional, precise, and report-ready.
- Do not use markdown tables unless explicitly requested.
- Do not overuse headings, boldface, or italics.
- Do not add decorative symbols unless explicitly requested.
- Do not output raw plain-text blocks when Markdown structure would improve readability.

Output quality check:
- The final text should look natural and well-formatted when rendered in Markdown.
- The text should remain readable even before rendering, with clear hierarchy and separation of points."""


@dataclass(slots=True)
class ReportSynthesisConfig:
    env_path: Path | None = None
    llm_api_key: str | None = None
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    enable_llm: bool = True
    smoke: bool = False
    llm_timeout_seconds: int = 600
    max_retries: int = 5
    short_report: bool = False
    loggers: tuple[Any, ...] = ()


def _log(config: ReportSynthesisConfig, level: str, message: str) -> None:
    for logger in config.loggers:
        getattr(logger, level)(message)


def _resolve_report_config(config: ReportSynthesisConfig) -> ReportSynthesisConfig:
    env_values = load_env_values(config.env_path)
    return ReportSynthesisConfig(
        env_path=Path(config.env_path).expanduser().resolve() if config.env_path else None,
        llm_api_key=first_non_empty(
            config.llm_api_key,
            env_values.get("DMX-API-KEY"),
            env_values.get("DMX_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
        )
        or None,
        llm_base_url=first_non_empty(config.llm_base_url, env_values.get("OPENAI_BASE_URL"), DEFAULT_LLM_BASE_URL),
        llm_model_name=first_non_empty(config.llm_model_name, DEFAULT_LLM_MODEL_NAME),
        llm_temperature=float(config.llm_temperature),
        enable_llm=bool(config.enable_llm),
        smoke=bool(config.smoke),
        llm_timeout_seconds=max(30, int(config.llm_timeout_seconds)),
        max_retries=max(1, int(config.max_retries)),
        short_report=bool(config.short_report),
        loggers=tuple(config.loggers or ()),
    )


def _make_client(config: ReportSynthesisConfig) -> JsonLLMClient | None:
    if config.smoke or not config.enable_llm or not config.llm_api_key:
        return None
    review_config = ReviewGenerationConfig(
        env_path=config.env_path,
        llm_api_key=config.llm_api_key,
        llm_base_url=config.llm_base_url,
        llm_model_name=config.llm_model_name,
        llm_temperature=config.llm_temperature,
        enable_llm=config.enable_llm,
        smoke=config.smoke,
        llm_timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.max_retries,
        loggers=config.loggers,
    )
    return JsonLLMClient(review_config)


def _point_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    points: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = normalize_whitespace(item.get("point") or item.get("text"))
        else:
            text = normalize_whitespace(item)
        if text:
            points.append(text)
    return _normalize_list(points)


def _point_citations(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    citations: list[str] = []
    for item in value:
        if isinstance(item, dict):
            citations.extend(_normalize_list(item.get("citations")))
    return _normalize_list(citations)


def _citation_lookup_from_papers(papers: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_ref = normalize_whitespace(paper.get("paper_ref"))
        citation_markdown = normalize_whitespace(
            paper.get("citation_markdown")
            or format_citation_markdown(
                paper.get("authors"),
                year=paper.get("year"),
                publication_date=paper.get("publication_date"),
                url=paper.get("url"),
            )
        )
        citation_label = normalize_whitespace(
            paper.get("citation_label")
            or format_citation_label(
                paper.get("authors"),
                year=paper.get("year"),
                publication_date=paper.get("publication_date"),
            )
        )
        for alias in (paper_ref, citation_markdown, citation_label):
            if alias and citation_markdown:
                lookup.setdefault(alias, citation_markdown)
                lookup.setdefault(alias.casefold(), citation_markdown)
    return lookup


def _resolve_citation_ref(citation: str, citation_lookup: dict[str, str], paper_ref_by_markdown: dict[str, str]) -> str:
    normalized = normalize_whitespace(citation)
    citation_markdown = citation_lookup.get(normalized) or citation_lookup.get(normalized.casefold()) or normalized
    return paper_ref_by_markdown.get(citation_markdown) or paper_ref_by_markdown.get(citation_markdown.casefold()) or ""


def build_section_summary(reviewer_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_section: dict[str, dict[str, Any]] = {}
    for review in reviewer_reviews:
        for section_review in review.get("section_reviews", []):
            if not isinstance(section_review, dict):
                continue
            section = normalize_whitespace(section_review.get("section")) or "Unknown"
            bucket = by_section.setdefault(section, {"strengths": [], "weaknesses": []})
            bucket["strengths"].extend(_point_texts(section_review.get("strengths")))
            bucket["weaknesses"].extend(_point_texts(section_review.get("weaknesses")))

    summary: list[dict[str, Any]] = []
    for section in REPORT_SECTION_ORDER:
        bucket = by_section.get(section, {"strengths": [], "weaknesses": []})
        summary.append(
            {
                "section": section,
                "main_strengths": _normalize_list(bucket["strengths"])[:5],
                "main_weaknesses": _normalize_list(bucket["weaknesses"])[:5],
            }
        )
    return summary


def build_dimension_summary(reviewer_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_section_summary(reviewer_reviews)


def _collect_revision_advice(
    reviewer_reviews: list[dict[str, Any]],
    evidence_bank: dict[str, Any],
) -> list[str]:
    advice: list[str] = []
    for review in reviewer_reviews:
        overall = review.get("overall") if isinstance(review.get("overall"), dict) else {}
        advice.extend(_normalize_list(overall.get("suggestions")))
    for card in evidence_bank.get("experiment_cards", [])[:12]:
        if not isinstance(card, dict):
            continue
        missing = card.get("missing_or_undercovered")
        if isinstance(missing, list):
            for item in missing:
                text = normalize_whitespace(item)
                if text:
                    advice.append(f"Address experiment coverage gap: {text}")
    return _normalize_list(advice)[:12]


def build_paper_references(
    reviewer_reviews: list[dict[str, Any]],
    evidence_bank: dict[str, Any],
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    raw_citations: list[str] = []
    seen_raw: set[str] = set()
    point_texts: list[str] = []
    for review in reviewer_reviews:
        for section_review in review.get("section_reviews", []):
            for point_group in ("strengths", "weaknesses"):
                for point in section_review.get(point_group, []):
                    if isinstance(point, dict):
                        point_text = normalize_whitespace(point.get("point") or point.get("text"))
                        if point_text:
                            point_texts.append(point_text)
                for citation in _point_citations(section_review.get(point_group)):
                    if citation not in seen_raw:
                        seen_raw.add(citation)
                        raw_citations.append(citation)
            for dimension in section_review.get("dimension_reviews", []):
                if isinstance(dimension, dict):
                    for citation in _normalize_list(dimension.get("citations")):
                        if citation not in seen_raw:
                            seen_raw.add(citation)
                            raw_citations.append(citation)

    papers = evidence_bank.get("searched_papers", {}).get("papers", [])
    paper_lookup = {
        normalize_whitespace(paper.get("paper_ref")): paper
        for paper in papers
        if isinstance(paper, dict) and normalize_whitespace(paper.get("paper_ref"))
    }
    citation_lookup = _citation_lookup_from_papers(papers)
    paper_ref_by_markdown = {
        citation_markdown: paper_ref
        for paper_ref, paper in paper_lookup.items()
        for citation_markdown in [
            normalize_whitespace(
                paper.get("citation_markdown")
                or format_citation_markdown(
                    paper.get("authors"),
                    year=paper.get("year"),
                    publication_date=paper.get("publication_date"),
                    url=paper.get("url"),
                )
            )
        ]
        if citation_markdown
    }
    paper_ref_by_markdown.update({key.casefold(): value for key, value in list(paper_ref_by_markdown.items())})

    cited_refs: list[str] = []
    seen_refs: set[str] = set()
    for citation in raw_citations:
        resolved_ref = _resolve_citation_ref(citation, citation_lookup, paper_ref_by_markdown)
        if resolved_ref and resolved_ref not in seen_refs:
            seen_refs.add(resolved_ref)
            cited_refs.append(resolved_ref)
    for paper_ref, paper in paper_lookup.items():
        if paper_ref in seen_refs:
            continue
        citation_candidates = [
            normalize_whitespace(paper.get("citation_markdown")),
            normalize_whitespace(paper.get("citation_label")),
            format_citation_markdown(
                paper.get("authors"),
                year=paper.get("year"),
                publication_date=paper.get("publication_date"),
                url=paper.get("url"),
            ),
            format_citation_label(
                paper.get("authors"),
                year=paper.get("year"),
                publication_date=paper.get("publication_date"),
            ),
        ]
        citation_candidates = [candidate for candidate in citation_candidates if candidate]
        if any(candidate in text for text in point_texts for candidate in citation_candidates):
            seen_refs.add(paper_ref)
            cited_refs.append(paper_ref)
    references: list[dict[str, Any]] = []
    for citation in cited_refs[:limit]:
        paper = paper_lookup.get(citation)
        if not paper:
            continue
        references.append(
            {
                "paper_ref": citation,
                "paper_title": normalize_whitespace(paper.get("title")) or None,
                "year": paper.get("year"),
                "publication_date": paper.get("publication_date"),
                "venue": paper.get("venue"),
                "authors": paper.get("authors"),
                "citation_label": paper.get("citation_label")
                or format_citation_label(
                    paper.get("authors"),
                    year=paper.get("year"),
                    publication_date=paper.get("publication_date"),
                ),
                "citation_markdown": paper.get("citation_markdown")
                or format_citation_markdown(
                    paper.get("authors"),
                    year=paper.get("year"),
                    publication_date=paper.get("publication_date"),
                    url=paper.get("url"),
                ),
                "url": paper.get("url"),
                "snippet": normalize_whitespace(paper.get("abstract"))[:700],
            }
        )
    return references


def _meta_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "reviewer_consensus": {"type": "string"},
            "reviewer_disagreements": {"type": "string"},
            "revision_advice": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "confidence",
            "reason",
            "strengths",
            "weaknesses",
            "reviewer_consensus",
            "reviewer_disagreements",
            "revision_advice",
        ],
    }


def _short_meta_schema() -> dict[str, Any]:
    dimension_schema = {
        "type": "object",
        "properties": {
            "standard_id": {"type": "string"},
            "dimension_name": {"type": "string"},
            "review_text": {"type": "string"},
        },
        "required": [
            "standard_id",
            "dimension_name",
            "review_text",
        ],
    }
    return {
        "type": "object",
        "properties": {
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "overall_assessment": {"type": "string"},
            "reviewer_consensus": {"type": "string"},
            "reviewer_disagreements": {"type": "string"},
            "shared_strengths": {"type": "array", "items": {"type": "string"}},
            "shared_weaknesses": {"type": "array", "items": {"type": "string"}},
            "rubric_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string"},
                        "dimensions": {"type": "array", "items": dimension_schema},
                    },
                    "required": ["section", "dimensions"],
                },
            },
            "revision_advice": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "confidence",
            "overall_assessment",
            "reviewer_consensus",
            "reviewer_disagreements",
            "shared_strengths",
            "shared_weaknesses",
            "rubric_summaries",
            "revision_advice",
        ],
    }


def _short_idea_overview_schema() -> dict[str, Any]:
    section_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["label", "text"],
    }
    return {
        "type": "object",
        "properties": {
            "sections": {"type": "array", "items": section_schema},
        },
        "required": ["sections"],
    }


def _format_reviewer_summaries(reviewer_reviews: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, review in enumerate(reviewer_reviews, start=1):
        overall = review.get("overall") if isinstance(review.get("overall"), dict) else {}
        section_lines = []
        for section in review.get("section_reviews", []):
            if not isinstance(section, dict):
                continue
            section_lines.append(f"- {section.get('section')}")
        blocks.append(
            "\n".join(
                [
                    f"Reviewer {index}",
                    f"Confidence: {overall.get('confidence')}",
                    f"Summary: {_clean_report_markdown_text(overall.get('summary'))}",
                    "Reviewed sections:",
                    "\n".join(section_lines),
                    f"Strengths: {json.dumps([_clean_report_markdown_text(item) for item in _normalize_list(overall.get('strengths'))], ensure_ascii=False)}",
                    f"Weaknesses: {json.dumps([_clean_report_markdown_text(item) for item in _normalize_list(overall.get('weaknesses'))], ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_review_evidence_by_rubric(reviewer_reviews: list[dict[str, Any]]) -> str:
    sections: list[dict[str, Any]] = []
    for reviewer_index, review in enumerate(reviewer_reviews, start=1):
        for section_review in review.get("section_reviews", []):
            if not isinstance(section_review, dict):
                continue
            section = normalize_whitespace(section_review.get("section")) or "Unknown"
            if isinstance(section_review.get("strengths"), list) or isinstance(section_review.get("weaknesses"), list):
                sections.append(
                    {
                        "reviewer": f"Reviewer {reviewer_index}",
                        "section": section,
                        "strengths": [
                            {
                                "point": _clean_report_markdown_text(point.get("point")),
                                "supporting_standard_ids": _normalize_list(point.get("supporting_standard_ids")),
                                "citations": [
                                    citation
                                    for citation in _normalize_list(point.get("citations"))
                                    if not contains_numeric_prose_citation(citation)
                                ],
                            }
                            for point in section_review.get("strengths", [])
                            if isinstance(point, dict) and normalize_whitespace(point.get("point"))
                        ],
                        "weaknesses": [
                            {
                                "point": _clean_report_markdown_text(point.get("point")),
                                "supporting_standard_ids": _normalize_list(point.get("supporting_standard_ids")),
                                "citations": [
                                    citation
                                    for citation in _normalize_list(point.get("citations"))
                                    if not contains_numeric_prose_citation(citation)
                                ],
                            }
                            for point in section_review.get("weaknesses", [])
                            if isinstance(point, dict) and normalize_whitespace(point.get("point"))
                        ],
                    }
                )
                continue
            dimensions = section_review.get("dimension_reviews", [])
            if not isinstance(dimensions, list):
                continue
            for dimension in dimensions:
                if not isinstance(dimension, dict):
                    continue
                sections.append(
                    {
                        "reviewer": f"Reviewer {reviewer_index}",
                        "section": section,
                        "standard_id": normalize_whitespace(dimension.get("standard_id")),
                        "dimension_name": normalize_whitespace(dimension.get("dimension_name")),
                        "assessment": _clean_report_markdown_text(dimension.get("assessment")),
                        "strengths": [_clean_report_markdown_text(item) for item in _normalize_list(dimension.get("strengths"))[:3]],
                        "weaknesses": [_clean_report_markdown_text(item) for item in _normalize_list(dimension.get("weaknesses"))[:3]],
                        "citations": [
                            citation
                            for citation in _normalize_list(dimension.get("citations"))
                            if not contains_numeric_prose_citation(citation)
                        ],
                    }
                )
    return json.dumps(sections, ensure_ascii=False, indent=2)


def _format_rubric_outline(normalized_rubric: dict[str, Any]) -> str:
    sections: list[dict[str, Any]] = []
    for section in normalized_rubric.get("sections", []):
        if not isinstance(section, dict):
            continue
        standards: list[dict[str, Any]] = []
        for standard in section.get("standards", []):
            if not isinstance(standard, dict):
                continue
            standards.append(
                {
                    "standard_id": normalize_whitespace(standard.get("standard_id")),
                    "dimension_name": normalize_whitespace(standard.get("dimension_name")),
                    "core_philosophy": normalize_whitespace(standard.get("core_philosophy")),
                    "required_evidence": normalize_whitespace(standard.get("required_evidence")),
                }
            )
        sections.append({"section": normalize_whitespace(section.get("section")), "standards": standards})
    return json.dumps(sections, ensure_ascii=False, indent=2)


def _paper_references_for_prompt(paper_references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_refs: list[dict[str, Any]] = []
    for paper in paper_references:
        if not isinstance(paper, dict):
            continue
        citation = normalize_whitespace(
            paper.get("citation_markdown")
            or format_citation_markdown(
                paper.get("authors"),
                year=paper.get("year"),
                publication_date=paper.get("publication_date"),
                url=paper.get("url"),
            )
        )
        if not citation:
            continue
        prompt_refs.append(
            {
                "internal_paper_ref_do_not_output": normalize_whitespace(paper.get("paper_ref")) or None,
                "citation_markdown": citation,
                "title": normalize_whitespace(paper.get("paper_title") or paper.get("title")) or "Unknown",
                "year": paper.get("year"),
                "venue": normalize_whitespace(paper.get("venue")) or None,
                "snippet": normalize_whitespace(paper.get("snippet"))[:500],
            }
        )
    return prompt_refs


def _figure_reference_label(card: dict[str, Any], index: int | None = None) -> str:
    raw_label = normalize_whitespace(card.get("figure_label")) or normalize_whitespace(card.get("file_name"))
    match = re.search(r"fig(?:ure)?\.?\s*([0-9]+(?:\([a-z]\)|[a-z])?)", raw_label, flags=re.IGNORECASE)
    if match:
        return f"Fig. {match.group(1)}"
    return ""


def _figure_cards_for_prompt(evidence_bank: dict[str, Any], *, limit: int = 12) -> list[dict[str, Any]]:
    cards = [
        card
        for card in (evidence_bank.get("figure_cards") if isinstance(evidence_bank.get("figure_cards"), list) else [])
        if isinstance(card, dict)
    ]
    prompt_cards: list[dict[str, Any]] = []
    for index, card in enumerate(cards[:limit], start=1):
        recommended_citation = _figure_reference_label(card, index)
        prompt_cards.append(
            {
                "recommended_figure_citation": recommended_citation or None,
                "figure_label": normalize_whitespace(card.get("figure_label")) or None,
                "paper_sections": _normalize_list(card.get("paper_sections")),
                "section_path_text": normalize_whitespace(card.get("section_path_text")),
                "related_claim": normalize_whitespace(card.get("related_claim")),
                "figure_grounded_summary": normalize_whitespace(card.get("evidence_text")),
                "figure_support_rationale": normalize_whitespace(
                    (card.get("paper_alignment") or {}).get("support_rationale")
                    if isinstance(card.get("paper_alignment"), dict)
                    else ""
                ),
                "supports_claim_well": bool(
                    (card.get("paper_alignment") or {}).get("supports_claim_well")
                    if isinstance(card.get("paper_alignment"), dict)
                    else False
                ),
                "reviewer_caution": normalize_whitespace(
                    (card.get("reviewer_caution") or {}).get("overclaim_risk")
                    if isinstance(card.get("reviewer_caution"), dict)
                    else ""
                ),
            }
        )
    return prompt_cards


def _build_meta_prompt(
    *,
    idea_text: str,
    reviewer_reviews: list[dict[str, Any]],
    dimension_summary: list[dict[str, Any]],
    paper_references: list[dict[str, Any]],
    revision_advice: list[str],
) -> str:
    prompt_paper_references = _paper_references_for_prompt(paper_references)
    return f"""
You are an ICLR-style Senior Meta-Reviewer / Area Chair.
Synthesize the reviewer reports into a final confidence judgment for a preliminary research idea. Be critical, domain-aware, precise, and evidence-grounded.

Rules:
- If evidence is missing or reviewers disagree strongly, lower confidence.
- Use only the provided reviewer reviews and paper reference table.
- Keep reviewers anonymous. Refer to them only as Reviewer 1, Reviewer 2, etc. Do not mention names, masked names, reviewer IDs, or identity hints.
- Keep the final reason professional, concise, and logically ordered.
- Summarize the strongest cross-reviewer positives in strengths and the main cross-reviewer concerns in weaknesses. Do not simply count reviewers; weigh the specificity and evidential force of their arguments.
- Do not generate numeric scores, ratings, recommendation labels, or score-like bins.

{CITATION_WRITING_REQUIREMENTS}

Reasoning requirements:
- Write the `reason` as a compact evidence chain: overall judgment, decisive strengths, decisive weaknesses, why those weaknesses matter, and the revision priority implied by them.
- In `reviewer_consensus`, identify the section-level issues where reviewers converge and explain why they are central to the final assessment.
- In `reviewer_disagreements`, distinguish disagreements about severity, scope, evidence interpretation, and reviewer emphasis. If only one reviewer report is available, explicitly say that cross-reviewer disagreement cannot be assessed.
- In `revision_advice`, propose concrete actions that directly address the strongest weaknesses; prioritize actions that would change the evidential status of the work.
- When mentioning prior work, benchmark precedent, feasibility evidence, or experiment coverage from the paper reference table, cite it using the citation requirements.
- Use only the `citation_markdown` strings from the paper reference table for citations. Do not output `internal_paper_ref_do_not_output` values or any numeric bracket citations.
- If reviewer evidence contains numeric manuscript-local references such as `[12]`, `[26]`, or `[53]`, do not copy them into the meta-review. Refer to the cited prior work descriptively unless it appears in the paper reference table with a `citation_markdown` value.
- Avoid vague phrases such as "promising", "interesting", or "needs improvement" unless the sentence names the concrete mechanism, dataset, baseline, metric, reproducibility condition, or literature overlap that justifies the claim.

{MARKDOWN_WRITING_REQUIREMENTS}

Additional rule for structured outputs:
- Return the required JSON structure exactly.
- Apply the Markdown writing requirements inside natural-language string fields only.
- Keep all string fields report-ready and directly readable when inserted into a Markdown report.

Research idea:
{idea_text}

Reviewer reviews:
{_format_reviewer_summaries(reviewer_reviews)}

Section-level feedback themes:
{json.dumps(dimension_summary, ensure_ascii=False, indent=2)}

Paper reference table:
{json.dumps(prompt_paper_references, ensure_ascii=False, indent=2)}

Candidate revision advice:
{json.dumps(revision_advice, ensure_ascii=False, indent=2)}
"""


def _build_short_meta_prompt(
    *,
    idea_text: str,
    normalized_rubric: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    dimension_summary: list[dict[str, Any]],
    paper_references: list[dict[str, Any]],
    revision_advice: list[str],
    evidence_bank: dict[str, Any],
) -> str:
    prompt_paper_references = _paper_references_for_prompt(paper_references)
    prompt_figure_cards = _figure_cards_for_prompt(evidence_bank)
    return f"""
You are an ICLR-style Senior Meta-Reviewer / Area Chair.
Write a concise but complete short meta-review for a preliminary research idea. The tone should read like a strong human-written review for ICLR or Nature-family venues: compact, analytical, and naturally phrased rather than mechanically templated.

This short report replaces the individual reviewer reports. It must therefore synthesize reviewer opinions at the rubric-dimension level.

Rules:
- Use only the provided reviewer reviews, rubric outline, paper reference table, and figure-grounded manuscript evidence.
- Keep reviewers anonymous. Refer to them only as Reviewer 1, Reviewer 2, etc. Do not mention names, masked names, reviewer IDs, or identity hints.
- Do not generate numeric scores, ratings, recommendation labels, or score-like bins.
- For rubric_summaries, preserve the rubric section order and rubric dimension names from the rubric outline.
- For each rubric dimension, write one compact review paragraph in `review_text`. Do not split it into labeled subfields such as strengths, weaknesses, or reviewer positions.
- Each `review_text` should sound like a human review comment: begin with the integrated judgment, then fold in consensus, disagreements in emphasis or severity, and the implication for the paper.
- Keep the writing fluent and report-ready. The output should read like a normal short evaluation report, not like a raw data dump.

{CITATION_WRITING_REQUIREMENTS}

Reasoning requirements:
- For each rubric dimension, make `review_text` follow this logic: integrated judgment, the most decisive supporting evidence from the paper, the main limitation or caveat, reviewer disagreement or difference in emphasis when it matters, and the implication for the final judgment.
- Preserve technical specificity from the reviewer evidence: algorithms, model components, objectives, datasets, splits, annotation protocols, baselines, metrics, ablations, statistical tests, compute/runtime constraints, reproducibility conditions, and failure modes when they are relevant.
- Do not merely restate individual reviewer comments. Explain how the combined evidence affects novelty, feasibility, experimental credibility, impact, or reproducibility.
- `shared_strengths`, `shared_weaknesses`, and `revision_advice` must align with the rubric-level integrated assessments.
- When mentioning prior work, benchmark precedent, feasibility evidence, or experiment coverage from the paper reference table, cite it using the citation requirements.
- Use only the `citation_markdown` strings from the paper reference table for citations. Do not output `internal_paper_ref_do_not_output` values or any numeric bracket citations.
- If reviewer evidence contains numeric manuscript-local references such as `[12]`, `[26]`, or `[53]`, do not copy them into the short meta-review. Refer to the cited prior work descriptively unless it appears in the paper reference table with a `citation_markdown` value.
- The figure-grounded manuscript evidence comes from the target paper's own figures. Use it to assess whether the visual evidence supports, weakens, or complicates the textual claim.
- If you mention a figure, cite it inline using only one of the provided `recommended_figure_citation` strings or the exact normalized form implied by the provided `figure_label`.
- Do not infer, reconstruct, or invent new manuscript figure numbers from context.
- Use figure references selectively. Cite a figure only when it materially supports a judgment about consistency between the manuscript text and the visual evidence, the logic of the experimental claim, or a mismatch/overclaim risk.
- Avoid vague phrases such as "promising", "interesting", or "needs improvement" unless the sentence names the concrete mechanism, dataset, baseline, metric, reproducibility condition, or literature overlap that justifies the claim.

{MARKDOWN_WRITING_REQUIREMENTS}

Additional rule for structured outputs:
- Return the required JSON structure exactly.
- Apply the Markdown writing requirements inside natural-language string fields only.
- Keep all string fields directly readable when inserted into a Markdown report.

Research idea:
{idea_text}

Rubric outline:
{_format_rubric_outline(normalized_rubric)}

Reviewer evidence organized by rubric:
{_format_review_evidence_by_rubric(reviewer_reviews)}

Section-level feedback themes:
{json.dumps(dimension_summary, ensure_ascii=False, indent=2)}

Paper reference table:
{json.dumps(prompt_paper_references, ensure_ascii=False, indent=2)}

Figure-grounded manuscript evidence:
{json.dumps(prompt_figure_cards, ensure_ascii=False, indent=2)}

Candidate revision advice:
{json.dumps(revision_advice, ensure_ascii=False, indent=2)}
"""


def _build_short_idea_overview_prompt(
    *,
    idea_text: str,
    normalized_rubric: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    evidence_bank: dict[str, Any],
) -> str:
    breakdown = normalized_rubric.get("idea_breakdown") if isinstance(normalized_rubric.get("idea_breakdown"), dict) else {}
    prompt_figure_cards = _figure_cards_for_prompt(evidence_bank)
    structured_idea = evidence_bank.get("structured_idea") if isinstance(evidence_bank.get("structured_idea"), dict) else {}
    return f"""
You are writing the `Idea Overview` section of a short academic review report.
Your task is to turn the structured idea notes into a compact, coherent overview that reads like a human-written review preamble, not like extracted bullets.

Rules:
- Use the whole idea and the whole review context. Do not summarize each subsection independently in isolation.
- Preserve the overview section structure. Use these section labels exactly when content is available: `Basic Idea`, `Motivation`, `Method`, `Experimental Focus`.
- Produce one main paragraph per section so that the final sequence reads coherently from problem to method to evidence.
- By default, write continuous prose. Small bullet lists are allowed only when the content is naturally enumerative, such as 2-4 distinct challenges, evidence strands, or contributions. Do not produce long bullet-heavy sections; never use 5 or more bullets in one section.
- The overview should describe what the paper is trying to do, why it matters, how it works, and what the main empirical evidence appears to be.
- Do not invent claims beyond the provided material.
- Use the original structured idea as the content skeleton, but rewrite each section with global awareness so that the final overview reads as one coherent whole.
- You may mention a figure only when it materially clarifies the main empirical setup or a core result trend. If you do, cite it using only the provided `recommended_figure_citation` or the exact normalized form implied by the provided `figure_label`.
- Do not infer, reconstruct, or invent new manuscript figure numbers from context.
- Keep the tone neutral, concise, and report-like.

{MARKDOWN_WRITING_REQUIREMENTS}

Return the required JSON structure exactly.

Structured idea:
{idea_text}

Idea breakdown:
{json.dumps(breakdown, ensure_ascii=False, indent=2)}

Structured idea by section:
{json.dumps(structured_idea, ensure_ascii=False, indent=2)}

Reviewer summaries:
{_format_reviewer_summaries(reviewer_reviews)}

Figure-grounded manuscript evidence:
{json.dumps(prompt_figure_cards, ensure_ascii=False, indent=2)}
"""


def _sanitize_meta(raw: dict[str, Any]) -> dict[str, Any]:
    confidence = normalize_whitespace(raw.get("confidence")).casefold()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "confidence": confidence,
        "reason": _clean_report_markdown_text(raw.get("reason")),
        "strengths": [_clean_report_markdown_text(item) for item in _normalize_list(raw.get("strengths"))[:8]],
        "weaknesses": [_clean_report_markdown_text(item) for item in _normalize_list(raw.get("weaknesses"))[:8]],
        "reviewer_consensus": _clean_report_markdown_text(raw.get("reviewer_consensus")),
        "reviewer_disagreements": _clean_report_markdown_text(raw.get("reviewer_disagreements")),
        "revision_advice": [_clean_report_markdown_text(item) for item in _normalize_list(raw.get("revision_advice"))[:12]],
    }


def _rewrite_meta_citations(meta: dict[str, Any], paper_references: list[dict[str, Any]]) -> dict[str, Any]:
    paper_lookup = _paper_reference_lookup(paper_references)
    rewritten = dict(meta)
    for key in ("reason", "reviewer_consensus", "reviewer_disagreements"):
        rewritten[key] = _clean_report_markdown_text(rewritten.get(key), paper_lookup)
    for key in ("strengths", "weaknesses", "revision_advice", "shared_strengths", "shared_weaknesses"):
        if isinstance(rewritten.get(key), list):
            rewritten[key] = [
                _clean_report_markdown_text(item, paper_lookup)
                for item in _normalize_list(rewritten.get(key))
            ]
    return rewritten


def _allowed_figure_citations(evidence_bank: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for card in (
        evidence_bank.get("figure_cards")
        if isinstance(evidence_bank.get("figure_cards"), list)
        else []
    ):
        if not isinstance(card, dict):
            continue
        citation = _figure_reference_label(card)
        if citation:
            allowed.add(citation)
    return allowed


def _strip_unknown_figure_citations(text: Any, allowed: set[str]) -> str:
    value = str(text or "")
    if not value:
        return ""

    def replace(match: re.Match[str]) -> str:
        citation = match.group(0)
        normalized = re.sub(r"\s+", " ", citation).strip()
        return normalized if normalized in allowed else "the corresponding figure"

    return re.sub(r"Fig\.\s*[0-9]+(?:\([a-z]\)|[a-z])?", replace, value)


def _sanitize_figure_references_in_short_meta(short_meta_review: dict[str, Any], evidence_bank: dict[str, Any]) -> dict[str, Any]:
    allowed = _allowed_figure_citations(evidence_bank)
    if not allowed:
        return short_meta_review
    sanitized = dict(short_meta_review)
    for key in ("overall_assessment", "reviewer_consensus", "reviewer_disagreements"):
        sanitized[key] = _strip_unknown_figure_citations(sanitized.get(key), allowed)
    for key in ("shared_strengths", "shared_weaknesses", "revision_advice"):
        if isinstance(sanitized.get(key), list):
            sanitized[key] = [_strip_unknown_figure_citations(item, allowed) for item in sanitized.get(key)]
    if isinstance(sanitized.get("rubric_summaries"), list):
        rewritten_sections: list[dict[str, Any]] = []
        for section in sanitized["rubric_summaries"]:
            if not isinstance(section, dict):
                continue
            new_section = dict(section)
            if isinstance(new_section.get("dimensions"), list):
                new_dims: list[dict[str, Any]] = []
                for dim in new_section["dimensions"]:
                    if not isinstance(dim, dict):
                        continue
                    new_dim = dict(dim)
                    new_dim["review_text"] = _strip_unknown_figure_citations(new_dim.get("review_text"), allowed)
                    new_dims.append(new_dim)
                new_section["dimensions"] = new_dims
            rewritten_sections.append(new_section)
        sanitized["rubric_summaries"] = rewritten_sections
    return sanitized


def _sanitize_figure_references_in_idea_overview(sections: list[dict[str, str]], evidence_bank: dict[str, Any]) -> list[dict[str, str]]:
    allowed = _allowed_figure_citations(evidence_bank)
    if not allowed:
        return sections
    sanitized: list[dict[str, str]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        sanitized.append(
            {
                "label": normalize_whitespace(section.get("label")),
                "text": _strip_unknown_figure_citations(section.get("text"), allowed),
            }
        )
    return sanitized


def _rubric_standard_lookup(normalized_rubric: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for section in normalized_rubric.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_name = normalize_whitespace(section.get("section"))
        for standard in section.get("standards", []):
            if not isinstance(standard, dict):
                continue
            standard_id = normalize_whitespace(standard.get("standard_id"))
            dimension_name = normalize_whitespace(standard.get("dimension_name"))
            if standard_id:
                lookup[(section_name.casefold(), standard_id.casefold())] = standard
            if dimension_name:
                lookup[(section_name.casefold(), dimension_name.casefold())] = standard
    return lookup


def _expected_rubric_dimensions(normalized_rubric: dict[str, Any]) -> list[dict[str, str]]:
    expected: list[dict[str, str]] = []
    for section in normalized_rubric.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_name = normalize_whitespace(section.get("section"))
        for standard in section.get("standards", []):
            if not isinstance(standard, dict):
                continue
            expected.append(
                {
                    "section": section_name,
                    "standard_id": normalize_whitespace(standard.get("standard_id")),
                    "dimension_name": normalize_whitespace(standard.get("dimension_name")),
                }
            )
    return expected


def _fallback_dimension_summary(
    expected_dimension: dict[str, str],
    reviewer_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    section_name = expected_dimension["section"]
    standard_id = expected_dimension["standard_id"]
    dimension_name = expected_dimension["dimension_name"]
    assessments: list[str] = []
    for reviewer_index, review in enumerate(reviewer_reviews, start=1):
        for section_review in review.get("section_reviews", []):
            if not isinstance(section_review, dict):
                continue
            if normalize_whitespace(section_review.get("section")).casefold() != section_name.casefold():
                continue
            for point_group in ("strengths", "weaknesses"):
                for point in section_review.get(point_group, []):
                    if not isinstance(point, dict):
                        continue
                    supporting_ids = {
                        normalize_whitespace(item).casefold()
                        for item in _normalize_list(point.get("supporting_standard_ids"))
                    }
                    if standard_id and standard_id.casefold() not in supporting_ids:
                        continue
                    text = normalize_whitespace(point.get("point"))
                    if not text:
                        continue
                    prefix = "supports" if point_group == "strengths" else "raises concern about"
                    assessments.append(f"Reviewer {reviewer_index} {prefix} {_truncate_text(text, max_chars=360)}")
            for dimension in section_review.get("dimension_reviews", []):
                if not isinstance(dimension, dict):
                    continue
                same_id = standard_id and normalize_whitespace(dimension.get("standard_id")).casefold() == standard_id.casefold()
                same_name = (
                    dimension_name
                    and normalize_whitespace(dimension.get("dimension_name")).casefold() == dimension_name.casefold()
                )
                if not same_id and not same_name:
                    continue
                assessment = normalize_whitespace(dimension.get("assessment"))
                if assessment:
                    assessments.append(f"Reviewer {reviewer_index}: {_truncate_text(assessment, max_chars=360)}")
    return {
        "standard_id": standard_id,
        "dimension_name": dimension_name,
        "review_text": _truncate_text(" ".join(assessments), max_chars=1100)
        or "No reviewer supplied a detailed assessment for this rubric dimension.",
    }


def _sanitize_short_meta(
    raw: dict[str, Any],
    *,
    normalized_rubric: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    revision_advice: list[str],
) -> dict[str, Any]:
    confidence = normalize_whitespace(raw.get("confidence")).casefold()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    by_section: dict[str, list[dict[str, Any]]] = {}
    for section in raw.get("rubric_summaries", []):
        if not isinstance(section, dict):
            continue
        section_name = normalize_whitespace(section.get("section")) or "Unknown"
        for dimension in section.get("dimensions", []):
            if not isinstance(dimension, dict):
                continue
            sanitized_dimension = {
                "standard_id": normalize_whitespace(dimension.get("standard_id")),
                "dimension_name": normalize_whitespace(dimension.get("dimension_name")) or "Unnamed Rubric Dimension",
                "review_text": _clean_report_markdown_block(dimension.get("review_text")),
            }
            by_section.setdefault(section_name.casefold(), []).append(sanitized_dimension)

    rubric_summaries: list[dict[str, Any]] = []
    for expected in _expected_rubric_dimensions(normalized_rubric):
        section_name = expected["section"]
        candidates = by_section.get(section_name.casefold(), [])
        match_index = -1
        for index, candidate in enumerate(candidates):
            candidate_id = normalize_whitespace(candidate.get("standard_id"))
            candidate_name = normalize_whitespace(candidate.get("dimension_name"))
            if expected["standard_id"] and candidate_id.casefold() == expected["standard_id"].casefold():
                match_index = index
                break
            if expected["dimension_name"] and candidate_name.casefold() == expected["dimension_name"].casefold():
                match_index = index
                break
        if match_index >= 0:
            dimension = candidates.pop(match_index)
            dimension["standard_id"] = expected["standard_id"] or dimension["standard_id"]
            dimension["dimension_name"] = expected["dimension_name"] or dimension["dimension_name"]
        else:
            dimension = _fallback_dimension_summary(expected, reviewer_reviews)
        if not rubric_summaries or rubric_summaries[-1]["section"] != section_name:
            rubric_summaries.append({"section": section_name, "dimensions": []})
        rubric_summaries[-1]["dimensions"].append(dimension)

    return {
        "confidence": confidence,
        "overall_assessment": _clean_report_markdown_text(raw.get("overall_assessment")),
        "reviewer_consensus": _clean_report_markdown_text(raw.get("reviewer_consensus")),
        "reviewer_disagreements": _clean_report_markdown_text(raw.get("reviewer_disagreements")),
        "shared_strengths": [
            _clean_report_markdown_text(item)
            for item in _normalize_list(raw.get("shared_strengths"))[:8]
        ],
        "shared_weaknesses": [
            _clean_report_markdown_text(item)
            for item in _normalize_list(raw.get("shared_weaknesses"))[:8]
        ],
        "rubric_summaries": rubric_summaries,
        "revision_advice": [
            _clean_report_markdown_text(item)
            for item in _normalize_list(raw.get("revision_advice"), fallback=revision_advice)[:12]
        ],
    }


def _sanitize_short_idea_overview(raw: dict[str, Any]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for item in raw.get("sections", []):
        if not isinstance(item, dict):
            continue
        label = normalize_whitespace(item.get("label"))
        text = _clean_report_markdown_block(item.get("text"))
        if label and text:
            sections.append({"label": label, "text": text})
    return sections


def generate_meta_review(
    *,
    idea_text: str,
    reviewer_reviews: list[dict[str, Any]],
    dimension_summary: list[dict[str, Any]],
    paper_references: list[dict[str, Any]],
    revision_advice: list[str],
    config: ReportSynthesisConfig,
    client: JsonLLMClient | None,
) -> dict[str, Any]:
    if client is None:
        strengths: list[str] = []
        weaknesses: list[str] = []
        for item in dimension_summary:
            strengths.extend(_normalize_list(item.get("main_strengths"))[:2])
            weaknesses.extend(_normalize_list(item.get("main_weaknesses"))[:2])
        return _rewrite_meta_citations({
            "confidence": "medium" if reviewer_reviews else "low",
            "reason": (
                f"Deterministic meta-review fallback based on {len(reviewer_reviews)} reviewer reports "
                f"and {len(dimension_summary)} section summaries."
            ),
            "strengths": _normalize_list(strengths)[:8],
            "weaknesses": _normalize_list(weaknesses)[:8],
            "reviewer_consensus": "Consensus is estimated from repeated feedback themes across reviewer reports.",
            "reviewer_disagreements": (
                "Only one reviewer report was available, so cross-reviewer disagreement cannot be assessed."
                if len(reviewer_reviews) <= 1
                else "Disagreement analysis is limited because the LLM meta-reviewer is disabled."
            ),
            "revision_advice": _normalize_list(revision_advice)[:12],
        }, paper_references)
    raw = client.generate_json(
        system_prompt="You are a strict academic meta-reviewer. Return only JSON.",
        user_prompt=_build_meta_prompt(
            idea_text=idea_text,
            reviewer_reviews=reviewer_reviews,
            dimension_summary=dimension_summary,
            paper_references=paper_references,
            revision_advice=revision_advice,
        ),
        schema=_meta_schema(),
    )
    meta = _sanitize_meta(raw)
    if len(reviewer_reviews) <= 1:
        meta["reviewer_disagreements"] = (
            "Only one reviewer report was available, so cross-reviewer disagreement cannot be assessed."
        )
    return _rewrite_meta_citations(meta, paper_references)


def generate_short_meta_review(
    *,
    idea_text: str,
    normalized_rubric: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    dimension_summary: list[dict[str, Any]],
    paper_references: list[dict[str, Any]],
    revision_advice: list[str],
    evidence_bank: dict[str, Any],
    config: ReportSynthesisConfig,
    client: JsonLLMClient | None,
) -> dict[str, Any]:
    if client is None:
        base_meta = generate_meta_review(
            idea_text=idea_text,
            reviewer_reviews=reviewer_reviews,
            dimension_summary=dimension_summary,
            paper_references=paper_references,
            revision_advice=revision_advice,
            config=config,
            client=None,
        )
        raw = {
            "confidence": base_meta["confidence"],
            "overall_assessment": base_meta["reason"],
            "reviewer_consensus": base_meta["reviewer_consensus"],
            "reviewer_disagreements": base_meta["reviewer_disagreements"],
            "shared_strengths": base_meta["strengths"],
            "shared_weaknesses": base_meta["weaknesses"],
            "rubric_summaries": [],
            "revision_advice": base_meta["revision_advice"],
        }
        return _sanitize_short_meta(
            raw,
            normalized_rubric=normalized_rubric,
            reviewer_reviews=reviewer_reviews,
            revision_advice=revision_advice,
        )

    raw = client.generate_json(
        system_prompt="You are a strict academic meta-reviewer. Return only JSON.",
        user_prompt=_build_short_meta_prompt(
            idea_text=idea_text,
            normalized_rubric=normalized_rubric,
            reviewer_reviews=reviewer_reviews,
            dimension_summary=dimension_summary,
            paper_references=paper_references,
            revision_advice=revision_advice,
            evidence_bank=evidence_bank,
        ),
        schema=_short_meta_schema(),
    )
    return _sanitize_figure_references_in_short_meta(_sanitize_short_meta(
        raw,
        normalized_rubric=normalized_rubric,
        reviewer_reviews=reviewer_reviews,
        revision_advice=revision_advice,
    ), evidence_bank)


def generate_short_idea_overview(
    *,
    idea_text: str,
    normalized_rubric: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    evidence_bank: dict[str, Any],
    client: JsonLLMClient | None,
) -> list[dict[str, str]]:
    fallback_sections = _split_idea_sections(idea_text)
    if client is None:
        if fallback_sections:
            return [
                {
                    "label": label,
                    "text": " ".join(re.sub(r"^-\s*", "", normalize_whitespace(item)) for item in items if normalize_whitespace(item)),
                }
                for label, items in fallback_sections
                if any(normalize_whitespace(item) for item in items)
            ]
        breakdown = normalized_rubric.get("idea_breakdown") if isinstance(normalized_rubric.get("idea_breakdown"), dict) else {}
        ordered = [
            ("Basic Idea", normalize_whitespace(normalized_rubric.get("idea_summary"))),
            ("Motivation", normalize_whitespace(breakdown.get("motivation_and_problem"))),
            ("Method", normalize_whitespace(breakdown.get("proposed_method"))),
            ("Experimental Focus", normalize_whitespace(breakdown.get("experiment_and_data"))),
        ]
        return [{"label": label, "text": text} for label, text in ordered if text]

    raw = client.generate_json(
        system_prompt="You are a concise academic reviewer. Return only JSON.",
        user_prompt=_build_short_idea_overview_prompt(
            idea_text=idea_text,
            normalized_rubric=normalized_rubric,
            reviewer_reviews=reviewer_reviews,
            evidence_bank=evidence_bank,
        ),
        schema=_short_idea_overview_schema(),
    )
    sanitized = _sanitize_figure_references_in_idea_overview(_sanitize_short_idea_overview(raw), evidence_bank)
    return sanitized or generate_short_idea_overview(
        idea_text=idea_text,
        normalized_rubric=normalized_rubric,
        reviewer_reviews=reviewer_reviews,
        evidence_bank=evidence_bank,
        client=None,
    )


def _format_rubric(normalized_rubric: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in normalized_rubric.get("sections", []):
        if not isinstance(section, dict):
            continue
        lines.append(f"### {section.get('section')}")
        standards = section.get("standards", []) if isinstance(section.get("standards"), list) else []
        if not standards:
            lines.append("- No section-specific standards were generated.")
            continue
        for standard in standards:
            if not isinstance(standard, dict):
                continue
            core_philosophy = _clean_report_markdown_text(standard.get("core_philosophy"))
            required_evidence = _clean_report_markdown_text(standard.get("required_evidence"))
            dimension_name = normalize_whitespace(standard.get("dimension_name")) or "Unnamed Rubric Dimension"
            lines.append(f"- **{dimension_name}**")
            if core_philosophy:
                lines.append(f"  - **Evaluation focus**: {core_philosophy}")
            if required_evidence:
                lines.append(f"  - **Required evidence**: {required_evidence}")
    return "\n".join(lines) if lines else "No rubric standards available."


def _format_review_board(normalized_reviewers: dict[str, Any]) -> str:
    lines: list[str] = []
    for index, reviewer in enumerate(normalized_reviewers.get("reviewers", []), start=1):
        if not isinstance(reviewer, dict):
            continue
        profile = normalize_whitespace(reviewer.get("academic_background"))
        lines.append(f"- **Reviewer {index}**: {profile or 'No profile available.'}")
    return "\n".join(lines) if lines else "No reviewer profiles available."


def _mask_name(value: Any) -> str:
    text = normalize_whitespace(value)
    if not text:
        return "Anonymous"
    parts = []
    for part in text.split():
        if len(part) <= 2:
            parts.append(part[0] + "x" if part else "")
        else:
            parts.append(part[0] + ("x" * (len(part) - 2)) + part[-1])
    return " ".join(parts)


def _format_grounding_stats(evidence_bank: dict[str, Any]) -> str:
    stats = evidence_bank.get("stats") if isinstance(evidence_bank.get("stats"), dict) else {}
    evidence_count = stats.get("evidence_card_count", len(evidence_bank.get("evidence_cards", [])))
    experiment_count = stats.get("experiment_card_count", len(evidence_bank.get("experiment_cards", [])))
    figure_count = stats.get("figure_card_count", len(evidence_bank.get("figure_cards", [])))
    grounding_status = stats.get("grounding_status") or evidence_bank.get("status") or "unknown"
    return "\n".join(
        [
            "**Grounding Stats**",
            f"- Evidence links: {evidence_count}",
            f"- Experiment links: {experiment_count}",
            f"- Figure cards: {figure_count}",
            f"- Grounding status: {grounding_status}",
        ]
    )


def _truncate_text(value: Any, *, max_chars: int) -> str:
    text = normalize_whitespace(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _grounding_score(card: dict[str, Any]) -> float:
    for key in ("rerank_score", "dense_score", "coverage_score"):
        try:
            if card.get(key) is not None:
                return float(card[key])
        except (TypeError, ValueError):
            continue
    return 0.0


GROUNDING_SECTION_ORDER = (
    "basic_idea",
    "motivation",
    "method",
    "experimental_focus",
    "experiment",
    "evaluation",
    "discussion",
)


def _coverage_rank(card: dict[str, Any]) -> int:
    label = normalize_whitespace(card.get("coverage_label")).casefold()
    return {
        "high": 4,
        "partial": 3,
        "well_covered": 3,
        "limited": 2,
        "undercovered": 2,
        "none": 1,
    }.get(label, 0)


def _section_display_name(section: str) -> str:
    normalized = normalize_whitespace(section).replace("_", " ")
    return normalized.title() if normalized else "Grounding"


def _select_grounding_highlights(evidence_bank: dict[str, Any]) -> list[dict[str, Any]]:
    cards = [card for card in evidence_bank.get("evidence_cards", []) if isinstance(card, dict)]
    cards = [
        card
        for card in cards
        if normalize_whitespace(card.get("idea_section"))
        and normalize_whitespace(card.get("idea_sentence"))
        and normalize_whitespace(card.get("evidence_text"))
    ]

    by_section: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        section = normalize_whitespace(card.get("idea_section")).casefold()
        by_section.setdefault(section, []).append(card)

    ordered_sections = list(GROUNDING_SECTION_ORDER)
    ordered_sections.extend(section for section in by_section if section not in set(ordered_sections))

    selected: list[dict[str, Any]] = []
    for section in ordered_sections:
        section_cards = by_section.get(section, [])
        if not section_cards:
            continue
        preferred = [card for card in section_cards if _coverage_rank(card) >= 3]
        candidates = preferred or section_cards
        candidates.sort(
            key=lambda card: (
                _coverage_rank(card),
                bool(card.get("shared_points")),
                _grounding_score(card),
                len(normalize_whitespace(card.get("evidence_text"))),
            ),
            reverse=True,
        )
        selected.append(candidates[0])
    return selected


def _format_grounding_points(value: Any) -> str:
    points = [_clean_report_markdown_text(item) for item in _normalize_list(value)]
    points = [point for point in points if point]
    return "\n".join(f"- {point}" for point in points) if points else "- None identified."


def _format_blockquote(value: Any, *, max_chars: int) -> str:
    text = _truncate_text(_clean_report_markdown_text(value), max_chars=max_chars)
    if not text:
        return "> No recalled passage available."
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _select_experiment_grounding(evidence_bank: dict[str, Any]) -> dict[str, Any] | None:
    cards = [card for card in evidence_bank.get("experiment_cards", []) if isinstance(card, dict)]
    cards = [
        card
        for card in cards
        if normalize_whitespace(card.get("recommended_goal"))
        and normalize_whitespace(card.get("coverage_rationale"))
    ]
    if not cards:
        return None
    cards.sort(
        key=lambda card: (
            _coverage_rank(card),
            bool(_normalize_list(card.get("overlap"))),
            _grounding_score(card),
            len(normalize_whitespace(card.get("coverage_rationale"))),
        ),
        reverse=True,
    )
    return cards[0]


def _format_grounding_list(value: Any, *, limit: int = 6) -> str:
    points = [_clean_report_markdown_text(item) for item in _normalize_list(value)]
    points = [point for point in points if point]
    if not points:
        return "- None identified."
    return "\n".join(f"- {point}" for point in points[:limit])


def _format_experiment_grounding(card: dict[str, Any] | None) -> str:
    if not card:
        return ""
    paper_title = normalize_whitespace(card.get("paper_title")) or "Unknown paper"
    lines = [
        "",
        f"### Experiment · {paper_title}",
        f"**Recommended experimental goal**: {_truncate_text(_clean_report_markdown_text(card.get('recommended_goal')), max_chars=320)}",
        "",
        "**Paper-grounded rationale**",
        _truncate_text(_clean_report_markdown_text(card.get("goal_rationale")), max_chars=520) or "None identified.",
        "",
        "**Coverage rationale**",
        _truncate_text(_clean_report_markdown_text(card.get("coverage_rationale")), max_chars=520) or "None identified.",
        "",
        "**Overlaps with the idea**",
        _format_grounding_list(card.get("overlap")),
        "",
        "**Missing or undercovered**",
        _format_grounding_list(card.get("missing_or_undercovered")),
        "",
        "**Additional focus in the idea**",
        _format_grounding_list(card.get("additional_focus_in_idea")),
    ]
    return "\n".join(lines)


def _format_grounding_evidence(evidence_bank: dict[str, Any]) -> str:
    highlights = _select_grounding_highlights(evidence_bank)
    experiment = _select_experiment_grounding(evidence_bank)
    if not highlights and not experiment:
        return "No grounding evidence was available.\n\n" + _format_grounding_stats(evidence_bank)

    lines: list[str] = [_format_grounding_stats(evidence_bank)]
    for card in highlights:
        section = _section_display_name(normalize_whitespace(card.get("idea_section")))
        paper_title = normalize_whitespace(card.get("paper_title")) or "Unknown paper"
        lines.extend(
            [
                "",
                f"### {section} · {paper_title}",
                f"**Idea claim**: {_truncate_text(_clean_report_markdown_text(card.get('idea_sentence')), max_chars=320)}",
                "",
                "**Recalled evidence**:",
                _format_blockquote(card.get("evidence_text"), max_chars=520),
                "",
                "**Shared points**",
                _format_grounding_points(card.get("shared_points")),
                "",
                "**Different points**",
                _format_grounding_points(card.get("different_points")),
            ]
        )
    experiment_block = _format_experiment_grounding(experiment)
    if experiment_block:
        lines.append(experiment_block)
    return "\n".join(lines)


def _format_publication_date(value: Any, fallback_year: Any = None) -> str:
    text = normalize_whitespace(value)
    if text:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return text
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return f"{text}-01"
        if re.fullmatch(r"\d{4}", text):
            return text
    year_text = normalize_whitespace(str(fallback_year)) if fallback_year is not None else ""
    if re.fullmatch(r"\d{4}", year_text):
        return year_text
    return "n.d."


def _format_reference_entries(papers: list[dict[str, Any]]) -> str:
    normalized_papers: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = normalize_whitespace(paper.get("paper_title") or paper.get("title")) or "Unknown"
        url = normalize_whitespace(paper.get("url") or paper.get("doi"))
        key = (url or title).casefold()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized_papers.append(paper)

    def sort_key(paper: dict[str, Any]) -> tuple[str, str]:
        authors = [normalize_whitespace(author) for author in _normalize_list(paper.get("authors"))]
        label = citation_author_label(authors)
        title = normalize_whitespace(paper.get("paper_title") or paper.get("title"))
        return (label.casefold(), title.casefold())

    lines: list[str] = []
    for paper in sorted(normalized_papers, key=sort_key):
        title = normalize_whitespace(paper.get("paper_title") or paper.get("title")) or "Unknown"
        venue = normalize_whitespace(paper.get("venue"))
        url = normalize_whitespace(paper.get("url") or paper.get("doi"))
        authors = [normalize_whitespace(author) for author in _normalize_list(paper.get("authors"))]
        authors = [author for author in authors if author]
        author_label = citation_author_label(authors)
        year = format_citation_year(paper.get("publication_date"), paper.get("year"))
        title_text = f"[{title}]({url})" if url else title
        author_separator = " " if author_label.endswith(".") else ". "
        suffix = " ".join(part for part in [f"{year}.", f"{venue}." if venue else ""] if part)
        lines.append(f"{author_label}{author_separator}{title_text}. {suffix}".rstrip())
    return "\n\n".join(lines) if lines else "No referenced literature available."


def _format_searched_papers(evidence_bank: dict[str, Any], paper_references: list[dict[str, Any]]) -> str:
    if paper_references:
        return _format_reference_entries(paper_references)
    return "No referenced literature available."


def _paper_reference_lookup(papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        normalize_whitespace(paper.get("paper_ref")): paper
        for paper in papers
        if isinstance(paper, dict) and normalize_whitespace(paper.get("paper_ref"))
    }


def _citation_markdown_for_ref(ref: str, paper_lookup: dict[str, dict[str, Any]]) -> str:
    paper = paper_lookup.get(ref)
    if not paper:
        return ref
    return normalize_whitespace(
        paper.get("citation_markdown")
        or format_citation_markdown(
            paper.get("authors"),
            year=paper.get("year"),
            publication_date=paper.get("publication_date"),
            url=paper.get("url"),
        )
    ) or ref


def _normalize_report_citations(text: Any, paper_lookup: dict[str, dict[str, Any]] | None = None) -> str:
    return strip_numeric_prose_citations(normalize_whitespace(text))


def _strip_math_backticks(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"`\s*(\$\$.*?\$\$)\s*`", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"`\s*(\$[^`\n]+\$)\s*`", r"\1", value)
    return value


def _clean_report_markdown_text(text: Any, paper_lookup: dict[str, dict[str, Any]] | None = None) -> str:
    citation_lookup = _citation_lookup_from_papers(list((paper_lookup or {}).values())) if paper_lookup else {}
    cleaned = _normalize_report_citations(_strip_math_backticks(text), paper_lookup)
    return link_author_year_citation_labels(cleaned, citation_lookup)


def _normalize_markdown_block_whitespace(text: Any) -> str:
    value = str(text or "")
    if "\n" not in value:
        return normalize_whitespace(value)
    lines: list[str] = []
    previous_blank = False
    for raw_line in value.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            if lines and not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def _normalize_inline_markdown_lists(text: str) -> str:
    value = text
    value = re.sub(r"(?m)^\*\s+(?=\*\*[^*\n]{1,160}:\*\*)", "- ", value)
    value = re.sub(r"\s+\*\s+(?=\*\*[^*\n]{1,160}:\*\*)", "\n- ", value)
    for marker in ("These points of consensus", "These points of disagreement", "In essence,"):
        marker_index = value.rfind(f" {marker}")
        last_bullet_index = value.rfind("\n- ")
        if last_bullet_index != -1 and marker_index > last_bullet_index:
            value = value[:marker_index].rstrip() + "\n\n" + value[marker_index + 1 :]
    return value.strip()


def _clean_report_markdown_block(text: Any, paper_lookup: dict[str, dict[str, Any]] | None = None) -> str:
    citation_lookup = _citation_lookup_from_papers(list((paper_lookup or {}).values())) if paper_lookup else {}
    cleaned = strip_numeric_prose_citations(_strip_math_backticks(text))
    cleaned = _normalize_markdown_block_whitespace(cleaned)
    cleaned = link_author_year_citation_labels(cleaned, citation_lookup)
    return _normalize_inline_markdown_lists(cleaned)


def _format_dimension_notes(value: Any, paper_lookup: dict[str, dict[str, Any]] | None = None) -> str:
    notes = [_clean_report_markdown_text(item, paper_lookup) for item in _normalize_list(value)]
    notes = [note for note in notes if note]
    return "\n".join(f"- {note}" for note in notes) if notes else "- None stated."


def _format_review_points(value: Any, paper_lookup: dict[str, dict[str, Any]] | None = None) -> str:
    notes: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = normalize_whitespace(item.get("point") or item.get("text"))
                citations = _normalize_list(item.get("citations"))
                if citations and paper_lookup:
                    citation_lookup = _citation_lookup_from_papers(list(paper_lookup.values()))
                    linked_text = link_author_year_citation_labels(text, citation_lookup)
                    cited_items = [
                        _citation_markdown_for_ref(citation, paper_lookup)
                        for citation in citations
                    ]
                    missing_citations = [
                        citation
                        for citation in _normalize_list(cited_items)
                        if citation and citation not in linked_text
                    ]
                    text = (
                        f"{linked_text} {'; '.join(missing_citations)}"
                        if missing_citations
                        else linked_text
                    )
            else:
                text = normalize_whitespace(item)
            cleaned = _clean_report_markdown_text(text, paper_lookup)
            if cleaned:
                notes.append(cleaned)
    return "\n".join(f"- {note}" for note in _normalize_list(notes)) if notes else "- None stated."


def _fallback_section_assessment(
    section: dict[str, Any],
    paper_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    strengths = _normalize_list(
        [
            item.get("point") or item.get("text")
            if isinstance(item, dict)
            else item
            for item in (section.get("strengths") if isinstance(section.get("strengths"), list) else [])
        ]
    )
    weaknesses = _normalize_list(
        [
            item.get("point") or item.get("text")
            if isinstance(item, dict)
            else item
            for item in (section.get("weaknesses") if isinstance(section.get("weaknesses"), list) else [])
        ]
    )
    if strengths and weaknesses:
        return _clean_report_markdown_text(
            f"Overall, this section has a clear positive signal: {strengths[0]} The main limiting concern is: {weaknesses[0]}",
            paper_lookup,
        )
    if strengths:
        return _clean_report_markdown_text(
            f"Overall, this section is primarily supported by the following positive signal: {strengths[0]}",
            paper_lookup,
        )
    if weaknesses:
        return _clean_report_markdown_text(
            f"Overall, this section is primarily limited by the following concern: {weaknesses[0]}",
            paper_lookup,
        )
    return "No section-level assessment was provided."


def _format_reviewer_reports(
    reviewer_reviews: list[dict[str, Any]],
    paper_references: list[dict[str, Any]] | None = None,
) -> str:
    paper_lookup = _paper_reference_lookup(paper_references or [])
    blocks: list[str] = []
    for index, review in enumerate(reviewer_reviews, start=1):
        overall = review.get("overall") if isinstance(review.get("overall"), dict) else {}
        parts = [
            f"### Reviewer {index}",
            f"**Confidence**: {overall.get('confidence')}",
            "",
            _clean_report_markdown_text(overall.get("summary"), paper_lookup),
        ]
        for section in review.get("section_reviews", []):
            if not isinstance(section, dict):
                continue
            parts.append(f"#### {section.get('section')}")
            assessment = _clean_report_markdown_block(section.get("assessment"), paper_lookup)
            parts.append(assessment or _fallback_section_assessment(section, paper_lookup))
            parts.append("")
            parts.append("**Strengths**")
            parts.append(_format_review_points(section.get("strengths"), paper_lookup))
            parts.append("")
            parts.append("**Weaknesses**")
            parts.append(_format_review_points(section.get("weaknesses"), paper_lookup))
            parts.append("")
        blocks.append("\n".join(part for part in parts if part is not None))
    return "\n\n".join(blocks)


def _format_bulleted_notes(value: Any, paper_references: list[dict[str, Any]] | None = None) -> str:
    paper_lookup = _paper_reference_lookup(paper_references or [])
    notes = [_clean_report_markdown_text(item, paper_lookup) for item in _normalize_list(value)]
    notes = [re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", note).strip() for note in notes if note]
    return "\n".join(f"- {note}" for note in notes) if notes else "- None stated."


SECTION_INTROS = {
    "idea": "This report begins with a concise overview of the idea so the central claim, motivation, and proposed approach are clear before the reviews are read.",
    "meta_review": "This section presents the overall judgment distilled from the full panel, highlighting where the reviewers converge, where they differ, and which strengths and weaknesses most shape the evaluation.",
    "grounding_evidence": "This section surfaces a small set of high-signal grounding matches so the link between the idea and recalled literature evidence is visible without expanding the full evidence bank.",
    "review_board": "The review board is summarized here to show the expertise and research backgrounds that shape the emphasis of the individual assessments below.",
    "reviewer_reports": "The detailed reviewer reports follow below. Each report is organized by evaluation section so that the reasoning behind the overall judgment remains traceable.",
    "rubric": "The rubric below records the evaluation dimensions used in the review process and makes explicit the standards against which the idea was assessed.",
    "searched_papers": "This section lists the literature records retained for the review context, with publication metadata shown in a uniform format for easier reference.",
}


def _section_intro(key: str) -> str:
    return SECTION_INTROS.get(key, "")


def _format_section_header(title: str, intro_key: str) -> str:
    intro = _section_intro(intro_key)
    return f"## {title}\n> {intro}\n" if intro else f"## {title}\n"


def _format_table_of_contents() -> str:
    return "\n".join(
        [
            "## Table of Contents",
            "- [Idea Overview](#idea-overview)",
            "- [Meta Review](#meta-review)",
            "- [Grounding Evidence](#grounding-evidence)",
            "- [Review Board](#review-board)",
            "- [Reviewer Reports](#reviewer-reports)",
            "- [Evaluation Rubric](#evaluation-rubric)",
            "- [References](#references)",
        ]
    )


def _split_idea_sections(idea_text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_label: str | None = None
    current_items: list[str] = []
    allowed_labels = {
        "basic idea": "Basic Idea",
        "motivation": "Motivation",
        "method": "Method",
        "experimental focus": "Experimental Focus",
        "evaluation": "Evaluation",
        "discussion": "Discussion",
    }
    for raw_line in str(idea_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(Basic idea|Motivation|Method|Experimental focus|Evaluation|Discussion)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            if current_label and current_items:
                sections.append((current_label, current_items))
            current_label = allowed_labels.get(match.group(1).strip().casefold(), match.group(1).strip().title())
            body = normalize_whitespace(match.group(2))
            current_items = []
            if body:
                current_items.append(body)
            continue
        bullet = re.match(r"^-\s+(.*)$", line)
        if bullet and current_label is not None:
            text = normalize_whitespace(bullet.group(1))
            if text:
                current_items.append(text)
            continue
        if current_label is None:
            current_label = "Overview"
            current_items = [normalize_whitespace(line)]
            continue
        current_items.append(normalize_whitespace(line))
    if current_label and current_items:
        sections.append((current_label, current_items))
    return sections


def _format_idea_block(idea_text: str, normalized_rubric: dict[str, Any]) -> str:
    sections = _split_idea_sections(idea_text)
    if sections:
        lines: list[str] = []
        for label, items in sections:
            lines.append(f"**{label}**")
            for item in items:
                cleaned = re.sub(r"^-\s*", "", normalize_whitespace(item))
                if cleaned:
                    lines.append(f"- {_strip_math_backticks(cleaned)}")
            lines.append("")
        if lines:
            lines.pop()
        return "\n".join(lines)

    breakdown = normalized_rubric.get("idea_breakdown") if isinstance(normalized_rubric.get("idea_breakdown"), dict) else {}
    breakdown_mapping = [
        ("Problem Motivation", breakdown.get("motivation_and_problem")),
        ("Proposed Method", breakdown.get("proposed_method")),
        ("Experiment And Data", breakdown.get("experiment_and_data")),
    ]
    breakdown_lines: list[str] = []
    for label, value in breakdown_mapping:
        text = normalize_whitespace(value)
        if not text or text.casefold() == "not mentioned":
            continue
        breakdown_lines.append(f"**{label}**")
        breakdown_lines.append(f"- {text}")
        breakdown_lines.append("")
    if breakdown_lines:
        breakdown_lines.pop()
        return "\n".join(breakdown_lines)
    return _strip_math_backticks(idea_text)


def _strip_numeric_prose_citations_preserving_line_prefix(markdown_report: str) -> str:
    lines: list[str] = []
    for raw_line in str(markdown_report or "").splitlines():
        prefix_match = re.match(r"^[ \t]*", raw_line)
        prefix = prefix_match.group(0) if prefix_match else ""
        stripped = strip_numeric_prose_citations(raw_line)
        if prefix and stripped:
            lines.append(prefix + stripped.lstrip())
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _finalize_markdown_report(markdown_report: str) -> tuple[str, list[str]]:
    cleaned = _strip_numeric_prose_citations_preserving_line_prefix(markdown_report).strip()
    warnings: list[str] = []
    if contains_numeric_prose_citation(cleaned):
        warnings.append("numeric_prose_citation_remaining")
    return cleaned, warnings


def build_markdown_report(
    *,
    idea_text: str,
    normalized_rubric: dict[str, Any],
    normalized_reviewers: dict[str, Any],
    evidence_bank: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    meta_review: dict[str, Any],
    revision_advice: list[str],
) -> str:
    paper_references = build_paper_references(reviewer_reviews, evidence_bank)
    paper_lookup = _paper_reference_lookup(paper_references)
    markdown, _ = _finalize_markdown_report(f"""# InnoEval Review Report

{_format_table_of_contents()}

{_format_section_header("Idea Overview", "idea")}
{_format_idea_block(idea_text, normalized_rubric)}

{_format_section_header("Meta Review", "meta_review")}
**Overall Assessment**
{_clean_report_markdown_block(meta_review.get('reason'), paper_lookup)}

**Reviewer Consensus**
{_clean_report_markdown_block(meta_review.get('reviewer_consensus'), paper_lookup)}

**Reviewer Disagreements**
{_clean_report_markdown_block(meta_review.get('reviewer_disagreements'), paper_lookup)}

**Shared Strengths**
{_format_bulleted_notes(meta_review.get('strengths'), paper_references)}

**Shared Weaknesses**
{_format_bulleted_notes(meta_review.get('weaknesses'), paper_references)}

{_format_section_header("Grounding Evidence", "grounding_evidence")}
{_format_grounding_evidence(evidence_bank)}

{_format_section_header("Review Board", "review_board")}
{_format_review_board(normalized_reviewers)}

{_format_section_header("Reviewer Reports", "reviewer_reports")}
{_format_reviewer_reports(reviewer_reviews, paper_references)}

{_format_section_header("Evaluation Rubric", "rubric")}
{_format_rubric(normalized_rubric)}

{_format_section_header("References", "searched_papers")}
{_format_searched_papers(evidence_bank, paper_references)}
""")
    return markdown


def _format_rubric_definition(standard: dict[str, Any] | None) -> str:
    if not standard:
        return "**Rubric definition**\n- No rubric definition was available."
    core_philosophy = _clean_report_markdown_text(standard.get("core_philosophy"))
    required_evidence = _clean_report_markdown_text(standard.get("required_evidence"))
    lines = ["**Rubric definition**"]
    if core_philosophy:
        lines.append(f"- **Evaluation focus**: {core_philosophy}")
    if required_evidence:
        lines.append(f"- **Expected evidence**: {required_evidence}")
    return "\n".join(lines) if len(lines) > 1 else "**Rubric definition**\n- No rubric definition was available."


def _format_short_dimension(
    *,
    section_name: str,
    dimension: dict[str, Any],
    paper_references: list[dict[str, Any]] | None = None,
) -> str:
    paper_lookup = _paper_reference_lookup(paper_references or [])
    dimension_name = normalize_whitespace(dimension.get("dimension_name")) or "Unnamed Rubric Dimension"
    review_text = _clean_report_markdown_block(dimension.get("review_text"), paper_lookup) or "No integrated assessment available."
    review_text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", review_text).strip()
    review_text = review_text.replace("\n\n", " ").replace("\n", " ")
    return f"- **{dimension_name}:** {review_text}"


def _format_short_detailed_evaluation(
    *,
    normalized_rubric: dict[str, Any],
    short_meta_review: dict[str, Any],
    paper_references: list[dict[str, Any]] | None = None,
) -> str:
    sections: list[str] = []
    for section in short_meta_review.get("rubric_summaries", []):
        if not isinstance(section, dict):
            continue
        section_name = normalize_whitespace(section.get("section")) or "Unknown"
        parts = [f"### {section_name}"]
        for dimension in section.get("dimensions", []):
            if not isinstance(dimension, dict):
                continue
            parts.append(
                _format_short_dimension(
                    section_name=section_name,
                    dimension=dimension,
                    paper_references=paper_references,
                )
            )
        sections.append("\n".join(parts))
    return "\n\n".join(sections) if sections else "No rubric-level synthesis was available."


def _format_short_idea_overview(sections: list[dict[str, str]]) -> str:
    if not sections:
        return "No concise idea overview was available."
    blocks: list[str] = []
    for section in sections:
        label = normalize_whitespace(section.get("label"))
        text = _clean_report_markdown_block(section.get("text"))
        if not label or not text:
            continue
        blocks.extend([f"**{label}**", text, ""])
    if blocks:
        blocks.pop()
    return "\n".join(blocks) if blocks else "No concise idea overview was available."


def build_short_markdown_report(
    *,
    short_idea_overview: list[dict[str, str]],
    normalized_reviewers: dict[str, Any],
    evidence_bank: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    short_meta_review: dict[str, Any],
) -> str:
    paper_references = build_paper_references(reviewer_reviews, evidence_bank)
    paper_lookup = _paper_reference_lookup(paper_references)
    markdown, _ = _finalize_markdown_report(f"""# InnoEval Short Meta Review

This short report distills the review process into a single meta-review. It preserves the rubric-level reasoning that supports the judgment, but omits the full individual reviewer reports to keep the document focused and readable.

{_format_section_header("Idea Overview", "idea")}
{_format_short_idea_overview(short_idea_overview)}

## Meta Review
This section summarizes the panel-level assessment, highlighting where reviewers converge, where their emphasis differs, and which strengths and weaknesses most affect the final judgment.

**Overall Assessment**
{_clean_report_markdown_block(short_meta_review.get('overall_assessment'), paper_lookup)}

**Reviewer Consensus**
{_clean_report_markdown_block(short_meta_review.get('reviewer_consensus'), paper_lookup)}

**Reviewer Disagreements**
{_clean_report_markdown_block(short_meta_review.get('reviewer_disagreements'), paper_lookup)}

**Shared Strengths**
{_format_bulleted_notes(short_meta_review.get('shared_strengths'), paper_references)}

**Shared Weaknesses**
{_format_bulleted_notes(short_meta_review.get('shared_weaknesses'), paper_references)}

## Section Review
This section consolidates the rubric-level judgments into compact reviewer-style prose under the four major evaluation sections.

{_format_short_detailed_evaluation(normalized_rubric={}, short_meta_review=short_meta_review, paper_references=paper_references)}

## Revision Advice
The following actions summarize the most important changes that would strengthen the work in response to the meta-review.

{_format_bulleted_notes(short_meta_review.get('revision_advice'), paper_references)}

## Review Board
The review board is summarized here to make the expertise behind the assessments transparent without exposing reviewer identities.

{_format_review_board(normalized_reviewers)}

## References
The retained literature records are listed here for context and traceability.

{_format_searched_papers(evidence_bank, paper_references)}
""")
    return markdown


def build_md_tree(markdown_report: str) -> dict[str, Any]:
    root = {"title": "InnoEval Review Report", "level": 1, "content": "", "children": []}
    current_h2: dict[str, Any] | None = None
    current_h3: dict[str, Any] | None = None
    for raw_line in markdown_report.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            root["title"] = line[2:].strip()
            continue
        if line.startswith("## "):
            current_h2 = {"title": line[3:].strip(), "level": 2, "content": "", "children": []}
            root["children"].append(current_h2)
            current_h3 = None
            continue
        if line.startswith("### ") and current_h2 is not None:
            current_h3 = {"title": line[4:].strip(), "level": 3, "content": "", "children": []}
            current_h2["children"].append(current_h3)
            continue
        if not line:
            continue
        target = current_h3 or current_h2 or root
        target["content"] = (target.get("content", "") + "\n" + raw_line).strip()
    return root


def synthesize_final_report(
    *,
    idea_text: str,
    normalized_rubric: dict[str, Any],
    normalized_reviewers: dict[str, Any],
    evidence_bank: dict[str, Any],
    reviewer_reviews: list[dict[str, Any]],
    output_dir: Path,
    artifact_paths: dict[str, Any],
    config: ReportSynthesisConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _resolve_report_config(config)
    if not reviewer_reviews:
        error_payload = {
            "status": "skipped",
            "reason": "No successful reviewer reviews are available for report synthesis.",
            "artifact_paths": artifact_paths,
        }
        write_json(output_dir / "final_report.json", error_payload)
        (output_dir / "final_report.md").write_text(
            "# InnoEval Review Report\n\nNo successful reviewer reviews are available.\n",
            encoding="utf-8",
        )
        write_json(output_dir / "md_tree.json", build_md_tree("# InnoEval Review Report\n"))
        return {
            **error_payload,
            "final_report_path": str((output_dir / "final_report.md").resolve()),
            "final_report_json_path": str((output_dir / "final_report.json").resolve()),
            "md_tree_path": str((output_dir / "md_tree.json").resolve()),
        }

    searched_papers_path = normalize_whitespace(artifact_paths.get("searched_papers_path"))
    if searched_papers_path:
        try:
            searched_papers_payload = json.loads(Path(searched_papers_path).read_text(encoding="utf-8"))
            if isinstance(searched_papers_payload, dict):
                evidence_bank = {**evidence_bank, "searched_papers": searched_papers_payload}
        except Exception:
            pass

    client = _make_client(resolved_config)
    started_at = time.perf_counter()
    section_summary = build_section_summary(reviewer_reviews)
    dimension_summary = section_summary
    revision_advice = _collect_revision_advice(reviewer_reviews, evidence_bank)
    paper_references = build_paper_references(reviewer_reviews, evidence_bank)
    _log(
        resolved_config,
        "info",
        (
            f"Starting report synthesis reviewer_count={len(reviewer_reviews)} "
            f"section_summary_count={len(section_summary)} paper_reference_count={len(paper_references)} "
            f"revision_advice_count={len(revision_advice)} llm_enabled={client is not None} "
            f"short_report={resolved_config.short_report}"
        ),
    )
    if resolved_config.short_report:
        short_idea_overview = generate_short_idea_overview(
            idea_text=idea_text,
            normalized_rubric=normalized_rubric,
            reviewer_reviews=reviewer_reviews,
            evidence_bank=evidence_bank,
            client=client,
        )
        meta_review = generate_short_meta_review(
            idea_text=idea_text,
            normalized_rubric=normalized_rubric,
            reviewer_reviews=reviewer_reviews,
            dimension_summary=section_summary,
            paper_references=paper_references,
            revision_advice=revision_advice,
            evidence_bank=evidence_bank,
            config=resolved_config,
            client=client,
        )
        revision_advice = _normalize_list(meta_review.get("revision_advice"))
        markdown_report = build_short_markdown_report(
            short_idea_overview=short_idea_overview,
            normalized_reviewers=normalized_reviewers,
            evidence_bank=evidence_bank,
            reviewer_reviews=reviewer_reviews,
            short_meta_review=meta_review,
        )
    else:
        meta_review = generate_meta_review(
            idea_text=idea_text,
            reviewer_reviews=reviewer_reviews,
            dimension_summary=section_summary,
            paper_references=paper_references,
            revision_advice=revision_advice,
            config=resolved_config,
            client=client,
        )
        revision_advice = _normalize_list(meta_review.get("revision_advice"))
        markdown_report = build_markdown_report(
            idea_text=idea_text,
            normalized_rubric=normalized_rubric,
            normalized_reviewers=normalized_reviewers,
            evidence_bank=evidence_bank,
            reviewer_reviews=reviewer_reviews,
            meta_review=meta_review,
            revision_advice=revision_advice,
        )
    md_tree = build_md_tree(markdown_report)

    final_report_path = output_dir / "final_report.md"
    final_report_json_path = output_dir / "final_report.json"
    md_tree_path = output_dir / "md_tree.json"

    final_payload = {
        "status": "ok",
        "title": "InnoEval Short Meta Review" if resolved_config.short_report else "InnoEval Review Report",
        "short_report": resolved_config.short_report,
        "report_prompt_version": REPORT_PROMPT_VERSION,
        "meta_review": meta_review,
        "idea_overview": short_idea_overview if resolved_config.short_report else None,
        "section_summary": section_summary,
        "dimension_summary": dimension_summary,
        "reviewer_reviews": [] if resolved_config.short_report else reviewer_reviews,
        "reviewer_review_count": len(reviewer_reviews),
        "searched_papers": evidence_bank.get("searched_papers"),
        "revision_advice": revision_advice,
        "paper_references": paper_references,
        "artifact_paths": artifact_paths,
        "warnings": [],
        "generation": {
            "llm_enabled": bool(resolved_config.enable_llm and resolved_config.llm_api_key and not resolved_config.smoke),
            "model": None if client is None else resolved_config.llm_model_name,
        },
        "final_report_path": str(final_report_path.resolve()),
        "final_report_json_path": str(final_report_json_path.resolve()),
        "md_tree_path": str(md_tree_path.resolve()),
    }
    final_report_path.write_text(markdown_report, encoding="utf-8")
    write_json(final_report_json_path, final_payload)
    write_json(md_tree_path, md_tree)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    _log(
        resolved_config,
        "info",
        (
            f"Finished report synthesis status=ok elapsed_ms={elapsed_ms:.1f} "
            f"final_report_path={final_report_path.resolve()}"
        ),
    )
    return final_payload
