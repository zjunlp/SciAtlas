from __future__ import annotations

import concurrent.futures
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    CITATION_WRITING_REQUIREMENTS,
    REVIEW_SECTIONS,
    first_non_empty,
    load_env_values,
    normalize_citation_spacing,
    normalize_whitespace,
    strip_numeric_prose_citations,
    write_json,
)
from .figure_cards import summarize_figure_card


DEFAULT_LLM_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_LLM_MODEL_NAME = "DeepSeek-V3.2"
DEFAULT_LLM_TEMPERATURE = 0.2
REVIEW_PROMPT_VERSION = 5

SECTION_EVIDENCE_PRIORITY = {
    "Motivation": {"basic_idea", "motivation"},
    "Method": {"method", "algorithm", "model", "protocol", "assumption"},
    "Result": {"experimental_focus", "experiment", "evaluation", "benchmark", "metric"},
    "Discussion": {"motivation", "method", "experimental_focus", "experiment", "discussion"},
}


@dataclass(slots=True)
class ReviewGenerationConfig:
    env_path: Path | None = None
    llm_api_key: str | None = None
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    enable_llm: bool = True
    smoke: bool = False
    max_workers: int = 5
    max_evidence_cards_per_section: int = 40
    max_evidence_cards_per_query: int = 4
    max_experiment_cards_per_section: int = 20
    max_figure_cards_per_section: int = 4
    max_total_evidence_chars_per_section: int = 60000
    max_source_full_text_chars: int = 100000
    llm_timeout_seconds: int = 600
    max_retries: int = 3
    loggers: tuple[Any, ...] = ()


def _log(config: ReviewGenerationConfig, level: str, message: str) -> None:
    for logger in config.loggers:
        getattr(logger, level)(message)


def _clean_json_response(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
        else:
            stripped = "\n".join(lines[1:]).strip()
    match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


def _parse_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(_clean_json_response(text))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _normalize_list(value: Any, *, fallback: list[str] | None = None) -> list[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = normalize_whitespace(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result or list(fallback or [])


def _resolve_config(config: ReviewGenerationConfig) -> ReviewGenerationConfig:
    env_values = load_env_values(config.env_path)
    return ReviewGenerationConfig(
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
        max_workers=max(1, int(config.max_workers)),
        max_evidence_cards_per_section=max(0, int(config.max_evidence_cards_per_section)),
        max_evidence_cards_per_query=max(1, int(config.max_evidence_cards_per_query)),
        max_experiment_cards_per_section=max(0, int(config.max_experiment_cards_per_section)),
        max_figure_cards_per_section=max(0, int(config.max_figure_cards_per_section)),
        max_total_evidence_chars_per_section=max(1000, int(config.max_total_evidence_chars_per_section)),
        max_source_full_text_chars=max(1000, int(config.max_source_full_text_chars)),
        llm_timeout_seconds=max(30, int(config.llm_timeout_seconds)),
        max_retries=max(1, int(config.max_retries)),
        loggers=tuple(config.loggers or ()),
    )


def _redact_api_key(api_key: str | None) -> str:
    text = normalize_whitespace(api_key)
    if not text:
        return "<missing>"
    if len(text) <= 8:
        return text[:2] + "***"
    return f"{text[:6]}...{text[-4:]}"


class JsonLLMClient:
    def __init__(self, config: ReviewGenerationConfig) -> None:
        if not config.llm_api_key:
            raise ValueError("LLM API key is missing")
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            started_at = time.perf_counter()
            try:
                _log(
                    self.config,
                    "info",
                    (
                        f"LLM request start model={self.config.llm_model_name} "
                        f"attempt={attempt + 1}/{self.config.max_retries} "
                        f"system_chars={len(system_prompt)} user_chars={len(user_prompt)} "
                        f"schema_chars={len(schema_text)}"
                    ),
                )
                response = self.client.chat.completions.create(
                    model=self.config.llm_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"{system_prompt}\n\n"
                                "Return only a JSON object. Follow this JSON schema:\n"
                                f"{schema_text}"
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.config.llm_temperature,
                    max_tokens=8192,
                    timeout=self.config.llm_timeout_seconds,
                )
                content = response.choices[0].message.content or ""
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                _log(
                    self.config,
                    "info",
                    (
                        f"LLM request success model={self.config.llm_model_name} "
                        f"attempt={attempt + 1}/{self.config.max_retries} "
                        f"elapsed_ms={elapsed_ms:.1f} response_chars={len(content)}"
                    ),
                )
                return _parse_json_object(content)
            except Exception as exc:
                last_error = exc
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                debug_context = (
                    f"base_url={self.config.llm_base_url} "
                    f"model={self.config.llm_model_name} "
                    f"api_key={_redact_api_key(self.config.llm_api_key)}"
                )
                _log(
                    self.config,
                    "warning",
                    (
                        f"LLM request failed model={self.config.llm_model_name} "
                        f"attempt={attempt + 1}/{self.config.max_retries} "
                        f"elapsed_ms={elapsed_ms:.1f} error_type={exc.__class__.__name__} "
                        f"{debug_context} error={exc}"
                    ),
                )
                if attempt + 1 < self.config.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            "LLM JSON generation failed: "
            f"base_url={self.config.llm_base_url} "
            f"model={self.config.llm_model_name} "
            f"api_key={_redact_api_key(self.config.llm_api_key)} "
            f"error={last_error}"
        ) from last_error


def _score_card(card: dict[str, Any]) -> float:
    for key in ("rerank_score", "dense_score", "coverage_score"):
        try:
            if card.get(key) is not None:
                return float(card[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def _is_section_relevant(card: dict[str, Any], section: str) -> bool:
    priorities = SECTION_EVIDENCE_PRIORITY.get(section, set())
    idea_section = normalize_whitespace(card.get("idea_section")).casefold()
    if idea_section in priorities:
        return True
    section_path = normalize_whitespace(card.get("section_path_text")).casefold()
    query = normalize_whitespace(card.get("query")).casefold()
    if section == "Motivation":
        return any(term in section_path or term in query for term in ("introduction", "motivation", "background"))
    if section == "Method":
        return any(term in section_path or term in query for term in ("method", "approach", "model", "algorithm"))
    if section == "Result":
        return any(term in section_path or term in query for term in ("result", "experiment", "evaluation", "benchmark"))
    if section == "Discussion":
        return any(term in section_path or term in query for term in ("discussion", "conclusion", "limitation"))
    return False


def select_evidence_for_section(
    evidence_bank: dict[str, Any],
    section: str,
    config: ReviewGenerationConfig,
) -> dict[str, list[dict[str, Any]]]:
    evidence_cards = [card for card in evidence_bank.get("evidence_cards", []) if isinstance(card, dict)]
    experiment_cards = [card for card in evidence_bank.get("experiment_cards", []) if isinstance(card, dict)]
    figure_cards = [card for card in evidence_bank.get("figure_cards", []) if isinstance(card, dict)]

    evidence_cards.sort(
        key=lambda card: (_is_section_relevant(card, section), _score_card(card)),
        reverse=True,
    )

    selected_evidence: list[dict[str, Any]] = []
    per_query_count: dict[str, int] = {}
    total_chars = 0
    for card in evidence_cards:
        if len(selected_evidence) >= config.max_evidence_cards_per_section:
            break
        query_id = normalize_whitespace(card.get("query_id")) or "_unknown"
        if per_query_count.get(query_id, 0) >= config.max_evidence_cards_per_query:
            continue
        text = normalize_whitespace(card.get("evidence_text"))
        if not text:
            continue
        if total_chars + len(text) > config.max_total_evidence_chars_per_section:
            continue
        selected_evidence.append(card)
        per_query_count[query_id] = per_query_count.get(query_id, 0) + 1
        total_chars += len(text)

    selected_experiments: list[dict[str, Any]] = []
    if section in {"Method", "Result", "Discussion"}:
        experiment_cards.sort(key=_score_card, reverse=True)
        selected_experiments = experiment_cards[: config.max_experiment_cards_per_section]

    def _figure_score(card: dict[str, Any]) -> tuple[float, float]:
        relevance = card.get("relevance") if isinstance(card.get("relevance"), dict) else {}
        try:
            section_relevance = float(relevance.get(section, 0.0))
        except (TypeError, ValueError):
            section_relevance = 0.0
        try:
            support_strength = float(card.get("support_strength", 0.0))
        except (TypeError, ValueError):
            support_strength = 0.0
        return (section_relevance, support_strength)

    selected_figures: list[dict[str, Any]] = []
    if config.max_figure_cards_per_section > 0:
        figure_cards.sort(key=_figure_score, reverse=True)
        for card in figure_cards:
            relevance, support_strength = _figure_score(card)
            paper_sections = [
                normalize_whitespace(item)
                for item in (card.get("paper_sections") if isinstance(card.get("paper_sections"), list) else [])
            ]
            if relevance <= 0 and section not in paper_sections:
                continue
            selected_figures.append(card)
            if len(selected_figures) >= config.max_figure_cards_per_section:
                break

    return {
        "evidence_cards": selected_evidence,
        "experiment_cards": selected_experiments,
        "figure_cards": selected_figures,
    }


def _format_reviewer_profile(reviewer: dict[str, Any]) -> str:
    trajectory = reviewer.get("research_trajectory") if isinstance(reviewer.get("research_trajectory"), list) else []
    arsenal = reviewer.get("technical_arsenal") if isinstance(reviewer.get("technical_arsenal"), list) else []
    return "\n".join(
        [
            f"Academic background: {normalize_whitespace(reviewer.get('academic_background')) or 'Not available.'}",
            f"Research trajectory: {json.dumps(trajectory, ensure_ascii=False)}",
            f"Technical arsenal: {json.dumps(arsenal, ensure_ascii=False)}",
            f"Persona: {normalize_whitespace(reviewer.get('persona')) or 'Not specified.'}",
        ]
    )


def _format_evidence_cards(selection: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for card in selection.get("evidence_cards", []):
        paper_ref = normalize_whitespace(card.get("paper_ref"))
        if not paper_ref:
            continue
        citation = normalize_whitespace(card.get("citation_markdown"))
        text = normalize_whitespace(card.get("evidence_text"))
        if len(text) > 1400:
            text = text[:1400].rstrip() + "..."
        lines.append(
            "\n".join(
                [
                    f"Paper: {normalize_whitespace(card.get('paper_title')) or 'Unknown'}",
                    f"Allowed citation: {citation or 'No allowed citation available; do not cite this paper by number.'}",
                    f"Query: {normalize_whitespace(card.get('query_id'))} | Idea section: {normalize_whitespace(card.get('idea_section'))}",
                    f"Section path: {normalize_whitespace(card.get('section_path_text'))}",
                    f"Grounded passage: {text}",
                ]
            )
        )
    for card in selection.get("experiment_cards", []):
        paper_ref = normalize_whitespace(card.get("paper_ref"))
        if not paper_ref:
            continue
        citation = normalize_whitespace(card.get("citation_markdown"))
        missing = card.get("missing_or_undercovered") if isinstance(card.get("missing_or_undercovered"), list) else []
        lines.append(
            "\n".join(
                [
                    f"Experiment relation from: {normalize_whitespace(card.get('paper_title')) or 'Unknown'}",
                    f"Allowed citation: {citation or 'No allowed citation available; do not cite this paper by number.'}",
                    f"Recommended goal: {normalize_whitespace(card.get('recommended_goal')) or 'Not specified'}",
                    f"Coverage: {normalize_whitespace(card.get('coverage_label')) or 'unknown'}",
                    f"Missing or undercovered: {json.dumps(missing, ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(lines) if lines else "No grounded evidence cards were available for this section."


def _format_figure_cards(selection: dict[str, list[dict[str, Any]]]) -> str:
    lines = [summarize_figure_card(card) for card in selection.get("figure_cards", []) if isinstance(card, dict)]
    return "\n\n".join(lines) if lines else "No figure-grounded manuscript evidence was available for this section."


def _format_source_full_text(source_full_text: str, max_chars: int) -> str:
    text = str(source_full_text or "").strip()
    if not text:
        return "Not provided."
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Full source text truncated by length limit.]"


def _section_schema() -> dict[str, Any]:
    point_schema = {
        "type": "object",
        "properties": {
            "point": {"type": "string"},
            "supporting_standard_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Internal rubric standard ids that informed this point. These ids are for traceability only and must not be mentioned inside point.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Linked author-year Markdown citations used in this point. Each item must exactly match one citation_markdown value from the allowed citation table, such as [Smith et al., 2024](https://example.com). Do not use numeric paper references.",
            },
        },
        "required": ["point", "supporting_standard_ids", "citations"],
    }
    return {
        "type": "object",
        "properties": {
            "section": {"type": "string"},
            "assessment": {"type": "string"},
            "strengths": {
                "type": "array",
                "items": point_schema,
            },
            "weaknesses": {
                "type": "array",
                "items": point_schema,
            },
            "rubric_coverage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "standard_id": {"type": "string"},
                        "used_in": {"type": "string", "enum": ["strength", "weakness", "both", "not_applicable"]},
                        "note": {"type": "string"},
                    },
                    "required": ["standard_id", "used_in", "note"],
                },
            },
        },
        "required": ["section", "assessment", "strengths", "weaknesses", "rubric_coverage"],
    }


def _build_section_prompt(
    *,
    idea_text: str,
    source_full_text: str,
    reviewer: dict[str, Any],
    section: dict[str, Any],
    evidence_selection: dict[str, list[dict[str, Any]]],
    config: ReviewGenerationConfig,
) -> str:
    standards = section.get("standards") if isinstance(section.get("standards"), list) else []
    allowed_citations = []
    seen_allowed: set[str] = set()
    for group_name in ("evidence_cards", "experiment_cards"):
        for card in evidence_selection.get(group_name, []):
            if not isinstance(card, dict):
                continue
            paper_ref = normalize_whitespace(card.get("paper_ref"))
            citation = normalize_whitespace(card.get("citation_markdown"))
            if not paper_ref or not citation or paper_ref in seen_allowed:
                continue
            seen_allowed.add(paper_ref)
            allowed_citations.append(
                {
                    "internal_paper_ref_do_not_output": paper_ref,
                    "citation_markdown": citation,
                    "title": normalize_whitespace(card.get("paper_title")) or "Unknown",
                    "year": card.get("paper_year"),
                    "url": normalize_whitespace(card.get("paper_url")) or None,
                }
            )
    return f"""
You are generating one section of a peer review. The target may be a preliminary research idea or a finished manuscript extracted from PDF.
Use the reviewer's academic background and persona to shape emphasis, but do not invent facts or evidence. Be critical, domain-aware, precise, and concrete.

Structured research idea:
{idea_text}

Full source text:
{_format_source_full_text(source_full_text, config.max_source_full_text_chars)}

Reviewer profile:
{_format_reviewer_profile(reviewer)}

Review section:
{section.get('section')}

Rubric standards for this section:
{json.dumps(standards, ensure_ascii=False, indent=2)}

Grounded paper context:
{_format_evidence_cards(evidence_selection)}

Figure-grounded manuscript evidence:
{_format_figure_cards(evidence_selection)}

Allowed paper citations:
{json.dumps(allowed_citations, ensure_ascii=False, indent=2)}

{CITATION_WRITING_REQUIREMENTS}

Instructions:
- Use the rubric standards as a private checklist for this section. The final output should be section-level reviewer comments, not one comment per rubric standard.
- Write `assessment` as a concise section-level paragraph before the strengths and weaknesses. It should synthesize the section's overall evidential status in 2 to 4 sentences, naming the main positive signal and the main limiting concern when both exist.
- Produce 3 to 6 strengths and 3 to 6 weaknesses when the source evidence supports that many points. Prefer fewer specific points over generic filler.
- Every strength or weakness must be grounded in at least one rubric standard and must include that standard id in `supporting_standard_ids`.
- You may synthesize multiple rubric standards into one point when they are naturally connected.
- Keep the reviewer anonymous. Do not mention reviewer names, author names, reviewer IDs, or any other identity markers in any generated field.
- Do not refer to rubric standards by internal IDs such as `motivation_01`, `method_02`, `result_01`, `discussion_01`, or `standard_id` inside any natural-language `point` or `note`.
- Do not explicitly name rubric dimensions inside the prose unless the phrase is naturally needed for readability. The rubric is the source of the judgment, not a visible report outline.
- Do not generate numeric scores, ratings, recommendation labels, or score-like bins.
- For the `citations` JSON array, return only linked author-year Markdown citations copied exactly from the allowed `citation_markdown` values, such as ["[Smith et al., 2024](https://example.com)"]. Do not return numeric internal paper references.
- The `internal_paper_ref_do_not_output` values are only for reading the context. Never output them in `point`, `citations`, `rubric_coverage`, or any other field.
- Do not cite paper numbers, bracketed reference numbers from the manuscript, or author-year links that are not in the allowed list.
- In natural-language points, use linked author-year citations from the allowed list, such as `[Smith et al., 2024](URL)`. Do not use numeric citations in prose.
- If the full source text contains manuscript-local reference numbers such as `[12]`, `[26]`, or `[53]`, do not copy those numbers into the output. Refer to the cited prior work descriptively unless the paper is also present in the allowed citation table.
- For prior-art similarity, method feasibility, experiment coverage, benchmark precedent, or reproducibility judgments, cite the relevant paper inline in the point text.
- Use the full source text as authoritative context for concrete methods, experimental design, datasets, baselines, metrics, results, limitations, and implementation details.
- The figure-grounded manuscript evidence comes from the target paper's own figures, not from prior work. Use it to judge whether manuscript claims are visually supported, but do not treat it as an external citation source.
- When the full source text contains concrete evidence, do not claim it is missing merely because it is absent from the structured idea or grounded paper context.
- If evidence is missing from both the structured idea and the full source text, say so and lower confidence in the assessment instead of inventing support.
- For each point, build a compact evidence chain: state the core judgment, identify the idea/source evidence that supports it, compare that evidence against grounded papers where relevant, and explain the implication for this section.
- Do not merely summarize related work. Explicitly state how each cited paper affects the current idea's novelty, feasibility, experimental credibility, or impact.
- Be granular when applicable: name algorithms, model components, objectives, datasets, sample sizes, splits, annotation protocols, baselines, metrics, statistical tests, ablations, hyperparameters, compute/runtime constraints, reproducibility requirements, and failure modes.
- Avoid generic praise or criticism such as "promising", "interesting", "weak", or "valuable" unless the sentence also names the concrete mechanism, evidence gap, benchmark, or design choice that justifies the judgment.
- Keep each point concise and actionable, but specific enough to stand alone in a Nature-family-style reviewer report.
- Write strengths and weaknesses in polished academic prose that can be directly embedded into Markdown.
- Use Markdown emphasis sparingly but deliberately for important concepts, for example **core limitation**, **main contribution**, *key assumption*, or *evidence gap*.
- When referring to formulas, symbols, variables, metrics, or mechanisms, use standard Markdown math notation such as `$R = |J(-1.0\\,V)/J(+1.0\\,V)|$`, `$\\eta_V$`, `$t_L$`, `$t_R$`, `$E_{{HOMO}}$`, or `$$ ... $$` for display equations when needed.
- Never wrap math expressions in backticks. Write `$...$` or `$$...$$`, not `` `$...$` ``, `` `$$...$$` ``, or any other code-formatted variant.
- Do not describe equations only in plain words when a compact mathematical expression would improve clarity.
- Keep strengths and weaknesses as standalone list-friendly statements of one to two sentences each; do not turn them into paragraphs.
- Preserve a formal reviewer tone. Do not use emojis, decorative headings, or markdown tables in the JSON fields.
"""


def _citation_aliases_for_card(card: dict[str, Any]) -> list[str]:
    citation = normalize_whitespace(card.get("citation_markdown"))
    aliases = [
        citation,
        normalize_whitespace(card.get("citation_label")),
    ]
    paper_ref = normalize_whitespace(card.get("paper_ref"))
    if paper_ref and citation:
        aliases.append(paper_ref)
    return [alias for alias in aliases if alias]


def _allowed_citation_lookup(evidence_selection: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group_name in ("evidence_cards", "experiment_cards"):
        for card in evidence_selection.get(group_name, []):
            if not isinstance(card, dict):
                continue
            citation = normalize_whitespace(card.get("citation_markdown"))
            if not citation:
                continue
            for alias in _citation_aliases_for_card(card):
                lookup.setdefault(alias, citation)
                lookup.setdefault(alias.casefold(), citation)
    return lookup


def _valid_citations(value: Any, allowed_citation_lookup: dict[str, str]) -> list[str]:
    raw_values: list[str] = []
    if isinstance(value, list):
        raw_values = [normalize_whitespace(item) for item in value]
    else:
        raw_values = [normalize_whitespace(value)]

    citations: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        normalized = allowed_citation_lookup.get(raw) or allowed_citation_lookup.get(raw.casefold())
        if normalized and normalized not in seen:
            seen.add(normalized)
            citations.append(normalized)
    return citations


def _inline_allowed_citations(text: str, allowed_citation_lookup: dict[str, str]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for normalized in set(allowed_citation_lookup.values()):
        if normalized and normalized in text and normalized not in seen:
            seen.add(normalized)
            citations.append(normalized)
    return citations


def _selected_paper_refs(
    section_name: str,
    evidence_selection: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 3,
) -> list[str]:
    refs = [
        normalize_whitespace(card.get("paper_ref"))
        for card in evidence_selection.get("evidence_cards", [])[:2]
        if normalize_whitespace(card.get("paper_ref"))
    ]
    if section_name in {"Method", "Result", "Discussion"}:
        refs.extend(
            normalize_whitespace(card.get("paper_ref"))
            for card in evidence_selection.get("experiment_cards", [])[:1]
            if normalize_whitespace(card.get("paper_ref"))
        )
    return _normalize_list(refs)[:limit]


def _selected_citation_markdowns(
    section_name: str,
    evidence_selection: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 3,
) -> list[str]:
    refs = set(_selected_paper_refs(section_name, evidence_selection, limit=limit))
    citations: list[str] = []
    for group_name in ("evidence_cards", "experiment_cards"):
        for card in evidence_selection.get(group_name, []):
            if not isinstance(card, dict):
                continue
            paper_ref = normalize_whitespace(card.get("paper_ref"))
            citation = normalize_whitespace(card.get("citation_markdown"))
            if paper_ref in refs and citation:
                citations.append(citation)
    return _normalize_list(citations)[:limit]


def _point_payload(point: str, standard_ids: list[str], citations: list[str]) -> dict[str, Any]:
    return {
        "point": normalize_citation_spacing(strip_numeric_prose_citations(point)),
        "supporting_standard_ids": _normalize_list(standard_ids),
        "citations": _normalize_list(citations),
    }


def _fallback_section_review(
    section: dict[str, Any],
    evidence_selection: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    standards = section.get("standards") if isinstance(section.get("standards"), list) else []
    section_name = normalize_whitespace(section.get("section"))
    standard_ids = [
        normalize_whitespace(standard.get("standard_id"))
        for standard in standards
        if isinstance(standard, dict) and normalize_whitespace(standard.get("standard_id"))
    ]
    citations = _selected_citation_markdowns(section_name, evidence_selection)
    if not standards:
        return {
            "section": section_name,
            "assessment": "Section-level fallback review generated after the reviewer LLM response was unavailable; no substantive assessment was produced.",
            "strengths": [],
            "weaknesses": [],
            "rubric_coverage": [],
        }
    return {
        "section": section_name,
        "assessment": (
            "Section-level fallback review generated after the reviewer LLM response was unavailable. "
            "The strengths and weaknesses below are placeholders indicating that the section should be regenerated for a substantive assessment."
        ),
        "strengths": [
            _point_payload(
                "Section-level fallback review generated after the reviewer LLM response was unavailable; no substantive strength was assessed.",
                standard_ids[:1],
                citations[:1],
            )
        ],
        "weaknesses": [
            _point_payload(
                "Rerun the review stage or increase review LLM timeout/retries to obtain a full section assessment grounded in the rubric.",
                standard_ids[:1],
                citations[:1],
            )
        ],
        "rubric_coverage": [
            {
                "standard_id": standard_id,
                "used_in": "not_applicable",
                "note": "Fallback review did not assess this standard.",
            }
            for standard_id in standard_ids
        ],
    }


def _sanitize_point_list(
    raw_points: Any,
    *,
    fallback_points: list[dict[str, Any]],
    allowed_standard_ids: set[str],
    allowed_citation_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_points, list):
        raw_points = []
    points: list[dict[str, Any]] = []
    for item in raw_points:
        if isinstance(item, dict):
            point = normalize_whitespace(item.get("point") or item.get("text") or item.get("assessment"))
            raw_standard_ids = item.get("supporting_standard_ids") or item.get("standard_ids")
            raw_citations = item.get("citations")
        else:
            point = normalize_whitespace(item)
            raw_standard_ids = []
            raw_citations = []
        if not point:
            continue
        standard_ids = [
            standard_id
            for standard_id in _normalize_list(raw_standard_ids)
            if standard_id in allowed_standard_ids
        ]
        if not standard_ids and allowed_standard_ids:
            standard_ids = [sorted(allowed_standard_ids)[0]]
        citations = _valid_citations(raw_citations, allowed_citation_lookup)
        inline_citations = _inline_allowed_citations(point, allowed_citation_lookup)
        for citation in inline_citations:
            if citation not in citations:
                citations.append(citation)
        point = normalize_citation_spacing(strip_numeric_prose_citations(point))
        points.append(
            {
                "point": point,
                "supporting_standard_ids": standard_ids,
                "citations": citations,
            }
        )
    return points or fallback_points


def _sanitize_section_review(
    raw: dict[str, Any],
    *,
    section: dict[str, Any],
    evidence_selection: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    allowed_citation_lookup = _allowed_citation_lookup(evidence_selection)
    standards = section.get("standards") if isinstance(section.get("standards"), list) else []
    allowed_standard_ids = {
        normalize_whitespace(standard.get("standard_id"))
        for standard in standards
        if isinstance(standard, dict) and normalize_whitespace(standard.get("standard_id"))
    }
    fallback = _fallback_section_review(section, evidence_selection)
    strengths = _sanitize_point_list(
        raw.get("strengths"),
        fallback_points=fallback["strengths"],
        allowed_standard_ids=allowed_standard_ids,
        allowed_citation_lookup=allowed_citation_lookup,
    )
    weaknesses = _sanitize_point_list(
        raw.get("weaknesses"),
        fallback_points=fallback["weaknesses"],
        allowed_standard_ids=allowed_standard_ids,
        allowed_citation_lookup=allowed_citation_lookup,
    )
    assessment = normalize_whitespace(raw.get("assessment") or raw.get("summary") or raw.get("overall_assessment"))
    if not assessment:
        assessment = fallback.get("assessment", "")
    assessment = normalize_citation_spacing(strip_numeric_prose_citations(assessment))

    used_by_standard: dict[str, set[str]] = {standard_id: set() for standard_id in allowed_standard_ids}
    for label, points in (("strength", strengths), ("weakness", weaknesses)):
        for point in points:
            for standard_id in point.get("supporting_standard_ids", []):
                if standard_id in used_by_standard:
                    used_by_standard[standard_id].add(label)

    raw_coverage = raw.get("rubric_coverage") if isinstance(raw.get("rubric_coverage"), list) else []
    raw_coverage_by_id = {
        normalize_whitespace(item.get("standard_id")): item
        for item in raw_coverage
        if isinstance(item, dict) and normalize_whitespace(item.get("standard_id")) in allowed_standard_ids
    }
    rubric_coverage: list[dict[str, Any]] = []
    for standard in standards:
        if not isinstance(standard, dict):
            continue
        standard_id = normalize_whitespace(standard.get("standard_id"))
        if not standard_id:
            continue
        raw_item = raw_coverage_by_id.get(standard_id, {})
        used_labels = used_by_standard.get(standard_id, set())
        if used_labels == {"strength", "weakness"}:
            used_in = "both"
        elif "strength" in used_labels:
            used_in = "strength"
        elif "weakness" in used_labels:
            used_in = "weakness"
        else:
            used_in = "not_applicable"
        raw_used_in = normalize_whitespace(raw_item.get("used_in")) if isinstance(raw_item, dict) else ""
        if raw_used_in in {"strength", "weakness", "both", "not_applicable"} and used_in == "not_applicable":
            used_in = raw_used_in
        rubric_coverage.append(
            {
                "standard_id": standard_id,
                "used_in": used_in,
                "note": normalize_whitespace(raw_item.get("note")) if isinstance(raw_item, dict) else "",
            }
        )

    return {
        "section": normalize_whitespace(section.get("section")) or normalize_whitespace(raw.get("section")),
        "assessment": assessment,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "rubric_coverage": rubric_coverage,
    }


def _evaluate_section(
    *,
    idea_text: str,
    source_full_text: str,
    reviewer: dict[str, Any],
    section: dict[str, Any],
    evidence_bank: dict[str, Any],
    config: ReviewGenerationConfig,
    client: JsonLLMClient | None,
) -> tuple[dict[str, Any], str, str | None]:
    evidence_selection = select_evidence_for_section(evidence_bank, normalize_whitespace(section.get("section")), config)
    if client is None:
        return _fallback_section_review(section, evidence_selection), "fallback", None

    try:
        raw = client.generate_json(
            system_prompt="You are a strict, evidence-grounded academic reviewer.",
            user_prompt=_build_section_prompt(
                idea_text=idea_text,
                source_full_text=source_full_text,
                reviewer=reviewer,
                section=section,
                evidence_selection=evidence_selection,
                config=config,
            ),
            schema=_section_schema(),
        )
        return _sanitize_section_review(raw, section=section, evidence_selection=evidence_selection), "llm", None
    except Exception as exc:
        return _fallback_section_review(section, evidence_selection), "fallback", str(exc)


def _build_overall(section_reviews: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []
    for section in section_reviews:
        section_name = normalize_whitespace(section.get("section")) or "this section"
        strengths.extend(
            normalize_whitespace(point.get("point"))
            for point in section.get("strengths", [])
            if isinstance(point, dict) and normalize_whitespace(point.get("point"))
        )
        weaknesses.extend(
            normalize_whitespace(point.get("point"))
            for point in section.get("weaknesses", [])
            if isinstance(point, dict) and normalize_whitespace(point.get("point"))
        )
        for point in section.get("weaknesses", []):
            if isinstance(point, dict) and normalize_whitespace(point.get("point")):
                suggestions.append(f"Address the {section_name} concern: {normalize_whitespace(point.get('point'))}")

    confidence = "medium"
    reviewed_sections = [review for review in section_reviews if review.get("strengths") or review.get("weaknesses")]
    if not reviewed_sections or warnings:
        confidence = "low"
    elif len(reviewed_sections) >= 3:
        confidence = "high"

    return {
        "confidence": confidence,
        "summary": (
            f"Reviewed {len(reviewed_sections)} rubric sections. "
            "The summary reflects recurring strengths, weaknesses, and evidence gaps across those sections."
        ),
        "strengths": _normalize_list(strengths)[:6],
        "weaknesses": _normalize_list(weaknesses)[:6],
        "suggestions": _normalize_list(suggestions)[:8],
    }


def evaluate_reviewer(
    *,
    idea_text: str,
    source_full_text: str,
    normalized_rubric: dict[str, Any],
    reviewer: dict[str, Any],
    evidence_bank: dict[str, Any],
    config: ReviewGenerationConfig,
    client: JsonLLMClient | None,
) -> dict[str, Any]:
    reviewer_started_at = time.perf_counter()
    section_reviews: list[dict[str, Any]] = []
    generation_modes: set[str] = set()
    warnings: list[dict[str, Any]] = []
    sections = [
        section
        for section in (
            normalized_rubric.get("sections")
            if isinstance(normalized_rubric.get("sections"), list)
            else []
        )
        if isinstance(section, dict)
    ]
    order_index = {section: index for index, section in enumerate(REVIEW_SECTIONS)}
    sections.sort(
        key=lambda section: order_index.get(
            normalize_whitespace(section.get("section")),
            len(order_index),
        )
    )

    def _run_section(index: int, section: dict[str, Any]) -> tuple[int, dict[str, Any], str, str | None]:
        section_name = normalize_whitespace(section.get("section")) or f"section_{index + 1}"
        section_started_at = time.perf_counter()
        _log(config, "info", f"Reviewer {reviewer.get('reviewer_id')} section start section={section_name}")
        section_client = JsonLLMClient(config) if client is not None else None
        review, mode, error = _evaluate_section(
            idea_text=idea_text,
            source_full_text=source_full_text,
            reviewer=reviewer,
            section=section,
            evidence_bank=evidence_bank,
            config=config,
            client=section_client,
        )
        elapsed_ms = (time.perf_counter() - section_started_at) * 1000.0
        _log(
            config,
            "info",
            (
                f"Reviewer {reviewer.get('reviewer_id')} section done section={section_name} "
                f"mode={mode} elapsed_ms={elapsed_ms:.1f} error={'none' if not error else error}"
            ),
        )
        return index, review, mode, error

    section_records: list[tuple[int, dict[str, Any], str, str | None]] = []
    if sections:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sections)) as executor:
            futures = [
                executor.submit(_run_section, index, section)
                for index, section in enumerate(sections)
            ]
            for future in concurrent.futures.as_completed(futures):
                section_records.append(future.result())

    for _, review, mode, error in sorted(section_records, key=lambda item: item[0]):
        if error:
            warnings.append(
                {
                    "section": normalize_whitespace(review.get("section")),
                    "type": "llm_fallback",
                    "error": error,
                }
            )
        generation_modes.add(mode)
        section_reviews.append(review)

    overall = _build_overall(section_reviews, warnings)
    generation_mode = "llm" if generation_modes == {"llm"} else "fallback" if generation_modes == {"fallback"} else "mixed"
    if config.smoke:
        generation_mode = "smoke"

    reviewer_elapsed_ms = (time.perf_counter() - reviewer_started_at) * 1000.0
    _log(
        config,
        "info",
        (
            f"Reviewer done reviewer_id={reviewer.get('reviewer_id')} "
            f"generation_mode={generation_mode} section_count={len(section_reviews)} "
            f"warning_count={len(warnings)} elapsed_ms={reviewer_elapsed_ms:.1f}"
        ),
    )

    return {
        "status": "ok",
        "reviewer_id": reviewer.get("reviewer_id"),
        "author_name": reviewer.get("author_name"),
        "author_id": reviewer.get("author_id"),
        "persona": reviewer.get("persona"),
        "reviewer_profile": reviewer,
        "generation_mode": generation_mode,
        "review_prompt_version": REVIEW_PROMPT_VERSION,
        "warnings": warnings,
        "section_reviews": section_reviews,
        "overall": overall,
    }


def _make_client(config: ReviewGenerationConfig) -> JsonLLMClient | None:
    if config.smoke or not config.enable_llm:
        return None
    if not config.llm_api_key:
        return None
    return JsonLLMClient(config)


def _cached_review_matches_reviewer(cached: dict[str, Any], reviewer: dict[str, Any]) -> bool:
    if normalize_whitespace(cached.get("status")) != "ok":
        return False
    if cached.get("review_prompt_version") != REVIEW_PROMPT_VERSION:
        return False
    overall = cached.get("overall") if isinstance(cached.get("overall"), dict) else {}
    if "score" in overall or "recommendation" in overall:
        return False
    for section in cached.get("section_reviews", []):
        if not isinstance(section, dict):
            continue
        if "section_score" in section:
            return False
        if "dimension_reviews" in section:
            return False
        if not normalize_whitespace(section.get("assessment")):
            return False
        if not isinstance(section.get("strengths"), list) or not isinstance(section.get("weaknesses"), list):
            return False
    if normalize_whitespace(cached.get("reviewer_id")) != normalize_whitespace(reviewer.get("reviewer_id")):
        return False
    cached_profile = cached.get("reviewer_profile")
    if not isinstance(cached_profile, dict):
        return False
    return all(
        normalize_whitespace(cached_profile.get(field)) == normalize_whitespace(reviewer.get(field))
        for field in ["author_name", "author_id", "academic_background", "persona"]
    )


def _load_cached_reviewer_review(output_dir: Path, reviewer: dict[str, Any]) -> dict[str, Any] | None:
    reviewer_id = normalize_whitespace(reviewer.get("reviewer_id"))
    if not reviewer_id:
        return None
    review_path = output_dir / f"{reviewer_id}.review.json"
    if not review_path.exists():
        return None
    try:
        cached = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(cached, dict) or not _cached_review_matches_reviewer(cached, reviewer):
        return None
    return {
        "reviewer_id": cached.get("reviewer_id"),
        "author_name": cached.get("author_name"),
        "status": cached["status"],
        "generation_mode": cached.get("generation_mode"),
        "review_path": str(review_path.resolve()),
        "review": cached,
        "cache": "hit",
    }


def run_reviewer_evaluations(
    *,
    idea_text: str,
    source_full_text: str,
    normalized_rubric: dict[str, Any],
    normalized_reviewers: dict[str, Any],
    evidence_bank: dict[str, Any],
    output_dir: Path,
    config: ReviewGenerationConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _resolve_config(config)
    source_full_text_chars = len(str(source_full_text or ""))
    reviewers = [
        reviewer for reviewer in normalized_reviewers.get("reviewers", []) if isinstance(reviewer, dict)
    ]

    review_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    index_path = output_dir / "reviewer_reviews.index.json"
    if not reviewers:
        payload = {
            "status": "skipped",
            "reviewer_count": 0,
            "successful_count": 0,
            "failed_count": 0,
            "llm_enabled": bool(resolved_config.enable_llm and resolved_config.llm_api_key and not resolved_config.smoke),
            "model": (
                resolved_config.llm_model_name
                if resolved_config.enable_llm and resolved_config.llm_api_key and not resolved_config.smoke
                else None
            ),
            "reviews": [],
            "errors": [],
            "reviewer_reviews": [],
            "source_full_text_chars": source_full_text_chars,
            "index_path": str(index_path.resolve()),
        }
        write_json(index_path, payload)
        return payload

    section_count = len(
        [
            section
            for section in (
                normalized_rubric.get("sections")
                if isinstance(normalized_rubric.get("sections"), list)
                else []
            )
            if isinstance(section, dict)
        ]
    )
    effective_reviewer_workers = min(max(1, resolved_config.max_workers), len(reviewers))
    _log(
        resolved_config,
        "info",
        (
            f"Starting reviewer evaluations reviewer_count={len(reviewers)} "
            f"configured_max_workers={resolved_config.max_workers} "
            f"effective_reviewer_workers={effective_reviewer_workers} "
            f"section_count_per_reviewer={section_count} "
            f"theoretical_max_parallel_llm_calls={effective_reviewer_workers * max(section_count, 1)}"
        ),
    )

    def _run_one(reviewer: dict[str, Any]) -> dict[str, Any]:
        cached_review = _load_cached_reviewer_review(output_dir, reviewer)
        if cached_review is not None:
            _log(resolved_config, "info", f"Reviewer cache hit reviewer_id={reviewer.get('reviewer_id')}")
            return cached_review

        client = _make_client(resolved_config)
        review = evaluate_reviewer(
            idea_text=idea_text,
            source_full_text=source_full_text,
            normalized_rubric=normalized_rubric,
            reviewer=reviewer,
            evidence_bank=evidence_bank,
            config=resolved_config,
            client=client,
        )
        review_path = output_dir / f"{reviewer.get('reviewer_id')}.review.json"
        write_json(review_path, review)
        return {
            "reviewer_id": reviewer.get("reviewer_id"),
            "author_name": reviewer.get("author_name"),
            "status": review["status"],
            "generation_mode": review["generation_mode"],
            "review_path": str(review_path.resolve()),
            "review": review,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_config.max_workers) as executor:
        future_to_reviewer = {executor.submit(_run_one, reviewer): reviewer for reviewer in reviewers}
        for future in concurrent.futures.as_completed(future_to_reviewer):
            reviewer = future_to_reviewer[future]
            try:
                review_records.append(future.result())
            except Exception as exc:
                reviewer_id = normalize_whitespace(reviewer.get("reviewer_id")) or "reviewer_unknown"
                error_payload = {
                    "status": "error",
                    "reviewer_id": reviewer_id,
                    "author_name": reviewer.get("author_name"),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
                error_path = output_dir / f"{reviewer_id}.review.json"
                write_json(error_path, error_payload)
                errors.append({**error_payload, "review_path": str(error_path.resolve())})

    review_records.sort(key=lambda item: normalize_whitespace(item.get("reviewer_id")))
    successful_reviews = [record for record in review_records if record.get("status") == "ok"]
    status = "ok"
    if not successful_reviews:
        status = "error"
    elif errors or len(successful_reviews) < len(reviewers):
        status = "partial_error"

    index_records = [
        {
            key: value
            for key, value in record.items()
            if key != "review"
        }
        for record in review_records
    ]
    payload = {
        "status": status,
        "reviewer_count": len(reviewers),
        "successful_count": len(successful_reviews),
        "failed_count": len(errors),
        "llm_enabled": bool(resolved_config.enable_llm and resolved_config.llm_api_key and not resolved_config.smoke),
        "model": (
            resolved_config.llm_model_name
            if resolved_config.enable_llm and resolved_config.llm_api_key and not resolved_config.smoke
            else None
        ),
        "reviews": index_records,
        "errors": errors,
        "reviewer_reviews": [record["review"] for record in review_records if "review" in record],
        "source_full_text_chars": source_full_text_chars,
        "index_path": str(index_path.resolve()),
    }
    write_json(index_path, payload)
    return payload
