#!/usr/bin/env python3
"""Generate a formal literature review and a deterministic diagnostic report.

This script accepts either:

1. an existing organized evidence map; or
2. a raw lr_search search_result.json, in which case it first organizes the result.

The pipeline intentionally produces two outputs:

- a reader-facing formal review assembled through multiple LLM calls;
- a developer-facing diagnostic report rendered mostly deterministically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import re
import threading
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LR_SEARCH_DIR = Path(__file__).resolve().parent
if str(LR_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(LR_SEARCH_DIR))

from literature_review_search import (  # noqa: E402
    DEFAULT_DMX_API_URL,
    DEFAULT_DMX_MODEL,
    DEFAULT_ENV_PATH,
    DmxJsonClient,
    normalize_whitespace,
    truncate_text,
    write_json,
)
from organize_search_result import organize  # noqa: E402

TEMPORAL_SECTION_ID = "temporal_development"
TEMPORAL_SECTION_TITLE = "Temporal Development of the Field"
METHOD_SUBSECTION_PROMPT_STYLE_VERSION = "bold_leadin_v1"
PLAIN_SUBSECTION_PROMPT_STYLE_VERSION = "plain_survey_prose_v1"
TEMPORAL_SUBSECTION_PROMPT_STYLE_VERSION = "temporal_bold_leadin_v3"
SUBJECT_CHEMISTRY = "chemistry"
SUBJECT_BIOLOGY = "biology"
SUBJECT_GENERAL = "general"
CASE_PATCH_NONE = "none"
CASE_PATCH_CASE1 = "case1"
CASE_PATCH_CASE2 = "case2"
CASE_PATCH_CASE5 = "case5"


GENERAL_SUBSECTION_RULES = [
    "Do not generalize from a single paper, a single catalyst, or a model substrate to an entire catalyst family, reaction platform, or material class.",
    "If the supplied evidence is mechanistically heterogeneous, explicitly separate subcategories or qualify the claim instead of forcing a unified mechanism.",
    "When evidence supports only local conclusions, write local conclusions. Prefer narrower but correct claims over broader but weak claims.",
    "If a paper's specific role, mechanism, or scope is uncertain from the supplied evidence, avoid asserting it as a factual detail.",
    "Do not treat adjacent application domains as interchangeable merely because they share catalytic, redox, or materials language.",
]

GENERAL_INTEGRATION_RULES = [
    "Do not strengthen claims beyond what the drafted sections support.",
    "Do not describe the field as broadly green, practical, scalable, general, or industrially relevant unless those conclusions are repeatedly supported in the drafted sections.",
]

CHEMISTRY_OUTLINE_RULES = [
    "Prefer sections built from comparable reaction families rather than broad platform labels alone.",
    "A section or subsection should ideally group papers that are comparable in catalyst system, reaction type, and operative mechanism.",
    "Do not mix environmental remediation, pollutant degradation, energy conversion, or surface electrocatalysis into the core narrative unless the topic explicitly includes those goals.",
    "When a catalyst platform supports multiple mechanistically distinct reaction classes, prefer splitting by reaction class rather than merging them under one platform-level advantage claim.",
    "Preserve fixed domain terms exactly when they are standard technical expressions; avoid paraphrasing them into loose near-synonyms.",
]

CHEMISTRY_SUBSECTION_RULES = [
    "For chemistry topics, synthesize at the level of reaction class, catalyst family, and mechanism. Avoid paper-by-paper retelling.",
    "Mechanistic claims must be tied to the specific reaction class and catalyst context that support them.",
    "Do not use one mechanistic vocabulary to cover catalyst systems whose operative steps differ substantially.",
    "Treat green, practical, scalable, and industrial claims as evidence-dependent and require concrete support such as selectivity, substrate scope, TON, TOF, catalyst lifetime, recyclability, Faradaic efficiency, current density, or scale-up evidence when relevant.",
]

CASE_OUTLINE_PATCHES: dict[str, list[str]] = {
    CASE_PATCH_CASE1: [
        "If zeolites are included alongside MOFs and COFs, avoid assuming that a narrower taxonomic term necessarily covers all of them; prefer broader wording when the evidence spans multiple crystalline porous material classes.",
    ],
    CASE_PATCH_CASE2: [
        "For selective aerobic oxidation, prefer a taxonomy that separates genuinely comparable catalyst families and reaction manifolds.",
        "Use a dedicated heterogeneous aerobic oxidation section rather than a generic emerging-systems bucket when the evidence supports that distinction.",
        "Within heterogeneous aerobic oxidation, separate supported metal or metal-oxide catalysts, MOF or COF-based systems, and heterogeneous photocatalytic systems when enough evidence exists.",
        "Exclude pollutant degradation, Cr(VI) removal, and energy-electrocatalysis papers from the core synthetic narrative.",
    ],
    CASE_PATCH_CASE5: [
        "Rebuild the cathodic section around product-oriented organic synthesis, such as electroreductive cross-coupling, electrocarboxylation, carbonyl reductive coupling, and cathodically generated carbanion or organometallic reactivity.",
        "Exclude pollutant dehalogenation, PFAS degradation, contrast-agent removal, nitrate treatment, microbial fuel cell studies, and other environmental-electrochemistry papers from the core organic electrosynthesis narrative.",
        "Treat Paired Electrolysis as a fixed technical term and keep it intact.",
    ],
}

CASE_SUBSECTION_PATCHES: dict[str, list[str]] = {
    CASE_PATCH_CASE1: [
        "If zeolites are included together with MOFs and COFs, avoid using 'porous framework materials' as a strict taxonomic term unless the evidence explicitly supports that usage; prefer broader wording such as 'crystalline porous materials' when needed.",
        "Do not attribute experimental-feasibility filtering, synthetic-accessibility prediction, or graph-neural-network modeling to a paper unless that function is clearly supported by the supplied evidence.",
        "Do not restate a general representation family as if every cited paper within that family uses the same input modality or architecture.",
        "When discussing model architecture, specify whether the evidence supports descriptor-based ML, ANN/MLP, GNN, geometric deep learning, or foundation models; do not collapse them.",
        "Do not describe Boyd et al. (2019) as performing prior experimental-viability filtering unless the supplied evidence explicitly states that.",
        "Do not describe Moghadam et al. (2019) as a GNN or atom-coordinate message-passing model unless the supplied evidence explicitly states that.",
        "Do not attribute synthetic-accessibility prediction to Vandenhaute et al. (2023) unless the supplied evidence explicitly supports that claim.",
    ],
    CASE_PATCH_CASE2: [
        "Only synthesize papers together when they target comparable selective organic synthesis outcomes.",
        "For Cu/nitroxyl chemistry, distinguish metal-free oxoammonium oxidation from Cu/nitroxyl cooperative alcohol dehydrogenation. Do not treat them as one unified mechanism.",
        "For Pd aerobic oxidation, separate alcohol oxidation, Wacker-type oxidation, oxidative Heck, and other distinct manifolds when discussing mechanism or ligand effects.",
        "Do not present ligand effects or selectivity features from one Pd reaction class as a platform-wide Pd advantage.",
        "For NHPI/PINO chemistry, do not uniformly describe the platform as metal-free; specify when metal co-catalysts are essential.",
        "Exclude pollutant degradation, Cr(VI) removal, and other non-synthetic oxidation papers from the core argument even if they use related catalyst labels.",
    ],
    CASE_PATCH_CASE5: [
        "Do not use CO2 electroreduction on metal surfaces as a mechanistic basis for molecular organic cross-electrophile coupling unless the evidence explicitly makes that connection.",
        "Use 'Redox Mediator' for reversible molecular electron shuttles such as TEMPO, halides, or triarylamines.",
        "Do not classify immobilized single-atom sites or electrode-bound catalytic centers as mediators by default; describe them as heterogeneous electrocatalysts or immobilized catalytic sites unless the evidence clearly defines a mediator role.",
        "Preserve the following terms as fixed units and do not paraphrase or split them: Paired Electrolysis, Electrophotocatalysis, Redox Mediator, Faradaic Efficiency, Constant-Current Electrolysis, Constant-Potential Electrolysis.",
        "Do not describe organic electrosynthesis as broadly green, practical, or scalable from isolated examples or single metrics.",
        "Exclude pollutant dehalogenation, PFAS degradation, contrast-agent removal, nitrate treatment, microbial fuel cell studies, and other environmental-electrochemistry papers from the core organic electrosynthesis narrative.",
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_text(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _normalize_base_url(base_url: str) -> str:
    trimmed = normalize_whitespace(base_url).rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def load_codex_gpt_override() -> dict[str, str]:
    config_path = Path("~/.codex/config.toml").expanduser()
    auth_path = Path("~/.codex/auth.json").expanduser()
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
                    block, 'base_url="', ('"', "\n")
                )
                wire_api = _extract_between(block, 'wire_api = "', ('"', "\n")) or _extract_between(
                    block, 'wire_api="', ('"', "\n")
                )
                if base_url:
                    values["base_url"] = _normalize_base_url(base_url)
                if wire_api:
                    values["wire_api"] = normalize_whitespace(wire_api)
                values["provider_name"] = provider
        if model:
            values["model"] = model
    if auth_path.exists():
        payload = _load_json(auth_path)
        api_key = normalize_whitespace(payload.get("OPENAI_API_KEY"))
        if api_key:
            values["api_key"] = api_key
    return values


class ProgressLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def log(self, stage: str, message: str, **fields: Any) -> None:
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        payload = {"timestamp": timestamp, "stage": stage, "message": message, **fields}
        stderr_line = f"[{timestamp}] [{stage}] {message}"
        if fields:
            extras = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
            stderr_line = f"{stderr_line} | {extras}"
        print(stderr_line, file=sys.stderr, flush=True)
        if self.path is None:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = normalize_whitespace(value)
    return [text] if text else []


def md_escape(value: Any, *, max_chars: int = 180) -> str:
    return truncate_text(normalize_whitespace(value), max_chars=max_chars).replace("|", " ")


def slugify(value: Any, *, fallback: str = "paper", max_chars: int = 24) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", normalize_whitespace(value).casefold())
    return slug[:max_chars] or fallback


def title_case_name(value: str) -> str:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return "Unknown"
    if "," in cleaned:
        last = normalize_whitespace(cleaned.split(",", 1)[0])
        return last or cleaned
    parts = cleaned.split()
    if not parts:
        return "Unknown"
    return parts[-1].strip(".") or cleaned


def strip_section_number(title: Any) -> str:
    return normalize_whitespace(re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", normalize_whitespace(title)))


def normalize_section_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", normalize_whitespace(value).casefold()).strip("_")


def make_section_id(title: Any, *, fallback: str) -> str:
    return normalize_section_id(strip_section_number(title)) or fallback


def coverage_bullets_from_section(section: dict[str, Any]) -> list[dict[str, Any]]:
    bullets: list[dict[str, Any]] = []
    raw_bullets = section.get("coverage_bullets")
    if isinstance(raw_bullets, list):
        bullet_items = raw_bullets
    elif isinstance(raw_bullets, dict):
        bullet_items = [raw_bullets]
    else:
        bullet_items = as_list(raw_bullets)
    for item in bullet_items:
        if isinstance(item, dict):
            point = normalize_whitespace(item.get("point") or item.get("description"))
            if not point:
                continue
            enriched = dict(item)
            enriched["point"] = point
            enriched.setdefault("description", normalize_whitespace(item.get("description")))
            enriched.setdefault("retrieval_queries", as_list(item.get("retrieval_queries"))[:4])
            enriched.setdefault("required_evidence_roles", as_list(item.get("required_evidence_roles"))[:4])
            enriched.setdefault("target_citation_count", item.get("target_citation_count") or 6)
            bullets.append(enriched)
        elif normalize_whitespace(item):
            bullets.append({"point": normalize_whitespace(item), "description": "", "target_citation_count": 6})
    if bullets:
        return bullets

    for item in as_list(section.get("must_cover")):
        point = normalize_whitespace(item)
        if point:
            bullets.append({"point": point, "description": "", "target_citation_count": 6})
    description = normalize_whitespace(section.get("description") or section.get("purpose") or section.get("notes"))
    if not bullets and description:
        bullets.append({"point": description, "description": description, "target_citation_count": 6})
    return bullets


def classify_outline_section(section: dict[str, Any]) -> str:
    section_id = normalize_section_id(section.get("section_id"))
    title = strip_section_number(section.get("section_title")).casefold()
    if "intro" in section_id or title == "introduction":
        return "intro"
    if "background" in section_id or "problem_formulation" in section_id or "problem formulation" in title:
        return "background"
    if (
        section_id == TEMPORAL_SECTION_ID
        or "temporal" in section_id
        or "chronological" in section_id
        or "time" in section_id
        or "temporal development" in title
        or "chronological" in title
        or "development" in title and "field" in title
    ):
        return "temporal"
    if "taxonomy" in section_id or "taxonomy" in title:
        return "taxonomy"
    if "comparative" in section_id or "comparative" in title:
        return "comparative"
    if "evaluation" in section_id or "benchmark" in section_id or "experimental practices" in title:
        return "evaluation"
    if "gap" in section_id or "future" in section_id or "open problems" in title:
        return "open_problems"
    if "conclusion" in section_id or title == "conclusion":
        return "conclusion"
    return "other"


def subsection_style_for_pack(pack: dict[str, Any]) -> str:
    section_id = normalize_section_id(pack.get("section_id"))
    if section_id == TEMPORAL_SECTION_ID:
        return TEMPORAL_SUBSECTION_PROMPT_STYLE_VERSION
    role = normalize_whitespace(pack.get("section_role")).casefold()
    section_kind = classify_outline_section(
        {
            "section_id": pack.get("section_id"),
            "section_title": pack.get("section_title"),
            "section_role": pack.get("section_role"),
        }
    )
    if role in {"method", "method_family"} or section_kind in {"taxonomy", "comparative"}:
        return METHOD_SUBSECTION_PROMPT_STYLE_VERSION
    return PLAIN_SUBSECTION_PROMPT_STYLE_VERSION


def canonicalize_outline(outline: dict[str, Any]) -> dict[str, Any]:
    sections = [item for item in outline.get("sections", []) if isinstance(item, dict)]
    family_sections = [item for item in outline.get("family_sections", []) if isinstance(item, dict)]

    normalized_sections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def normalize_section(item: dict[str, Any], *, index: int, default_role: str = "other") -> dict[str, Any]:
        enriched = dict(item)
        title = (
            strip_section_number(enriched.get("section_title"))
            or strip_section_number(enriched.get("title"))
            or strip_section_number(enriched.get("family_name"))
            or f"Section {index}"
        )
        enriched["section_title"] = title
        section_id = normalize_section_id(enriched.get("section_id")) or make_section_id(title, fallback=f"section_{index}")
        base_id = section_id
        suffix = 2
        while section_id in seen_ids:
            section_id = f"{base_id}_{suffix}"
            suffix += 1
        seen_ids.add(section_id)
        enriched["section_id"] = section_id
        enriched["section_role"] = normalize_whitespace(enriched.get("section_role")) or default_role
        enriched.setdefault("description", normalize_whitespace(enriched.get("purpose") or enriched.get("notes")))
        enriched.setdefault("coverage_bullets", coverage_bullets_from_section(enriched))
        enriched.setdefault("render_mode", "with_subsections" if enriched.get("subsections") else "section_body")
        enriched.setdefault("render_as", "major_section")
        return enriched

    for index, item in enumerate(sections, start=1):
        normalized_sections.append(normalize_section(item, index=index))

    for item in family_sections:
        if normalize_whitespace(item.get("section_title") or item.get("family_name")):
            index = len(normalized_sections) + 1
            enriched = normalize_section(item, index=index, default_role="method")
            enriched.setdefault("render_as", "major_section")
            normalized_sections.append(enriched)

    normalized = dict(outline)
    normalized["sections"] = normalized_sections
    normalized["family_sections"] = []
    return normalized


def preserve_markdown_block(text: Any) -> str:
    raw = str(text or "")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in raw.split("\n")]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line.strip():
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False
    return "\n".join(cleaned).strip()


def assignment_rows(evidence_map: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in evidence_map.get("paper_assignments", [])
        if isinstance(item, dict) and item.get("paper_id")
    ]


def assignments_by_id(evidence_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["paper_id"]: item for item in assignment_rows(evidence_map)}


def paper_cards_by_id(search_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["paper_id"]: item
        for item in search_result.get("paper_cards", [])
        if isinstance(item, dict) and item.get("paper_id")
    }


def normalize_search_actions(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    actions_payload = search_result.get("search_actions", {})
    normalized: list[dict[str, Any]] = []
    if isinstance(actions_payload, list):
        for item in actions_payload:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized
    if not isinstance(actions_payload, dict):
        return normalized
    for round_name, value in actions_payload.items():
        for item in as_list(value):
            if isinstance(item, dict):
                enriched = dict(item)
                enriched.setdefault("round", round_name)
                normalized.append(enriched)
    return normalized


def representative_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -(item.get("citation_count") or 0),
        item.get("year") is None,
        -(item.get("year") or 0),
        normalize_whitespace(item.get("title")).casefold(),
    )


def time_window_sort_key(label: Any, search_result: dict[str, Any]) -> tuple[int, int, str]:
    normalized = normalize_whitespace(label)
    for index, window in enumerate(search_result.get("time_windows", [])):
        if isinstance(window, dict) and normalize_whitespace(window.get("label")) == normalized:
            return (0, index, normalized)
    if normalized == "outside_windows":
        return (1, 0, normalized)
    if normalized == "unknown":
        return (2, 0, normalized)
    return (3, 0, normalized)


def time_window_year_text(window: dict[str, Any]) -> str:
    start = window.get("start_year")
    end = window.get("end_year")
    if isinstance(start, int) and isinstance(end, int):
        return f"{start}-{end}"
    if isinstance(start, int):
        return f"{start}-"
    if isinstance(end, int):
        return f"up to {end}"
    return "undated"


def time_window_display_name(label: Any) -> str:
    text = normalize_whitespace(label).replace("_", "-").casefold()
    if "foundational" in text or "emergence" in text:
        return "Foundational Emergence"
    if "expansion" in text or "specialization" in text:
        return "Expansion and Specialization"
    if "recent" in text or "frontier" in text:
        return "Recent Frontier"
    parts = [part for part in re.split(r"[-\s]+", text) if part]
    if not parts:
        return "Undated Period"
    if len(normalize_whitespace(label)) >= 36 and len(parts) > 1:
        parts = parts[:-1]
    stop = {"of", "and", "the", "to", "in", "for"}
    words = [part if part in stop else part.capitalize() for part in parts[:6]]
    return " ".join(words)


def temporal_dominant_cells(evidence_map: dict[str, Any], label: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    normalized = normalize_whitespace(label)
    cells = [
        cell
        for cell in evidence_map.get("method_time_matrix", [])
        if isinstance(cell, dict) and normalize_whitespace(cell.get("time_bucket_label")) == normalized
    ]
    cells.sort(key=lambda item: (-(item.get("paper_count") or 0), normalize_whitespace(item.get("cluster_name"))))
    return cells[:limit]


def temporal_period_placeholder_title(label: Any, window: dict[str, Any] | None) -> str:
    normalized = normalize_whitespace(label)
    if normalized == "outside_windows":
        return "Outside the Main Time Windows"
    if normalized == "unknown":
        return "Undated or Weakly Dated Work"
    years = time_window_year_text(window or {})
    return f"Chronological Phase ({years})" if years and years != "undated" else "Chronological Phase"


def format_temporal_subsection_title(
    label: Any,
    window: dict[str, Any] | None,
    *,
    dominant_cells: list[dict[str, Any]] | None = None,
) -> str:
    del dominant_cells
    return temporal_period_placeholder_title(label, window)


def choose_top_clusters(evidence_map: dict[str, Any], *, max_clusters: int) -> list[dict[str, Any]]:
    clusters = [
        cluster
        for cluster in evidence_map.get("method_clusters", [])
        if isinstance(cluster, dict) and cluster.get("cluster_id") != "OUT" and (cluster.get("paper_count") or 0) > 0
    ]
    clusters.sort(key=lambda item: (-(item.get("paper_count") or 0), normalize_whitespace(item.get("name"))))
    return clusters[:max_clusters]


def build_bibliography(search_result: dict[str, Any], evidence_map: dict[str, Any]) -> list[dict[str, Any]]:
    cards_by_id = paper_cards_by_id(search_result)
    assignments = assignments_by_id(evidence_map)
    used_keys: set[str] = set()
    entries: list[dict[str, Any]] = []

    def make_cite_key(card: dict[str, Any], assignment: dict[str, Any]) -> str:
        authors = card.get("authors") if isinstance(card.get("authors"), list) else []
        first_author = title_case_name(authors[0]) if authors else "unknown"
        year = card.get("year") or assignment.get("year") or "nd"
        title_word = slugify(normalize_whitespace(card.get("title")).split(" ")[0] if normalize_whitespace(card.get("title")) else "paper")
        base = f"{slugify(first_author, fallback='unknown', max_chars=16)}{year}{title_word}"
        key = base
        suffix = 1
        while key in used_keys:
            suffix += 1
            key = f"{base}{suffix}"
        used_keys.add(key)
        return key

    for paper_id, assignment in assignments.items():
        card = cards_by_id.get(paper_id, {})
        authors = card.get("authors") if isinstance(card.get("authors"), list) else []
        year = card.get("year") or assignment.get("year")
        cite_key = make_cite_key(card, assignment)
        first_author = title_case_name(authors[0]) if authors else "Unknown"
        if len(authors) <= 1:
            author_display = first_author
        elif len(authors) == 2:
            author_display = f"{first_author} and {title_case_name(authors[1])}"
        else:
            author_display = f"{first_author} et al."
        venue = normalize_whitespace(card.get("venue"))
        doi = normalize_whitespace(card.get("doi"))
        url = normalize_whitespace(card.get("url"))
        entries.append(
            {
                "paper_id": paper_id,
                "cite_key": cite_key,
                "authors": authors,
                "author_display": author_display,
                "year": year,
                "title": normalize_whitespace(card.get("title") or assignment.get("title")),
                "venue": venue,
                "doi": doi,
                "url": url,
                "is_preprint": "arxiv" in venue.casefold() or "10.48550/arxiv" in doi.casefold(),
                "citation_label": f"{author_display}, {year}" if year else author_display,
                "role": assignment.get("role"),
                "method_cluster_id": assignment.get("method_cluster_id"),
                "method_cluster_name": assignment.get("method_cluster_name"),
                "time_bucket_label": assignment.get("time_bucket_label"),
                "citation_count": assignment.get("citation_count"),
                "source_actions": assignment.get("source_actions", []),
            }
        )
    entries.sort(
        key=lambda item: (
            item.get("year") is None,
            item.get("year") or 9999,
            normalize_whitespace(item.get("author_display")).casefold(),
            normalize_whitespace(item.get("title")).casefold(),
        )
    )
    return entries


def paper_payload_for_id(
    paper_id: str,
    *,
    biblio_by_id: dict[str, dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entry = biblio_by_id[paper_id]
    assignment = assignments.get(paper_id, {})
    card = cards.get(paper_id, {})
    return {
        "paper_id": paper_id,
        "cite_key": entry.get("cite_key"),
        "citation_label": entry.get("citation_label"),
        "title": entry.get("title"),
        "authors": entry.get("authors", [])[:6],
        "year": entry.get("year"),
        "venue": entry.get("venue"),
        "role": assignment.get("role"),
        "method_cluster_id": assignment.get("method_cluster_id"),
        "method_cluster_name": assignment.get("method_cluster_name"),
        "time_bucket_label": assignment.get("time_bucket_label"),
        "citation_count": assignment.get("citation_count"),
        "source_actions": assignment.get("source_actions", []),
        "source": card.get("source", []),
        "evidence_source_tier": card.get("evidence_source_tier", assignment.get("evidence_source_tier", "pipeline")),
        "expert_seed": bool(card.get("expert_seed") or assignment.get("expert_seed")),
        "expert_neighbor": bool(card.get("expert_neighbor") or assignment.get("expert_neighbor")),
        "source_seed_title": card.get("source_seed_title"),
        "source_seed_s2_paper_id": card.get("source_seed_s2_paper_id"),
        "abstract": truncate_text(card.get("abstract"), max_chars=1200),
    }


def bibliography_by_paper_id(bibliography: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["paper_id"]: item for item in bibliography if item.get("paper_id")}


def bibliography_by_cite_key(bibliography: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["cite_key"]: item for item in bibliography if item.get("cite_key")}


def compact_outline_context(
    search_result: dict[str, Any],
    evidence_map: dict[str, Any],
    bibliography: list[dict[str, Any]],
    *,
    max_clusters: int,
    max_representatives_per_cluster: int,
) -> dict[str, Any]:
    top_clusters = choose_top_clusters(evidence_map, max_clusters=max_clusters)
    biblio_by_id = bibliography_by_paper_id(bibliography)
    card_by_id = paper_cards_by_id(search_result)
    cluster_summaries = []
    for cluster in top_clusters:
        reps = []
        for paper_id in cluster.get("representative_paper_ids", [])[:max_representatives_per_cluster]:
            entry = biblio_by_id.get(normalize_whitespace(paper_id))
            if not entry:
                continue
            reps.append(
                {
                    "paper_id": entry["paper_id"],
                    "cite_key": entry["cite_key"],
                    "citation_label": entry["citation_label"],
                    "title": truncate_text(entry["title"], max_chars=160),
                    "year": entry["year"],
                    "role": entry["role"],
                    "evidence_source_tier": card_by_id.get(entry["paper_id"], {}).get("evidence_source_tier", "pipeline"),
                }
            )
        cluster_summaries.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "name": cluster.get("name"),
                "definition": truncate_text(cluster.get("definition"), max_chars=260),
                "paper_count": cluster.get("paper_count"),
                "role_counts": cluster.get("role_counts"),
                "representative_papers": reps,
            }
        )

    time_windows = []
    for window in search_result.get("time_windows", []):
        if isinstance(window, dict):
            time_windows.append(
                {
                    "label": window.get("label"),
                    "start_year": window.get("start_year"),
                    "end_year": window.get("end_year"),
                    "rationale": window.get("rationale"),
                }
            )

    return {
        "topic": search_result.get("topic"),
        "topic_profile": {
            "normalized_topic": (search_result.get("topic_profile") or {}).get("normalized_topic"),
            "scope_inclusions": as_list((search_result.get("topic_profile") or {}).get("scope_inclusions"))[:8],
            "scope_exclusions": as_list((search_result.get("topic_profile") or {}).get("scope_exclusions"))[:8],
            "key_facets": as_list((search_result.get("topic_profile") or {}).get("key_facets"))[:10],
        },
        "paper_count": evidence_map.get("paper_count"),
        "time_windows": time_windows,
        "method_clusters": cluster_summaries,
        "role_counts": {
            role: len(ids)
            for role, ids in evidence_map.get("role_groups", {}).items()
            if isinstance(ids, list)
        },
    }


def build_outline_prompt(
    context: dict[str, Any],
    *,
    family_limit: int,
    subject_domain: str,
    case_patch: str,
) -> str:
    family_limit_instruction = (
        f"If the topic clearly supports multiple major method or catalyst families, use {family_limit} or fewer of them as main sections when possible.\n"
        if family_limit > 0
        else "Choose a compact set of main sections and merge weakly supported method families when they do not sustain a full section.\n"
    )
    subject_rules = subject_outline_rule_block(subject_domain)
    case_rules = case_outline_rule_block(case_patch)
    return (
        "You are planning a formal academic literature review.\n"
        "Return strict JSON only.\n"
        "Do not invent papers or topics beyond the supplied evidence.\n"
        "First infer whether the topic is best treated as chemistry or biology/omics, then follow the corresponding field conventions when designing the outline.\n"
        "Use the field's own vocabulary and review logic. Choose section names that sound natural for that discipline.\n"
        "Keep the main scientific outline short, linear, and topic-specific. Real domain reviews usually have about 3 to 7 main scientific sections.\n"
        "Prefer one dominant organizing axis rather than several abstract survey axes at once.\n"
        f"{family_limit_instruction}"
        "Write one ordered `sections` list only. Put all main scientific sections directly in that list, including method or catalyst family sections when they belong in the main narrative.\n"
        "Do not split the outline into a separate family-section block.\n"
        "Do not generate subsection trees or coverage_bullets in the outline. Later stages will expand each section into evidence packs and drafts.\n\n"
        "Chemistry outline prior:\n"
        "- Organize around catalyst systems, reaction classes, mechanistic distinctions, scope expansion, and synthetic utility.\n"
        "- A good chemistry review often moves from platform to mechanism to optimization to scope to utility.\n"
        "- Natural chemistry section patterns include: Introduction; core catalytic system or main catalyst families; mechanistic understanding; catalyst variants or platform evolution; expanded substrate scope or reaction classes; practical synthetic considerations; summary and outlook.\n"
        "- If multiple catalyst families are equally central, they may appear as separate main sections in the body before any final mechanism/practicality synthesis.\n\n"
        "Biology / omics outline prior:\n"
        "- Organize around biological need, technology background, methods or computational analysis, biological applications, and limitations or artifacts.\n"
        "- A good biology review often moves from capability to interpretation to application to caution.\n"
        "- Natural biology section patterns include: Introduction; technology or conceptual background; methods or computational analysis; applications or biological discoveries; limitations, artifacts, or quality issues; conclusion and outlook.\n\n"
        "For any discipline:\n"
        "- Merge challenges, limitations, and outlook when that produces a cleaner review.\n"
        "- Only introduce a dedicated historical or temporal section when the evidence shows that chronology is itself a core scientific story.\n"
        "- Include a forward-looking section near the end of the review.\n"
        "- Its title does not need to be literally `Future Work`; choose a field-appropriate and topic-specific title such as `Future Work`, `Open Problems and Future Directions`, `Outlook`, or `Challenges and Opportunities` when suitable.\n"
        "- This forward-looking section should appear in one of the final positions of the outline, typically immediately before the Conclusion or integrated with a late-stage synthesis section.\n"
        "- The section should synthesize unresolved technical bottlenecks, credible research opportunities, methodological or evaluation gaps, and evidence-supported future directions rather than generic speculation.\n"
        "- For this section, provide concrete `purpose`, `must_cover`, `paper_ids`, and `cluster_ids` so later drafting stages can write it from evidence.\n"
        "- A short Conclusion section may follow this forward-looking section if that produces a cleaner review structure.\n"
        "- Only introduce abstract survey sections if they are genuinely natural for the field and topic.\n\n"
        f"{subject_rules}"
        f"{case_rules}"
        "Return this schema:\n"
        "{\n"
        '  "review_title": "...",\n'
        '  "review_subtitle": "...",\n'
        '  "one_sentence_summary": "...",\n'
        '  "abstract_outline": ["3-5 points that the abstract must cover"],\n'
        '  "method_family_section_count": 0,\n'
        '  "sections": [\n'
        '    {\n'
        '      "section_id": "intro",\n'
        '      "section_title": "1. Introduction",\n'
        '      "section_role": "introduction",\n'
        '      "purpose": "...",\n'
        '      "must_cover": ["..."],\n'
        '      "paper_ids": ["P001"],\n'
        '      "cluster_ids": ["C1"],\n'
        '      "notes": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Evidence summary JSON:\n"
        f"{json.dumps(context, ensure_ascii=False)}\n"
    )


def build_temporal_section(
    search_result: dict[str, Any],
    evidence_map: dict[str, Any],
) -> dict[str, Any]:
    time_windows = [item for item in search_result.get("time_windows", []) if isinstance(item, dict)]
    time_buckets = [item for item in evidence_map.get("time_buckets", []) if isinstance(item, dict)]
    bucket_by_label = {normalize_whitespace(item.get("label")): item for item in time_buckets}
    labels = [normalize_whitespace(item.get("label")) for item in time_windows if normalize_whitespace(item.get("label"))]
    for bucket in sorted(time_buckets, key=lambda item: time_window_sort_key(item.get("label"), search_result)):
        label = normalize_whitespace(bucket.get("label"))
        if label and label not in labels and label not in {"unknown", "outside_windows"}:
            labels.append(label)

    subsections: list[dict[str, Any]] = []
    for label in labels:
        window = next((item for item in time_windows if normalize_whitespace(item.get("label")) == label), None)
        bucket = bucket_by_label.get(label, {})
        title = temporal_period_placeholder_title(label, window)
        paper_count = bucket.get("paper_count") or len(bucket.get("paper_ids", []))
        subsections.append(
            {
                "subsection_id": f"time_{normalize_section_id(label)}",
                "subsection_title": title,
                "description": (
                    "Analyze this chronological phase of the topic: identify the period's dominant themes, representative works, "
                    "new method families or evaluation concerns, and how this phase changes the field trajectory."
                ),
                "coverage_bullets": [
                    {
                        "point": "Define the technical character of this period.",
                        "description": "Explain why this time window is a coherent phase and what research questions dominated it.",
                        "target_citation_count": 4,
                    },
                    {
                        "point": "Identify representative works and dominant themes.",
                        "description": "Cite original method papers, benchmarks, surveys, and frontier works from this period.",
                        "target_citation_count": 8,
                    },
                    {
                        "point": "Explain the transition into or out of this period.",
                        "description": "Describe what changed relative to adjacent phases and which problems motivated the next stage.",
                        "target_citation_count": 4,
                    },
                ],
                "paper_ids": bucket.get("representative_paper_ids", []),
                "time_bucket_label": label,
                "target_citation_count": min(14, max(8, int(paper_count or 0) // 4)),
            }
        )

    return {
        "section_id": TEMPORAL_SECTION_ID,
        "section_title": TEMPORAL_SECTION_TITLE,
        "section_role": "temporal",
        "description": (
            "Trace the topic's development across the search-derived time windows, showing how major themes, "
            "representative papers, and research priorities changed from one period to the next."
        ),
        "coverage_bullets": [
            {
                "point": "Period-by-period intellectual development of the topic.",
                "description": "Use time windows as the organizing spine, while explaining dominant method families and representative works within each period.",
                "target_citation_count": 12,
            }
        ],
        "subsections": subsections,
        "render_mode": "with_subsections",
        "render_as": "major_section",
        "paper_ids": [],
        "cluster_ids": [],
        "notes": "This fixed section is generated from lr_search time windows, time buckets, and the method-by-time matrix.",
    }


def enrich_outline_with_temporal_section(
    outline: dict[str, Any],
    search_result: dict[str, Any],
    evidence_map: dict[str, Any],
) -> dict[str, Any]:
    sections = [item for item in outline.get("sections", []) if isinstance(item, dict)]
    enriched_sections: list[dict[str, Any]] = []
    for section in sections:
        if classify_outline_section(section) == "temporal":
            temporal_section = build_temporal_section(search_result, evidence_map)
            enriched_sections.append({**section, **temporal_section})
        else:
            enriched_sections.append(section)
    enriched = dict(outline)
    enriched["sections"] = enriched_sections
    enriched["family_sections"] = []
    return enriched


def temporal_section_evidence_pack(
    search_result: dict[str, Any],
    evidence_map: dict[str, Any],
    bibliography: list[dict[str, Any]],
    *,
    max_papers: int,
) -> dict[str, Any]:
    biblio_by_id = bibliography_by_paper_id(bibliography)
    assignments = assignments_by_id(evidence_map)
    cards = paper_cards_by_id(search_result)
    windows_by_label = {
        normalize_whitespace(item.get("label")): item
        for item in search_result.get("time_windows", [])
        if isinstance(item, dict) and normalize_whitespace(item.get("label"))
    }
    buckets = [
        item
        for item in evidence_map.get("time_buckets", [])
        if isinstance(item, dict) and normalize_whitespace(item.get("label")) not in {"unknown", "outside_windows"}
    ]
    buckets.sort(key=lambda item: time_window_sort_key(item.get("label"), search_result))
    matrix_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in evidence_map.get("method_time_matrix", []):
        if not isinstance(cell, dict):
            continue
        label = normalize_whitespace(cell.get("time_bucket_label"))
        if label:
            matrix_by_label[label].append(cell)
    for cells in matrix_by_label.values():
        cells.sort(key=lambda item: (-(item.get("paper_count") or 0), normalize_whitespace(item.get("cluster_name"))))

    selected_ids: list[str] = []

    def add_paper_id(paper_id: Any) -> None:
        normalized = normalize_whitespace(paper_id)
        if normalized and normalized in biblio_by_id and normalized not in selected_ids:
            selected_ids.append(normalized)

    period_contexts: list[dict[str, Any]] = []
    previous_top_clusters: set[str] = set()
    for bucket in buckets:
        label = normalize_whitespace(bucket.get("label"))
        window = windows_by_label.get(label, {})
        cells = matrix_by_label.get(label, [])
        dominant_cells = cells[:5]
        period_ids: list[str] = []

        def add_period_id(paper_id: Any) -> None:
            normalized = normalize_whitespace(paper_id)
            if normalized and normalized in biblio_by_id and normalized not in period_ids:
                period_ids.append(normalized)
                add_paper_id(normalized)

        for paper_id in bucket.get("representative_paper_ids", [])[:8]:
            add_period_id(paper_id)
        for cell in dominant_cells:
            for paper_id in cell.get("representative_paper_ids", [])[:4]:
                add_period_id(paper_id)
        for paper_id in bucket.get("paper_ids", []):
            if len(period_ids) >= max_papers:
                break
            role = normalize_whitespace(assignments.get(normalize_whitespace(paper_id), {}).get("role"))
            if role in {"core_method", "benchmark_evaluation", "survey", "recent_frontier", "gap_refinement"}:
                add_period_id(paper_id)

        top_cluster_ids = {normalize_whitespace(cell.get("cluster_id")) for cell in dominant_cells if normalize_whitespace(cell.get("cluster_id"))}
        emerging_cluster_ids = [cluster_id for cluster_id in top_cluster_ids if cluster_id not in previous_top_clusters]
        previous_top_clusters = top_cluster_ids
        title = temporal_period_placeholder_title(label, window)
        period_contexts.append(
            {
                "label": label,
                "title": title,
                "start_year": window.get("start_year"),
                "end_year": window.get("end_year"),
                "rationale": window.get("rationale"),
                "paper_count": bucket.get("paper_count"),
                "role_counts": bucket.get("role_counts", {}),
                "method_counts": bucket.get("method_counts", {}),
                "dominant_method_cells": [
                    {
                        "cluster_id": cell.get("cluster_id"),
                        "cluster_name": cell.get("cluster_name"),
                        "paper_count": cell.get("paper_count"),
                        "role_counts": cell.get("role_counts", {}),
                        "representative_papers": [
                            paper_payload_for_id(
                                normalize_whitespace(paper_id),
                                biblio_by_id=biblio_by_id,
                                assignments=assignments,
                                cards=cards,
                            )
                            for paper_id in cell.get("representative_paper_ids", [])[:4]
                            if normalize_whitespace(paper_id) in biblio_by_id
                        ],
                    }
                    for cell in dominant_cells
                ],
                "emerging_cluster_ids": emerging_cluster_ids,
                "representative_papers": [
                    paper_payload_for_id(
                        paper_id,
                        biblio_by_id=biblio_by_id,
                        assignments=assignments,
                        cards=cards,
                    )
                    for paper_id in period_ids[:max_papers]
                ],
            }
        )

    selected_ids = selected_ids[: max(max_papers, len(period_contexts) * 8)]
    papers = [
        paper_payload_for_id(paper_id, biblio_by_id=biblio_by_id, assignments=assignments, cards=cards)
        for paper_id in selected_ids
    ]
    return {
        "topic": search_result.get("topic"),
        "section_id": TEMPORAL_SECTION_ID,
        "section_title": TEMPORAL_SECTION_TITLE,
        "section_role": "temporal",
        "prompt_style_version": TEMPORAL_SUBSECTION_PROMPT_STYLE_VERSION,
        "description": "Period-by-period synthesis of the topic's development using search-derived time windows.",
        "time_periods": period_contexts,
        "time_windows": [
            {
                "label": label,
                "start_year": window.get("start_year"),
                "end_year": window.get("end_year"),
                "rationale": window.get("rationale"),
            }
            for label, window in windows_by_label.items()
        ],
        "papers": papers,
    }


def section_evidence_pack(
    section: dict[str, Any],
    search_result: dict[str, Any],
    evidence_map: dict[str, Any],
    bibliography: list[dict[str, Any]],
    *,
    max_papers: int,
) -> dict[str, Any]:
    biblio_by_id = bibliography_by_paper_id(bibliography)
    assignments = assignments_by_id(evidence_map)
    cards = paper_cards_by_id(search_result)
    if normalize_section_id(section.get("section_id")) == TEMPORAL_SECTION_ID:
        return temporal_section_evidence_pack(
            search_result,
            evidence_map,
            bibliography,
            max_papers=max_papers,
        )
    search_clusters = {
        normalize_whitespace(item.get("cluster_id")): item
        for item in search_result.get("method_clusters", [])
        if isinstance(item, dict) and normalize_whitespace(item.get("cluster_id"))
    }
    organized_clusters = {
        normalize_whitespace(item.get("cluster_id")): item
        for item in evidence_map.get("method_clusters", [])
        if isinstance(item, dict) and normalize_whitespace(item.get("cluster_id"))
    }
    retained_cluster_paper_ids = {
        normalize_whitespace(item)
        for item in evidence_map.get("retained_cluster_paper_ids", [])
        if normalize_whitespace(item)
    }
    excluded_outlier_paper_ids = {
        normalize_whitespace(item)
        for item in evidence_map.get("excluded_outlier_paper_ids", [])
        if normalize_whitespace(item)
    }
    expert_seed_ids = {
        normalize_whitespace(card.get("paper_id"))
        for card in search_result.get("paper_cards", [])
        if isinstance(card, dict)
        and normalize_whitespace(card.get("paper_id"))
        and normalize_whitespace(card.get("evidence_source_tier")) == "expert_seed"
    }
    allowed_review_paper_ids = (retained_cluster_paper_ids | expert_seed_ids) - excluded_outlier_paper_ids
    time_windows = {
        normalize_whitespace(item.get("label")): item
        for item in search_result.get("time_windows", [])
        if isinstance(item, dict) and normalize_whitespace(item.get("label"))
    }

    selected_ids: list[str] = []
    bucket_ids: dict[str, list[str]] = defaultdict(list)

    def add_paper_id(paper_id: Any, *, bucket: str | None = None) -> None:
        normalized = normalize_whitespace(paper_id)
        if normalized and allowed_review_paper_ids and normalized not in allowed_review_paper_ids:
            return
        if normalized and normalized in biblio_by_id and normalized not in selected_ids:
            selected_ids.append(normalized)
        if normalized and normalized in biblio_by_id and bucket:
            if normalized not in bucket_ids[bucket]:
                bucket_ids[bucket].append(normalized)

    def paper_payload(paper_id: str) -> dict[str, Any]:
        return paper_payload_for_id(paper_id, biblio_by_id=biblio_by_id, assignments=assignments, cards=cards)

    for paper_id in section.get("paper_ids", []):
        add_paper_id(paper_id, bucket="seed_papers")

    section_cluster_ids = set(normalize_whitespace(x) for x in section.get("cluster_ids", []))
    if not section_cluster_ids and normalize_whitespace(section.get("cluster_id")):
        section_cluster_ids.add(normalize_whitespace(section.get("cluster_id")))

    if section_cluster_ids:
        for cluster in evidence_map.get("method_clusters", []):
            if not isinstance(cluster, dict):
                continue
            if normalize_whitespace(cluster.get("cluster_id")) in section_cluster_ids:
                for paper_id in cluster.get("representative_paper_ids", []):
                    add_paper_id(paper_id, bucket="representative_method_papers")
                for paper_id in cluster.get("paper_ids", [])[: max_papers * 2]:
                    role = normalize_whitespace(assignments.get(normalize_whitespace(paper_id), {}).get("role"))
                    if role == "core_method":
                        add_paper_id(paper_id, bucket="representative_method_papers")
                    elif role == "benchmark_evaluation":
                        add_paper_id(paper_id, bucket="benchmark_or_evaluation_papers")
                    elif role == "survey":
                        add_paper_id(paper_id, bucket="survey_or_framing_papers")
                    else:
                        add_paper_id(paper_id, bucket="supporting_or_boundary_papers")

    for role in ("benchmark_evaluation",):
        for paper_id in evidence_map.get("role_groups", {}).get(role, [])[: max(4, max_papers // 4)]:
            add_paper_id(paper_id, bucket="benchmark_or_evaluation_papers")

    if normalize_section_id(section.get("section_id")).startswith("intro"):
        for paper_id in evidence_map.get("role_groups", {}).get("survey", [])[:4]:
            add_paper_id(paper_id, bucket="survey_or_framing_papers")
    if normalize_section_id(section.get("section_id")) in {"taxonomy", "open", "background"}:
        for paper_id in evidence_map.get("role_groups", {}).get("survey", [])[:3]:
            add_paper_id(paper_id, bucket="survey_or_framing_papers")

    for cluster_id in section_cluster_ids:
        cluster = organized_clusters.get(cluster_id, {})
        for paper_id in cluster.get("representative_paper_ids", [])[:2]:
            add_paper_id(paper_id, bucket="foundational_papers")

    selected_ids = selected_ids[:max_papers]
    selected_id_set = set(selected_ids)
    papers = [paper_payload(paper_id) for paper_id in selected_ids]

    grouped_buckets: dict[str, list[dict[str, Any]]] = {}
    for bucket_name, paper_ids in bucket_ids.items():
        kept = [paper_payload(paper_id) for paper_id in paper_ids if paper_id in selected_id_set]
        if kept:
            grouped_buckets[bucket_name] = kept

    time_slice_groups: dict[str, list[dict[str, Any]]] = {}
    for paper_id in selected_ids:
        time_label = normalize_whitespace(assignments.get(paper_id, {}).get("time_bucket_label"))
        if not time_label:
            continue
        time_slice_groups.setdefault(time_label, []).append(paper_payload(paper_id))

    cluster_context = []
    for cluster_id in section_cluster_ids:
        cluster_search = search_clusters.get(cluster_id, {})
        cluster_org = organized_clusters.get(cluster_id, {})
        if not cluster_search and not cluster_org:
            continue
        cluster_context.append(
            {
                "cluster_id": cluster_id,
                "name": cluster_org.get("name") or cluster_search.get("name"),
                "definition": cluster_search.get("definition") or cluster_org.get("definition"),
                "distinguishing_features": cluster_search.get("distinguishing_features", []),
                "missing_signals": cluster_org.get("missing_signals") or cluster_search.get("missing_signals", []),
                "paper_count": cluster_org.get("paper_count"),
                "role_counts": cluster_org.get("role_counts", {}),
                "time_counts": cluster_org.get("time_counts", {}),
            }
        )

    return {
        "topic": search_result.get("topic"),
        "section_id": section.get("section_id"),
        "section_title": section.get("section_title"),
        "purpose": section.get("purpose"),
        "must_cover": section.get("must_cover", []),
        "notes": section.get("notes"),
        "cluster_ids": list(section_cluster_ids),
        "cluster_context": cluster_context,
        "time_window_context": [
            {
                "label": label,
                "start_year": time_windows[label].get("start_year"),
                "end_year": time_windows[label].get("end_year"),
                "rationale": time_windows[label].get("rationale"),
                "papers": payloads,
            }
            for label, payloads in time_slice_groups.items()
            if label in time_windows
        ],
        "foundational_papers": grouped_buckets.get("foundational_papers", []),
        "representative_method_papers": grouped_buckets.get("representative_method_papers", []),
        "benchmark_or_evaluation_papers": grouped_buckets.get("benchmark_or_evaluation_papers", []),
        "survey_or_framing_papers": grouped_buckets.get("survey_or_framing_papers", []),
        "supporting_or_boundary_papers": grouped_buckets.get("supporting_or_boundary_papers", []),
        "seed_papers": grouped_buckets.get("seed_papers", []),
        "papers": papers,
    }


def normalize_subsections(section: dict[str, Any]) -> list[dict[str, Any]]:
    if normalize_section_id(section.get("section_id")) == TEMPORAL_SECTION_ID and section.get("time_periods"):
        normalized = []
        for index, period in enumerate([item for item in section.get("time_periods", []) if isinstance(item, dict)], start=1):
            label = normalize_whitespace(period.get("label")) or f"period_{index}"
            normalized.append(
                {
                    "subsection_id": f"time_{normalize_section_id(label)}",
                    "subsection_title": normalize_whitespace(period.get("title")) or temporal_period_placeholder_title(label, period),
                    "description": (
                        f"Analyze this chronological phase: {normalize_whitespace(period.get('rationale'))}"
                        if normalize_whitespace(period.get("rationale"))
                        else "Analyze this chronological phase of the topic."
                    ),
                    "coverage_bullets": [
                        {
                            "point": "Characterize the period's dominant research themes.",
                            "description": "Explain what problems, methods, or evaluation concerns defined this period.",
                            "target_citation_count": 5,
                        },
                        {
                            "point": "Discuss representative works from this period.",
                            "description": "Use multiple citations from the supplied period evidence, prioritizing original method and benchmark papers.",
                            "target_citation_count": 8,
                        },
                        {
                            "point": "Explain how the period changed the topic's trajectory.",
                            "description": "Compare with adjacent periods and identify transitions, maturation, or emerging directions.",
                            "target_citation_count": 5,
                        },
                    ],
                    "paper_ids": [
                        normalize_whitespace(paper.get("paper_id"))
                        for paper in period.get("representative_papers", [])
                        if isinstance(paper, dict) and normalize_whitespace(paper.get("paper_id"))
                    ],
                    "cluster_ids": [],
                    "time_bucket_label": label,
                    "hidden_title": False,
                }
            )
        return normalized

    raw_subsections = [item for item in section.get("subsections", []) if isinstance(item, dict)]
    if not raw_subsections or normalize_whitespace(section.get("render_mode")) == "section_body":
        return [
            {
                "subsection_id": f"{normalize_section_id(section.get('section_id'))}_body",
                "subsection_title": strip_section_number(section.get("section_title")) or "Section Body",
                "description": normalize_whitespace(section.get("description") or section.get("purpose")),
                "coverage_bullets": coverage_bullets_from_section(section),
                "paper_ids": as_list(section.get("paper_ids")),
                "cluster_ids": as_list(section.get("cluster_ids")),
                "hidden_title": True,
            }
        ]

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_subsections, start=1):
        title = strip_section_number(item.get("subsection_title") or item.get("title")) or f"Subsection {index}"
        subsection_id = (
            normalize_section_id(item.get("subsection_id"))
            or make_section_id(title, fallback=f"{normalize_section_id(section.get('section_id'))}_sub_{index}")
        )
        base_id = subsection_id
        suffix = 2
        while subsection_id in seen_ids:
            subsection_id = f"{base_id}_{suffix}"
            suffix += 1
        seen_ids.add(subsection_id)
        enriched = dict(item)
        enriched["subsection_id"] = subsection_id
        enriched["subsection_title"] = title
        enriched.setdefault("description", normalize_whitespace(item.get("description")))
        enriched["coverage_bullets"] = coverage_bullets_from_section(enriched)
        enriched.setdefault("paper_ids", [])
        enriched.setdefault("cluster_ids", [])
        enriched["hidden_title"] = False
        normalized.append(enriched)
    return normalized


def words_in_markdown(text: Any) -> int:
    return len(re.findall(r"\b[\w'-]+\b", preserve_markdown_block(text)))


def cite_keys_in_text(text: Any) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\[@([^\]]+)\]", str(text or "")):
        for part in re.split(r"[;,]", normalize_whitespace(match.group(1))):
            key = normalize_whitespace(part).lstrip("@")
            if key and key not in keys:
                keys.append(key)
    return keys


def build_paper_mounts(
    outline: dict[str, Any],
    section_packs: list[dict[str, Any]],
    bibliography: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cite_by_paper = bibliography_by_paper_id(bibliography)
    mounts: list[dict[str, Any]] = []
    pack_by_section = {normalize_section_id(pack.get("section_id")): pack for pack in section_packs}
    for section in [item for item in outline.get("sections", []) if isinstance(item, dict)]:
        section_id = normalize_section_id(section.get("section_id"))
        pack = pack_by_section.get(section_id, {})
        section_cluster_ids = set(normalize_whitespace(x) for x in as_list(section.get("cluster_ids")) if normalize_whitespace(x))
        for subsection in normalize_subsections(section):
            subsection_id = normalize_section_id(subsection.get("subsection_id"))
            temporal_label = normalize_whitespace(subsection.get("time_bucket_label"))
            subsection_cluster_ids = set(normalize_whitespace(x) for x in as_list(subsection.get("cluster_ids")) if normalize_whitespace(x))
            seed_ids = set(normalize_whitespace(x) for x in as_list(subsection.get("paper_ids")) if normalize_whitespace(x))
            active_cluster_ids = subsection_cluster_ids or section_cluster_ids
            for paper in pack.get("papers", []):
                paper_id = normalize_whitespace(paper.get("paper_id"))
                if not paper_id or paper_id not in cite_by_paper:
                    continue
                paper_cluster_id = normalize_whitespace(paper.get("method_cluster_id"))
                if section_id == TEMPORAL_SECTION_ID and temporal_label:
                    if normalize_whitespace(paper.get("time_bucket_label")) != temporal_label:
                        continue
                    reason = "time_bucket"
                elif seed_ids and paper_id in seed_ids:
                    reason = "seed"
                elif active_cluster_ids and paper_cluster_id in active_cluster_ids:
                    reason = "cluster"
                elif not subsection_cluster_ids:
                    reason = "section_pack"
                else:
                    continue
                role = normalize_whitespace(paper.get("role"))
                support_type = "method"
                if role == "benchmark_evaluation":
                    support_type = "benchmark"
                elif role == "survey":
                    support_type = "definition"
                elif role in {"gap_refinement", "recent_frontier"}:
                    support_type = "future_direction"
                mounts.append(
                    {
                        "paper_id": paper_id,
                        "cite_key": cite_by_paper[paper_id].get("cite_key"),
                        "section_id": section_id,
                        "subsection_id": subsection_id,
                        "evidence_role": role or "supporting",
                        "support_type": support_type,
                        "key_information": truncate_text(paper.get("abstract"), max_chars=360),
                        "match_reason": reason,
                    }
                )
    return mounts


def citation_groups_from_papers(papers: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups = {
        "expert_seed_anchors": [],
        "expert_recommendations": [],
        "foundational": [],
        "representative_methods": [],
        "benchmarks": [],
        "recent_frontier": [],
        "limitations": [],
        "survey_framing": [],
    }
    for paper in papers:
        cite_key = normalize_whitespace(paper.get("cite_key"))
        if not cite_key:
            continue
        tier = normalize_whitespace(paper.get("evidence_source_tier"))
        if tier == "expert_seed":
            groups["expert_seed_anchors"].append(cite_key)
        elif tier == "expert_recommendation":
            groups["expert_recommendations"].append(cite_key)
        role = normalize_whitespace(paper.get("role"))
        if role == "benchmark_evaluation":
            groups["benchmarks"].append(cite_key)
        elif role == "survey":
            groups["survey_framing"].append(cite_key)
        elif role in {"recent_frontier", "gap_refinement"}:
            groups["recent_frontier"].append(cite_key)
        elif role == "core_method":
            groups["representative_methods"].append(cite_key)
        else:
            groups["limitations"].append(cite_key)
    for group_name, values in groups.items():
        groups[group_name] = list(dict.fromkeys(values))
    return groups


def build_temporal_subsection_pack(
    section: dict[str, Any],
    subsection: dict[str, Any],
    section_pack: dict[str, Any],
    *,
    max_papers: int,
    target_citations: int,
) -> dict[str, Any]:
    label = normalize_whitespace(subsection.get("time_bucket_label"))
    period = next(
        (
            item
            for item in section_pack.get("time_periods", [])
            if isinstance(item, dict) and normalize_whitespace(item.get("label")) == label
        ),
        {},
    )
    selected: list[dict[str, Any]] = []

    def add(paper: dict[str, Any]) -> None:
        paper_id = normalize_whitespace(paper.get("paper_id"))
        if paper_id and all(normalize_whitespace(existing.get("paper_id")) != paper_id for existing in selected):
            selected.append(paper)

    for cell in period.get("dominant_method_cells", []):
        if not isinstance(cell, dict):
            continue
        for paper in cell.get("representative_papers", []):
            if isinstance(paper, dict):
                add(paper)
    for paper in period.get("representative_papers", []):
        if isinstance(paper, dict):
            add(paper)
    if len(selected) < max(8, target_citations):
        for paper in section_pack.get("papers", []):
            if normalize_whitespace(paper.get("time_bucket_label")) == label:
                add(paper)
            if len(selected) >= max_papers:
                break

    selected = selected[:max_papers]
    allowed_cite_keys = [normalize_whitespace(paper.get("cite_key")) for paper in selected if normalize_whitespace(paper.get("cite_key"))]
    citation_groups = citation_groups_from_papers(selected)
    period_representatives = [
        normalize_whitespace(paper.get("cite_key"))
        for paper in period.get("representative_papers", [])[: max(4, target_citations // 2)]
        if isinstance(paper, dict) and normalize_whitespace(paper.get("cite_key")) in allowed_cite_keys
    ]
    dominant_representatives: list[str] = []
    for cell in period.get("dominant_method_cells", []):
        if not isinstance(cell, dict):
            continue
        for paper in cell.get("representative_papers", [])[:2]:
            if isinstance(paper, dict):
                cite_key = normalize_whitespace(paper.get("cite_key"))
                if cite_key and cite_key in allowed_cite_keys and cite_key not in dominant_representatives:
                    dominant_representatives.append(cite_key)
    must_use = list(dict.fromkeys(period_representatives + dominant_representatives))[:target_citations]
    return {
        "topic": section_pack.get("topic"),
        "section_id": TEMPORAL_SECTION_ID,
        "section_title": TEMPORAL_SECTION_TITLE,
        "section_role": "temporal",
        "prompt_style_version": section_pack.get("prompt_style_version") or TEMPORAL_SUBSECTION_PROMPT_STYLE_VERSION,
        "subsection_id": subsection.get("subsection_id"),
        "subsection_title": temporal_period_placeholder_title(label, period),
        "hidden_title": False,
        "description": normalize_whitespace(subsection.get("description") or period.get("rationale")),
        "coverage_bullets": coverage_bullets_from_section(subsection),
        "target_word_count": section_pack.get("target_word_count"),
        "target_citation_count": target_citations,
        "time_bucket_label": label,
        "time_period": period,
        "llm_should_generate_title": True,
        "allowed_cite_keys": allowed_cite_keys,
        "citation_plan": {
            "section_id": TEMPORAL_SECTION_ID,
            "subsection_id": subsection.get("subsection_id"),
            "target_citation_count": min(target_citations, len(allowed_cite_keys)),
            "must_use_cite_keys": must_use,
            "should_use_cite_keys": allowed_cite_keys[: max(target_citations * 2, target_citations)],
            "citation_groups": citation_groups,
        },
        "papers": selected,
    }


def build_subsection_pack(
    section: dict[str, Any],
    subsection: dict[str, Any],
    section_pack: dict[str, Any],
    *,
    max_papers: int,
    target_citations: int,
) -> dict[str, Any]:
    if normalize_section_id(section.get("section_id")) == TEMPORAL_SECTION_ID:
        return build_temporal_subsection_pack(
            section,
            subsection,
            section_pack,
            max_papers=max_papers,
            target_citations=target_citations,
        )

    section_cluster_ids = set(normalize_whitespace(x) for x in as_list(section.get("cluster_ids")) if normalize_whitespace(x))
    subsection_cluster_ids = set(normalize_whitespace(x) for x in as_list(subsection.get("cluster_ids")) if normalize_whitespace(x))
    active_cluster_ids = subsection_cluster_ids or section_cluster_ids
    seed_ids = set(normalize_whitespace(x) for x in as_list(subsection.get("paper_ids")) if normalize_whitespace(x))
    selected: list[dict[str, Any]] = []

    def add(paper: dict[str, Any]) -> None:
        paper_id = normalize_whitespace(paper.get("paper_id"))
        if paper_id and all(normalize_whitespace(existing.get("paper_id")) != paper_id for existing in selected):
            selected.append(paper)

    for paper in section_pack.get("papers", []):
        paper_id = normalize_whitespace(paper.get("paper_id"))
        paper_cluster_id = normalize_whitespace(paper.get("method_cluster_id"))
        if paper_id in seed_ids or (active_cluster_ids and paper_cluster_id in active_cluster_ids):
            add(paper)

    if len(selected) < max(8, target_citations):
        for bucket_name in (
            "representative_method_papers",
            "benchmark_or_evaluation_papers",
            "foundational_papers",
            "survey_or_framing_papers",
            "supporting_or_boundary_papers",
            "papers",
        ):
            for paper in section_pack.get(bucket_name, []):
                add(paper)
                if len(selected) >= max_papers:
                    break
            if len(selected) >= max_papers:
                break

    selected = selected[:max_papers]
    allowed_cite_keys = [normalize_whitespace(paper.get("cite_key")) for paper in selected if normalize_whitespace(paper.get("cite_key"))]
    coverage_bullets = coverage_bullets_from_section(subsection) or coverage_bullets_from_section(section)
    citation_groups = citation_groups_from_papers(selected)
    must_use: list[str] = []
    for group_name in ("expert_seed_anchors", "expert_recommendations", "representative_methods", "benchmarks", "recent_frontier"):
        for cite_key in citation_groups[group_name][: max(2, target_citations // 3)]:
            if cite_key not in must_use:
                must_use.append(cite_key)
    must_use = [key for key in must_use if key in allowed_cite_keys][:target_citations]
    return {
        "topic": section_pack.get("topic"),
        "section_id": section.get("section_id"),
        "section_title": section.get("section_title"),
        "section_role": section.get("section_role"),
        "prompt_style_version": subsection_style_for_pack(
            {
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_role": section.get("section_role"),
            }
        ),
        "subsection_id": subsection.get("subsection_id"),
        "subsection_title": subsection.get("subsection_title"),
        "hidden_title": subsection.get("hidden_title", False),
        "description": normalize_whitespace(subsection.get("description") or section.get("description")),
        "coverage_bullets": coverage_bullets,
        "target_word_count": section_pack.get("target_word_count"),
        "target_citation_count": target_citations,
        "allowed_cite_keys": allowed_cite_keys,
        "citation_plan": {
            "section_id": section.get("section_id"),
            "subsection_id": subsection.get("subsection_id"),
            "target_citation_count": min(target_citations, len(allowed_cite_keys)),
            "must_use_cite_keys": must_use,
            "should_use_cite_keys": allowed_cite_keys[: max(target_citations * 2, target_citations)],
            "citation_groups": citation_groups,
        },
        "papers": selected,
    }


def build_section_prompt(section_pack: dict[str, Any]) -> str:
    section_id = normalize_whitespace(section_pack.get("section_id")).casefold()
    is_method_family = section_id.startswith("family_")
    structure_hint = (
        "This is a method-family section. Synthesize the family at the level of ideas and sub-branches, "
        "not one paragraph per paper. Explain core idea, representative lines of work, strengths, limitations, "
        "and relation to neighboring families."
        if is_method_family
        else "Write this section as polished academic prose with clear internal flow. Avoid bullet-list dumping."
    )
    markdown_hint = (
        "Use polished Markdown for readability. Prefer prose, but avoid walls of plain text. "
        "Use a few short `###` subheadings where helpful. Use `**bold lead-ins**` to anchor key moves. "
        "Use compact bullet lists only when they improve scanability. Tables are allowed only for true comparisons.\n"
    )
    content_hint = (
        "Make the section content more substantial than a generic overview. "
        "Each section should clearly explain distinctions, tradeoffs, and representative patterns grounded in the evidence.\n"
    )
    citation_hint = (
        "Prefer original method papers, benchmark papers, or canonical technical papers as primary support. "
        "Use survey papers sparingly, mainly for framing or broad context, not as the main evidence for method claims.\n"
    )
    return (
        "You are writing one section of a formal academic literature review.\n"
        "Return strict JSON only.\n"
        "Use only the supplied evidence.\n"
        "Do not invent papers, venues, or claims.\n"
        "Do not mention pipeline internals, clustering procedures, or evidence-map artifacts.\n"
        "Use inline citation placeholders in the form [@cite_key].\n"
        f"{structure_hint}\n\n"
        f"{markdown_hint}"
        f"{content_hint}\n"
        f"{citation_hint}\n"
        "Return this schema:\n"
        "{\n"
        '  "section_title": "...",\n'
        '  "section_summary": "...",\n'
        '  "section_text": "well-structured Markdown section body using paragraphs, optional ### subheadings, optional compact lists, and [@cite_key] citations",\n'
        '  "used_cite_keys": ["smith2024chain"]\n'
        "}\n\n"
        "Section evidence pack JSON:\n"
        f"{json.dumps(section_pack, ensure_ascii=False)}\n"
    )


def build_subsection_prompt(
    subsection_pack: dict[str, Any],
    *,
    target_words: int,
    target_citations: int,
    subject_domain: str,
    case_patch: str,
) -> str:
    if normalize_section_id(subsection_pack.get("section_id")) == TEMPORAL_SECTION_ID:
        return build_temporal_subsection_prompt(
            subsection_pack,
            target_words=target_words,
            target_citations=target_citations,
            subject_domain=subject_domain,
            case_patch=case_patch,
        )
    style_version = normalize_whitespace(subsection_pack.get("prompt_style_version"))
    if style_version == METHOD_SUBSECTION_PROMPT_STYLE_VERSION:
        structure_instruction = (
            "Maintain polished academic prose with clear transitions. Do not add a Markdown heading for the subsection title. "
            "Use 3-6 coherent paragraphs rather than a flat paper-by-paper list. Use concise bold lead-ins selectively when they clarify "
            "parallel method categories, technical mechanisms, benchmark families, limitations, or transition points, for example "
            "`**Search-based decomposition.** ...`. Do not force a bold lead-in on every paragraph; use it only when the paragraph's function "
            "is categorical or thesis-like. Avoid generic labels such as `**Overview.**`, `**Methods.**`, `**Evaluation.**`, or `**Conclusion.**`.\n\n"
        )
    else:
        structure_instruction = (
            "Maintain polished academic prose with clear transitions. Do not add a Markdown heading for the subsection title. "
            "Write in natural survey prose with 3-5 coherent paragraphs. Do not begin paragraphs with bold lead-ins. "
            "For introduction, background, evaluation framing, and conclusion-style sections, use ordinary topic sentences and narrative synthesis; "
            "bold emphasis may appear inside a sentence only when it is genuinely needed for a key term.\n\n"
        )
    general_rules = general_subsection_rule_block()
    subject_rules = subject_subsection_rule_block(subject_domain)
    case_rules = case_subsection_rule_block(case_patch)
    return (
        "You are writing one subsection of a formal academic survey.\n"
        "Return strict JSON only.\n"
        "Use only the supplied evidence pack. Do not invent papers, venues, datasets, results, or claims.\n"
        "Do not mention pipeline internals, retrieval, clustering, packs, or audits.\n"
        "Use inline citation placeholders exactly as [@cite_key]. Cite only keys in `allowed_cite_keys`.\n"
        "When choosing evidence, use `evidence_source_tier` as a soft priority signal: treat `expert_seed` papers as anchor works and preferential representative examples when they fit the subsection topic; treat `expert_recommendation` papers as expert-neighborhood support, stronger than ordinary retrieved papers but still subordinate to topical fit and evidence quality; use ordinary pipeline papers for breadth, chronology, comparison, and coverage.\n"
        f"Write approximately {target_words} words. If the evidence pack is small, stay shorter rather than inventing support.\n"
        f"Use {target_citations} distinct cite keys when enough evidence is available; use at least 6 for method/evaluation subsections with sufficient papers.\n"
        "Address every coverage bullet. Map each substantial claim to citations. Prefer original method, benchmark, and recent technical papers; "
        "use survey papers mainly for framing. Synthesize and compare approaches rather than writing one paragraph per paper.\n"
        f"{general_rules}"
        f"{subject_rules}"
        f"{case_rules}"
        f"{structure_instruction}"
        "Return this schema:\n"
        "{\n"
        '  "section_id": "...",\n'
        '  "subsection_id": "...",\n'
        '  "subsection_title": "...",\n'
        '  "subsection_summary": "...",\n'
        '  "subsection_text": "polished Markdown prose using [@cite_key] placeholders",\n'
        '  "used_cite_keys": ["smith2024chain"],\n'
        '  "unused_must_use_cite_keys": [{"cite_key": "...", "reason": "..."}],\n'
        '  "coverage_bullet_status": [{"point": "...", "covered": true, "cite_keys": ["smith2024chain"]}]\n'
        "}\n\n"
        "Subsection evidence pack JSON:\n"
        f"{json.dumps(subsection_pack, ensure_ascii=False)}\n"
    )


def build_temporal_subsection_prompt(
    subsection_pack: dict[str, Any],
    *,
    target_words: int,
    target_citations: int,
    subject_domain: str,
    case_patch: str,
) -> str:
    period = subsection_pack.get("time_period", {}) if isinstance(subsection_pack.get("time_period"), dict) else {}
    start_year = period.get("start_year")
    end_year = period.get("end_year")
    years_text = time_window_year_text(period) if period else "undated"
    general_rules = general_subsection_rule_block()
    subject_rules = subject_subsection_rule_block(subject_domain)
    case_rules = case_subsection_rule_block(case_patch)
    return (
        "You are writing one chronological phase subsection of a formal academic literature review.\n"
        "Return strict JSON only.\n"
        "Use only the supplied evidence pack. Do not invent papers, venues, datasets, results, or claims.\n"
        "Do not mention pipeline internals, retrieval, clustering, packs, audits, or search actions.\n"
        "Use inline citation placeholders exactly as [@cite_key]. Cite only keys in `allowed_cite_keys`.\n"
        "When choosing representative works for the period, use `evidence_source_tier` as a soft priority signal: `expert_seed` papers are anchor works when they fall in this period and match the technical theme; `expert_recommendation` papers are expert-neighborhood support; ordinary pipeline papers remain important for breadth and transitions.\n"
        f"Write approximately {target_words} words. This subsection should be substantive and citation-rich when enough evidence is available.\n"
        f"Use {target_citations} distinct cite keys when possible; use at least 8 if the pack provides enough papers.\n\n"
        "You must generate a domain-appropriate subsection title for this chronological phase and return it in `subsection_title`. "
        "The title should be specific to the topic and evidence in this period, not a generic era label and not a recycled title from another field. "
        "Do not use stock phrases such as `Prompting-Based Reasoning Foundations`, `Specialized Reasoning Methods and Evaluation`, "
        "`Foundational Emergence`, `Expansion`, or `Recent Era` unless the supplied evidence is literally about those topics. "
        "Use the time window, dominant method cells, representative papers, and period rationale as signals for naming the phase. "
        f"Include the year span `({years_text})` at the end of the title when the years are known.\n"
        "Organize the prose around the time period and the title you generate. Make the years explicit in the opening sentence. "
        "Explain the period as a phase in the field's development, not as a paper-by-paper list. Cover: "
        "1) the dominant research themes of the period; "
        "2) representative works and why they mattered; "
        "3) which method families, benchmarks, or evaluation concerns became prominent; "
        "4) how this period differs from the previous phase or sets up the next phase.\n"
        "Prefer original method papers, benchmark papers, and frontier papers as evidence. Use surveys mainly for framing. "
        "Synthesize transitions and intellectual momentum with professional survey prose.\n"
        "Every paragraph must begin with a concise bold lead-in followed by a short topic sentence, for example "
        "`**Chain-of-thought as the organizing primitive.** The 2020-2022 period ...`. "
        "Use 4-6 paragraphs. Each paragraph should have a distinct function: phase framing, representative methods, evaluation/benchmarks, transition/limitations, and optionally a closing synthesis. "
        "Avoid generic lead-ins such as `**Overview.**`, `**Methods.**`, `**Evaluation.**`, or `**Conclusion.**`; make each lead-in name a concrete technical theme, method family, benchmark turn, limitation, or transition.\n\n"
        f"{general_rules}"
        f"{subject_rules}"
        f"{case_rules}"
        "Return this schema:\n"
        "{\n"
        '  "section_id": "temporal_development",\n'
        '  "subsection_id": "...",\n'
        '  "subsection_title": "...",\n'
        '  "subsection_summary": "...",\n'
        '  "subsection_text": "polished chronological survey prose using [@cite_key] placeholders",\n'
        '  "used_cite_keys": ["smith2024chain"],\n'
        '  "unused_must_use_cite_keys": [{"cite_key": "...", "reason": "..."}],\n'
        '  "coverage_bullet_status": [{"point": "...", "covered": true, "cite_keys": ["smith2024chain"]}]\n'
        "}\n\n"
        "Temporal subsection evidence pack JSON:\n"
        f"{json.dumps(subsection_pack, ensure_ascii=False)}\n"
    )


def build_integration_metadata_prompt(
    title: str,
    subtitle: str,
    one_sentence_summary: str,
    abstract_outline: list[Any],
    outline: dict[str, Any],
    drafted_sections: list[dict[str, Any]],
    *,
    case_patch: str,
) -> str:
    compact_sections = [
        {
            "section_id": item.get("section_id"),
            "section_title": item.get("section_title"),
            "section_summary": item.get("section_summary"),
            "subsection_summaries": [
                {
                    "subsection_id": sub.get("subsection_id"),
                    "subsection_title": sub.get("subsection_title"),
                    "subsection_summary": sub.get("subsection_summary"),
                    "used_cite_keys": sub.get("used_cite_keys", []),
                }
                for sub in item.get("subsections", [])
                if isinstance(sub, dict)
            ],
            "used_cite_keys": item.get("used_cite_keys", []),
        }
        for item in drafted_sections
        if isinstance(item, dict)
    ]
    general_rules = general_integration_rule_block()
    case_rules = case_subsection_rule_block(case_patch)
    return (
        "You are preparing the metadata for a formal academic literature review.\n"
        "Return strict JSON only.\n"
        "Do not rewrite section bodies. Do not invent papers or cite keys.\n"
        "You may use citation placeholders in the abstract only if they appear in the drafted section summaries.\n"
        "Your job is limited to title/subtitle polish, one abstract paragraph, concise executive takeaways, and an optional conclusion bridge.\n\n"
        f"{general_rules}"
        f"{case_rules}"
        "Return this schema:\n"
        "{\n"
        '  "title": "...",\n'
        '  "subtitle": "...",\n'
        '  "one_sentence_summary": "...",\n'
        '  "abstract_text": "one polished abstract paragraph",\n'
        '  "conclusion": "short synthesis paragraph, optional citations only from provided used cite keys",\n'
        '  "used_cite_keys": ["smith2024chain"]\n'
        "}\n\n"
        f"Requested title: {title}\n"
        f"Requested subtitle: {subtitle}\n"
        f"Requested one-sentence summary: {one_sentence_summary}\n"
        f"Abstract outline: {json.dumps(abstract_outline, ensure_ascii=False)}\n\n"
        f"Outline JSON:\n{json.dumps(outline, ensure_ascii=False)}\n\n"
        f"Drafted section summaries JSON:\n{json.dumps(compact_sections, ensure_ascii=False)}\n"
    )


def build_section_from_subsection_drafts(
    section: dict[str, Any],
    subsection_drafts: list[dict[str, Any]],
) -> dict[str, Any]:
    used_cite_keys: list[str] = []
    for draft in subsection_drafts:
        for key in as_list(draft.get("used_cite_keys")) + cite_keys_in_text(draft.get("subsection_text")):
            key = normalize_whitespace(key)
            if key and key not in used_cite_keys:
                used_cite_keys.append(key)
    section_text_parts = []
    for draft in subsection_drafts:
        title = strip_section_number(draft.get("subsection_title"))
        text = preserve_markdown_block(draft.get("subsection_text"))
        if not text:
            continue
        if draft.get("hidden_title"):
            section_text_parts.append(text)
        else:
            section_text_parts.append(f"### {title}\n\n{text}" if title else text)
    return {
        "section_id": section.get("section_id"),
        "section_title": strip_section_number(section.get("section_title")),
        "section_summary": " ".join(
            normalize_whitespace(draft.get("subsection_summary")) for draft in subsection_drafts if normalize_whitespace(draft.get("subsection_summary"))
        ),
        "section_text": "\n\n".join(section_text_parts),
        "subsections": subsection_drafts,
        "used_cite_keys": used_cite_keys,
    }


def build_subsection_citation_audit(subsection_pack: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    text_keys = cite_keys_in_text(draft.get("subsection_text"))
    declared_keys = [normalize_whitespace(key) for key in as_list(draft.get("used_cite_keys")) if normalize_whitespace(key)]
    used_keys = list(dict.fromkeys(text_keys + declared_keys))
    allowed_keys = set(normalize_whitespace(key) for key in as_list(subsection_pack.get("allowed_cite_keys")) if normalize_whitespace(key))
    must_use_keys = set(
        normalize_whitespace(key)
        for key in subsection_pack.get("citation_plan", {}).get("must_use_cite_keys", [])
        if normalize_whitespace(key)
    )
    target_citations = subsection_pack.get("citation_plan", {}).get("target_citation_count") or subsection_pack.get("target_citation_count") or 0
    coverage_status = [item for item in draft.get("coverage_bullet_status", []) if isinstance(item, dict)]
    return {
        "section_id": subsection_pack.get("section_id"),
        "subsection_id": subsection_pack.get("subsection_id"),
        "subsection_title": subsection_pack.get("subsection_title"),
        "word_count": words_in_markdown(draft.get("subsection_text")),
        "target_word_count": subsection_pack.get("target_word_count"),
        "citation_count": len(used_keys),
        "target_citation_count": min(int(target_citations or 0), len(allowed_keys)),
        "used_cite_keys": used_keys,
        "pack_outside_cite_keys": [key for key in used_keys if key not in allowed_keys],
        "unused_must_use_cite_keys": [key for key in must_use_keys if key not in used_keys],
        "coverage_bullets": len(subsection_pack.get("coverage_bullets", [])),
        "covered_bullets": sum(1 for item in coverage_status if item.get("covered") is True),
    }


def build_section_quality_audit(
    drafted_sections: list[dict[str, Any]],
    integrated_report: dict[str, Any],
) -> dict[str, Any]:
    final_by_section = {
        normalize_section_id(item.get("section_id")): item
        for item in integrated_report.get("sections", [])
        if isinstance(item, dict) and normalize_section_id(item.get("section_id"))
    }
    sections = []
    for draft in drafted_sections:
        section_id = normalize_section_id(draft.get("section_id"))
        final = final_by_section.get(section_id, {})
        draft_text = preserve_markdown_block(draft.get("section_text"))
        final_text = preserve_markdown_block(final.get("section_text")) or draft_text
        sections.append(
            {
                "section_id": section_id,
                "section_title": draft.get("section_title"),
                "draft_word_count": words_in_markdown(draft_text),
                "final_word_count": words_in_markdown(final_text),
                "draft_citation_count": len(cite_keys_in_text(draft_text)),
                "final_citation_count": len(cite_keys_in_text(final_text)),
                "subsection_count": len([item for item in draft.get("subsections", []) if isinstance(item, dict)]),
            }
        )
    return {"sections": sections}


def build_citation_flow_audit(
    drafted_sections: list[dict[str, Any]],
    integrated_report: dict[str, Any],
    citation_audit: dict[str, Any],
) -> dict[str, Any]:
    draft_keys: list[str] = []
    for section in drafted_sections:
        for key in cite_keys_in_text(section.get("section_text")):
            if key not in draft_keys:
                draft_keys.append(key)
    integrated_keys: list[str] = []
    for section in integrated_report.get("sections", []):
        if not isinstance(section, dict):
            continue
        for key in cite_keys_in_text(section.get("section_text")):
            if key not in integrated_keys:
                integrated_keys.append(key)
    final_keys = [normalize_whitespace(key) for key in citation_audit.get("used_cite_keys", []) if normalize_whitespace(key)]
    return {
        "draft_cite_keys": draft_keys,
        "integrated_cite_keys": integrated_keys,
        "final_rendered_cite_keys": final_keys,
        "lost_after_integration": [key for key in draft_keys if key not in integrated_keys],
        "lost_after_rendering": [key for key in integrated_keys if key not in final_keys],
    }


def build_integration_prompt(
    title: str,
    subtitle: str,
    one_sentence_summary: str,
    outline: dict[str, Any],
    drafted_sections: list[dict[str, Any]],
) -> str:
    return (
        "You are integrating multiple drafted sections into a coherent formal literature review.\n"
        "Return strict JSON only.\n"
        "Do not invent papers or facts.\n"
        "Preserve citation placeholders in the form [@cite_key].\n"
        "Your task is to remove duplication, align terminology, improve transitions, and produce clean section bodies.\n"
        "Preserve or improve useful Markdown structure. The final review should be easy to scan, not a wall of plain text.\n"
        "Use `###` subheadings where helpful, `**bold lead-ins**` for emphasis, and compact lists only when they improve readability.\n"
        "Do not flatten all section drafts into undifferentiated paragraphs.\n\n"
        "Prefer original method papers, benchmark papers, and canonical technical papers over survey papers when preserving citations.\n"
        "Return this schema:\n"
        "{\n"
        '  "title": "...",\n'
        '  "subtitle": "...",\n'
        '  "one_sentence_summary": "...",\n'
        '  "abstract_text": "one polished abstract paragraph",\n'
        '  "executive_summary": ["4-6 concise takeaways"],\n'
        '  "sections": [\n'
        '    {\n'
        '      "section_title": "...",\n'
        '      "section_text": "...",\n'
        '      "used_cite_keys": ["smith2024chain"]\n'
        "    }\n"
        "  ],\n"
        '  "conclusion": "...",\n'
        '  "used_cite_keys": ["smith2024chain"]\n'
        "}\n\n"
        f"Requested title: {title}\n"
        f"Requested subtitle: {subtitle}\n"
        f"Requested one-sentence summary: {one_sentence_summary}\n\n"
        f"Outline JSON:\n{json.dumps(outline, ensure_ascii=False)}\n\n"
        f"Drafted sections JSON:\n{json.dumps(drafted_sections, ensure_ascii=False)}\n"
    )


def render_reference_entry(entry: dict[str, Any]) -> str:
    authors = entry.get("authors", [])
    if not authors:
        author_text = "Unknown"
    elif len(authors) <= 3:
        author_text = ", ".join(authors[:-1]) + (f", and {authors[-1]}" if len(authors) > 1 else authors[0])
    else:
        author_text = ", ".join(authors[:3]) + ", et al."
    title = normalize_whitespace(entry.get("title")) or "Untitled"
    year = entry.get("year") or "n.d."
    venue = normalize_whitespace(entry.get("venue"))
    doi = normalize_whitespace(entry.get("doi"))
    url = normalize_whitespace(entry.get("url"))
    tail_parts = []
    if venue:
        tail_parts.append(venue)
    if doi:
        tail_parts.append(f"DOI: {doi}")
    elif url:
        tail_parts.append(url)
    tail = ". ".join(tail_parts)
    return f"{author_text} ({year}). {title}." + (f" {tail}." if tail else "")


def replace_cite_placeholders(text: str, bibliography: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    by_key = bibliography_by_cite_key(bibliography)
    used_keys: list[str] = []
    missing_keys: list[str] = []

    def repl(match: re.Match[str]) -> str:
        raw = normalize_whitespace(match.group(1))
        candidates = []
        for part in re.split(r"[;,]", raw):
            key = normalize_whitespace(part).lstrip("@")
            if key:
                candidates.append(key)
        labels: list[str] = []
        for key in candidates:
            if key in by_key:
                if key not in used_keys:
                    used_keys.append(key)
                labels.append(by_key[key]["citation_label"])
            else:
                if key not in missing_keys:
                    missing_keys.append(key)
                labels.append(f"MISSING:{key}")
        if not labels:
            return match.group(0)
        return "(" + "; ".join(labels) + ")"

    rendered = re.sub(r"\[@([^\]]+)\]", repl, text)
    audit = {
        "used_cite_keys": used_keys,
        "missing_cite_keys": missing_keys,
    }
    return rendered, audit


def choose_reference_order(cite_keys: list[str], bibliography: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = bibliography_by_cite_key(bibliography)
    return [by_key[key] for key in cite_keys if key in by_key]


def build_formal_review_markdown(
    integrated_report: dict[str, Any],
    outline: dict[str, Any],
    bibliography: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    raw_sections = [item for item in integrated_report.get("sections", []) if isinstance(item, dict)]
    raw_by_section_id = {
        normalize_section_id(item.get("section_id")): item
        for item in raw_sections
        if normalize_section_id(item.get("section_id"))
    }
    used_keys_combined: list[str] = []
    missing_keys_combined: list[str] = []

    def absorb_audit(audit: dict[str, Any]) -> None:
        for key in audit.get("used_cite_keys", []):
            if key not in used_keys_combined:
                used_keys_combined.append(key)
        for key in audit.get("missing_cite_keys", []):
            if key not in missing_keys_combined:
                missing_keys_combined.append(key)

    title = normalize_whitespace(integrated_report.get("title")) or "Literature Review"
    subtitle = normalize_whitespace(integrated_report.get("subtitle") or integrated_report.get("one_sentence_summary"))
    abstract_text, abstract_audit = replace_cite_placeholders(
        normalize_whitespace(integrated_report.get("abstract_text")),
        bibliography,
    )
    absorb_audit(abstract_audit)

    lines = [f"# {title}"]
    if subtitle:
        lines.extend(["", f"_{subtitle}_"])
    if abstract_text:
        lines.extend(["", "## Abstract", "", abstract_text, ""])

    outline_sections = [item for item in outline.get("sections", []) if isinstance(item, dict)]
    major_index = 1
    raw_cursor = 0
    rendered_section_ids: set[str] = set()
    conclusion_fallback = preserve_markdown_block(integrated_report.get("conclusion"))

    def next_raw_section(section_id: str) -> dict[str, Any]:
        nonlocal raw_cursor
        if section_id in raw_by_section_id:
            return raw_by_section_id[section_id]
        if raw_cursor < len(raw_sections):
            item = raw_sections[raw_cursor]
            raw_cursor += 1
            return item
        return {}

    for section_outline in outline_sections:
        section_id = normalize_section_id(section_outline.get("section_id"))
        if not section_id:
            section_id = f"section_{major_index}"
        section = next_raw_section(section_id)
        rendered_section_ids.add(section_id)
        canonical_title = strip_section_number(section_outline.get("section_title") or section.get("section_title")) or "Section"
        section_title = f"{major_index}. {canonical_title}"
        lines.extend([f"## {section_title}", ""])

        raw_subsections = [item for item in section.get("subsections", []) if isinstance(item, dict)]
        if raw_subsections:
            for sub_index, subsection in enumerate(raw_subsections, start=1):
                subsection_title = strip_section_number(subsection.get("subsection_title")) or f"Subsection {sub_index}"
                subsection_text = preserve_markdown_block(subsection.get("subsection_text") or subsection.get("section_text"))
                subsection_text, subsection_audit = replace_cite_placeholders(subsection_text, bibliography)
                absorb_audit(subsection_audit)
                if not subsection.get("hidden_title"):
                    lines.extend([f"### {major_index}.{sub_index} {subsection_title}", ""])
                if subsection_text:
                    lines.extend([subsection_text, ""])
        else:
            section_text = preserve_markdown_block(section.get("section_text"))
            if not section_text and classify_outline_section(section_outline) == "conclusion":
                section_text = conclusion_fallback
                conclusion_fallback = ""
            section_text, section_audit = replace_cite_placeholders(section_text, bibliography)
            absorb_audit(section_audit)
            if section_text:
                lines.extend([section_text, ""])
        major_index += 1

    for section in raw_sections:
        section_id = normalize_section_id(section.get("section_id"))
        if section_id and section_id in rendered_section_ids:
            continue
        section_title = f"{major_index}. {strip_section_number(section.get('section_title')) or 'Section'}"
        section_text, section_audit = replace_cite_placeholders(
            preserve_markdown_block(section.get("section_text")),
            bibliography,
        )
        absorb_audit(section_audit)
        if section_text:
            lines.extend([f"## {section_title}", "", section_text, ""])
            major_index += 1

    if conclusion_fallback and not any(classify_outline_section(section) == "conclusion" for section in outline_sections):
        conclusion_text, conclusion_audit = replace_cite_placeholders(conclusion_fallback, bibliography)
        absorb_audit(conclusion_audit)
        lines.extend(["## Conclusion", "", conclusion_text, ""])

    ordered_refs = choose_reference_order(used_keys_combined, bibliography)
    lines.extend(["## References", ""])
    for entry in ordered_refs:
        lines.append(f"- {render_reference_entry(entry)}")
    lines.append("")

    audit = {
        "used_cite_keys": used_keys_combined,
        "missing_cite_keys": missing_keys_combined,
        "unused_bibliography_keys": [
            item["cite_key"] for item in bibliography if item["cite_key"] not in set(used_keys_combined)
        ],
    }
    return "\n".join(lines).rstrip() + "\n", audit


def render_diagnostic_report(
    search_result: dict[str, Any],
    evidence_map: dict[str, Any],
    bibliography: list[dict[str, Any]],
    outline: dict[str, Any],
    section_packs: list[dict[str, Any]],
    citation_audit: dict[str, Any],
    *,
    formal_model: str,
    section_quality_audit: dict[str, Any] | None = None,
    citation_flow_audit: dict[str, Any] | None = None,
    subsection_citation_audit: dict[str, Any] | None = None,
) -> str:
    assignments = assignment_rows(evidence_map)
    biblio_by_id = bibliography_by_paper_id(bibliography)
    actions = normalize_search_actions(search_result)
    topic = normalize_whitespace(search_result.get("topic"))
    paper_count = evidence_map.get("paper_count") or len(assignments)
    method_clusters = [item for item in evidence_map.get("method_clusters", []) if isinstance(item, dict)]
    top_clusters = [item for item in method_clusters if item.get("cluster_id") != "OUT"]
    quality_notes = search_result.get("coverage_report", {}) if isinstance(search_result.get("coverage_report"), dict) else {}
    profile = search_result.get("topic_profile", {}) if isinstance(search_result.get("topic_profile"), dict) else {}

    lines = [f"# Diagnostic Report: {topic}", ""]
    lines.extend(
        [
            "## Topic and Run Metadata",
            "",
            f"- Topic: {topic}",
            f"- Evidence paper count: {paper_count}",
            f"- Method cluster count: {len(top_clusters)}",
            f"- Time bucket count: {len(evidence_map.get('time_buckets', []))}",
            f"- Formal review model: {formal_model}",
            "",
        ]
    )

    lines.extend(["## Topic Profile", ""])
    if profile:
        for key in (
            "normalized_topic",
            "scope",
            "possible_method_families",
            "upstream_foundations",
            "application_contexts",
            "likely_keywords",
            "ambiguous_terms",
            "exclusion_rules",
        ):
            value = profile.get(key)
            if isinstance(value, list):
                lines.append(f"- {key.replace('_', ' ').title()}: " + "; ".join(normalize_whitespace(x) for x in value if normalize_whitespace(x)))
            elif normalize_whitespace(value):
                lines.append(f"- {key.replace('_', ' ').title()}: {normalize_whitespace(value)}")
        lines.append("")
    else:
        lines.extend(["No topic profile available.", ""])

    lines.extend(["## Time Window Design", ""])
    lines.extend(["| Label | Start | End | Rationale |", "| --- | ---: | ---: | --- |"])
    for window in search_result.get("time_windows", []):
        if not isinstance(window, dict):
            continue
        lines.append(
            f"| {md_escape(window.get('label'), max_chars=48)} | {window.get('start_year') or ''} | "
            f"{window.get('end_year') or ''} | {md_escape(window.get('rationale'), max_chars=180)} |"
        )
    lines.append("")

    lines.extend(["## Search Actions", ""])
    lines.extend(["| Round | Action | Intent | Time Window | Query | Rationale |", "| --- | --- | --- | --- | --- | --- |"])
    for action in actions:
        time_window = action.get("time_window") if isinstance(action.get("time_window"), dict) else {}
        time_label = normalize_whitespace(time_window.get("label"))
        if not time_label:
            start = time_window.get("start_year")
            end = time_window.get("end_year")
            time_label = f"{start}-{end}" if start or end else ""
        lines.append(
            f"| {md_escape(action.get('round'), max_chars=32)} | {md_escape(action.get('action_id'), max_chars=40)} | "
            f"{md_escape(action.get('intent'), max_chars=40)} | {md_escape(time_label, max_chars=40)} | "
            f"{md_escape(action.get('query'), max_chars=80)} | {md_escape(action.get('rationale'), max_chars=120)} |"
        )
    lines.append("")

    lines.extend(["## Coverage Summary", ""])
    lines.append(f"- Coverage status: {normalize_whitespace(quality_notes.get('coverage_status')) or 'unknown'}")
    for key in ("method_gaps", "time_gaps", "representative_gaps", "off_topic_notes"):
        values = quality_notes.get(key)
        if isinstance(values, list) and values:
            lines.append(f"- {key.replace('_', ' ').title()}: " + "; ".join(normalize_whitespace(x) for x in values if normalize_whitespace(x)))
    lines.append("")

    lines.extend(["## Method Clusters", ""])
    for cluster in top_clusters:
        lines.extend(
            [
                f"### {normalize_whitespace(cluster.get('name'))}",
                "",
                normalize_whitespace(cluster.get("definition")) or "No definition available.",
                "",
                f"- Paper count: {cluster.get('paper_count') or 0}",
                f"- Role counts: {json.dumps(cluster.get('role_counts', {}), ensure_ascii=False)}",
                f"- Time counts: {json.dumps(cluster.get('time_counts', {}), ensure_ascii=False)}",
            ]
        )
        reps = []
        for paper_id in cluster.get("representative_paper_ids", [])[:8]:
            entry = biblio_by_id.get(normalize_whitespace(paper_id))
            if not entry:
                continue
            reps.append(f"{entry['citation_label']} - {entry['title']}")
        if reps:
            lines.append("- Representative papers: " + "; ".join(reps))
        missing_signals = [normalize_whitespace(x) for x in cluster.get("missing_signals", []) if normalize_whitespace(x)]
        if missing_signals:
            lines.append("- Missing signals: " + "; ".join(missing_signals))
        lines.append("")

    lines.extend(["## Method x Time Matrix", ""])
    lines.extend(["| Cluster | Time Bucket | Papers | Representative Papers |", "| --- | --- | ---: | --- |"])
    for cell in evidence_map.get("method_time_matrix", []):
        if not isinstance(cell, dict):
            continue
        reps = []
        for paper_id in cell.get("representative_paper_ids", [])[:3]:
            entry = biblio_by_id.get(normalize_whitespace(paper_id))
            if entry:
                reps.append(f"{entry['author_display']} ({entry['year']})")
        lines.append(
            f"| {md_escape(cell.get('cluster_name'), max_chars=48)} | {md_escape(cell.get('time_bucket_label'), max_chars=32)} | "
            f"{cell.get('paper_count') or 0} | {md_escape('; '.join(reps), max_chars=180)} |"
        )
    lines.append("")

    lines.extend(["## Formal Outline and Section Packs", ""])
    for section in [item for item in outline.get("sections", []) if isinstance(item, dict)]:
        coverage_points = [
            normalize_whitespace(item.get("point"))
            for item in coverage_bullets_from_section(section)
            if normalize_whitespace(item.get("point"))
        ]
        subsections = normalize_subsections(section)
        lines.extend(
            [
                f"### {normalize_whitespace(section.get('section_title'))}",
                "",
                f"- Role: {normalize_whitespace(section.get('section_role')) or 'other'}",
                f"- Description: {normalize_whitespace(section.get('description') or section.get('purpose'))}",
                f"- Coverage bullets: {'; '.join(coverage_points) if coverage_points else 'n/a'}",
                f"- Subsections: {len(subsections)}",
                f"- Assigned papers: {len(as_list(section.get('paper_ids')))}",
                f"- Assigned clusters: {', '.join(normalize_whitespace(x) for x in section.get('cluster_ids', []) if normalize_whitespace(x)) or 'n/a'}",
                "",
            ]
        )
    for pack in section_packs:
        lines.append(f"- Pack `{normalize_whitespace(pack.get('section_id'))}` contains {len(pack.get('papers', []))} papers.")
    lines.append("")

    lines.extend(["## Citation Audit", ""])
    lines.append(f"- Used cite keys: {len(citation_audit.get('used_cite_keys', []))}")
    lines.append(f"- Missing cite keys: {len(citation_audit.get('missing_cite_keys', []))}")
    lines.append(f"- Unused bibliography entries: {len(citation_audit.get('unused_bibliography_keys', []))}")
    if citation_audit.get("missing_cite_keys"):
        lines.append("- Missing keys: " + ", ".join(citation_audit["missing_cite_keys"]))
    lines.append("")

    if section_quality_audit:
        lines.extend(["## Section Quality Audit", ""])
        lines.extend(["| Section | Draft Words | Final Words | Draft Cites | Final Cites | Subsections |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for item in section_quality_audit.get("sections", []):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {md_escape(item.get('section_title') or item.get('section_id'), max_chars=60)} | "
                f"{item.get('draft_word_count') or 0} | {item.get('final_word_count') or 0} | "
                f"{item.get('draft_citation_count') or 0} | {item.get('final_citation_count') or 0} | "
                f"{item.get('subsection_count') or 0} |"
            )
        lines.append("")

    if subsection_citation_audit:
        lines.extend(["## Subsection Citation Audit", ""])
        lines.extend(["| Section | Subsection | Words | Citations | Target | Pack-Outside | Unused Must-Use |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for item in subsection_citation_audit.get("subsections", []):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {md_escape(item.get('section_id'), max_chars=32)} | {md_escape(item.get('subsection_title') or item.get('subsection_id'), max_chars=60)} | "
                f"{item.get('word_count') or 0} | {item.get('citation_count') or 0} | {item.get('target_citation_count') or 0} | "
                f"{len(item.get('pack_outside_cite_keys', []))} | {len(item.get('unused_must_use_cite_keys', []))} |"
            )
        lines.append("")

    if citation_flow_audit:
        lines.extend(["## Citation Flow Audit", ""])
        lines.append(f"- Draft cite keys: {len(citation_flow_audit.get('draft_cite_keys', []))}")
        lines.append(f"- Integrated cite keys: {len(citation_flow_audit.get('integrated_cite_keys', []))}")
        lines.append(f"- Final rendered cite keys: {len(citation_flow_audit.get('final_rendered_cite_keys', []))}")
        lines.append(f"- Lost after integration: {len(citation_flow_audit.get('lost_after_integration', []))}")
        lines.append(f"- Lost after rendering: {len(citation_flow_audit.get('lost_after_rendering', []))}")
        lines.append("")

    lines.extend(["## Full Paper Registry", ""])
    lines.extend(
        [
            "| Paper ID | Cite Key | Year | Role | Cluster | Time Bucket | Title | Venue | Citations |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for assignment in sorted(assignments, key=representative_sort_key):
        entry = biblio_by_id.get(assignment["paper_id"], {})
        lines.append(
            f"| {assignment['paper_id']} | {md_escape(entry.get('cite_key'), max_chars=32)} | {assignment.get('year') or ''} | "
            f"{md_escape(assignment.get('role'), max_chars=28)} | {md_escape(assignment.get('method_cluster_name'), max_chars=42)} | "
            f"{md_escape(assignment.get('time_bucket_label'), max_chars=28)} | {md_escape(assignment.get('title'), max_chars=80)} | "
            f"{md_escape(entry.get('venue'), max_chars=40)} | {assignment.get('citation_count') or 0} |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def llm_json(client: DmxJsonClient, *, system_prompt: str, user_prompt: str, label: str) -> dict[str, Any]:
    return client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt, label=label)


def llm_json_logged(
    client: DmxJsonClient,
    *,
    system_prompt: str,
    user_prompt: str,
    label: str,
    logger: ProgressLogger,
    attempt: int,
) -> dict[str, Any]:
    if not hasattr(client, "chat_json_text"):
        return llm_json(client, system_prompt=system_prompt, user_prompt=user_prompt, label=label)
    raw_dir = (logger.path.parent / "llm_raw") if logger.path else None
    try:
        content = client.chat_json_text(system_prompt=system_prompt, user_prompt=user_prompt, label=label)
    except Exception:
        raise
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{label}_attempt{attempt}.txt"
        raw_path.write_text(content, encoding="utf-8")
    try:
        from literature_review_search import parse_json_object

        return parse_json_object(content)
    except Exception as exc:
        if raw_dir:
            meta_path = raw_dir / f"{label}_attempt{attempt}.error.json"
            write_json_text(
                meta_path,
                {
                    "label": label,
                    "attempt": attempt,
                    "error": str(exc),
                    "raw_path": str(raw_dir / f"{label}_attempt{attempt}.txt"),
                    "content_chars": len(content),
                },
            )
        raise


def llm_json_with_retries(
    client: DmxJsonClient,
    *,
    system_prompt: str,
    user_prompt: str,
    label: str,
    logger: ProgressLogger,
    attempts: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                logger.log(label, "Retrying LLM JSON request", attempt=attempt, total_attempts=attempts)
            return llm_json_logged(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                label=label,
                logger=logger,
                attempt=attempt,
            )
        except Exception as exc:
            last_error = exc
            logger.log(label, "LLM JSON request failed", attempt=attempt, total_attempts=attempts, error=str(exc))
            if attempt >= attempts:
                break
    raise RuntimeError(f"LLM JSON call failed for {label} after {attempts} attempts: {last_error}") from last_error


def build_client(args: argparse.Namespace, *, timeout: int | None = None) -> DmxJsonClient:
    api_url = args.llm_api_url
    model = args.llm_model
    api_key_override = None
    wire_api = "chat_completions"
    if getattr(args, "use_gpt", False):
        override = load_codex_gpt_override()
        api_url = normalize_whitespace(override.get("base_url")) or api_url
        model = normalize_whitespace(override.get("model")) or model
        api_key_override = normalize_whitespace(override.get("api_key")) or None
        wire_api = normalize_whitespace(override.get("wire_api")) or "responses"
        if not api_url or not model or not api_key_override:
            raise RuntimeError(
                "GPT override is enabled but Codex config/auth is incomplete. "
                "Expected base_url and model in ~/.codex/config.toml plus OPENAI_API_KEY in ~/.codex/auth.json."
            )
    return DmxJsonClient(
        env_path=Path(args.env).expanduser().resolve(),
        api_url=api_url,
        model=model,
        timeout=timeout if timeout is not None else args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        use_env_proxy=args.use_env_proxy,
        api_key_override=api_key_override,
        wire_api=wire_api,
    )


def ensure_evidence_map(
    *,
    evidence_map_path: Path | None,
    search_result_path: Path | None,
    min_cluster_score: float,
    representatives_per_group: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, bool]:
    if evidence_map_path is None and search_result_path is None:
        raise ValueError("One of --evidence-map or --search-result must be provided.")

    if search_result_path is None and evidence_map_path is not None:
        evidence_map = read_json(evidence_map_path)
        inferred_search = evidence_map_path.parent / "search_result.json"
        if not inferred_search.exists():
            raise ValueError(f"Could not infer search_result.json next to evidence map: {inferred_search}")
        search_result = read_json(inferred_search)
        return search_result, evidence_map, evidence_map_path, False

    assert search_result_path is not None
    search_result = read_json(search_result_path)
    sibling_evidence_path = search_result_path.parent / "organized_search_result.json"
    if evidence_map_path is None and sibling_evidence_path.exists():
        evidence_map = read_json(sibling_evidence_path)
        return search_result, evidence_map, sibling_evidence_path, False
    if evidence_map_path is not None and evidence_map_path.exists():
        evidence_map = read_json(evidence_map_path)
        return search_result, evidence_map, evidence_map_path, False

    evidence_map = organize(
        search_result,
        clusters_payload=None,
        min_cluster_score=min_cluster_score,
        representatives_per_group=representatives_per_group,
    )
    generated_path = search_result_path.parent / "organized_search_result.json"
    write_json(generated_path, evidence_map)
    return search_result, evidence_map, generated_path, True


def load_if_exists(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return read_json(path)
    return None


def find_existing_json_by_suffix(directory: Path, suffix: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(f"*_{suffix}.json"))
    return matches[0] if matches else None


def apply_subsection_pack_style_defaults(pack: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(pack)
    if normalize_section_id(enriched.get("section_id")) == TEMPORAL_SECTION_ID:
        enriched["prompt_style_version"] = TEMPORAL_SUBSECTION_PROMPT_STYLE_VERSION
    else:
        enriched["prompt_style_version"] = subsection_style_for_pack(enriched)
    return enriched


def integrated_matches_outline(integrated: dict[str, Any], outline: dict[str, Any]) -> bool:
    outline_ids = [
        normalize_section_id(item.get("section_id"))
        for item in outline.get("sections", [])
        if isinstance(item, dict) and normalize_section_id(item.get("section_id"))
    ]
    signature_ids = [
        normalize_section_id(item)
        for item in as_list(integrated.get("outline_section_ids"))
        if normalize_section_id(item)
    ]
    if signature_ids != outline_ids:
        return False
    integrated_ids = [
        normalize_section_id(item.get("section_id"))
        for item in integrated.get("sections", [])
        if isinstance(item, dict) and normalize_section_id(item.get("section_id"))
    ]
    return outline_ids == integrated_ids


def draft_matches_pack(draft: dict[str, Any], pack: dict[str, Any]) -> bool:
    expected = normalize_whitespace(pack.get("prompt_style_version"))
    if expected and normalize_whitespace(draft.get("prompt_style_version")) != expected:
        return False
    if not pack.get("llm_should_generate_title"):
        expected_title = strip_section_number(pack.get("subsection_title"))
        draft_title = strip_section_number(draft.get("subsection_title"))
        if expected_title and draft_title != expected_title:
            return False
    else:
        draft_title = strip_section_number(draft.get("subsection_title"))
        if not draft_title:
            return False
    if expected:
        text = preserve_markdown_block(draft.get("subsection_text"))
        paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", text) if normalize_whitespace(paragraph)]
        starts_with_bold = [paragraph.lstrip().startswith("**") for paragraph in paragraphs]
        if expected == TEMPORAL_SUBSECTION_PROMPT_STYLE_VERSION and any(not value for value in starts_with_bold):
            return False
        if expected == METHOD_SUBSECTION_PROMPT_STYLE_VERSION and paragraphs and not any(starts_with_bold):
            return False
        if expected == PLAIN_SUBSECTION_PROMPT_STYLE_VERSION and any(starts_with_bold):
            return False
    return True


def normalize_draft_to_pack(draft: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(draft)
    normalized["section_id"] = pack.get("section_id")
    normalized["subsection_id"] = pack.get("subsection_id")
    if normalize_whitespace(pack.get("subsection_title")) and not pack.get("llm_should_generate_title"):
        normalized["subsection_title"] = pack.get("subsection_title")
    normalized["hidden_title"] = pack.get("hidden_title", False)
    if normalize_whitespace(pack.get("prompt_style_version")):
        normalized["prompt_style_version"] = normalize_whitespace(pack.get("prompt_style_version"))
    return normalized


def render_rule_block(title: str, rules: list[str]) -> str:
    cleaned = [normalize_whitespace(rule) for rule in rules if normalize_whitespace(rule)]
    if not cleaned:
        return ""
    lines = [f"{title}\n"]
    lines.extend(f"- {rule}\n" for rule in cleaned)
    lines.append("\n")
    return "".join(lines)


def general_subsection_rule_block() -> str:
    return render_rule_block("General writing and evidence rules:", GENERAL_SUBSECTION_RULES)


def general_integration_rule_block() -> str:
    return render_rule_block("General integration rules:", GENERAL_INTEGRATION_RULES)


def subject_outline_rule_block(subject_domain: str) -> str:
    if subject_domain == SUBJECT_CHEMISTRY:
        return render_rule_block("Subject-specific chemistry outline rules:", CHEMISTRY_OUTLINE_RULES)
    return ""


def subject_subsection_rule_block(subject_domain: str) -> str:
    if subject_domain == SUBJECT_CHEMISTRY:
        return render_rule_block("Subject-specific chemistry writing rules:", CHEMISTRY_SUBSECTION_RULES)
    return ""


def case_outline_rule_block(case_patch: str) -> str:
    return render_rule_block("Case-specific outline rules:", CASE_OUTLINE_PATCHES.get(case_patch, []))


def case_subsection_rule_block(case_patch: str) -> str:
    return render_rule_block("Case-specific writing rules:", CASE_SUBSECTION_PATCHES.get(case_patch, []))


def generate_reports(args: argparse.Namespace) -> dict[str, Any]:
    subject_domain = normalize_whitespace(args.subject_domain).casefold() or SUBJECT_CHEMISTRY
    if subject_domain not in {SUBJECT_CHEMISTRY, SUBJECT_BIOLOGY, SUBJECT_GENERAL}:
        raise ValueError(f"Unsupported --subject-domain: {args.subject_domain}")
    case_patch = normalize_whitespace(args.case_prompt_patch).casefold() or CASE_PATCH_NONE
    if case_patch not in {CASE_PATCH_NONE, CASE_PATCH_CASE1, CASE_PATCH_CASE2, CASE_PATCH_CASE5}:
        raise ValueError(f"Unsupported --case-prompt-patch: {args.case_prompt_patch}")

    evidence_map_path = Path(args.evidence_map).expanduser().resolve() if args.evidence_map else None
    search_result_path = Path(args.search_result).expanduser().resolve() if args.search_result else None
    search_result, evidence_map, resolved_evidence_path, generated_evidence = ensure_evidence_map(
        evidence_map_path=evidence_map_path,
        search_result_path=search_result_path,
        min_cluster_score=args.min_cluster_score,
        representatives_per_group=args.representatives_per_group,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolved_evidence_path.parent / "lr_review"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(output_dir / "progress.jsonl")
    section_packs_dir = output_dir / "section_packs"
    subsection_packs_dir = output_dir / "subsection_packs"
    section_drafts_dir = output_dir / "section_drafts"
    subsection_drafts_dir = output_dir / "subsection_drafts"
    section_packs_dir.mkdir(parents=True, exist_ok=True)
    subsection_packs_dir.mkdir(parents=True, exist_ok=True)
    section_drafts_dir.mkdir(parents=True, exist_ok=True)
    subsection_drafts_dir.mkdir(parents=True, exist_ok=True)
    logger.log(
        "init",
        "Initialized report generation",
        output_dir=str(output_dir),
        paper_count=evidence_map.get("paper_count"),
        organized_evidence_map_generated=generated_evidence,
    )

    bibliography_path = output_dir / "bibliography.json"
    if args.resume and bibliography_path.exists():
        bibliography = _load_json(bibliography_path)
        if not isinstance(bibliography, list):
            raise ValueError(f"Expected bibliography list in {bibliography_path}")
        logger.log("bibliography", "Reusing existing bibliography", path=str(bibliography_path))
    else:
        bibliography = build_bibliography(search_result, evidence_map)
        write_json(bibliography_path, bibliography)
        logger.log("bibliography", "Built bibliography", bibliography_count=len(bibliography), path=str(bibliography_path))

    if args.stop_after == "bibliography":
        logger.log("stop", "Stopped after bibliography stage by request")
        return {
            "bibliography_path": str(bibliography_path),
            "organized_evidence_map_path": str(resolved_evidence_path),
            "organized_evidence_map_generated": generated_evidence,
            "paper_count": evidence_map.get("paper_count"),
            "stop_after": args.stop_after,
        }

    client = build_client(args)

    outline_context = compact_outline_context(
        search_result,
        evidence_map,
        bibliography,
        max_clusters=args.outline_cluster_limit,
        max_representatives_per_cluster=args.outline_reps_per_cluster,
    )
    outline_path = output_dir / "formal_outline.json"
    if args.resume and outline_path.exists():
        outline = canonicalize_outline(read_json(outline_path))
        logger.log("outline", "Reusing existing formal outline", path=str(outline_path))
    else:
        logger.log(
            "outline",
            "Requesting formal outline",
            cluster_count=len(outline_context.get("method_clusters", [])),
            family_limit=("auto" if args.method_family_limit <= 0 else args.method_family_limit),
        )
        outline_client = build_client(args, timeout=args.outline_timeout)
        outline = llm_json_with_retries(
            outline_client,
            system_prompt="You plan evidence-grounded academic survey structures and return strict JSON only.",
            user_prompt=build_outline_prompt(
                outline_context,
                family_limit=args.method_family_limit,
                subject_domain=subject_domain,
                case_patch=case_patch,
            ),
            label="formal_review_outline",
            logger=logger,
            attempts=args.outline_attempts,
        )
        outline = canonicalize_outline(outline)
        logger.log("outline", "Formal outline ready", path=str(outline_path))
    outline = enrich_outline_with_temporal_section(outline, search_result, evidence_map)
    write_json(outline_path, outline)

    family_sections = [item for item in outline.get("family_sections", []) if isinstance(item, dict)]
    sections = [item for item in outline.get("sections", []) if isinstance(item, dict)]
    all_sections = sections + family_sections
    logger.log(
        "outline",
        "Outline summary",
        section_count=len(sections),
        family_section_count=len(family_sections),
    )

    if args.stop_after == "outline":
        logger.log("stop", "Stopped after outline stage by request")
        return {
            "bibliography_path": str(bibliography_path),
            "formal_outline_path": str(outline_path),
            "organized_evidence_map_path": str(resolved_evidence_path),
            "organized_evidence_map_generated": generated_evidence,
            "paper_count": evidence_map.get("paper_count"),
            "llm_model": args.llm_model,
            "stop_after": args.stop_after,
        }

    section_packs: list[dict[str, Any]] = []
    for index, section in enumerate(all_sections, start=1):
        temp_pack = section_evidence_pack(
            section,
            search_result,
            evidence_map,
            bibliography,
            max_papers=args.section_max_papers,
        )
        section_id = normalize_whitespace(temp_pack.get("section_id")) or f"section_{index}"
        section_pack_path = section_packs_dir / f"{index:02d}_{section_id}.json"
        reusable_section_pack_path = section_pack_path if section_pack_path.exists() else find_existing_json_by_suffix(section_packs_dir, section_id)
        if args.resume and reusable_section_pack_path and reusable_section_pack_path.exists() and section_id != TEMPORAL_SECTION_ID:
            pack = read_json(reusable_section_pack_path)
            logger.log("packs", "Reusing existing section evidence pack", index=index, section_id=section_id, path=str(reusable_section_pack_path))
            if reusable_section_pack_path != section_pack_path:
                write_json(section_pack_path, pack)
        else:
            pack = temp_pack
            write_json(section_pack_path, pack)
            logger.log(
                "packs",
                "Prepared section evidence pack",
                index=index,
                section_id=section_id,
                paper_count=len(pack.get("papers", [])),
                path=str(section_pack_path),
            )
        section_packs.append(pack)

    paper_mounts = build_paper_mounts(outline, section_packs, bibliography)
    paper_mounts_path = output_dir / "paper_mounts.json"
    write_json(paper_mounts_path, {"paper_mounts": paper_mounts})
    logger.log("mounts", "Built deterministic paper mounts", mount_count=len(paper_mounts), path=str(paper_mounts_path))

    subsection_jobs: list[dict[str, Any]] = []
    citation_plan: list[dict[str, Any]] = []
    pack_by_section = {normalize_section_id(pack.get("section_id")): pack for pack in section_packs}
    for section_index, section in enumerate(all_sections, start=1):
        section_id = normalize_section_id(section.get("section_id")) or f"section_{section_index}"
        section_pack = pack_by_section.get(section_id)
        if not section_pack:
            continue
        for subsection_index, subsection in enumerate(normalize_subsections(section), start=1):
            target_citations = max(
                1,
                int(args.subsection_target_citations),
                max((int(item.get("target_citation_count") or 0) for item in coverage_bullets_from_section(subsection)), default=0),
            )
            subsection_pack = build_subsection_pack(
                section,
                subsection,
                section_pack,
                max_papers=args.subsection_max_papers,
                target_citations=target_citations,
            )
            subsection_pack["target_word_count"] = args.subsection_target_words
            safe_subsection_id = normalize_section_id(subsection_pack.get("subsection_id")) or f"sub_{subsection_index}"
            path = subsection_packs_dir / f"{section_index:02d}_{subsection_index:02d}_{section_id}_{safe_subsection_id}.json"
            reusable_subsection_pack_path = (
                path if path.exists() else find_existing_json_by_suffix(subsection_packs_dir, f"{section_id}_{safe_subsection_id}")
            )
            if args.resume and reusable_subsection_pack_path and reusable_subsection_pack_path.exists() and section_id != TEMPORAL_SECTION_ID:
                subsection_pack = apply_subsection_pack_style_defaults(read_json(reusable_subsection_pack_path))
                logger.log("subsection_packs", "Reusing subsection pack", section_id=section_id, subsection_id=safe_subsection_id, path=str(reusable_subsection_pack_path))
                write_json(path, subsection_pack)
            else:
                subsection_pack = apply_subsection_pack_style_defaults(subsection_pack)
                write_json(path, subsection_pack)
                logger.log(
                    "subsection_packs",
                    "Prepared subsection pack",
                    section_id=section_id,
                    subsection_id=safe_subsection_id,
                    paper_count=len(subsection_pack.get("papers", [])),
                    path=str(path),
                )
            citation_plan.append(subsection_pack.get("citation_plan", {}))
            subsection_jobs.append(
                {
                    "section_index": section_index,
                    "subsection_index": subsection_index,
                    "section": section,
                    "subsection": subsection,
                    "pack": subsection_pack,
                }
            )
    citation_plan_path = output_dir / "citation_plan.json"
    write_json(citation_plan_path, {"citation_plan": citation_plan})
    logger.log("citation_plan", "Built citation plan", subsection_count=len(citation_plan), path=str(citation_plan_path))

    if args.stop_after == "packs":
        logger.log("stop", "Stopped after section pack stage by request")
        return {
            "bibliography_path": str(bibliography_path),
            "formal_outline_path": str(outline_path),
            "section_packs_dir": str(section_packs_dir),
            "subsection_packs_dir": str(subsection_packs_dir),
            "paper_mounts_path": str(paper_mounts_path),
            "citation_plan_path": str(citation_plan_path),
            "organized_evidence_map_path": str(resolved_evidence_path),
            "organized_evidence_map_generated": generated_evidence,
            "paper_count": evidence_map.get("paper_count"),
            "llm_model": args.llm_model,
            "stop_after": args.stop_after,
        }

    logger.log(
        "sections",
        "Starting parallel subsection drafting",
        subsection_count=len(subsection_jobs),
        max_parallel_sections=(len(subsection_jobs) if args.max_parallel_sections <= 0 else args.max_parallel_sections),
    )

    def draft_one_subsection(job: dict[str, Any]) -> tuple[tuple[int, int], dict[str, Any], Path, dict[str, Any], bool]:
        section_index = job["section_index"]
        subsection_index = job["subsection_index"]
        pack = job["pack"]
        section_id = normalize_section_id(pack.get("section_id")) or f"section_{section_index}"
        subsection_id = normalize_section_id(pack.get("subsection_id")) or f"subsection_{subsection_index}"
        path = subsection_drafts_dir / f"{section_index:02d}_{subsection_index:02d}_{section_id}_{subsection_id}.json"
        reusable_path = path if path.exists() else find_existing_json_by_suffix(subsection_drafts_dir, f"{section_id}_{subsection_id}")
        if args.resume and reusable_path and reusable_path.exists():
            reusable_draft = read_json(reusable_path)
            if draft_matches_pack(reusable_draft, pack):
                return (section_index, subsection_index), normalize_draft_to_pack(reusable_draft, pack), reusable_path, pack, True
        local_client = build_client(args)
        draft = llm_json(
            local_client,
            system_prompt="You write one subsection of a formal literature review and return strict JSON only.",
            user_prompt=build_subsection_prompt(
                pack,
                target_words=args.subsection_target_words,
                target_citations=pack.get("citation_plan", {}).get("target_citation_count") or args.subsection_target_citations,
                subject_domain=subject_domain,
                case_patch=case_patch,
            ),
            label=f"subsection_{section_id}_{subsection_id}",
        )
        draft = normalize_draft_to_pack(draft, pack)
        return (section_index, subsection_index), draft, path, pack, False

    subsection_drafts_by_index: dict[tuple[int, int], dict[str, Any]] = {}
    subsection_audits: list[dict[str, Any]] = []
    actual_parallelism = len(subsection_jobs) if args.max_parallel_sections <= 0 else args.max_parallel_sections
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, actual_parallelism)) as executor:
        future_to_meta = {}
        for job in subsection_jobs:
            section_index = job["section_index"]
            subsection_index = job["subsection_index"]
            pack = job["pack"]
            section_id = normalize_section_id(pack.get("section_id")) or f"section_{section_index}"
            subsection_id = normalize_section_id(pack.get("subsection_id")) or f"subsection_{subsection_index}"
            existing_path = subsection_drafts_dir / f"{section_index:02d}_{subsection_index:02d}_{section_id}_{subsection_id}.json"
            reusable_path = existing_path if existing_path.exists() else find_existing_json_by_suffix(subsection_drafts_dir, f"{section_id}_{subsection_id}")
            if args.resume and reusable_path and reusable_path.exists() and draft_matches_pack(read_json(reusable_path), pack):
                logger.log("sections", "Dispatching subsection draft reuse", section_id=section_id, subsection_id=subsection_id, path=str(reusable_path))
            else:
                logger.log("sections", "Dispatching subsection draft request", section_id=section_id, subsection_id=subsection_id)
            future = executor.submit(draft_one_subsection, job)
            future_to_meta[future] = {"section_index": section_index, "subsection_index": subsection_index, "section_id": section_id, "subsection_id": subsection_id}
        for future in concurrent.futures.as_completed(future_to_meta):
            meta = future_to_meta[future]
            section_id = meta["section_id"]
            subsection_id = meta["subsection_id"]
            index_out, subsection_draft, path, pack, reused = future.result()
            subsection_drafts_by_index[index_out] = subsection_draft
            subsection_audits.append(build_subsection_citation_audit(pack, subsection_draft))
            target_path = subsection_drafts_dir / f"{index_out[0]:02d}_{index_out[1]:02d}_{section_id}_{subsection_id}.json"
            if reused:
                if path != target_path:
                    write_json(target_path, subsection_draft)
                logger.log("sections", "Reused existing subsection draft", section_id=section_id, subsection_id=subsection_id, path=str(path))
            else:
                write_json(target_path, subsection_draft)
                logger.log("sections", "Completed subsection draft", section_id=section_id, subsection_id=subsection_id, path=str(target_path))

    drafted_sections: list[dict[str, Any]] = []
    for section_index, section in enumerate(all_sections, start=1):
        subsection_drafts = [
            subsection_drafts_by_index[(sec_idx, sub_idx)]
            for sec_idx, sub_idx in sorted(subsection_drafts_by_index)
            if sec_idx == section_index
        ]
        section_draft = build_section_from_subsection_drafts(section, subsection_drafts)
        section_id = normalize_section_id(section_draft.get("section_id")) or f"section_{section_index}"
        write_json(section_drafts_dir / f"{section_index:02d}_{section_id}.json", section_draft)
        drafted_sections.append(section_draft)
    subsection_citation_audit_path = output_dir / "subsection_citation_audit.json"
    subsection_citation_audit = {"subsections": subsection_audits}
    write_json(subsection_citation_audit_path, subsection_citation_audit)
    logger.log("sections", "Built subsection citation audit", path=str(subsection_citation_audit_path))

    logger.log("integration", "Starting metadata integration without body rewrite", drafted_section_count=len(drafted_sections))
    integrated_path = output_dir / "integrated_review.json"
    if args.resume and integrated_path.exists():
        existing_integrated = read_json(integrated_path)
        if integrated_matches_outline(existing_integrated, outline):
            integrated = existing_integrated
            logger.log("integration", "Reusing existing integrated review", path=str(integrated_path))
        else:
            logger.log("integration", "Existing integrated review is stale for the current outline; regenerating metadata", path=str(integrated_path))
            integration_client = build_client(args, timeout=args.integration_timeout)
            integrated = llm_json_with_retries(
                integration_client,
                system_prompt="You prepare formal review metadata and return strict JSON only.",
                user_prompt=build_integration_metadata_prompt(
                    normalize_whitespace(outline.get("review_title")),
                    normalize_whitespace(outline.get("review_subtitle")),
                    normalize_whitespace(outline.get("one_sentence_summary")),
                    as_list(outline.get("abstract_outline")),
                    outline,
                    drafted_sections,
                    case_patch=case_patch,
                ),
                label="formal_review_integration",
                logger=logger,
                attempts=args.integration_attempts,
            )
            integrated["sections"] = drafted_sections
            integrated["outline_section_ids"] = [
                normalize_section_id(item.get("section_id"))
                for item in outline.get("sections", [])
                if isinstance(item, dict) and normalize_section_id(item.get("section_id"))
            ]
            write_json(integrated_path, integrated)
            logger.log("integration", "Integrated review ready", path=str(integrated_path))
    else:
        integration_client = build_client(args, timeout=args.integration_timeout)
        integrated = llm_json_with_retries(
            integration_client,
            system_prompt="You prepare formal review metadata and return strict JSON only.",
            user_prompt=build_integration_metadata_prompt(
                normalize_whitespace(outline.get("review_title")),
                normalize_whitespace(outline.get("review_subtitle")),
                normalize_whitespace(outline.get("one_sentence_summary")),
                as_list(outline.get("abstract_outline")),
                outline,
                drafted_sections,
                case_patch=case_patch,
            ),
            label="formal_review_integration",
            logger=logger,
            attempts=args.integration_attempts,
        )
        integrated["sections"] = drafted_sections
        integrated["outline_section_ids"] = [
            normalize_section_id(item.get("section_id"))
            for item in outline.get("sections", [])
            if isinstance(item, dict) and normalize_section_id(item.get("section_id"))
        ]
        write_json(integrated_path, integrated)
        logger.log("integration", "Integrated review ready", path=str(integrated_path))
    integrated["sections"] = drafted_sections
    formal_review_md, citation_audit = build_formal_review_markdown(integrated, outline, bibliography)
    section_quality_audit = build_section_quality_audit(drafted_sections, integrated)
    citation_flow_audit = build_citation_flow_audit(drafted_sections, integrated, citation_audit)
    write_json(integrated_path, integrated)
    diagnostic_md = render_diagnostic_report(
        search_result,
        evidence_map,
        bibliography,
        outline,
        section_packs,
        citation_audit,
        formal_model=args.llm_model,
        section_quality_audit=section_quality_audit,
        citation_flow_audit=citation_flow_audit,
        subsection_citation_audit=subsection_citation_audit,
    )

    formal_review_path = output_dir / "formal_review.md"
    diagnostic_report_path = output_dir / "diagnostic_report.md"
    write_text(formal_review_path, formal_review_md)
    write_text(diagnostic_report_path, diagnostic_md)
    write_json(output_dir / "citation_audit.json", citation_audit)
    write_json(output_dir / "section_quality_audit.json", section_quality_audit)
    write_json(output_dir / "citation_flow_audit.json", citation_flow_audit)
    logger.log(
        "render",
        "Rendered final outputs",
        formal_review_path=str(formal_review_path),
        diagnostic_report_path=str(diagnostic_report_path),
        citation_audit_path=str(output_dir / "citation_audit.json"),
        section_quality_audit_path=str(output_dir / "section_quality_audit.json"),
        citation_flow_audit_path=str(output_dir / "citation_flow_audit.json"),
    )

    return {
        "formal_review_path": str(formal_review_path),
        "diagnostic_report_path": str(diagnostic_report_path),
        "bibliography_path": str(bibliography_path),
        "formal_outline_path": str(outline_path),
        "citation_audit_path": str((output_dir / "citation_audit.json").resolve()),
        "section_quality_audit_path": str((output_dir / "section_quality_audit.json").resolve()),
        "citation_flow_audit_path": str((output_dir / "citation_flow_audit.json").resolve()),
        "subsection_citation_audit_path": str(subsection_citation_audit_path.resolve()),
        "paper_mounts_path": str(paper_mounts_path.resolve()),
        "citation_plan_path": str(citation_plan_path.resolve()),
        "organized_evidence_map_path": str(resolved_evidence_path),
        "organized_evidence_map_generated": generated_evidence,
        "paper_count": evidence_map.get("paper_count"),
        "llm_model": args.llm_model,
        "progress_log_path": str((output_dir / "progress.jsonl").resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a formal literature review and deterministic diagnostic report from lr_search artifacts."
    )
    parser.add_argument("--search-result", default=None, help="Path to lr_search search_result.json.")
    parser.add_argument("--evidence-map", default=None, help="Path to organized_search_result.json.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <artifact_dir>/lr_review.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env with DMX credentials.")
    parser.add_argument(
        "--subject-domain",
        choices=(SUBJECT_CHEMISTRY, SUBJECT_BIOLOGY, SUBJECT_GENERAL),
        default=SUBJECT_CHEMISTRY,
        help=(
            "Subject-specific prompt constraint set to apply during review generation. "
            "Use 'general' to skip chemistry- and biology-specific rules and rely only on the "
            "domain-agnostic GENERAL_SUBSECTION_RULES / GENERAL_INTEGRATION_RULES."
        ),
    )
    parser.add_argument(
        "--case-prompt-patch",
        choices=(CASE_PATCH_NONE, CASE_PATCH_CASE1, CASE_PATCH_CASE2, CASE_PATCH_CASE5),
        default=CASE_PATCH_NONE,
        help="Optional case-specific prompt patch to inject for targeted review refinement.",
    )
    parser.add_argument(
        "--use-gpt",
        action="store_true",
        help="Override the default DMX client config with model/base_url/api_key loaded from ~/.codex/config.toml and ~/.codex/auth.json.",
    )
    parser.add_argument("--llm-api-url", default=DEFAULT_DMX_API_URL, help="DMX chat completions endpoint.")
    parser.add_argument("--llm-model", default=DEFAULT_DMX_MODEL, help="DMX model name.")
    parser.add_argument("--llm-timeout", type=int, default=180, help="LLM timeout in seconds.")
    parser.add_argument("--outline-timeout", type=int, default=600, help="Outline-stage LLM timeout in seconds.")
    parser.add_argument("--integration-timeout", type=int, default=600, help="Integration-stage LLM timeout in seconds.")
    parser.add_argument("--outline-attempts", type=int, default=3, help="Number of outline retries on timeout or invalid JSON.")
    parser.add_argument("--integration-attempts", type=int, default=3, help="Number of integration retries on timeout or invalid JSON.")
    parser.add_argument("--llm-max-tokens", type=int, default=6000, help="LLM max tokens.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use HTTP(S)_PROXY for DMX.")
    parser.add_argument("--min-cluster-score", type=float, default=0.18, help="Minimum lexical score for auto-organization.")
    parser.add_argument("--representatives-per-group", type=int, default=5, help="Representative count when auto-organizing.")
    parser.add_argument("--outline-cluster-limit", type=int, default=5, help="Maximum clusters sent to the outline planner.")
    parser.add_argument("--outline-reps-per-cluster", type=int, default=4, help="Representative papers per cluster in the outline context.")
    parser.add_argument(
        "--method-family-limit",
        type=int,
        default=4,
        help="Maximum number of method-family sections to plan. <=0 means no fixed upper bound; let the evidence decide.",
    )
    parser.add_argument("--section-max-papers", type=int, default=40, help="Maximum evidence papers per section pack.")
    parser.add_argument("--subsection-max-papers", type=int, default=40, help="Maximum evidence papers per subsection pack.")
    parser.add_argument("--subsection-target-words", type=int, default=600, help="Target words per subsection draft.")
    parser.add_argument("--subsection-target-citations", type=int, default=8, help="Target citations per subsection draft when evidence is available.")
    parser.add_argument(
        "--max-parallel-sections",
        type=int,
        default=0,
        help="Maximum number of section drafts to request in parallel. <=0 means use all sections in parallel.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("bibliography", "outline", "packs", "full"),
        default="full",
        help="Stop after an intermediate stage for cheaper debugging. 'packs' is the cheap front-half mode.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing outline, section packs, section drafts, and integration outputs when present.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = generate_reports(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
