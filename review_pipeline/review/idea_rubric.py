from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

from cuda_devices import default_torch_device, device_at, first_configured_cuda_devices
from .common import (
    LEGACY_SECTION_ALIASES,
    REVIEW_SECTIONS,
    RUBRIC_STANDARD_KEYS,
    ensure_directory,
    first_non_empty,
    load_env_values,
    normalize_whitespace,
    slugify,
    write_json,
)
from .llm_retry import call_llm_json_with_retry


DEFAULT_LLM_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_LLM_MODEL_NAME = "DeepSeek-V3.2"
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_EMBED_MODEL_PATH = "/data1/bge-model/AI-ModelScope/bge-large-en-v1.5"
DEFAULT_RERANK_MODEL_PATH = "/data1/bge-reranker-large"
DEFAULT_FAISS_INDEX_PATH = "/data1/nc_dataset/merged_nc_dataset/faiss_nc.index"
DEFAULT_NC_META_PATH = "/data1/nc_dataset/merged_nc_dataset/nc_meta.json"
DEFAULT_SEARCH_TOP_K = 50
DEFAULT_SEARCH_FINAL_K = 15
DEFAULT_MAX_WORKERS = 8
_DEFAULT_CUDA_DEVICES = first_configured_cuda_devices()
DEFAULT_EMBED_DEVICE = device_at(_DEFAULT_CUDA_DEVICES, 0) or default_torch_device(index=0)
DEFAULT_RERANK_DEVICE = device_at(_DEFAULT_CUDA_DEVICES, 1) or default_torch_device(index=1)
RUBRIC_SCHEMA_VERSION = 5
GENERAL_RUBRIC_SYSTEM_PROMPT = """You are a senior academic reviewer for a top journal.
CREATE A PROFESSIONAL EVALUATION RUBRIC divided into Motivation, Method, Result, and Discussion.
Synthesize historical standards and adjust them tightly to the user's idea. The rubric should be balanced across the four sections and should not read like generic journal-review criteria.
Each section should contain 3 to 5 technically specific standards. Prefer standards grounded in the idea's concrete problem framing, mechanism, assumptions, data, evaluation plan, expected result claims, limitations, or deployment context.
Avoid broad standalone dimension names such as "Novelty", "Clarity", "Significance", "Validity", or "Effectiveness". If such a concept is needed, make it domain- and idea-specific.
If a rubric item mentions a formula, variable, metric definition, or symbolic mechanism, express it using Markdown math notation such as `$x$`, `$R = a/b$`, or `$$ ... $$` rather than plain-text equations.
Never wrap math expressions in backticks or code formatting. Use `$...$` or `$$...$$` only.
Because the response must be JSON, every backslash inside a string must be doubled. For example, write `\\\\alpha` or `\\\\hat{x}`, never `\\alpha` or `\\hat{x}`.
All output MUST be strictly in English."""

DETAILED_RUBRIC_SYSTEM_PROMPT = """You are an expert reviewer.
The user provided a Research Idea text. Create a strict list of idea-specific review rules.
IMPORTANT RULES:
1. Only use what is clearly written in the text.
2. DO NOT divide into sections. Put ALL rules into a single list.
3. Use professional, domain-appropriate reviewer language instead of generic or simplified wording.
4. Prefer technical terms, model/component names, optimization objectives, evaluation protocols, theoretical assumptions, and metric names that are explicitly present in the idea.
5. Each rule must be tightly anchored to a concrete mechanism, claim, experimental setup, or failure mode in the idea. Avoid generic criteria that could apply to almost any paper.
6. When possible, name the rule using specialist terminology rather than broad labels such as "novelty", "clarity", or "effectiveness" alone.
7. If the idea contains formulas, variable names, metrics, or symbolic expressions, preserve them using Markdown math notation such as `$x$`, `$dE/dW$`, or `$$ ... $$`.
8. Prefer plain-English names for symbolic expressions when possible, for example "z-hat" instead of a LaTeX command.
9. Never wrap math expressions in backticks or code formatting. Use `$...$` or `$$...$$` only.
10. Because the response must be JSON, every backslash inside a string must be doubled. For example, write `\\\\alpha` or `\\\\hat{x}`, never `\\alpha` or `\\hat{x}`.
All output MUST be strictly in English."""

SECTIONED_SYNTHESIS_SYSTEM_PROMPT = """You are a Senior Reviewer and Area Expert for a top science journal.
You are provided with a user's Research Idea and TWO preliminary review rubrics:
1. [General Rubric]: classified by Motivation, Method, Result, and Discussion.
2. [Detailed Rubric]: a strict, non-hallucinated list of idea-specific rules.

Your task: create the final selected review standards for the report rubric, grouped into exactly four sections: Motivation, Method, Result, and Discussion.

CRITICAL RULES:
1. Use the Detailed Rubric as the main source for idea-specific constraints.
2. Use the General Rubric only when it contains a clearly idea-relevant and technically meaningful standard.
3. Refine and deduplicate overlapping standards.
4. For every final standard, assign source_tag strictly as "General" or "Detailed".
   If a standard merges overlapping concepts from both rubrics, tag it "Detailed".
5. For every final standard, assign target_section strictly as one of "Motivation", "Method", "Result", or "Discussion".
6. All Detailed standards must be assigned to one of the four sections; there is no Idea-Specific section.
7. Keep the four sections balanced. Produce 3 to 5 final standards per section unless the idea lacks enough evidence for a section.
8. The dimension_name MUST NOT contain the "&" symbol and MUST focus on one single concept.
9. Do not hallucinate concepts not present in the Research Idea.
10. Standards must be professional and idea-specific. Avoid generic journal-review clichés.
11. Avoid broad standalone dimension names such as "Novelty", "Clarity", "Significance", "Validity", or "Effectiveness". If such a concept is necessary, narrow it to a concrete mechanism, claim, experimental protocol, or literature contrast in the idea.
12. Motivation standards should focus on problem framing, literature positioning, domain need, and the specificity of the claimed contribution.
13. Method standards should focus on mechanism, algorithmic design, assumptions, privacy/safety constraints, implementation details, and theoretical consistency.
14. Result standards should focus on evaluation protocols, datasets, baselines, metrics, ablations, statistical reliability, and whether the expected evidence can support the claims.
15. Discussion standards should focus on limitations, boundary conditions, failure modes, generalizability, deployment implications, ethical/societal risks, and future work.
16. If required_evidence mentions formulas, variables, metrics, or symbolic definitions, write them using Markdown math notation rather than plain-text equations.
17. Never wrap math expressions in backticks or code formatting. Use `$...$` or `$$...$$` only.
18. Because the response must be JSON, every backslash inside a string must be doubled. For example, write `\\\\alpha` or `\\\\hat{x}`, never `\\alpha` or `\\hat{x}`.
All output MUST be strictly in English."""


class PaperSummary(BaseModel):
    paper_title: str = Field(description="The full title of the paper. Provide 'Unknown' if not explicitly stated.")
    abstract_content: list[str] = Field(description="Factual summary of the Abstract.")
    introduction_content: list[str] = Field(description="Factual summary of the Introduction section.")
    methods_content: list[str] = Field(description="Factual summary of the Methods section.")
    results_content: list[str] = Field(description="Factual summary of the Results section.")
    discussion_content: list[str] = Field(description="Factual summary of the Discussion/Conclusion section.")


class ExtractedDimension(BaseModel):
    dimension_name: str = Field(description="A simple and clear name for this rule.")
    core_reason: str = Field(description="Why does the reviewer care about this?")
    required_evidence: list[str] = Field(description="What exact proof did the reviewer ask for?")


class ModuleStandards(BaseModel):
    standards: list[ExtractedDimension] = Field(description="List of evaluation standards for this module.")


class ReviewStandard(BaseModel):
    dimension_name: str = Field(description="A concise evaluation dimension.")
    core_philosophy: str = Field(description="Why reviewers will care about this dimension.")
    required_evidence: str = Field(description="The evidence needed to justify this dimension.")


class GeneralReviewRubric(BaseModel):
    Idea_Summary: str = Field(description="A brief one-sentence summary of the proposed idea.")
    Motivation_Standards: list[ReviewStandard] = Field(description="Professional criteria for Motivation.")
    Method_Standards: list[ReviewStandard] = Field(description="Professional criteria for Method.")
    Result_Standards: list[ReviewStandard] = Field(description="Professional criteria for Result.")
    Discussion_Standards: list[ReviewStandard] = Field(description="General criteria for Discussion/Conclusion.")


class DetailedUnifiedRubric(BaseModel):
    Evaluation_Standards: list[ReviewStandard] = Field(
        description="A strict flat list of idea-specific review rules. DO NOT divide into paper sections."
    )


class IdeaBreakdown(BaseModel):
    motivation_and_problem: str = Field(
        description="The main problem the idea wants to solve. If not found, write 'Not mentioned'."
    )
    proposed_method: str = Field(
        description="The exact method or model in the idea. If not found, write 'Not mentioned'."
    )
    experiment_and_data: str = Field(
        description="The dataset or tests clearly written in the idea. If not found, write 'Not mentioned'."
    )


class RefinedStandard(BaseModel):
    dimension_name: str = Field(
        description="A clear rule name. DO NOT use the '&' symbol. Each name must focus on one single concept."
    )
    core_philosophy: str = Field(description="Why this rule is important.")
    required_evidence: str = Field(description="What the author must do to pass this rule.")
    source_tag: str = Field(description="The origin of this standard. Must be strictly 'General' or 'Detailed'.")
    target_section: str = Field(
        description=(
            "The section this standard belongs to. Must be strictly one of "
            "'Motivation', 'Method', 'Result', or 'Discussion'."
        )
    )


class SectionedSynthesisRubric(BaseModel):
    Idea_Summary: str = Field(description="A brief one-sentence summary of the proposed idea.")
    Idea_Breakdown: IdeaBreakdown = Field(description="Break down the idea to understand it clearly.")
    Motivation_Standards: list[RefinedStandard] = Field(description="Final standards for Motivation.")
    Method_Standards: list[RefinedStandard] = Field(description="Final standards for Method.")
    Result_Standards: list[RefinedStandard] = Field(description="Final standards for Result.")
    Discussion_Standards: list[RefinedStandard] = Field(description="Final standards for Discussion.")


@dataclass(slots=True)
class IdeaRubricConfig:
    target_idea: str
    artifact_root: Path
    output_path: Path
    sources_root: Path | None = None
    env_path: Path | None = None
    llm_api_key: str | None = None
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_timeout_seconds: int = 300
    embed_model_path: str = DEFAULT_EMBED_MODEL_PATH
    rerank_model_path: str = DEFAULT_RERANK_MODEL_PATH
    faiss_index_path: str = DEFAULT_FAISS_INDEX_PATH
    nc_meta_path: str = DEFAULT_NC_META_PATH
    search_top_k: int = DEFAULT_SEARCH_TOP_K
    search_final_k: int = DEFAULT_SEARCH_FINAL_K
    max_workers: int = DEFAULT_MAX_WORKERS
    embed_device: str = DEFAULT_EMBED_DEVICE
    rerank_device: str = DEFAULT_RERANK_DEVICE


def _clean_json_response(result_str: str) -> str:
    stripped = result_str.strip()
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if match:
        return match.group(0)
    return stripped


def _compact_text(value: Any) -> str:
    return normalize_whitespace(value)


def _summarize_general_rubric_for_synthesis(general_result: dict[str, Any]) -> str:
    section_keys = [(section, RUBRIC_STANDARD_KEYS[section]) for section in REVIEW_SECTIONS]
    lines: list[str] = []
    summary = _compact_text(general_result.get("Idea_Summary"))
    if summary:
        lines.append(f"Idea Summary: {summary}")
    for section_name, key in section_keys:
        lines.append(f"{section_name}:")
        standards = general_result.get(key) if isinstance(general_result.get(key), list) else []
        if not standards:
            lines.append("- No standards.")
            continue
        for standard in standards:
            if not isinstance(standard, dict):
                continue
            dimension = _compact_text(standard.get("dimension_name")) or "Unnamed dimension"
            philosophy = _compact_text(standard.get("core_philosophy"))
            evidence = _compact_text(standard.get("required_evidence"))
            line = f"- {dimension}"
            if philosophy:
                line += f": {philosophy}"
            if evidence:
                line += f" | Evidence: {evidence}"
            lines.append(line)
    return "\n".join(lines)


def _summarize_detailed_rubric_for_synthesis(detailed_result: dict[str, Any]) -> str:
    lines = ["Detailed standards:"]
    standards = detailed_result.get("Evaluation_Standards") if isinstance(detailed_result.get("Evaluation_Standards"), list) else []
    if not standards:
        lines.append("- No standards.")
        return "\n".join(lines)
    for standard in standards:
        if not isinstance(standard, dict):
            continue
        dimension = _compact_text(standard.get("dimension_name")) or "Unnamed dimension"
        philosophy = _compact_text(standard.get("core_philosophy"))
        evidence = _compact_text(standard.get("required_evidence"))
        line = f"- {dimension}"
        if philosophy:
            line += f": {philosophy}"
        if evidence:
            line += f" | Evidence: {evidence}"
        lines.append(line)
    return "\n".join(lines)


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        document = fitz.open(pdf_path)
        return "\n".join(page.get_text("text") for page in document)
    except Exception:
        return ""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _filter_review_report(full_text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", full_text)
    start_patterns = [
        r"(?i)(Reviewers['’]?\s*comments\s*:?)",
        r"(?i)(Reviewer\s*(?:#)?\s*1\s*:?)",
        r"(?i)(Comments to the Author(?:s)?\s*:?)",
    ]
    end_patterns = [
        r"(?i)(Author(?:['’]s)?\s*Rebuttal)",
        r"(?i)(Point-by-point\s*response)",
        r"(?i)(Publisher['’]s note)",
    ]

    start_index = 0
    for pattern in start_patterns:
        match = re.search(pattern, text)
        if match:
            start_index = match.start()
            break

    end_index = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text[start_index:])
        if match and start_index + match.start() < end_index:
            end_index = start_index + match.start()

    clean_text = text[start_index:end_index].strip()
    if len(clean_text) >= 200:
        return clean_text
    return text[min(500, len(text)) : end_index].strip()


def _fetch_pmc_xml_by_doi(doi: str) -> bytes | None:
    clean_doi = normalize_whitespace(doi)
    if not clean_doi:
        return None
    clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean_doi, flags=re.IGNORECASE)
    clean_doi = clean_doi.strip()

    search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    try:
        response = requests.get(
            search_url,
            params={"query": f"DOI:{clean_doi}", "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        result_list = response.json().get("resultList", {}).get("result", [])
        pmcid = normalize_whitespace(result_list[0].get("pmcid")) if result_list else ""
        if not pmcid:
            return None
        fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        xml_response = requests.get(fulltext_url, timeout=15)
        xml_response.raise_for_status()
        return xml_response.content
    except Exception:
        return None


def _parse_jats_xml_sections(xml_content: bytes) -> dict[str, str]:
    soup = BeautifulSoup(xml_content, "xml")
    sections = {
        "Abstract": "",
        "Introduction": "",
        "Results": "",
        "Discussion": "",
        "Methods": "",
    }

    abstract_tag = soup.find("abstract")
    if abstract_tag:
        sections["Abstract"] = abstract_tag.get_text(separator="\n", strip=True)

    body_tag = soup.find("body")
    if body_tag:
        for sec in body_tag.find_all("sec", recursive=False):
            title_tag = sec.find("title")
            if not title_tag:
                continue
            title_text = title_tag.get_text(strip=True).lower()
            content = sec.get_text(separator="\n", strip=True)[len(title_tag.get_text(strip=True)) :].strip()
            if "introduction" in title_text or "background" in title_text:
                sections["Introduction"] = content
            elif "result" in title_text:
                sections["Results"] = content
            elif "discussion" in title_text or "conclusion" in title_text:
                sections["Discussion"] = content
            elif "method" in title_text or "experimental" in title_text:
                sections["Methods"] = content
    return sections


GENERAL_SECTION_KEYS = tuple(RUBRIC_STANDARD_KEYS[section] for section in REVIEW_SECTIONS)
GENERAL_SECTION_NAME_TO_KEY = dict(RUBRIC_STANDARD_KEYS)


def _canonical_rubric_section(value: Any) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    for section in REVIEW_SECTIONS:
        if text.casefold() == section.casefold():
            return section
    for legacy, canonical in LEGACY_SECTION_ALIASES.items():
        if text.casefold() == legacy.casefold():
            return canonical
    return ""


def _standards_with_source(payload: dict[str, Any], key: str, source_tag: str) -> list[dict[str, Any]]:
    standards = payload.get(key) if isinstance(payload.get(key), list) else []
    tagged: list[dict[str, Any]] = []
    for standard in standards:
        if not isinstance(standard, dict):
            continue
        item = dict(standard)
        item["source_tag"] = source_tag
        tagged.append(item)
    return tagged


def _clean_refined_standard(raw: dict[str, Any]) -> dict[str, Any]:
    standard = dict(raw)
    source_tag = normalize_whitespace(standard.get("source_tag"))
    if source_tag not in {"General", "Detailed"}:
        source_tag = "Detailed"
    standard["source_tag"] = source_tag
    target_section = _canonical_rubric_section(standard.get("target_section"))
    standard["target_section"] = target_section
    dimension_name = normalize_whitespace(standard.get("dimension_name")).replace("&", "and")
    if dimension_name:
        standard["dimension_name"] = dimension_name
    return standard


def _standard_dedup_key(raw: dict[str, Any]) -> str:
    return normalize_whitespace(
        " | ".join(
            [
                normalize_whitespace(raw.get("dimension_name")).lower(),
                normalize_whitespace(raw.get("core_philosophy")).lower(),
                normalize_whitespace(raw.get("required_evidence")).lower(),
            ]
        )
    )


def _build_canonical_rubric(
    *,
    target_idea: str,
    general_result: dict[str, Any],
    synthesis_result: dict[str, Any],
) -> dict[str, Any]:
    min_per_section = 3
    max_per_section = 5
    rubric: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "Idea_Summary": normalize_whitespace(synthesis_result.get("Idea_Summary"))
        or normalize_whitespace(general_result.get("Idea_Summary"))
        or normalize_whitespace(target_idea)[:500],
        "Idea_Breakdown": synthesis_result.get("Idea_Breakdown")
        if isinstance(synthesis_result.get("Idea_Breakdown"), dict)
        else None,
    }
    candidates_by_key: dict[str, list[dict[str, Any]]] = {}
    for key in GENERAL_SECTION_KEYS:
        candidates_by_key[key] = _standards_with_source(general_result, key, "General")
        rubric[key] = []

    seen_keys_by_section: dict[str, set[str]] = {key: set() for key in GENERAL_SECTION_KEYS}

    for section, key in GENERAL_SECTION_NAME_TO_KEY.items():
        raw_standards = synthesis_result.get(key) if isinstance(synthesis_result.get(key), list) else []
        for standard in raw_standards:
            if not isinstance(standard, dict):
                continue
            cleaned = _clean_refined_standard({**standard, "target_section": section})
            dedup_key = _standard_dedup_key(cleaned)
            if not dedup_key or dedup_key in seen_keys_by_section[key]:
                continue
            if len(rubric[key]) >= max_per_section:
                continue
            seen_keys_by_section[key].add(dedup_key)
            rubric[key].append(cleaned)

    for key in GENERAL_SECTION_KEYS:
        if len(rubric[key]) >= min_per_section:
            continue
        for standard in candidates_by_key[key]:
            if len(rubric[key]) >= min_per_section:
                break
            if not isinstance(standard, dict):
                continue
            dedup_key = _standard_dedup_key(standard)
            if not dedup_key or dedup_key in seen_keys_by_section[key]:
                continue
            section_name = next((section for section, section_key in GENERAL_SECTION_NAME_TO_KEY.items() if section_key == key), "")
            rubric[key].append(_clean_refined_standard({**standard, "target_section": section_name}))
            seen_keys_by_section[key].add(dedup_key)

    for key in GENERAL_SECTION_KEYS:
        rubric[key] = rubric[key][:max_per_section]
    return rubric


class PaperSearchEngine:
    def __init__(self, config: IdeaRubricConfig) -> None:
        self.index = faiss.read_index(config.faiss_index_path)
        self.metadata_store = json.loads(Path(config.nc_meta_path).read_text(encoding="utf-8"))
        self.embed_model = SentenceTransformer(config.embed_model_path, device=config.embed_device)
        self.rerank_model = CrossEncoder(config.rerank_model_path, device=config.rerank_device)

    def search(self, idea: str, *, top_k: int, final_k: int) -> list[dict[str, Any]]:
        query_vector = self.embed_model.encode(
            ["Represent this sentence for searching relevant passages: " + idea],
            normalize_embeddings=True,
        )
        distances, indices = self.index.search(np.array(query_vector).astype("float32"), top_k + 5)

        retrieved_docs: list[dict[str, Any]] = []
        idea_lower = idea.strip().lower()
        for idx_pos, index in enumerate(indices[0]):
            doc = dict(self.metadata_store[index])
            title = normalize_whitespace(doc.get("Title"))
            if title and title.lower() in idea_lower:
                continue
            if float(distances[0][idx_pos]) > 0.9:
                continue
            retrieved_docs.append(doc)
            if len(retrieved_docs) >= top_k:
                break

        cross_inputs = [[idea, doc["doc_text"]] for doc in retrieved_docs]
        rerank_scores = self.rerank_model.predict(cross_inputs, batch_size=16)
        for index, doc in enumerate(retrieved_docs):
            doc["rerank_score"] = float(rerank_scores[index])
        return sorted(retrieved_docs, key=lambda item: item["rerank_score"], reverse=True)[:final_k]


class IdeaRubricRunner:
    def __init__(self, config: IdeaRubricConfig) -> None:
        self.config = self._resolve_config(config)
        self._timing_lock = threading.Lock()
        self.timing_log_path = self.config.artifact_root / "timing.log"
        if not self.config.target_idea:
            raise ValueError("target_idea must not be empty")
        if not self.config.llm_api_key:
            raise ValueError("LLM API key is required for idea rubric generation")

    def _log_timing(self, event: str, **fields: Any) -> None:
        payload = {"ts": round(time.time(), 3), "event": event, **fields}
        line = json.dumps(payload, ensure_ascii=False)
        with self._timing_lock:
            with self.timing_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _resolve_config(self, config: IdeaRubricConfig) -> IdeaRubricConfig:
        env_values = load_env_values(config.env_path)
        return IdeaRubricConfig(
            target_idea=normalize_whitespace(config.target_idea),
            artifact_root=Path(config.artifact_root).expanduser().resolve(),
            output_path=Path(config.output_path).expanduser().resolve(),
            sources_root=Path(config.sources_root).expanduser().resolve() if config.sources_root else None,
            env_path=Path(config.env_path).expanduser().resolve() if config.env_path else None,
            llm_api_key=first_non_empty(
                config.llm_api_key,
                env_values.get("DMX-API-KEY"),
                env_values.get("DMX_API_KEY"),
                env_values.get("OPENAI_API_KEY"),
            )
            or None,
            llm_base_url=first_non_empty(
                config.llm_base_url,
                env_values.get("OPENAI_BASE_URL"),
                DEFAULT_LLM_BASE_URL,
            ),
            llm_model_name=first_non_empty(config.llm_model_name, DEFAULT_LLM_MODEL_NAME),
            llm_temperature=float(config.llm_temperature),
            llm_timeout_seconds=int(config.llm_timeout_seconds),
            embed_model_path=first_non_empty(config.embed_model_path, DEFAULT_EMBED_MODEL_PATH),
            rerank_model_path=first_non_empty(config.rerank_model_path, DEFAULT_RERANK_MODEL_PATH),
            faiss_index_path=first_non_empty(config.faiss_index_path, DEFAULT_FAISS_INDEX_PATH),
            nc_meta_path=first_non_empty(config.nc_meta_path, DEFAULT_NC_META_PATH),
            search_top_k=int(config.search_top_k),
            search_final_k=int(config.search_final_k),
            max_workers=int(config.max_workers),
            embed_device=first_non_empty(config.embed_device, DEFAULT_EMBED_DEVICE),
            rerank_device=first_non_empty(config.rerank_device, DEFAULT_RERANK_DEVICE),
        )

    @property
    def sources_root(self) -> Path:
        return self.config.sources_root or self.config.artifact_root

    def call_llm_with_retry(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_model_schema: Any,
        max_retries: int = 3,
        temperature: float | None = None,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        effective_temperature = self.config.llm_temperature if temperature is None else temperature
        return call_llm_json_with_retry(
            api_key=self.config.llm_api_key or "",
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model_schema=response_model_schema,
            timeout_seconds=self.config.llm_timeout_seconds,
            max_retries=max_retries,
            temperature=effective_temperature,
            max_tokens=8192,
            helper_path=Path(__file__).with_name("llm_json_call_worker.py"),
            label=label,
            debug_dir=self.config.artifact_root / "llm_failures",
            log_attempt=lambda **kwargs: self._log_timing("llm_call", **kwargs),
        )

    def _build_doc_artifact_dir(self, index: int, doc: dict[str, Any]) -> Path:
        title = normalize_whitespace(doc.get("Title")) or normalize_whitespace(doc.get("title")) or f"paper-{index}"
        return ensure_directory(self.sources_root / "papers" / f"{index:02d}_{slugify(title, limit=60)}")

    def process_single_retrieved_paper_sources(self, doc: dict[str, Any]) -> dict[str, Any]:
        paper_started_at = time.perf_counter()
        source_folder = Path(normalize_whitespace(doc.get("source_folder_path") or doc.get("folder_path")))
        artifact_dir = Path(doc["artifact_dir"])
        title = normalize_whitespace(doc.get("Title"))
        doi = normalize_whitespace(doc.get("DOI"))
        timings_ms: dict[str, Any] = {}

        def finish(status: str, reason: str | None = None) -> dict[str, Any]:
            elapsed_ms = round((time.perf_counter() - paper_started_at) * 1000, 1)
            payload = {
                "title": title,
                "artifact_dir": str(artifact_dir),
                "status": status,
                "elapsed_ms": elapsed_ms,
                "timings_ms": timings_ms,
            }
            if reason:
                payload["reason"] = reason
            self._log_timing(
                "paper_done",
                title=title,
                artifact_dir=str(artifact_dir),
                status=status,
                elapsed_ms=elapsed_ms,
                reason=reason,
            )
            return payload

        if not source_folder.exists():
            return finish("skipped", "source folder does not exist")

        sections_path = artifact_dir / "extracted_sections.json"
        review_text_path = artifact_dir / "review_text.txt"
        source_meta_path = artifact_dir / "source_meta.json"

        if not sections_path.exists():
            step_started_at = time.perf_counter()
            xml_content = _fetch_pmc_xml_by_doi(doi)
            if not xml_content:
                timings_ms["sections"] = round((time.perf_counter() - step_started_at) * 1000, 1)
                return finish("skipped", "PMC XML unavailable")
            sections = _parse_jats_xml_sections(xml_content)
            if not any(sections.values()):
                timings_ms["sections"] = round((time.perf_counter() - step_started_at) * 1000, 1)
                return finish("skipped", "no parsed sections")
            write_json(sections_path, sections)
            timings_ms["sections"] = round((time.perf_counter() - step_started_at) * 1000, 1)

        if not review_text_path.exists():
            step_started_at = time.perf_counter()
            review_pdf_path = next(source_folder.glob("review*.pdf"), None)
            if review_pdf_path is None:
                return finish("skipped", "review PDF missing")
            peer_raw_text = _extract_pdf_text(review_pdf_path)
            if not peer_raw_text:
                return finish("skipped", "review PDF text extraction failed")
            clean_review = _filter_review_report(peer_raw_text)
            _write_text(review_text_path, clean_review)
            timings_ms["review_text"] = round((time.perf_counter() - step_started_at) * 1000, 1)

        if not source_meta_path.exists():
            write_json(
                source_meta_path,
                {
                    "title": title,
                    "doi": doi,
                    "source_folder_path": str(source_folder.resolve()),
                    "artifact_dir": str(artifact_dir.resolve()),
                },
            )

        return finish("ok")

    def process_single_retrieved_paper_llm(self, doc: dict[str, Any]) -> dict[str, Any]:
        paper_started_at = time.perf_counter()
        artifact_dir = Path(doc["artifact_dir"])
        title = normalize_whitespace(doc.get("Title") or doc.get("title"))
        timings_ms: dict[str, Any] = {}

        def finish(status: str, reason: str | None = None) -> dict[str, Any]:
            elapsed_ms = round((time.perf_counter() - paper_started_at) * 1000, 1)
            payload = {
                "title": title,
                "artifact_dir": str(artifact_dir),
                "status": status,
                "elapsed_ms": elapsed_ms,
                "timings_ms": timings_ms,
            }
            if reason:
                payload["reason"] = reason
            self._log_timing(
                "paper_llm_done",
                title=title,
                artifact_dir=str(artifact_dir),
                status=status,
                elapsed_ms=elapsed_ms,
                reason=reason,
            )
            return payload

        sections_path = Path(normalize_whitespace(doc.get("sections_path"))) if normalize_whitespace(doc.get("sections_path")) else artifact_dir / "extracted_sections.json"
        review_text_path = Path(normalize_whitespace(doc.get("review_text_path"))) if normalize_whitespace(doc.get("review_text_path")) else artifact_dir / "review_text.txt"
        summary_path = artifact_dir / "summary.json"
        dimensions_path = artifact_dir / "dimensions.json"

        if not sections_path.exists():
            return finish("skipped", "sections file missing")
        if not review_text_path.exists():
            return finish("skipped", "review text file missing")

        if not summary_path.exists():
            step_started_at = time.perf_counter()
            sections_text = sections_path.read_text(encoding="utf-8")
            summary_data = self.call_llm_with_retry(
                system_prompt="Extract factual summaries for EXACTLY 5 specific sections based on the provided JSON.",
                user_content=sections_text,
                response_model_schema=PaperSummary,
                temperature=0.1,
                label=f"summary:{title}",
            )
            if not summary_data:
                timings_ms["summary"] = round((time.perf_counter() - step_started_at) * 1000, 1)
                return finish("skipped", "summary LLM failed")
            write_json(summary_path, summary_data)
            timings_ms["summary"] = round((time.perf_counter() - step_started_at) * 1000, 1)

        if not dimensions_path.exists():
            step_started_at = time.perf_counter()
            summary_dict = json.loads(summary_path.read_text(encoding="utf-8"))
            clean_review = review_text_path.read_text(encoding="utf-8")

            def extract_module(name: str, json_key: str, text: str) -> tuple[str, list[dict[str, Any]], float]:
                module_started_at = time.perf_counter()
                if not text.strip():
                    return json_key, [], 0.0
                result = self.call_llm_with_retry(
                    system_prompt=f"Extract reviewer standards specifically for the {name} section.",
                    user_content=f"### Summary:\n{text}\n### Review:\n{clean_review}",
                    response_model_schema=ModuleStandards,
                    temperature=0.1,
                    label=f"module:{title}:{name}",
                )
                standards = result.get("standards") if isinstance(result, dict) else None
                elapsed_ms = round((time.perf_counter() - module_started_at) * 1000, 1)
                self._log_timing(
                    "module_done",
                    title=title,
                    module=name,
                    status="ok" if isinstance(standards, list) else "skipped",
                    elapsed_ms=elapsed_ms,
                    standard_count=len(standards) if isinstance(standards, list) else 0,
                )
                return json_key, standards if isinstance(standards, list) else [], elapsed_ms

            intro_text = "\n".join(
                summary_dict.get("abstract_content", []) + summary_dict.get("introduction_content", [])
            )
            module_inputs = [
                ("Introduction", "introduction_standards", intro_text),
                ("Methods", "methods_standards", "\n".join(summary_dict.get("methods_content", []))),
                ("Results", "results_standards", "\n".join(summary_dict.get("results_content", []))),
                ("Discussion", "discussion_standards", "\n".join(summary_dict.get("discussion_content", []))),
            ]
            final_criteria: dict[str, list[dict[str, Any]]] = {key: [] for _, key, _ in module_inputs}
            module_timings_ms: dict[str, float] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(module_inputs)) as executor:
                future_map = {
                    executor.submit(extract_module, name, json_key, text): name
                    for name, json_key, text in module_inputs
                }
                for future in concurrent.futures.as_completed(future_map):
                    json_key, standards, elapsed_ms = future.result()
                    final_criteria[json_key] = standards
                    module_timings_ms[json_key] = elapsed_ms
            timings_ms["dimensions"] = round((time.perf_counter() - step_started_at) * 1000, 1)
            timings_ms["dimension_modules"] = module_timings_ms
            if not any(final_criteria.values()):
                return finish("skipped", "dimension extraction returned no standards")
            write_json(dimensions_path, final_criteria)

        return finish("ok")

    def extract_criteria_context(self, retrieved_docs: list[dict[str, Any]]) -> str:
        extracted_data: dict[str, set[str]] = {
            "Motivation": set(),
            "Method": set(),
            "Result": set(),
            "Discussion": set(),
        }
        key_mapping = {
            "introduction_standards": "Motivation",
            "methods_standards": "Method",
            "results_standards": "Result",
            "discussion_standards": "Discussion",
        }

        for doc in retrieved_docs:
            artifact_dir = Path(doc["artifact_dir"])
            dim_file = artifact_dir / "dimensions.json"
            if not dim_file.exists():
                continue
            try:
                data = json.loads(dim_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for json_key, module_name in key_mapping.items():
                for standard in data.get(json_key, []):
                    if standard.get("dimension_name") and standard.get("core_reason"):
                        extracted_data[module_name].add(
                            f"- Dimension: {standard.get('dimension_name')}\n"
                            f"  Reason: {standard.get('core_reason')}"
                        )

        context_blocks: list[str] = []
        for module_name, items in extracted_data.items():
            if items:
                context_blocks.append(
                    f"=== {module_name} Historical Standards ===\n" + "\n\n".join(sorted(items))
                )
            else:
                context_blocks.append(f"=== {module_name} Historical Standards ===\nNo data.")
        return "\n\n".join(context_blocks)

    def build_sources(self) -> dict[str, Any]:
        ensure_directory(self.sources_root)
        engine = PaperSearchEngine(self.config)
        retrieved_docs = engine.search(
            self.config.target_idea,
            top_k=self.config.search_top_k,
            final_k=self.config.search_final_k,
        )
        if not retrieved_docs:
            raise RuntimeError("Idea rubric search returned no candidate papers")

        for index, doc in enumerate(retrieved_docs, start=1):
            doc["source_folder_path"] = normalize_whitespace(doc.get("folder_path"))
            doc["artifact_dir"] = str(self._build_doc_artifact_dir(index, doc).resolve())

        retrieved_papers_path = self.sources_root / "retrieved_papers.json"
        write_json(retrieved_papers_path, {"papers": retrieved_docs})

        processing_results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {executor.submit(self.process_single_retrieved_paper_sources, doc): doc for doc in retrieved_docs}
            for future in concurrent.futures.as_completed(future_map):
                processing_results.append(future.result())

        processing_index_path = self.sources_root / "processing_index.json"
        write_json(processing_index_path, {"papers": processing_results})

        paper_contexts_path = self.sources_root / "paper_contexts.json"
        write_json(
            paper_contexts_path,
            {
                "papers": [
                    {
                        "title": normalize_whitespace(doc.get("Title")),
                        "doi": normalize_whitespace(doc.get("DOI")),
                        "artifact_dir": doc["artifact_dir"],
                        "sections_path": str((Path(doc["artifact_dir"]) / "extracted_sections.json").resolve()),
                        "review_text_path": str((Path(doc["artifact_dir"]) / "review_text.txt").resolve()),
                        "source_meta_path": str((Path(doc["artifact_dir"]) / "source_meta.json").resolve()),
                    }
                    for doc in retrieved_docs
                ]
            },
        )
        return {
            "status": "ok",
            "retrieved_paper_count": len(retrieved_docs),
            "retrieved_papers_path": str(retrieved_papers_path.resolve()),
            "processing_index_path": str(processing_index_path.resolve()),
            "paper_contexts_path": str(paper_contexts_path.resolve()),
            "timing_log_path": str(self.timing_log_path.resolve()),
            "sources_root": str(self.sources_root.resolve()),
            "result_path": str(paper_contexts_path.resolve()),
        }

    def run_llm(self) -> dict[str, Any]:
        ensure_directory(self.config.artifact_root)
        ensure_directory(self.config.output_path.parent)
        paper_contexts_path = self.sources_root / "paper_contexts.json"
        if not paper_contexts_path.exists():
            raise RuntimeError("Rubric sources are missing paper_contexts.json")

        paper_contexts_payload = json.loads(paper_contexts_path.read_text(encoding="utf-8"))
        retrieved_docs = paper_contexts_payload.get("papers") if isinstance(paper_contexts_payload, dict) else None
        if not isinstance(retrieved_docs, list) or not retrieved_docs:
            raise RuntimeError("Rubric sources returned no paper contexts")

        llm_docs: list[dict[str, Any]] = []
        for item in retrieved_docs:
            if not isinstance(item, dict):
                continue
            source_artifact_dir = Path(normalize_whitespace(item.get("artifact_dir")))
            title = normalize_whitespace(item.get("title")) or source_artifact_dir.name
            llm_docs.append(
                {
                    "Title": title,
                    "DOI": normalize_whitespace(item.get("doi")),
                    "artifact_dir": str(
                        ensure_directory(self.config.artifact_root / "papers" / source_artifact_dir.name).resolve()
                    ),
                    "sections_path": normalize_whitespace(item.get("sections_path")),
                    "review_text_path": normalize_whitespace(item.get("review_text_path")),
                }
            )

        processing_results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {executor.submit(self.process_single_retrieved_paper_llm, doc): doc for doc in llm_docs}
            for future in concurrent.futures.as_completed(future_map):
                processing_results.append(future.result())

        processing_index_path = self.config.artifact_root / "processing_index.json"
        write_json(processing_index_path, {"papers": processing_results})

        historical_context = self.extract_criteria_context(retrieved_docs)
        historical_context_path = self.config.artifact_root / "historical_context.txt"
        _write_text(historical_context_path, historical_context)
        general_user_content = (
            f"=== IDEA ===\n{self.config.target_idea}\n\n"
            f"=== HISTORICAL STANDARDS ===\n{historical_context}"
        )
        detailed_user_content = (
            f"=== USER IDEA ===\n{self.config.target_idea}\n\n"
            f"=== HISTORICAL STANDARDS ===\n{historical_context}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_general = executor.submit(
                self.call_llm_with_retry,
                system_prompt=GENERAL_RUBRIC_SYSTEM_PROMPT,
                user_content=general_user_content,
                response_model_schema=GeneralReviewRubric,
                label="general_rubric",
            )
            future_detailed = executor.submit(
                self.call_llm_with_retry,
                system_prompt=DETAILED_RUBRIC_SYSTEM_PROMPT,
                user_content=detailed_user_content,
                response_model_schema=DetailedUnifiedRubric,
                label="detailed_rubric",
            )
            general_result = future_general.result()
            detailed_result = future_detailed.result()

        if not general_result:
            raise RuntimeError("Failed to generate general rubric")
        if not detailed_result:
            raise RuntimeError("Failed to generate detailed rubric")

        general_rubric_path = self.config.artifact_root / "general_rubric.json"
        detailed_rubric_path = self.config.artifact_root / "detailed_rubric.json"
        write_json(general_rubric_path, general_result)
        write_json(detailed_rubric_path, detailed_result)

        synthesis_user_content = "\n".join(
            [
                "=== USER RESEARCH IDEA ===",
                self.config.target_idea,
                "",
                "=== PRELIMINARY RUBRIC 1 (General) ===",
                _summarize_general_rubric_for_synthesis(general_result),
                "",
                "=== PRELIMINARY RUBRIC 2 (Detailed) ===",
                _summarize_detailed_rubric_for_synthesis(detailed_result),
            ]
        )
        synthesis_result = self.call_llm_with_retry(
            system_prompt=SECTIONED_SYNTHESIS_SYSTEM_PROMPT,
            user_content=synthesis_user_content,
            response_model_schema=SectionedSynthesisRubric,
            label="sectioned_rubric_synthesis",
        )
        if not synthesis_result:
            raise RuntimeError("Failed to synthesize sectioned rubric")

        synthesis_rubric_path = self.config.artifact_root / "sectioned_rubric_synthesis.json"
        write_json(synthesis_rubric_path, synthesis_result)

        rubric = _build_canonical_rubric(
            target_idea=self.config.target_idea,
            general_result=general_result,
            synthesis_result=synthesis_result,
        )
        if not any(rubric.get(key) for key in GENERAL_SECTION_KEYS):
            raise RuntimeError("Sectioned rubric synthesis returned no standards")

        write_json(self.config.output_path, rubric)
        return {
            "status": "ok",
            "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
            "processing_index_path": str(processing_index_path.resolve()),
            "historical_context_path": str(historical_context_path.resolve()),
            "general_rubric_path": str(general_rubric_path.resolve()),
            "detailed_rubric_path": str(detailed_rubric_path.resolve()),
            "synthesis_rubric_path": str(synthesis_rubric_path.resolve()),
            "timing_log_path": str(self.timing_log_path.resolve()),
            "result_path": str(self.config.output_path.resolve()),
        }

    def run(self) -> dict[str, Any]:
        self.build_sources()
        return self.run_llm()


def run_idea_rubric(config: IdeaRubricConfig) -> dict[str, Any]:
    runner = IdeaRubricRunner(config)
    return runner.run()


def run_rubric_sources(config: IdeaRubricConfig) -> dict[str, Any]:
    runner = IdeaRubricRunner(config)
    return runner.build_sources()


def run_rubric_llm(config: IdeaRubricConfig) -> dict[str, Any]:
    runner = IdeaRubricRunner(config)
    return runner.run_llm()
