#!/usr/bin/env python3
"""Ground idea text or a paper PDF against paragraph-level evidence from retrieved papers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..llm import resolve_llm_settings
from ..llm.client import build_llm_client

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "runs" / "pdf_manifest" / "manifest.json"
DEFAULT_TARGET_DIR = Path("/tmp/scischolar_grounding/target")
DEFAULT_RESULT_DIR = Path("/tmp/scischolar_grounding/result")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert scientific analysis agent. "
    "Transform a research idea or paper into structured, atomic, evaluation-ready research components. "
    "Every output must be factual, concise, self-contained, and independently understandable. "
    "Return strict JSON only."
)
EXTRACTION_USER_PROMPT = """Analyze the academic input below and extract structured scientific content.

Requirements:
- Extract these sections only: `basic_idea`, `motivation`, `method`, `experimental_focus`.
- Every item must be atomic, self-contained, and independently understandable.
- Do not use vague references such as "the framework", "this method", "the proposed approach", or "this component" unless the referent is explicitly named in the same sentence.
- Do not use undefined acronyms or paper-internal shorthand without explanation.
- Stay faithful to the input. Do not invent details, datasets, baselines, metrics, claims, or terminology.
- Different items should not be near-duplicates.
- `basic_idea` should summarize the core innovation, task, and key mechanism.
- `motivation` should contain concrete limitations, gaps, or needs only when present.
- `method` should be the main focus and should break the approach into atomic components such as overall framework, component role, implementation detail, component interaction, training objective, or inference procedure.
- `experimental_focus` must come after `method` conceptually and explain what each experiment is meant to validate, test, resolve, or demonstrate.
- To write `experimental_focus`, combine the experimental part of the input with the earlier `motivation` and `method`.
- Focus on the purpose of the experiment only: what research problem it is addressing, what claim/mechanism it is trying to validate, what comparison or analysis is meant to establish, or what question it is intended to answer.
- Write each `experimental_focus` item as the underlying experimental goal, not as the path used to reach that goal.
- Remove phrasing such as "use/apply/through/by ... to validate/evaluate/test ..." and rewrite it as the goal itself.
- Different experimental paths that serve the same validation purpose should be normalized into the same goal-level focus whenever possible.
- Do not focus on the technical setup itself. Avoid datasets, baselines, metrics, hardware, split names, implementation details, procedural descriptions, or named evaluation protocols unless they are strictly necessary to state the experimental goal.
- Do not focus on outcome statements. Avoid result phrasing such as "outperforms baselines", "achieves better performance", or "aligns with human experts".
- If the source mentions setup, procedure, or outcomes but the underlying purpose is clear, rewrite the item as that purpose only.
- If a section is absent, return an empty array for that section.

Return JSON with this schema only:
{{
  "basic_idea": ["..."],
  "motivation": ["..."],
  "method": ["..."],
  "experimental_focus": ["..."]
}}

Academic input:
{idea_text}
"""
QUERY_GENERATION_SYSTEM_PROMPT = (
    "You are a scientific grounding agent. "
    "Generate retrieval-oriented paragraph search queries from structured scientific extraction results. "
    "Return strict JSON only."
)
QUERY_GENERATION_USER_PROMPT = """Generate dense-retrieval queries from the structured extraction below.

Requirements:
- Use only the `motivation` and `method` sections as query sources.
- Consider all provided sentences from those two sections before deciding which queries to emit.
- Select the most retrieval-useful items yourself. Do not mirror every input sentence if some are redundant.
- Produce at most {max_queries} total queries across both sections combined.
- You may allocate the total freely across `motivation` and `method`.
- Each output item must contain:
  - `section`: either `motivation` or `method`
  - `sentence`: the source sentence you selected
  - `query`: the final retrieval-oriented rewrite
- Keep the selected sentence meaning exactly.
- Write concise academic retrieval phrases or sentences likely to match paper paragraphs.
- Preserve task, object, method, mechanism, training objective, dataset, baseline, metric, or analysis anchors when present.
- Avoid vague wording such as "the framework", "this method", "evaluation", "performance improvement", or "how it works".
- Do not introduce unsupported facts.

Return JSON with this schema only:
{{
  "items": [
    {{
      "section": "motivation | method",
      "sentence": "selected source sentence",
      "query": "retrieval-oriented rewrite optimized for semantic paragraph retrieval"
    }}
  ]
}}

Structured extraction:
{items_json}
"""
GROUNDING_REFINEMENT_SYSTEM_PROMPT = (
    "You are a scientific grounding alignment agent. "
    "Given a research idea unit, a retrieved paragraph, and local paper context, analyze how the retrieved evidence aligns with the target idea. "
    "Use only the provided evidence. Return strict JSON only."
)
GROUNDING_REFINEMENT_USER_PROMPT = """Analyze how the retrieved evidence aligns with a specific research idea unit.

Research idea:
{idea_text}

Target query section:
{query_section}

Target query sentence:
{query_sentence}

Retrieval query:
{query_text}

Paper title:
{paper_title}

Paper abstract:
{paper_abstract}

Section path:
{section_path_text}

Retrieved paragraph:
{paragraph_text}

Previous paragraphs:
{previous_paragraphs}

Next paragraphs:
{next_paragraphs}

Requirements:
- Judge whether the retrieved evidence supports the target query unit.
- Use the paragraph as the primary evidence and use the surrounding context only to clarify scope, subject, or omitted details.
- Do not invent claims that are not grounded in the provided paragraph or context.
- If the paragraph is weakly related, say so explicitly.
- `focus_aspect` should state which specific aspect of the target query is actually addressed by the evidence.
- `grounded_passage` should be a concise evidence-focused passage, usually 1-3 sentences, that is more context-aware and better aligned to the target query than the raw paragraph alone.
- `evidence_span` should quote or closely paraphrase the most directly relevant span from the retrieved paragraph or immediate context.
- `shared_points` should list concrete aspects that are aligned between the target idea unit and the evidence.
- `different_points` should list concrete mismatches, missing parts, narrower scope, or different emphasis between the evidence and the target idea unit.
- `coverage_label` must be one of: `high`, `partial`, `limited`, `none`.
- Use `high` only when the evidence covers most of the target idea unit.
- Use `partial` when there is clear overlap but also important missing coverage.
- Use `limited` when the evidence is only weakly related or covers a small sub-aspect.
- Use `none` when it is essentially irrelevant.
- `why_this_matches` should briefly explain the overall judgment.

Return JSON with this schema only:
{{
  "status": "supported | partially_supported | weak_match | irrelevant",
  "focus_aspect": "which specific aspect of the target query is grounded",
  "grounded_passage": "context-aware grounding passage",
  "evidence_span": "most relevant evidence span",
  "shared_points": ["..."],
  "different_points": ["..."],
  "coverage_label": "high | partial | limited | none",
  "why_this_matches": "brief explanation of the match quality"
}}
"""
EXPERIMENT_RECOMMENDATION_SYSTEM_PROMPT = (
    "You are a scientific experiment-grounding agent. "
    "Given a target idea and one related paper, identify experiment goals that the target idea should focus on "
    "after learning from that paper's motivation, method, and experiment design focus. "
    "Return strict JSON only."
)
EXPERIMENT_RECOMMENDATION_USER_PROMPT = """Recommend experiment goals for the target idea by grounding on one related paper.

Target idea structured extraction:
{idea_extraction_json}

Related paper title:
{paper_title}

Related paper structured extraction:
{paper_extraction_json}

Requirements:
- Focus on experiment goals that the target idea should validate, compare, analyze, stress-test, or explain.
- Use the target idea's `motivation` and `method` as the main anchor.
- Use the related paper's `motivation`, `method`, and `experimental_focus` only as inspiration.
- The output must be about the target idea, not a summary of the paper's own experiments.
- Each goal must be atomic, concrete, self-contained, and written as the underlying validation purpose rather than the experimental path.
- State what should be established or examined, not how to do it.
- Remove path wording such as "use/apply/through/by comparing/ablating/measuring ... to validate ..." and rewrite to the shared goal that those designs aim to test.
- Different possible experiment designs that support the same purpose should be collapsed into one goal-level statement whenever possible.
- Avoid setup details such as datasets, hyperparameters, split names, hardware, exact metrics, ablation protocol details, or named evaluation procedures unless they are absolutely necessary to express the goal itself.
- Avoid outcome wording such as "show that it outperforms" unless the real goal is to validate a comparative claim.
- If the paper offers limited useful inspiration, return fewer goals instead of inventing weak ones.

Return JSON with this schema only:
{{
  "recommended_experimental_goals": [
    {{
      "goal": "experiment goal for the target idea",
      "rationale": "why this goal matters for the target idea in light of the paper",
      "inspired_by": ["paper sentence 1", "paper sentence 2"]
    }}
  ]
}}
"""
EXPERIMENT_COVERAGE_SYSTEM_PROMPT = (
    "You are a scientific experiment coverage analysis agent. "
    "Compare a target idea's original experimental focus with paper-inspired recommended experiment goals, "
    "then judge how well the original design covers those goals. "
    "Return strict JSON only."
)
EXPERIMENT_COVERAGE_USER_PROMPT = """Analyze experiment coverage for the target idea.

Target idea original experimental focus:
{idea_experimental_focus_json}

Related paper title:
{paper_title}

Paper-inspired recommended experiment goals:
{recommended_goals_json}

Requirements:
- `overlap` should capture experiment goals that are already covered or strongly aligned.
- `missing_or_undercovered` should capture important goals suggested by the paper that the target idea does not clearly cover.
- `additional_focus_in_idea` should capture idea experiment goals that go beyond what this paper suggests.
- Compare at the level of experimental purpose, not implementation path.
- Treat differently worded experiments as overlapping when they aim to validate the same underlying claim, mechanism, comparison, robustness question, or analysis target.
- Ignore differences in setup, procedure, dataset, metric, baseline choice, ablation design, or other experimental path details unless they change the actual goal being tested.
- `coverage_label` must be one of: `well_covered`, `partially_covered`, `many_gaps`.
- Use `well_covered` only when the target idea covers most important recommended goals with little missing coverage.
- Use `partially_covered` when there is meaningful overlap but also notable missing coverage.
- Use `many_gaps` when several important recommended goals are not covered.
- `coverage_score` must be a float between 0 and 1 where higher means better coverage.
- Be concrete and concise. Do not invent experiments unsupported by the inputs.

Return JSON with this schema only:
{{
  "overlap": ["..."],
  "missing_or_undercovered": ["..."],
  "additional_focus_in_idea": ["..."],
  "coverage_label": "well_covered | partially_covered | many_gaps",
  "coverage_score": 0.0,
  "coverage_rationale": "brief explanation"
}}
"""


from .utils import extract_text_from_pdf
from .vendor.pdf_extraction.parser import parse_tei_document


class GroundingError(RuntimeError):
    """Raised when the grounding pipeline cannot continue."""


@dataclass(frozen=True)
class GroundingInput:
    input_type: str
    text: str
    pdf_path: str | None = None


@dataclass(frozen=True)
class StructuredExtraction:
    basic_idea: list[str]
    motivation: list[str]
    method: list[str]
    experimental_focus: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "basic_idea": list(self.basic_idea),
            "motivation": list(self.motivation),
            "method": list(self.method),
            "experimental_focus": list(self.experimental_focus),
        }


@dataclass(slots=True)
class AtomicQuery:
    query_id: str
    section: str
    sentence: str
    query_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "section": self.section,
            "sentence": self.sentence,
            "query": self.query_text,
        }


def build_fallback_queries_from_extraction(
    extraction: StructuredExtraction,
    *,
    max_queries: int,
    idea_text: str | None = None,
) -> tuple[list[AtomicQuery], str | None]:
    candidates: list[tuple[str, str]] = [
        ("motivation", sentence) for sentence in extraction.motivation
    ] + [
        ("method", sentence) for sentence in extraction.method
    ]
    reason = "source_sentence_fallback"

    if not candidates:
        candidates = [("method", sentence) for sentence in extraction.basic_idea]
        reason = "basic_idea_fallback"

    if not candidates:
        normalized_idea_text = normalize_whitespace(idea_text)
        if normalized_idea_text:
            candidates = [("method", normalized_idea_text)]
            reason = "idea_text_fallback"

    queries: list[AtomicQuery] = []
    seen: set[str] = set()
    for section, sentence in candidates:
        if len(queries) >= max_queries:
            break
        normalized_sentence = normalize_whitespace(sentence)
        if not normalized_sentence:
            continue
        dedupe_key = f"{section}\t{normalized_sentence.casefold()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        queries.append(
            AtomicQuery(
                query_id=f"q{len(queries) + 1}",
                section=section,
                sentence=normalized_sentence,
                query_text=normalized_sentence,
            )
        )

    if not queries:
        return [], None
    return queries, reason


@dataclass(slots=True)
class ParagraphRecord:
    paragraph_id: str
    paper_rank: int
    paper_title: str
    paper_dir: str
    section_path: list[str]
    paragraph_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperContextRecord:
    paper_title: str
    paper_dir: str
    abstract: str
    paragraphs: list[ParagraphRecord]


@dataclass(frozen=True)
class QueryGeneratorConfig:
    provider: str | None = None
    api_url: str | None = None
    model: str | None = None
    env_path: Path = DEFAULT_ENV_PATH
    timeout: int = 60
    max_tokens: int = 1000
    use_env_proxy: bool = False


class GroundingLlmClient:
    def __init__(self, config: QueryGeneratorConfig | None = None) -> None:
        self.config = config or QueryGeneratorConfig()
        self.settings = resolve_llm_settings(
            self.config.env_path,
            {
                "query_provider": self.config.provider,
                "query_api_url": self.config.api_url,
                "query_model": self.config.model,
                "query_timeout": self.config.timeout,
            },
            provider_keys=("query_provider",),
            base_url_keys=("query_base_url", "query_api_url"),
            model_keys=("query_model",),
            timeout_keys=("query_timeout",),
        )
        self.client = build_llm_client(
            self.config.env_path,
            {
                "query_provider": self.config.provider,
                "query_api_url": self.config.api_url,
                "query_model": self.config.model,
                "query_timeout": self.config.timeout,
            },
            provider_keys=("query_provider",),
            base_url_keys=("query_base_url", "query_api_url"),
            model_keys=("query_model",),
            timeout_keys=("query_timeout",),
            use_env_proxy=self.config.use_env_proxy,
        )
        self.provider = self.settings.provider
        self.model_name = self.settings.model

    def _request_content(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        try:
            return self.client.chat_text(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise GroundingError(f"Grounding LLM request failed: {exc}") from exc

    def extract_structure(self, idea_text: str) -> StructuredExtraction:
        content = self._request_content(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=EXTRACTION_USER_PROMPT.format(idea_text=idea_text.strip()),
            max_tokens=max(self.config.max_tokens, 1600),
        )
        return parse_structured_extraction_response(content)

    def generate_queries_from_extraction(
        self,
        extraction: StructuredExtraction,
        *,
        max_queries: int,
        idea_text: str | None = None,
    ) -> tuple[list[AtomicQuery], dict[str, Any]]:
        payload = {
            "motivation": list(extraction.motivation),
            "method": list(extraction.method),
        }
        fallback_queries, fallback_reason = build_fallback_queries_from_extraction(
            extraction,
            max_queries=max_queries,
            idea_text=idea_text,
        )

        if not extraction.motivation and not extraction.method:
            if fallback_queries:
                return fallback_queries, {
                    "strategy": "fallback",
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                    "llm_error": "No motivation/method extraction available for query generation.",
                }
            raise GroundingError("No usable extraction content available for query generation.")

        llm_error: str | None = None
        try:
            content = self._request_content(
                system_prompt=QUERY_GENERATION_SYSTEM_PROMPT,
                user_prompt=QUERY_GENERATION_USER_PROMPT.format(
                    max_queries=max_queries,
                    items_json=json.dumps(payload, ensure_ascii=False, indent=2),
                ),
                max_tokens=min(max(self.config.max_tokens, 1000), 1800),
            )
            queries = parse_query_generation_response(content, max_queries=max_queries)
            filtered = [query for query in queries if query.section in {"motivation", "method"}]
            if filtered:
                return filtered[:max_queries], {
                    "strategy": "llm",
                    "fallback_used": False,
                    "fallback_reason": None,
                    "llm_error": None,
                }
            llm_error = "No usable motivation/method queries found in LLM response."
        except Exception as exc:
            llm_error = str(exc)

        if fallback_queries:
            return fallback_queries, {
                "strategy": "fallback",
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "llm_error": llm_error,
            }
        raise GroundingError(llm_error or "No usable motivation/method queries found in LLM response.")

    def refine_grounding_match(
        self,
        *,
        idea_text: str,
        query: AtomicQuery,
        paper_title: str,
        paper_abstract: str,
        section_path_text: str,
        paragraph_text: str,
        previous_paragraphs: list[dict[str, str]],
        next_paragraphs: list[dict[str, str]],
    ) -> dict[str, Any]:
        content = self._request_content(
            system_prompt=GROUNDING_REFINEMENT_SYSTEM_PROMPT,
            user_prompt=GROUNDING_REFINEMENT_USER_PROMPT.format(
                idea_text=idea_text.strip(),
                query_section=query.section,
                query_sentence=query.sentence,
                query_text=query.query_text,
                paper_title=paper_title,
                paper_abstract=paper_abstract or "",
                section_path_text=section_path_text or "",
                paragraph_text=paragraph_text,
                previous_paragraphs=json.dumps(previous_paragraphs, ensure_ascii=False, indent=2),
                next_paragraphs=json.dumps(next_paragraphs, ensure_ascii=False, indent=2),
            ),
        )
        return parse_refined_grounding_response(content)

    def recommend_experiment_goals(
        self,
        *,
        idea_extraction: StructuredExtraction,
        paper_title: str,
        paper_extraction: StructuredExtraction,
    ) -> dict[str, Any]:
        idea_payload = {
            "motivation": list(idea_extraction.motivation),
            "method": list(idea_extraction.method),
        }
        paper_payload = {
            "motivation": list(paper_extraction.motivation),
            "method": list(paper_extraction.method),
            "experimental_focus": list(paper_extraction.experimental_focus),
        }
        content = self._request_content(
            system_prompt=EXPERIMENT_RECOMMENDATION_SYSTEM_PROMPT,
            user_prompt=EXPERIMENT_RECOMMENDATION_USER_PROMPT.format(
                idea_extraction_json=json.dumps(idea_payload, ensure_ascii=False, indent=2),
                paper_title=paper_title,
                paper_extraction_json=json.dumps(paper_payload, ensure_ascii=False, indent=2),
            ),
            max_tokens=max(self.config.max_tokens, 1600),
        )
        return parse_experiment_recommendation_response(content)

    def analyze_experiment_coverage(
        self,
        *,
        idea_experimental_focus: list[str],
        paper_title: str,
        recommended_goals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        content = self._request_content(
            system_prompt=EXPERIMENT_COVERAGE_SYSTEM_PROMPT,
            user_prompt=EXPERIMENT_COVERAGE_USER_PROMPT.format(
                idea_experimental_focus_json=json.dumps(idea_experimental_focus, ensure_ascii=False, indent=2),
                paper_title=paper_title,
                recommended_goals_json=json.dumps(recommended_goals, ensure_ascii=False, indent=2),
            ),
            max_tokens=max(self.config.max_tokens, 1400),
        )
        return parse_experiment_coverage_response(content)


class DenseEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int,
        query_prefix: str,
        device: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name_or_path, **kwargs)
        self.batch_size = batch_size
        self.query_prefix = query_prefix

    def encode_paragraphs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"{self.query_prefix}{text}" if self.query_prefix else text for text in texts]
        return self.encode_paragraphs(prefixed)


class ParagraphReranker:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int,
        device: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = CrossEncoder(model_name_or_path, **kwargs)
        self.batch_size = batch_size

    def score(self, query_text: str, paragraph_texts: list[str]) -> list[float]:
        if not paragraph_texts:
            return []
        pairs = [[query_text, paragraph_text] for paragraph_text in paragraph_texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ground idea text or a paper PDF against paragraph-level evidence from papers produced by "
            "search/result/pdf_manifest."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-text", help="Idea text to be grounded.")
    input_group.add_argument("--pdf-path", help="Local PDF path used as the grounding input.")
    input_group.add_argument("--idea-file", help="Path to a UTF-8 text file containing the idea text.")

    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Manifest JSON from pdf_manifest.")
    parser.add_argument(
        "--papers-root",
        default=None,
        help="Optional override for the pdf_manifest run directory. Defaults to manifest parent.",
    )
    parser.add_argument("--top-k-papers", type=int, default=None, help="Optional cap on manifest papers to use.")
    parser.add_argument("--min-paragraph-chars", type=int, default=80, help="Minimum paragraph character length.")
    parser.add_argument("--min-paragraph-words", type=int, default=8, help="Minimum paragraph word count.")

    parser.add_argument(
        "--query-provider",
        default=None,
        help="LLM provider for grounding calls. Defaults to LLM_PROVIDER or openai_compatible.",
    )
    parser.add_argument("--query-model", default=None, help="Model used for grounding LLM calls.")
    parser.add_argument(
        "--query-api-url",
        default=None,
        help="Optional OpenAI-compatible chat completions endpoint override for grounding LLM calls.",
    )
    parser.add_argument("--query-timeout", type=int, default=60, help="Grounding query generation timeout in seconds.")
    parser.add_argument("--query-max-tokens", type=int, default=1000, help="Grounding query generation max tokens.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=8,
        help="Maximum atomic queries to generate. Higher values are often useful for paper-PDF inputs.",
    )
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to .env file with grounding LLM credentials.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the environment for grounding LLM requests.",
    )

    parser.add_argument(
        "--embedding-model",
        "--embedding-model-path",
        dest="embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face repo id or local sentence-transformers model path for dense retrieval.",
    )
    parser.add_argument(
        "--query-prefix",
        default=DEFAULT_QUERY_PREFIX,
        help="Prefix added to query text before BGE encoding.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="Batch size for dense embedding.")
    parser.add_argument("--dense-candidate-k", type=int, default=40, help="Dense retrieval candidate depth.")

    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Skip cross-encoder reranking and rely on dense retrieval only.",
    )
    parser.add_argument(
        "--reranker-model",
        "--reranker-model-path",
        dest="reranker_model",
        default=DEFAULT_RERANKER_MODEL,
        help="Hugging Face repo id or local cross-encoder model path for paragraph reranking.",
    )
    parser.add_argument("--reranker-top-n", type=int, default=20, help="Dense candidates sent to the reranker.")
    parser.add_argument("--reranker-batch-size", type=int, default=8, help="Batch size for reranker scoring.")

    parser.add_argument("--final-top-k", type=int, default=8, help="Final grounded paragraphs per query.")
    parser.add_argument(
        "--max-paragraphs-per-paper",
        type=int,
        default=2,
        help="Maximum number of final grounded paragraphs kept from the same paper per query.",
    )
    parser.add_argument(
        "--enable-grounding-refinement",
        action="store_true",
        help="Use the configured LLM to refine final grounded matches with local paragraph context from the source paper.",
    )
    parser.add_argument(
        "--refinement-context-window",
        type=int,
        default=2,
        help="How many previous and next paragraphs to include around each refined match.",
    )
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device override.")
    parser.add_argument(
        "--disable-experiment-grounding",
        action="store_true",
        help="Skip experiment-focused per-paper grounding analysis.",
    )
    parser.add_argument(
        "--paper-extraction-max-chars",
        type=int,
        default=0,
        help="Optional character cap for serialized paper text sent to extraction. Non-positive means no cap.",
    )

    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR), help="Cache directory for corpus embeddings.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--result-tag", default="grounding", help="Tag used in the default output filename.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the final JSON output.")
    return parser


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def normalize_model_name_or_path(value: str | None) -> str:
    model_name_or_path = normalize_whitespace(value)
    if not model_name_or_path:
        return ""
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return model_name_or_path


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise GroundingError(f"Invalid JSON response: {text!r}") from None
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise GroundingError(f"Expected a JSON object, got: {type(payload).__name__}")
    return payload


VALID_QUERY_SECTIONS = {"motivation", "method"}
VALID_EXTRACTION_SECTIONS = {
    "basic_idea",
    "motivation",
    "method",
    "experimental_focus",
}


def normalize_query_section(value: Any) -> str:
    text = normalize_whitespace(value).casefold().replace("-", "_").replace(" ", "_")
    if not text:
        return "method"

    alias_map = {
        "motivation": "motivation",
        "problem": "motivation",
        "method": "method",
        "approach": "method",
    }
    normalized = alias_map.get(text, text)
    if normalized in VALID_QUERY_SECTIONS:
        return normalized
    return "method"


def normalize_extraction_section(value: Any) -> str:
    text = normalize_whitespace(value).casefold().replace("-", "_").replace(" ", "_")
    if not text:
        return "method"

    alias_map = {
        "basicidea": "basic_idea",
        "basic_idea": "basic_idea",
        "summary": "basic_idea",
        "overview": "basic_idea",
        "motivation": "motivation",
        "problem": "motivation",
        "method": "method",
        "approach": "method",
        "experimentalfocus": "experimental_focus",
        "experimental_focus": "experimental_focus",
        "experimentalsetting": "experimental_focus",
        "experimental_setting": "experimental_focus",
        "experiment": "experimental_focus",
        "evaluation": "experimental_focus",
    }
    normalized = alias_map.get(text, text)
    if normalized in VALID_EXTRACTION_SECTIONS:
        return normalized
    return "method"


def normalize_extracted_sentences(value: Any) -> list[str]:
    raw_items: list[Any]
    if value is None:
        raw_items = []
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    sentences: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            text = normalize_whitespace(
                item.get("sentence")
                or item.get("statement")
                or item.get("claim")
                or item.get("text")
                or item.get("description")
            )
        else:
            text = normalize_whitespace(item)
        if not text:
            continue
        dedupe_key = text.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        sentences.append(text)
    return sentences


def parse_structured_extraction_response(content: str) -> StructuredExtraction:
    payload = parse_json_object(content)
    sections: dict[str, list[str]] = {name: [] for name in VALID_EXTRACTION_SECTIONS}

    if "items" in payload and isinstance(payload.get("items"), list):
        for item in payload["items"]:
            if not isinstance(item, dict):
                continue
            section = normalize_extraction_section(item.get("section"))
            sentence = normalize_whitespace(
                item.get("sentence")
                or item.get("statement")
                or item.get("claim")
                or item.get("text")
            )
            if not sentence:
                continue
            if sentence.casefold() not in {value.casefold() for value in sections[section]}:
                sections[section].append(sentence)
    else:
        for section in VALID_EXTRACTION_SECTIONS:
            for sentence in normalize_extracted_sentences(payload.get(section)):
                sections[section].append(sentence)

    return StructuredExtraction(
        basic_idea=dedupe_sentences(sections["basic_idea"]),
        motivation=dedupe_sentences(sections["motivation"]),
        method=dedupe_sentences(sections["method"]),
        experimental_focus=dedupe_sentences(sections["experimental_focus"]),
    )


def dedupe_sentences(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_whitespace(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result

def parse_query_generation_response(content: str, *, max_queries: int) -> list[AtomicQuery]:
    payload = parse_json_object(content)
    raw_queries = payload.get("items")
    if not isinstance(raw_queries, list):
        raise GroundingError(f"Missing queries list in response: {payload!r}")

    parsed_queries: list[AtomicQuery] = []
    seen: set[str] = set()

    for item in raw_queries:
        if len(parsed_queries) >= max_queries:
            break

        section = "method"
        sentence = ""
        query_text = ""

        if not isinstance(item, dict):
            continue
        section = normalize_query_section(item.get("section"))
        sentence = normalize_whitespace(item.get("sentence"))
        query_text = normalize_whitespace(item.get("query"))

        if not sentence and not query_text:
            continue
        if not sentence:
            sentence = query_text
        if not query_text:
            query_text = sentence

        dedupe_key = f"{section}\t{query_text.casefold()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        parsed_queries.append(
            AtomicQuery(
                query_id=f"q{len(parsed_queries) + 1}",
                section=section,
                sentence=sentence,
                query_text=query_text,
            )
        )

    if not parsed_queries:
        raise GroundingError(f"No usable queries found in response: {payload!r}")
    return parsed_queries

def parse_refined_grounding_response(content: str) -> dict[str, Any]:
    payload = parse_json_object(content)
    status = normalize_whitespace(payload.get("status")).casefold().replace(" ", "_")
    if status not in {"supported", "partially_supported", "weak_match", "irrelevant"}:
        status = "weak_match"
    coverage_label = normalize_whitespace(payload.get("coverage_label")).casefold().replace("-", "_").replace(" ", "_")
    coverage_aliases = {
        "high": "high",
        "strong": "high",
        "partial": "partial",
        "partially": "partial",
        "limited": "limited",
        "weak": "limited",
        "low": "limited",
        "none": "none",
        "irrelevant": "none",
    }
    normalized_coverage = coverage_aliases.get(coverage_label, coverage_label)
    if normalized_coverage not in {"high", "partial", "limited", "none"}:
        default_by_status = {
            "supported": "high",
            "partially_supported": "partial",
            "weak_match": "limited",
            "irrelevant": "none",
        }
        normalized_coverage = default_by_status[status]
    return {
        "status": status,
        "focus_aspect": normalize_whitespace(payload.get("focus_aspect")),
        "grounded_passage": normalize_whitespace(payload.get("grounded_passage")),
        "evidence_span": normalize_whitespace(payload.get("evidence_span")),
        "shared_points": dedupe_sentences(normalize_extracted_sentences(payload.get("shared_points"))),
        "different_points": dedupe_sentences(normalize_extracted_sentences(payload.get("different_points"))),
        "coverage_label": normalized_coverage,
        "why_this_matches": normalize_whitespace(payload.get("why_this_matches")),
    }


def parse_experiment_recommendation_response(content: str) -> dict[str, Any]:
    payload = parse_json_object(content)
    raw_items = payload.get("recommended_experimental_goals")
    if not isinstance(raw_items, list):
        raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    recommendations: list[dict[str, Any]] = []
    seen_goals: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            goal = normalize_whitespace(
                item.get("goal")
                or item.get("experimental_goal")
                or item.get("focus")
                or item.get("text")
            )
            rationale = normalize_whitespace(item.get("rationale") or item.get("why"))
            inspired_by = dedupe_sentences(normalize_extracted_sentences(item.get("inspired_by")))
        else:
            goal = normalize_whitespace(item)
            rationale = ""
            inspired_by = []
        if not goal:
            continue
        dedupe_key = goal.casefold()
        if dedupe_key in seen_goals:
            continue
        seen_goals.add(dedupe_key)
        recommendations.append(
            {
                "goal": goal,
                "rationale": rationale,
                "inspired_by": inspired_by,
            }
        )
    return {"recommended_experimental_goals": recommendations}


def parse_experiment_coverage_response(content: str) -> dict[str, Any]:
    payload = parse_json_object(content)
    label = normalize_whitespace(payload.get("coverage_label")).casefold().replace("-", "_").replace(" ", "_")
    label_aliases = {
        "well": "well_covered",
        "well_covered": "well_covered",
        "good": "well_covered",
        "covered_well": "well_covered",
        "partially": "partially_covered",
        "partial": "partially_covered",
        "partially_covered": "partially_covered",
        "medium": "partially_covered",
        "many_gaps": "many_gaps",
        "gap": "many_gaps",
        "gaps": "many_gaps",
        "poor": "many_gaps",
    }
    normalized_label = label_aliases.get(label, label)
    if normalized_label not in {"well_covered", "partially_covered", "many_gaps"}:
        normalized_label = "partially_covered"

    raw_score = payload.get("coverage_score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        default_scores = {
            "well_covered": 0.85,
            "partially_covered": 0.55,
            "many_gaps": 0.2,
        }
        score = default_scores[normalized_label]
    score = max(0.0, min(1.0, score))

    return {
        "overlap": dedupe_sentences(normalize_extracted_sentences(payload.get("overlap"))),
        "missing_or_undercovered": dedupe_sentences(
            normalize_extracted_sentences(payload.get("missing_or_undercovered"))
        ),
        "additional_focus_in_idea": dedupe_sentences(
            normalize_extracted_sentences(payload.get("additional_focus_in_idea"))
        ),
        "coverage_label": normalized_label,
        "coverage_score": score,
        "coverage_rationale": normalize_whitespace(payload.get("coverage_rationale") or payload.get("rationale")),
    }

def load_grounding_input(args: argparse.Namespace) -> GroundingInput:
    if args.idea_text:
        raw_text = args.idea_text
        input_type = "idea_text"
        pdf_path: str | None = None
    elif getattr(args, "pdf_path", None):
        pdf_path_obj = Path(args.pdf_path).expanduser().resolve()
        if not pdf_path_obj.exists():
            raise GroundingError(f"PDF file not found: {pdf_path_obj}")
        extracted_text = extract_text_from_pdf(pdf_path_obj)
        if not extracted_text:
            raise GroundingError(f"Failed to extract text from PDF: {pdf_path_obj}")
        raw_text = extracted_text
        input_type = "pdf_path"
        pdf_path = str(pdf_path_obj)
    else:
        raw_text = Path(args.idea_file).read_text(encoding="utf-8")
        input_type = "idea_file"
        pdf_path = None

    normalized = normalize_whitespace(raw_text)
    if not normalized:
        raise GroundingError("Input text is empty after normalization.")
    if len(normalized) < 50:
        raise GroundingError("Extracted input text is too short or empty.")
    return GroundingInput(
        input_type=input_type,
        text=normalized,
        pdf_path=pdf_path,
    )


def resolve_corpus_root(manifest_path: Path, papers_root: str | None) -> Path:
    if papers_root:
        return Path(papers_root).expanduser().resolve()
    return manifest_path.resolve().parent


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise GroundingError(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GroundingError(f"Manifest must be a JSON object: {manifest_path}")
    return payload


def select_paper_entries(manifest_payload: dict[str, Any], *, top_k_papers: int | None) -> list[dict[str, Any]]:
    raw_entries = manifest_payload.get("papers")
    if not isinstance(raw_entries, list):
        raise GroundingError("Manifest is missing a top-level 'papers' list.")

    selected_entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        if normalize_whitespace(entry.get("status")) != "ok":
            continue
        selected_entries.append(entry)
        if top_k_papers is not None and len(selected_entries) >= top_k_papers:
            break

    if not selected_entries:
        raise GroundingError("Manifest contains no usable papers with status='ok'.")
    return selected_entries


def resolve_artifact_path(corpus_root: Path, value: str | None) -> Path | None:
    text = normalize_whitespace(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return corpus_root / path


def load_document_payload(corpus_root: Path, paper_entry: dict[str, Any]) -> tuple[dict[str, Any], str]:
    tei_payload = paper_entry.get("tei")
    if not isinstance(tei_payload, dict):
        raise GroundingError(f"Missing 'tei' payload for paper: {paper_entry.get('title')}")

    parsed_path = resolve_artifact_path(corpus_root, tei_payload.get("parsed_json_path"))
    if parsed_path and parsed_path.exists():
        payload = json.loads(parsed_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise GroundingError(f"Parsed payload must be a JSON object: {parsed_path}")
        return payload, "parsed_json"

    tei_xml_path = resolve_artifact_path(corpus_root, tei_payload.get("tei_xml_path"))
    if tei_xml_path and tei_xml_path.exists():
        tei_xml = tei_xml_path.read_text(encoding="utf-8")
        return parse_tei_document(tei_xml, preserve_bibr_refs=False).to_dict(), "tei_xml"

    raise GroundingError(
        "Neither parsed_json_path nor tei_xml_path is available for paper: "
        f"{paper_entry.get('title')}"
    )


def paper_context_key(*, paper_dir: str, paper_title: str) -> str:
    normalized_dir = normalize_whitespace(paper_dir)
    if normalized_dir:
        return normalized_dir
    return normalize_whitespace(paper_title).casefold()


def extract_abstract_text(document_payload: dict[str, Any]) -> str:
    raw_abstract = document_payload.get("abstract")
    if isinstance(raw_abstract, str):
        return normalize_whitespace(raw_abstract)
    if isinstance(raw_abstract, list):
        parts = [normalize_whitespace(item) for item in raw_abstract if normalize_whitespace(item)]
        return normalize_whitespace(" ".join(parts))
    if isinstance(raw_abstract, dict):
        parts: list[str] = []
        for key in ("text", "abstract", "content"):
            value = raw_abstract.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)
        return normalize_whitespace(" ".join(parts))
    return ""


def serialize_document_for_extraction(document_payload: dict[str, Any], *, max_chars: int = 0) -> str:
    parts: list[str] = []
    title = normalize_whitespace(document_payload.get("title"))
    if title:
        parts.append(f"Title: {title}")

    abstract = extract_abstract_text(document_payload)
    if abstract:
        parts.append(f"Abstract: {abstract}")

    def walk(section: dict[str, Any], path_prefix: list[str]) -> None:
        heading = normalize_whitespace(section.get("heading"))
        current_path = list(path_prefix)
        if heading:
            current_path.append(heading)

        raw_paragraphs = section.get("paragraphs")
        if isinstance(raw_paragraphs, list):
            for raw_paragraph in raw_paragraphs:
                paragraph_text = normalize_whitespace(raw_paragraph)
                if not paragraph_text:
                    continue
                if current_path:
                    parts.append(f"Section: {' > '.join(current_path)}")
                parts.append(paragraph_text)

        raw_subsections = section.get("subsections")
        if isinstance(raw_subsections, list):
            for subsection in raw_subsections:
                if isinstance(subsection, dict):
                    walk(subsection, current_path)

    raw_body = document_payload.get("body")
    if isinstance(raw_body, list):
        for section in raw_body:
            if isinstance(section, dict):
                walk(section, [])

    serialized = "\n\n".join(parts).strip()
    if max_chars > 0 and len(serialized) > max_chars:
        truncated = serialized[:max_chars].rstrip()
        last_break = truncated.rfind("\n\n")
        if last_break >= max_chars // 2:
            truncated = truncated[:last_break].rstrip()
        serialized = truncated
    return serialized


def should_keep_paragraph(text: str, *, min_chars: int, min_words: int) -> bool:
    if len(text) < min_chars:
        return False
    if len(text.split()) < min_words:
        return False
    if not any(character.isalpha() for character in text):
        return False
    return True


def collect_paragraph_records_from_sections(
    sections: list[dict[str, Any]],
    *,
    paper_rank: int,
    paper_title: str,
    paper_dir: str,
    min_chars: int,
    min_words: int,
) -> list[ParagraphRecord]:
    records: list[ParagraphRecord] = []
    next_index = 0

    def walk(section: dict[str, Any], path_prefix: list[str]) -> None:
        nonlocal next_index

        heading = normalize_whitespace(section.get("heading"))
        current_path = list(path_prefix)
        if heading:
            current_path.append(heading)

        raw_paragraphs = section.get("paragraphs")
        if isinstance(raw_paragraphs, list):
            for raw_paragraph in raw_paragraphs:
                text = normalize_whitespace(raw_paragraph)
                if not should_keep_paragraph(text, min_chars=min_chars, min_words=min_words):
                    continue
                records.append(
                    ParagraphRecord(
                        paragraph_id=f"paper{paper_rank:02d}-p{next_index:04d}",
                        paper_rank=paper_rank,
                        paper_title=paper_title,
                        paper_dir=paper_dir,
                        section_path=list(current_path),
                        paragraph_index=next_index,
                        text=text,
                    )
                )
                next_index += 1

        raw_subsections = section.get("subsections")
        if isinstance(raw_subsections, list):
            for subsection in raw_subsections:
                if isinstance(subsection, dict):
                    walk(subsection, current_path)

    for section in sections:
        if isinstance(section, dict):
            walk(section, [])
    return records


def build_paragraph_records(
    entries: list[dict[str, Any]],
    *,
    corpus_root: Path,
    min_chars: int,
    min_words: int,
) -> tuple[list[ParagraphRecord], dict[str, int]]:
    records: list[ParagraphRecord] = []
    source_counts = {"parsed_json": 0, "tei_xml": 0}

    for entry in entries:
        document_payload, source_name = load_document_payload(corpus_root, entry)
        body_sections = document_payload.get("body")
        if not isinstance(body_sections, list):
            body_sections = []

        paper_rank = int(entry.get("rank") or len(records) + 1)
        paper_title = normalize_whitespace(entry.get("title") or document_payload.get("title"))
        paper_dir = normalize_whitespace(entry.get("paper_dir"))
        paper_records = collect_paragraph_records_from_sections(
            body_sections,
            paper_rank=paper_rank,
            paper_title=paper_title,
            paper_dir=paper_dir,
            min_chars=min_chars,
            min_words=min_words,
        )
        records.extend(paper_records)
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

    if not records:
        raise GroundingError("No usable body paragraphs were extracted from the selected papers.")
    return records, source_counts


def build_paper_context_records(
    entries: list[dict[str, Any]],
    *,
    corpus_root: Path,
    paragraphs: list[ParagraphRecord],
) -> dict[str, PaperContextRecord]:
    paragraphs_by_paper: dict[str, list[ParagraphRecord]] = {}
    for paragraph in paragraphs:
        key = paper_context_key(paper_dir=paragraph.paper_dir, paper_title=paragraph.paper_title)
        paragraphs_by_paper.setdefault(key, []).append(paragraph)

    for items in paragraphs_by_paper.values():
        items.sort(key=lambda item: item.paragraph_index)

    contexts: dict[str, PaperContextRecord] = {}
    for entry in entries:
        document_payload, _ = load_document_payload(corpus_root, entry)
        paper_title = normalize_whitespace(entry.get("title") or document_payload.get("title"))
        paper_dir = normalize_whitespace(entry.get("paper_dir"))
        key = paper_context_key(paper_dir=paper_dir, paper_title=paper_title)
        contexts[key] = PaperContextRecord(
            paper_title=paper_title,
            paper_dir=paper_dir,
            abstract=extract_abstract_text(document_payload),
            paragraphs=list(paragraphs_by_paper.get(key, [])),
        )
    return contexts


def serialize_context_paragraphs(paragraphs: list[ParagraphRecord]) -> list[dict[str, str]]:
    return [
        {
            "section_path_text": " > ".join(paragraph.section_path),
            "text": paragraph.text,
        }
        for paragraph in paragraphs
    ]


def get_match_context(
    paper_contexts: dict[str, PaperContextRecord],
    paragraph: ParagraphRecord,
    *,
    context_window: int,
) -> dict[str, Any]:
    key = paper_context_key(paper_dir=paragraph.paper_dir, paper_title=paragraph.paper_title)
    paper_context = paper_contexts.get(key)
    if paper_context is None:
        return {
            "paper_title": paragraph.paper_title,
            "paper_abstract": "",
            "section_path_text": " > ".join(paragraph.section_path),
            "previous_paragraphs": [],
            "next_paragraphs": [],
        }

    ordered = paper_context.paragraphs
    current_position = 0
    for index, item in enumerate(ordered):
        if item.paragraph_index == paragraph.paragraph_index:
            current_position = index
            break

    start = max(0, current_position - max(0, context_window))
    end = min(len(ordered), current_position + max(0, context_window) + 1)
    previous_items = ordered[start:current_position]
    next_items = ordered[current_position + 1 : end]
    return {
        "paper_title": paper_context.paper_title,
        "paper_abstract": paper_context.abstract,
        "section_path_text": " > ".join(paragraph.section_path),
        "previous_paragraphs": serialize_context_paragraphs(previous_items),
        "next_paragraphs": serialize_context_paragraphs(next_items),
    }

def refine_retrieval_results_with_context(
    *,
    idea_text: str,
    retrieval_results: list[dict[str, Any]],
    paragraphs: list[ParagraphRecord],
    paper_contexts: dict[str, PaperContextRecord],
    generator: GroundingLlmClient,
    context_window: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    work_items: list[tuple[dict[str, Any], AtomicQuery, ParagraphRecord, dict[str, Any]]] = []

    for result in retrieval_results:
        query = AtomicQuery(
            query_id=str(result["query_id"]),
            section=str(result["section"]),
            sentence=str(result["sentence"]),
            query_text=str(result["query"]),
        )
        for match in result["matches"]:
            corpus_paragraph_index = match.get("corpus_paragraph_index")
            if not isinstance(corpus_paragraph_index, int):
                corpus_paragraph_index = match["paragraph_index"]
            paragraph = paragraphs[corpus_paragraph_index]
            context = get_match_context(
                paper_contexts,
                paragraph,
                context_window=context_window,
            )
            match["context"] = context
            work_items.append((match, query, paragraph, context))

    if not work_items:
        return retrieval_results, stats

    stats["attempted"] = len(work_items)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(work_items)) as executor:
        future_to_match = {
            executor.submit(
                generator.refine_grounding_match,
                idea_text=idea_text,
                query=query,
                paper_title=str(context["paper_title"]),
                paper_abstract=str(context["paper_abstract"]),
                section_path_text=str(context["section_path_text"]),
                paragraph_text=paragraph.text,
                previous_paragraphs=list(context["previous_paragraphs"]),
                next_paragraphs=list(context["next_paragraphs"]),
            ): match
            for match, query, paragraph, context in work_items
        }
        for future in concurrent.futures.as_completed(future_to_match):
            match = future_to_match[future]
            try:
                match["refined_grounding"] = future.result()
                stats["succeeded"] += 1
            except Exception as exc:
                match["refined_grounding"] = {
                    "status": "error",
                    "focus_aspect": "",
                    "grounded_passage": "",
                    "evidence_span": "",
                    "shared_points": [],
                    "different_points": [],
                    "coverage_label": "none",
                    "why_this_matches": "",
                    "error": str(exc),
                }
                stats["failed"] += 1

    return retrieval_results, stats


def analyze_experiment_grounding_for_paper(
    *,
    paper_entry: dict[str, Any],
    corpus_root: Path,
    idea_extraction: StructuredExtraction,
    generator: GroundingLlmClient,
    paper_extraction_max_chars: int,
) -> dict[str, Any]:
    paper_rank = int(paper_entry.get("rank") or 0)
    paper_dir = normalize_whitespace(paper_entry.get("paper_dir"))

    try:
        document_payload, source_name = load_document_payload(corpus_root, paper_entry)
        paper_title = normalize_whitespace(paper_entry.get("title") or document_payload.get("title"))
        paper_text = serialize_document_for_extraction(
            document_payload,
            max_chars=max(0, paper_extraction_max_chars),
        )
        if not paper_text:
            raise GroundingError(f"No usable paper text found for experiment grounding: {paper_title}")

        paper_extraction = generator.extract_structure(paper_text)
        recommendation = generator.recommend_experiment_goals(
            idea_extraction=idea_extraction,
            paper_title=paper_title,
            paper_extraction=paper_extraction,
        )
        coverage_analysis = generator.analyze_experiment_coverage(
            idea_experimental_focus=list(idea_extraction.experimental_focus),
            paper_title=paper_title,
            recommended_goals=recommendation["recommended_experimental_goals"],
        )
        return {
            "paper_rank": paper_rank,
            "paper_title": paper_title,
            "paper_dir": paper_dir,
            "document_source": source_name,
            "paper_text_length": len(paper_text),
            "status": "ok",
            "error": None,
            "paper_extraction": paper_extraction.to_dict(),
            "paper_inspired_recommendation": recommendation,
            "coverage_analysis": coverage_analysis,
        }
    except Exception as exc:
        return {
            "paper_rank": paper_rank,
            "paper_title": normalize_whitespace(paper_entry.get("title")),
            "paper_dir": paper_dir,
            "document_source": None,
            "paper_text_length": 0,
            "status": "failed",
            "error": str(exc),
            "paper_extraction": None,
            "paper_inspired_recommendation": {"recommended_experimental_goals": []},
            "coverage_analysis": None,
        }


def run_experiment_grounding(
    *,
    args: argparse.Namespace,
    selected_entries: list[dict[str, Any]],
    corpus_root: Path,
    idea_extraction: StructuredExtraction,
) -> dict[str, Any]:
    if getattr(args, "disable_experiment_grounding", False):
        return {
            "requested": False,
            "status": "disabled",
            "error": None,
            "model": None,
            "paper_count": len(selected_entries),
            "idea_experimental_focus": list(idea_extraction.experimental_focus),
            "results": [],
            "stats": {"attempted": 0, "succeeded": 0, "failed": 0},
        }

    config = QueryGeneratorConfig(
        provider=getattr(args, "query_provider", None),
        api_url=args.query_api_url,
        model=args.query_model,
        env_path=Path(args.env),
        timeout=args.query_timeout,
        max_tokens=args.query_max_tokens,
        use_env_proxy=args.use_env_proxy,
    )
    model_name = None
    try:
        generator = GroundingLlmClient(config)
        model_name = generator.model_name
    except Exception as exc:
        return {
            "requested": True,
            "status": "setup_failed",
            "error": str(exc),
            "model": model_name or args.query_model,
            "paper_count": len(selected_entries),
            "idea_experimental_focus": list(idea_extraction.experimental_focus),
            "results": [],
            "stats": {"attempted": 0, "succeeded": 0, "failed": 0},
        }

    results: list[dict[str, Any] | None] = [None] * len(selected_entries)
    stats = {"attempted": len(selected_entries), "succeeded": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_entries)) as executor:
        future_to_index = {
            executor.submit(
                analyze_experiment_grounding_for_paper,
                paper_entry=entry,
                corpus_root=corpus_root,
                idea_extraction=idea_extraction,
                generator=generator,
                paper_extraction_max_chars=max(0, int(getattr(args, "paper_extraction_max_chars", 0) or 0)),
            ): index
            for index, entry in enumerate(selected_entries)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            if result["status"] == "ok":
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1

    finalized_results = [result for result in results if result is not None]
    if stats["failed"] == 0:
        status = "ok"
        error = None
    elif stats["succeeded"] == 0:
        status = "failed"
        error = "Experiment grounding failed for all selected papers."
    else:
        status = "partial_failed"
        error = "Experiment grounding failed for some selected papers."

    return {
        "requested": True,
        "status": status,
        "error": error,
        "model": generator.model_name,
        "paper_count": len(selected_entries),
        "idea_experimental_focus": list(idea_extraction.experimental_focus),
        "results": finalized_results,
        "stats": stats,
    }


def artifact_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def compute_corpus_signature(
    entries: list[dict[str, Any]],
    *,
    corpus_root: Path,
    embedding_model: str,
    min_chars: int,
    min_words: int,
) -> str:
    signature_payload = {
        "corpus_root": str(corpus_root),
        "embedding_model": embedding_model,
        "min_paragraph_chars": min_chars,
        "min_paragraph_words": min_words,
        "papers": [],
    }

    papers_payload = signature_payload["papers"]
    assert isinstance(papers_payload, list)
    for entry in entries:
        tei_payload = entry.get("tei") if isinstance(entry.get("tei"), dict) else {}
        parsed_path = resolve_artifact_path(corpus_root, tei_payload.get("parsed_json_path"))
        tei_path = resolve_artifact_path(corpus_root, tei_payload.get("tei_xml_path"))
        papers_payload.append(
            {
                "rank": entry.get("rank"),
                "title": entry.get("title"),
                "paper_dir": entry.get("paper_dir"),
                "parsed_json": artifact_fingerprint(parsed_path),
                "tei_xml": artifact_fingerprint(tei_path),
            }
        )

    serialized = json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def load_cached_corpus(
    target_dir: Path,
    signature: str,
) -> tuple[list[ParagraphRecord], np.ndarray, dict[str, int]] | None:
    meta_path = target_dir / f"corpus_{signature}.json"
    emb_path = target_dir / f"corpus_{signature}.npy"
    if not meta_path.exists() or not emb_path.exists():
        return None

    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_paragraphs = meta_payload.get("paragraphs")
    if not isinstance(raw_paragraphs, list):
        return None

    paragraphs: list[ParagraphRecord] = []
    for item in raw_paragraphs:
        if not isinstance(item, dict):
            return None
        paragraphs.append(
            ParagraphRecord(
                paragraph_id=str(item["paragraph_id"]),
                paper_rank=int(item["paper_rank"]),
                paper_title=str(item["paper_title"]),
                paper_dir=str(item["paper_dir"]),
                section_path=[str(part) for part in item.get("section_path", [])],
                paragraph_index=int(item["paragraph_index"]),
                text=str(item["text"]),
            )
        )

    raw_source_counts = meta_payload.get("source_counts")
    source_counts: dict[str, int] = {}
    if isinstance(raw_source_counts, dict):
        source_counts = {
            str(key): int(value)
            for key, value in raw_source_counts.items()
            if isinstance(value, int)
        }

    embeddings = np.load(emb_path)
    if embeddings.shape[0] != len(paragraphs):
        return None
    return paragraphs, np.asarray(embeddings, dtype=np.float32), source_counts


def save_cached_corpus(
    target_dir: Path,
    signature: str,
    *,
    paragraphs: list[ParagraphRecord],
    embeddings: np.ndarray,
    selected_paper_count: int,
    source_counts: dict[str, int],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    meta_path = target_dir / f"corpus_{signature}.json"
    emb_path = target_dir / f"corpus_{signature}.npy"

    meta_payload = {
        "signature": signature,
        "selected_paper_count": selected_paper_count,
        "paragraph_count": len(paragraphs),
        "source_counts": source_counts,
        "paragraphs": [paragraph.to_dict() for paragraph in paragraphs],
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(emb_path, embeddings)


def dense_search(query_embedding: np.ndarray, paragraph_embeddings: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    if paragraph_embeddings.size == 0:
        return []

    scores = paragraph_embeddings @ query_embedding
    requested = min(int(top_k), int(scores.shape[0]))
    if requested <= 0:
        return []

    if requested == scores.shape[0]:
        top_indices = np.argsort(scores)[::-1]
    else:
        partial = np.argpartition(scores, -requested)[-requested:]
        top_indices = partial[np.argsort(scores[partial])[::-1]]

    return [
        {
            "paragraph_index": int(index),
            "dense_score": float(scores[index]),
        }
        for index in top_indices
    ]


def select_diverse_matches(
    candidates: list[dict[str, Any]],
    paragraphs: list[ParagraphRecord],
    *,
    final_top_k: int,
    max_paragraphs_per_paper: int,
) -> list[dict[str, Any]]:
    if final_top_k <= 0:
        return []

    selected: list[dict[str, Any]] = []
    per_paper_counts: dict[str, int] = {}

    for candidate in candidates:
        paragraph = paragraphs[candidate["paragraph_index"]]
        paper_key = paragraph.paper_dir or paragraph.paper_title.casefold()
        current_count = per_paper_counts.get(paper_key, 0)
        if max_paragraphs_per_paper > 0 and current_count >= max_paragraphs_per_paper:
            continue
        per_paper_counts[paper_key] = current_count + 1
        selected.append(candidate)
        if len(selected) >= final_top_k:
            break

    return selected


def generate_queries(
    idea_text: str,
    *,
    args: argparse.Namespace,
) -> tuple[list[AtomicQuery], StructuredExtraction, dict[str, Any]]:
    generator = GroundingLlmClient(
        QueryGeneratorConfig(
            provider=getattr(args, "query_provider", None),
            api_url=args.query_api_url,
            model=args.query_model,
            env_path=Path(args.env),
            timeout=args.query_timeout,
            max_tokens=args.query_max_tokens,
            use_env_proxy=args.use_env_proxy,
        )
    )

    extraction = generator.extract_structure(idea_text)
    queries, query_meta = generator.generate_queries_from_extraction(
        extraction,
        max_queries=args.max_queries,
        idea_text=idea_text,
    )
    return queries, extraction, {
        "status": "ok",
        "error": None,
        "extraction": extraction.to_dict(),
        "strategy": query_meta.get("strategy"),
        "fallback_used": bool(query_meta.get("fallback_used")),
        "fallback_reason": query_meta.get("fallback_reason"),
        "llm_error": query_meta.get("llm_error"),
        "model": generator.model_name,
        "provider": generator.provider,
    }


def run_grounding(args: argparse.Namespace) -> dict[str, Any]:
    grounding_input = load_grounding_input(args)
    idea_text = grounding_input.text
    manifest_path = Path(args.manifest).expanduser().resolve()
    corpus_root = resolve_corpus_root(manifest_path, args.papers_root)
    target_dir = Path(args.target_dir).expanduser().resolve()
    embedding_model = normalize_model_name_or_path(args.embedding_model)
    reranker_model = normalize_model_name_or_path(args.reranker_model)

    manifest_payload = load_manifest(manifest_path)
    selected_entries = select_paper_entries(manifest_payload, top_k_papers=args.top_k_papers)

    corpus_signature = compute_corpus_signature(
        selected_entries,
        corpus_root=corpus_root,
        embedding_model=embedding_model,
        min_chars=args.min_paragraph_chars,
        min_words=args.min_paragraph_words,
    )

    dense_encoder = DenseEncoder(
        embedding_model,
        batch_size=args.embedding_batch_size,
        query_prefix=args.query_prefix,
        device=args.device,
    )
    cache_hit = False
    source_counts: dict[str, int] = {}
    cached_corpus = load_cached_corpus(target_dir, corpus_signature)
    if cached_corpus is not None:
        paragraphs, paragraph_embeddings, source_counts = cached_corpus
        cache_hit = True
    else:
        paragraphs, source_counts = build_paragraph_records(
            selected_entries,
            corpus_root=corpus_root,
            min_chars=args.min_paragraph_chars,
            min_words=args.min_paragraph_words,
        )
        paragraph_embeddings = dense_encoder.encode_paragraphs([paragraph.text for paragraph in paragraphs])
        save_cached_corpus(
            target_dir,
            corpus_signature,
            paragraphs=paragraphs,
            embeddings=paragraph_embeddings,
            selected_paper_count=len(selected_entries),
            source_counts=source_counts,
        )

    queries, idea_extraction, query_generation_meta = generate_queries(idea_text, args=args)
    query_embeddings = dense_encoder.encode_queries([query.query_text for query in queries])
    refinement_generator: GroundingLlmClient | None = None
    paper_contexts: dict[str, PaperContextRecord] = {}
    refinement_requested = bool(args.enable_grounding_refinement)
    refinement_stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    refinement_status = "disabled"
    refinement_error: str | None = None

    if refinement_requested:
        try:
            refinement_generator = GroundingLlmClient(
                QueryGeneratorConfig(
                    provider=getattr(args, "query_provider", None),
                    api_url=args.query_api_url,
                    model=args.query_model,
                    env_path=Path(args.env),
                    timeout=args.query_timeout,
                    max_tokens=args.query_max_tokens,
                    use_env_proxy=args.use_env_proxy,
                )
            )
            paper_contexts = build_paper_context_records(
                selected_entries,
                corpus_root=corpus_root,
                paragraphs=paragraphs,
            )
            refinement_status = "ok"
        except Exception as exc:
            refinement_status = "setup_failed"
            refinement_error = str(exc)
            refinement_generator = None

    reranker: ParagraphReranker | None = None
    reranker_status = "disabled" if args.disable_reranker else "ok"
    reranker_error: str | None = None
    if not args.disable_reranker:
        try:
            reranker = ParagraphReranker(
                reranker_model,
                batch_size=args.reranker_batch_size,
                device=args.device,
            )
        except Exception as exc:
            reranker_status = "dense_only_fallback"
            reranker_error = str(exc)

    retrieval_results: list[dict[str, Any]] = []
    for query, query_embedding in zip(queries, query_embeddings):
        dense_candidates = dense_search(
            query_embedding,
            paragraph_embeddings,
            top_k=args.dense_candidate_k,
        )

        reranked_candidates = list(dense_candidates)
        if reranker is not None and dense_candidates:
            rerank_pool = dense_candidates[: min(len(dense_candidates), args.reranker_top_n)]
            rerank_scores = reranker.score(
                query.query_text,
                [paragraphs[candidate["paragraph_index"]].text for candidate in rerank_pool],
            )
            reranked_candidates = []
            for candidate, rerank_score in zip(rerank_pool, rerank_scores):
                reranked_candidate = dict(candidate)
                reranked_candidate["rerank_score"] = rerank_score
                reranked_candidates.append(reranked_candidate)
            reranked_candidates.sort(
                key=lambda item: (item.get("rerank_score", float("-inf")), item["dense_score"]),
                reverse=True,
            )

        final_candidates = select_diverse_matches(
            reranked_candidates,
            paragraphs,
            final_top_k=args.final_top_k,
            max_paragraphs_per_paper=args.max_paragraphs_per_paper,
        )

        matches: list[dict[str, Any]] = []
        for rank, candidate in enumerate(final_candidates, start=1):
            paragraph = paragraphs[candidate["paragraph_index"]]
            matches.append(
                {
                    "rank": rank,
                    "paragraph_id": paragraph.paragraph_id,
                    "corpus_paragraph_index": candidate["paragraph_index"],
                    "paper_rank": paragraph.paper_rank,
                    "paper_title": paragraph.paper_title,
                    "paper_dir": paragraph.paper_dir,
                    "section_path": list(paragraph.section_path),
                    "section_path_text": " > ".join(paragraph.section_path),
                    "paragraph_index": paragraph.paragraph_index,
                    "dense_score": candidate["dense_score"],
                    "rerank_score": candidate.get("rerank_score"),
                    "text": paragraph.text,
                }
            )

        retrieval_results.append(
            {
                "query_id": query.query_id,
                "section": query.section,
                "sentence": query.sentence,
                "query": query.query_text,
                "matches": matches,
            }
        )

    if refinement_generator is not None:
        retrieval_results, refinement_stats = refine_retrieval_results_with_context(
            idea_text=idea_text,
            retrieval_results=retrieval_results,
            paragraphs=paragraphs,
            paper_contexts=paper_contexts,
            generator=refinement_generator,
            context_window=args.refinement_context_window,
        )

    if refinement_requested and refinement_status == "ok":
        if refinement_stats["attempted"] == 0:
            refinement_status = "no_matches_refined"
        elif refinement_stats["failed"] == 0:
            refinement_status = "ok"
        elif refinement_stats["succeeded"] == 0:
            refinement_status = "failed"
        else:
            refinement_status = "partial_failed"

    experiment_grounding = run_experiment_grounding(
        args=args,
        selected_entries=selected_entries,
        corpus_root=corpus_root,
        idea_extraction=idea_extraction,
    )

    return {
        "status": "ok",
        "input": {
            "type": grounding_input.input_type,
            "pdf_path": grounding_input.pdf_path,
        },
        "idea_text": idea_text,
        "pdf_path": grounding_input.pdf_path,
        "manifest_path": str(manifest_path),
        "papers_root": str(corpus_root),
        "corpus": {
            "signature": corpus_signature,
            "selected_paper_count": len(selected_entries),
            "paragraph_count": len(paragraphs),
            "embedding_model": embedding_model,
            "embedding_model_path": embedding_model,
            "cache_hit": cache_hit,
            "source_counts": source_counts,
        },
        "query_generation": {
            "status": query_generation_meta["status"],
            "error": query_generation_meta["error"],
            "extraction": query_generation_meta["extraction"],
            "strategy": query_generation_meta.get("strategy"),
            "fallback_used": bool(query_generation_meta.get("fallback_used")),
            "fallback_reason": query_generation_meta.get("fallback_reason"),
            "llm_error": query_generation_meta.get("llm_error"),
            "model": query_generation_meta.get("model"),
            "provider": query_generation_meta.get("provider"),
            "query_count": len(queries),
            "queries": [query.to_dict() for query in queries],
        },
        "retrieval": {
            "dense_candidate_k": args.dense_candidate_k,
            "final_top_k": args.final_top_k,
            "max_paragraphs_per_paper": args.max_paragraphs_per_paper,
            "reranker_requested": not args.disable_reranker,
            "reranker_status": reranker_status,
            "reranker_error": reranker_error,
            "reranker_model": None if args.disable_reranker else reranker_model,
            "reranker_model_path": None if args.disable_reranker else reranker_model,
            "refinement_requested": refinement_requested,
            "refinement_status": refinement_status,
            "refinement_error": refinement_error,
            "refinement_model": refinement_generator.model_name if refinement_generator is not None else None,
            "refinement_context_window": args.refinement_context_window,
            "refinement_stats": refinement_stats,
            "results": retrieval_results,
        },
        "experiment_grounding": experiment_grounding,
    }


def default_output_path(result_dir: Path, result_tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalize_whitespace(result_tag)) or "grounding"
    return result_dir / f"{timestamp}_{safe_tag}.json"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    payload = run_grounding(args)
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(
        Path(DEFAULT_RESULT_DIR), args.result_tag
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    output_path.write_text(f"{text}\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
