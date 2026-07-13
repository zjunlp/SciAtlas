"""Dataclasses passed between pipeline steps.

``Paper.from_api`` is intentionally defensive: the SciAtlas ``result`` payload
is a free-form dict, so we probe a list of plausible field names for each
attribute and keep the raw record around for anything we missed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _first(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def _id_sources(record: dict) -> list[dict]:
    """The record itself plus any nested ``raw``/``paper`` dicts to probe for ids."""
    sources = [record]
    for key in ("raw", "paper"):
        nested = record.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    return sources


def _extract_paper_id(record: dict) -> str:
    """Resolve a stable, non-empty paper id.

    Different backends name the id differently: OpenAlex uses ``id`` (a ``W…``
    URL), the SciAtlas REST result uses ``paper_id``/``doi``, and the *local* KG
    record exposes neither — only ``group_id`` plus a resolvable ``paper_url``
    (a DOI) / ``pdf_url`` (an arXiv URL). Without this fallback every local-KG
    paper got ``paper_id=''`` and collapsed into a single graph node. We prefer a
    resolvable URL (DOI/arXiv) over the internal ``group_id`` so Step 2 can later
    resolve the paper to OpenAlex; a title hash is the last resort so distinct
    papers never collide on the empty string.
    """
    for source in _id_sources(record):
        explicit = _first(source, "paper_id", "id", "pid", "doi", "openalex_id", "work_id")
        if explicit:
            return str(explicit).strip()
    for source in _id_sources(record):
        for key in ("paper_url", "url", "pdf_url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for source in _id_sources(record):
        gid = source.get("group_id")
        if isinstance(gid, str) and gid.strip():
            return gid.strip()
    title = _first(record, "title", "name", default="")
    if isinstance(title, str) and title.strip():
        return "title:" + hashlib.sha1(title.strip().lower().encode("utf-8")).hexdigest()[:12]
    return ""


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    keywords: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)   # cited paper ids
    citations: list[str] = field(default_factory=list)    # citing paper ids
    is_influential: bool = False
    score: float | None = None                            # SciAtlas rank score
    idea_fields: dict[str, str] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_api(cls, record: dict) -> "Paper":
        idea_fields = _first(record, "idea_fields", default={}) or {}
        return cls(
            paper_id=_extract_paper_id(record),
            title=_first(record, "title", "name", default=""),
            abstract=_first(record, "abstract", "summary", "text", default=""),
            year=_first(record, "year", "publication_year", "pub_year"),
            keywords=_first(record, "keywords", "key_words", "tags", default=[]) or [],
            references=_first(record, "references", "reference_ids", "cites",
                              "outgoing_citations", default=[]) or [],
            citations=_first(record, "citations", "cited_by", "incoming_citations",
                             default=[]) or [],
            is_influential=bool(_first(record, "is_influential", "influential",
                                       default=False)),
            score=_first(record, "score", "rank_score", "relevance"),
            idea_fields={str(k): str(v) for k, v in idea_fields.items()} if isinstance(idea_fields, dict) else {},
            raw=record,
        )

    def short(self) -> str:
        return f"[{self.paper_id}] {self.title} ({self.year or 'n.d.'})"


@dataclass
class SeedSet:
    """Output of Step 1 — the refined query plus the top-k seed papers.

    Replaces the former ``PaperPool``: Step 1 no longer returns a broad pool but
    a small set of seed papers (``k_step1``, default 1) plus the LLM-refined
    query. ``refined_query`` supersedes the old ``seed_query`` and
    ``seed_papers`` supersedes ``target_papers``.
    """
    seed_papers: list[Paper] = field(default_factory=list)
    research_problem: str = ""
    keywords: list[str] = field(default_factory=list)
    refined_query: str = ""


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str               # explicit_pred | parallel_pred | direct_to_seed | has_keyword
    weight: float = 1.0
    edge_type: str = ""
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchGraph:
    """Output of Step 2 — paper-evolution DAG with evidence scores."""
    papers: dict[str, Paper] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    evidence_scores: dict[str, float] = field(default_factory=dict)  # paper_id -> score2
    foundational_ids: list[str] = field(default_factory=list)
    phases: list[list[str]] = field(default_factory=list)            # topo layers (paper ids)
    seed_paper_id: str | None = None                                 # primary seed (back-compat)
    seed_paper_ids: list[str] = field(default_factory=list)          # all Step-1 seeds

    def ordered_papers(self) -> list[Paper]:
        out: list[Paper] = []
        for layer in self.phases:
            for pid in layer:
                if pid in self.papers:
                    out.append(self.papers[pid])
        return out


@dataclass
class Inspiration:
    """A single same/near/cross-domain inspiration and its later analysis."""
    domain: str
    paper_title: str
    paper_abstract: str
    radius: str                 # "R1" same field / "R2" cross domain
    is_combination_plausible: bool = False
    combination_points: list[str] = field(default_factory=list)
    combination_plan: str = ""
    justification: str = ""
    source_paper: Paper | None = None
    candidate_id: int | None = None


@dataclass
class Idea:
    title: str
    description: str = ""
    novelty: str = ""
    significance: str = ""
    motivation: str = ""
    methods: str = ""
    # Structured idea body (Step 9 emits these; ``methods`` is synthesized from them
    # for back-compat with the novelty check + plain-text consumers).
    method_overview: str = ""   # one-paragraph overview shown before the numbered components
    proposed_method: list[dict] = field(default_factory=list)   # [{"name","description"}]
    analogy_mapping: list[dict] = field(default_factory=list)    # [{"inspiration","concept","adaptation","citation"}]
    experiment_plan: list[str] = field(default_factory=list)
    inspiration_sources: list[str] = field(default_factory=list)
    key_references: list[str] = field(default_factory=list)
    # citations actually used in motivation/methods, rendered as the
    # References section. Each item: {"citation": "Surname et al., Year", "url": "..."}.
    references: list[dict] = field(default_factory=list)
    # filled by post-Step-9 novelty check
    novelty_level: str | None = None
    novelty_justification: str | None = None
    improvement_suggestions: list[str] = field(default_factory=list)
