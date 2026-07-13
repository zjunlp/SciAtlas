from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractedKeyword:
    text: str
    score: int

    @property
    def normalized_score(self) -> float:
        return max(0.0, min(1.0, self.score / 10.0))


@dataclass(slots=True)
class ExtractedTitle:
    title: str
    confidence: float


@dataclass(slots=True)
class QueryUnderstandingResult:
    cleaned_text: str
    keywords: list[ExtractedKeyword]
    titles: list[ExtractedTitle]
    keyword_source: str
    title_source: str
    source_type: str = "idea_text"
    source_title: str | None = None
    reference_titles: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KeywordMatchEvidence:
    input_keyword: str
    input_score: float
    kg_keyword_id: str
    kg_keyword_text: str
    match_score: float
    match_type: str
    edge_relevance_score: float | None = None


@dataclass(slots=True)
class TitleMatchEvidence:
    extracted_title: str
    confidence: float
    matched_title: str
    match_score: float
    match_type: str


@dataclass(slots=True)
class CandidatePaper:
    paper_id: str
    title: str
    abstract: str | None
    publication_year: int | None
    cited_by_count: int
    score_kw_path: float = 0.0
    score_emb_path: float = 0.0
    score_title_path: float = 0.0
    pre_graph_score: float = 0.0
    graph_score: float = 0.0
    importance: float = 0.0
    final_score: float = 0.0
    hit_sources: set[str] = field(default_factory=set)
    keyword_evidence: list[KeywordMatchEvidence] = field(default_factory=list)
    title_evidence: list[TitleMatchEvidence] = field(default_factory=list)
    graph_evidence: list[str] = field(default_factory=list)
    score_breakdown: dict[str, Any] = field(default_factory=dict)

    def merge_metadata(self, other: CandidatePaper) -> None:
        if not self.title and other.title:
            self.title = other.title
        if not self.abstract and other.abstract:
            self.abstract = other.abstract
        if self.publication_year is None and other.publication_year is not None:
            self.publication_year = other.publication_year
        self.cited_by_count = max(self.cited_by_count, other.cited_by_count)


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: str
    title: str | None = None
    text: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    cited_by_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_text(self) -> str:
        return self.title or self.text or self.node_id


@dataclass(slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredAuthor:
    author_id: str
    name: str
    score: float


@dataclass(slots=True)
class SearchResult:
    query: str
    understanding: QueryUnderstandingResult
    results: list[CandidatePaper]
    authors: list[ScoredAuthor] = field(default_factory=list)
