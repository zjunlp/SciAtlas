from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .config import SearchConfig
from .encoder import QueryEncoder
from .candidate_fusion import fuse_candidates, refresh_pre_graph_scores
from .graph_rerank import rerank_with_graph
from .models import CandidatePaper, QueryUnderstandingResult, ScoredAuthor, SearchResult
from .neo4j_repository import Neo4jSearchRepository
from .pdf_extraction import extract_pdf_to_dict
from .pdf_reference_selector import PdfReferenceSelector
from .query_understanding import QueryUnderstandingService
from .recall_embedding_path import recall_from_embeddings
from .recall_keyword_path import recall_from_keywords
from .recall_title_path import recall_from_titles
from .reranker import EmbeddingReranker
from .text_utils import min_max_normalize, normalize_text, normalize_title_exact, percentile, resolve_importance, title_similarity


def _log_stage(name: str, phase: str, start_time: float | None = None, **fields: object) -> None:
    parts = [f"[stage] name={name}", f"phase={phase}"]
    if start_time is not None:
        parts.append(f"elapsed_ms={(time.perf_counter() - start_time) * 1000:.1f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts), file=sys.stderr, flush=True)


class PaperSearchPipeline:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.config.validate()
        self.query_understanding = QueryUnderstandingService(config)
        self.pdf_reference_selector = PdfReferenceSelector(config)
        self.encoder = QueryEncoder(
            model_path=config.embedding_model_path,
            device=config.embedding_device,
            paper_embed_dim=config.paper_embed_dim,
        )
        self.reranker = (
            EmbeddingReranker(
                model_path=config.reranker_model_path,
                device=config.reranker_device or config.embedding_device,
            )
            if config.enable_embedding_rerank
            else None
        )

    def search(self, idea_text: str) -> SearchResult:
        understanding = self.query_understanding.understand(idea_text)
        return self._search_from_understanding(understanding)

    def search_pdf(self, pdf_path: str) -> SearchResult:
        try:
            extracted = extract_pdf_to_dict(pdf_path, preserve_bibr_refs=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to extract title/abstract/references from PDF: {pdf_path}") from exc
        pdf_title = str(extracted.get("title") or "").strip()
        abstract = str(extracted.get("abstract") or "").strip()
        body_sections = extracted.get("body") or []
        references = extracted.get("references") or []
        raw_references = list(references) if isinstance(references, list) else []
        self._write_pdf_debug(
            "01_extracted_pdf.json",
            {
                "pdf_path": pdf_path,
                "title": pdf_title,
                "abstract": abstract,
                "body_section_count": len(body_sections),
                "reference_count": len(raw_references),
                "references": raw_references,
            },
        )
        print(
            "[pdf_extract] "
            f"title={pdf_title or 'N/A'} | "
            f"abstract[:100]={abstract[:100] or 'N/A'} | "
            f"reference_count={len(references)}",
            file=sys.stderr,
            flush=True,
        )
        if not abstract:
            raise RuntimeError(f"PDF extraction succeeded but abstract is empty: {pdf_path}")
        selector_start = time.perf_counter()
        _log_stage(
            "select_pdf_references",
            "start",
            body_section_count=len(body_sections),
            reference_count=len(references),
            mode=self.config.pdf_reference_selection_mode,
        )
        if self.config.pdf_reference_selection_mode == "all":
            references = raw_references[: self.config.max_titles_from_pdf_references]
            selection_groups = []
            selected_ref_ids = [
                str(reference.get("ref_id") or "").strip()
                for reference in references
                if isinstance(reference, dict) and str(reference.get("ref_id") or "").strip()
            ]
        else:
            references, selection_groups = self.pdf_reference_selector.filter_references(
                pdf_title,
                abstract,
                body_sections,
                references,
            )
            selected_ref_ids = self.pdf_reference_selector.flatten_ref_ids(selection_groups)
        _log_stage(
            "select_pdf_references",
            "end",
            selector_start,
            selected_ref_count=len(selected_ref_ids),
            filtered_reference_count=len(references),
            selection_group_count=len(selection_groups),
        )
        reference_titles = [
            str(item.get("title") or "").strip()
            for item in references
            if isinstance(item, dict)
        ]
        self._write_pdf_debug(
            "02_selected_references.json",
            {
                "pdf_path": pdf_path,
                "raw_reference_count": len(raw_references),
                "selected_reference_count": len(references),
                "selection_group_count": len(selection_groups),
                "selection_groups": selection_groups,
                "selected_references": references,
                "reference_titles": reference_titles,
                "max_titles_from_pdf_references": self.config.max_titles_from_pdf_references,
                "pdf_reference_selection_mode": self.config.pdf_reference_selection_mode,
            },
        )
        understand_start = time.perf_counter()
        _log_stage("understand_pdf", "start")
        understanding = self.query_understanding.understand_pdf(
            pdf_title=pdf_title,
            abstract=abstract,
            reference_titles=reference_titles,
        )
        self._write_pdf_debug(
            "03_understanding.json",
            {
                "query": understanding.cleaned_text,
                "source_title": understanding.source_title,
                "keyword_source": understanding.keyword_source,
                "title_source": understanding.title_source,
                "keywords": [
                    {
                        "text": item.text,
                        "score": item.score,
                    }
                    for item in understanding.keywords
                ],
                "titles": [
                    {
                        "title": item.title,
                        "confidence": item.confidence,
                    }
                    for item in understanding.titles
                ],
                "reference_titles": list(understanding.reference_titles),
            },
        )
        decided_references = []
        reference_title_set = {item.title for item in understanding.titles}
        for reference in references:
            if not isinstance(reference, dict):
                continue
            title = str(reference.get("title") or "").strip()
            if not title or title not in reference_title_set:
                continue
            decided_references.append(
                {
                    "ref_id": str(reference.get("ref_id") or "").strip() or None,
                    "title": title,
                    "score": 1.0,
                }
            )
        print(
            f"[pdf_references_final] count={len(decided_references)} references={json.dumps(decided_references, ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )
        _log_stage(
            "understand_pdf",
            "end",
            understand_start,
            keyword_count=len(understanding.keywords),
            title_count=len(understanding.titles),
        )
        return self._search_from_understanding(understanding)

    def _search_from_understanding(self, understanding: QueryUnderstandingResult) -> SearchResult:
        ranked: list[CandidatePaper]
        query_vector = self.encoder.paper_query_vector(understanding.cleaned_text)
        with Neo4jSearchRepository(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password or "",
            database=self.config.neo4j_database,
        ) as repository:
            index_start = time.perf_counter()
            _log_stage("wait_for_indexes", "start")
            repository.wait_for_indexes()
            _log_stage("wait_for_indexes", "end", index_start)

            keyword_start = time.perf_counter()
            _log_stage("recall_keywords", "start")
            matched_keyword_scores = recall_from_keywords(
                understanding=understanding,
                config=self.config,
                repository=repository,
                encoder=self.encoder,
            )
            _log_stage(
                "recall_keywords",
                "end",
                keyword_start,
                matched_keyword_count=len(matched_keyword_scores),
            )

            embedding_start = time.perf_counter()
            _log_stage("recall_embeddings", "start")
            embedding_candidates = recall_from_embeddings(
                query_text=understanding.cleaned_text,
                query_vector=query_vector,
                config=self.config,
                repository=repository,
                reranker=self.reranker,
            )
            _log_stage(
                "recall_embeddings",
                "end",
                embedding_start,
                candidate_count=len(embedding_candidates),
            )

            title_start = time.perf_counter()
            _log_stage("recall_titles", "start", input_title_count=len(understanding.titles))
            title_candidates = recall_from_titles(
                understanding=understanding,
                config=self.config,
                repository=repository,
            )
            if understanding.title_source == "pdf_references":
                self._write_pdf_debug(
                    "04_title_recall_candidates.json",
                    {
                        "input_title_count": len(understanding.titles),
                        "candidate_count": len(title_candidates),
                        "candidates": [
                            {
                                "paper_id": item.paper_id,
                                "title": item.title,
                                "publication_year": item.publication_year,
                                "cited_by_count": item.cited_by_count,
                                "score_title_path": item.score_title_path,
                                "hit_sources": sorted(item.hit_sources),
                                "title_evidence": [
                                    {
                                        "extracted_title": evidence.extracted_title,
                                        "confidence": evidence.confidence,
                                        "matched_title": evidence.matched_title,
                                        "match_score": evidence.match_score,
                                        "match_type": evidence.match_type,
                                    }
                                    for evidence in item.title_evidence
                                ],
                            }
                            for item in title_candidates.values()
                        ],
                    },
                )
            _log_stage(
                "recall_titles",
                "end",
                title_start,
                input_title_count=len(understanding.titles),
                candidate_count=len(title_candidates),
            )

            fuse_start = time.perf_counter()
            candidates = fuse_candidates(
                embedding_candidates=embedding_candidates,
                title_candidates=title_candidates,
            )
            refresh_pre_graph_scores(
                candidates=candidates,
                config=self.config,
                repository=repository,
                query_vector=query_vector,
                encoder=self.encoder,
            )
            _log_stage("fuse_candidates", "end", fuse_start, merged_count=len(candidates))

            graph_scores: dict[str, float] = {}
            graph_node_scores: dict[str, float] = {}
            graph_nodes = {}
            graph_explanations: dict[str, list[str]] = {}
            if self.config.graph_method != "none":
                positive_candidate_count = sum(1 for item in candidates.values() if item.pre_graph_score > 0)
                title_seed_count = sum(1 for item in candidates.values() if item.pre_graph_score > 0 and item.title_evidence)
                graph_start = time.perf_counter()
                _log_stage(
                    "graph_rerank",
                    "start",
                    candidate_count=len(candidates),
                    positive_candidate_count=positive_candidate_count,
                    title_seed_count=title_seed_count,
                    selected_seed_paper_count=min(self.config.max_seed_papers, positive_candidate_count),
                    max_seed_papers=self.config.max_seed_papers,
                    seed_keyword_count=len(matched_keyword_scores),
                )
                graph_scores, graph_node_scores, graph_nodes, graph_explanations = rerank_with_graph(
                    candidates=candidates,
                    matched_keyword_scores=matched_keyword_scores,
                    config=self.config,
                    repository=repository,
                )
                _log_stage(
                    "graph_rerank",
                    "end",
                    graph_start,
                    graph_score_count=len(graph_scores),
                    graph_node_count=len(graph_nodes),
                    explanation_count=len(graph_explanations),
                )

            for paper_id, node in graph_nodes.items():
                if node.node_type != "Paper":
                    continue
                if paper_id not in candidates:
                    candidates[paper_id] = CandidatePaper(
                        paper_id=paper_id,
                        title=node.title or "",
                        abstract=node.abstract,
                        publication_year=node.publication_year,
                        cited_by_count=node.cited_by_count,
                    )

            finalize_start = time.perf_counter()
            _log_stage("finalize_results", "start", candidate_count=len(candidates))
            refresh_pre_graph_scores(
                candidates=candidates,
                config=self.config,
                repository=repository,
                query_vector=query_vector,
                encoder=self.encoder,
            )
            self._finalize_scores(candidates, graph_scores, graph_explanations)
            ranked = sorted(candidates.values(), key=lambda item: item.final_score, reverse=True)
            ranked = self._filter_ranked_by_target_field(ranked, repository)
            ranked = self._filter_ranked_by_source_title(ranked, understanding)
            ranked = self._dedupe_ranked_by_title(ranked)
            authors = self._collect_scored_authors(graph_nodes, graph_node_scores)
            _log_stage(
                "finalize_results",
                "end",
                finalize_start,
                final_result_count=len(ranked),
            )

        return SearchResult(
            query=understanding.cleaned_text,
            understanding=understanding,
            results=ranked[: self.config.final_top_k],
            authors=authors,
        )

    def _finalize_scores(
        self,
        candidates: dict[str, CandidatePaper],
        graph_scores: dict[str, float],
        graph_explanations: dict[str, list[str]],
    ) -> None:
        citation_p95 = percentile([item.cited_by_count for item in candidates.values()], 0.95)
        for candidate in candidates.values():
            candidate.importance = resolve_importance(
                candidate.cited_by_count,
                citation_p95,
                uniform_importance=self.config.uniform_importance,
            )
            if self.config.graph_method == "none":
                candidate.graph_score = candidate.pre_graph_score
            else:
                candidate.graph_score = graph_scores.get(candidate.paper_id, 0.0)
                if candidate.graph_score > 0:
                    candidate.hit_sources.add("graph_rerank")
            candidate.graph_evidence = graph_explanations.get(candidate.paper_id, [])

        pre_norm = min_max_normalize({paper_id: item.pre_graph_score for paper_id, item in candidates.items()})
        graph_norm = min_max_normalize({paper_id: item.graph_score for paper_id, item in candidates.items()})

        for paper_id, candidate in candidates.items():
            if self.config.graph_method == "none":
                graph_support_factor = 1.0
            else:
                graph_support_factor = max(0.25, pre_norm.get(paper_id, 0.0))
            title_bonus = self.config.final_title_bonus if _has_exact_title_hit(candidate) else 0.0
            candidate.final_score = min(
                1.0,
                (
                self.config.final_weight_pre_graph * pre_norm.get(paper_id, 0.0)
                + self.config.final_weight_graph * graph_norm.get(paper_id, 0.0) * graph_support_factor
                + self.config.final_weight_importance * candidate.importance
                + title_bonus
                ),
            )
            candidate.score_breakdown.update(
                {
                    "norm_pre_graph_score": pre_norm.get(paper_id, 0.0),
                    "norm_graph_score": graph_norm.get(paper_id, 0.0),
                    "graph_support_factor": graph_support_factor,
                    "title_bonus": title_bonus,
                    "final_score": candidate.final_score,
                }
            )

    def _write_pdf_debug(self, filename: str, payload: object) -> None:
        if not self.config.pdf_debug_dir:
            return
        debug_dir = Path(self.config.pdf_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _filter_ranked_by_target_field(
        self,
        ranked: list[CandidatePaper],
        repository: Neo4jSearchRepository,
    ) -> list[CandidatePaper]:
        if not self.config.target_field:
            return ranked
        target_field = _normalize_field_name(self.config.target_field)
        if not target_field:
            return ranked
        fields_by_paper_id = repository.fetch_paper_fields([item.paper_id for item in ranked])
        return [
            item
            for item in ranked
            if any(_normalize_field_name(field_name) == target_field for field_name in fields_by_paper_id.get(item.paper_id, []))
        ]

    def _dedupe_ranked_by_title(self, ranked: list[CandidatePaper]) -> list[CandidatePaper]:
        exact_deduped: list[CandidatePaper] = []
        seen_normalized_titles: set[str] = set()
        for candidate in ranked:
            title = (candidate.title or "").strip()
            if not title:
                exact_deduped.append(candidate)
                continue

            normalized_title = normalize_title_exact(title)
            if normalized_title and normalized_title in seen_normalized_titles:
                continue
            if normalized_title:
                seen_normalized_titles.add(normalized_title)
            exact_deduped.append(candidate)

        target_count = min(len(exact_deduped), self.config.final_top_k)
        if target_count <= 0:
            return []

        window_size = min(len(exact_deduped), max(target_count * 5, 50))
        window_step = max(target_count * 5, 50)
        while True:
            deduped = self._fuzzy_dedupe_prefix(
                exact_deduped[:window_size],
                stop_after=target_count,
            )
            if len(deduped) >= target_count or window_size >= len(exact_deduped):
                return deduped[:target_count]
            window_size = min(len(exact_deduped), window_size + window_step)

    def _fuzzy_dedupe_prefix(
        self,
        ranked: list[CandidatePaper],
        stop_after: int,
    ) -> list[CandidatePaper]:
        deduped: list[CandidatePaper] = []
        threshold = self.config.final_title_dedup_threshold
        for candidate in ranked:
            title = (candidate.title or "").strip()
            if not title:
                deduped.append(candidate)
            elif any(title_similarity(title, kept.title or "") >= threshold for kept in deduped):
                continue
            else:
                deduped.append(candidate)
            if len(deduped) >= stop_after:
                break
        return deduped

    def _filter_ranked_by_source_title(
        self,
        ranked: list[CandidatePaper],
        understanding: QueryUnderstandingResult,
    ) -> list[CandidatePaper]:
        if understanding.source_type != "pdf":
            return ranked
        source_title = (understanding.source_title or "").strip()
        if not source_title:
            return ranked
        threshold = self.config.final_pdf_source_title_filter_threshold
        return [
            candidate
            for candidate in ranked
            if not candidate.title or title_similarity(candidate.title, source_title) < threshold
        ]

    def _collect_scored_authors(
        self,
        graph_nodes: dict[str, object],
        graph_node_scores: dict[str, float],
    ) -> list[ScoredAuthor]:
        authors: list[ScoredAuthor] = []
        for node_id, node in graph_nodes.items():
            if getattr(node, "node_type", None) != "Author":
                continue
            score = float(graph_node_scores.get(node_id, 0.0))
            name = (getattr(node, "text", None) or getattr(node, "title", None) or node_id).strip()
            authors.append(
                ScoredAuthor(
                    author_id=node_id,
                    name=name,
                    score=score,
                )
            )
        authors.sort(key=lambda item: (-item.score, item.name.casefold(), item.author_id))
        return authors


def _has_exact_title_hit(candidate: CandidatePaper) -> bool:
    return any(item.match_type == "exact_normalized" for item in candidate.title_evidence)


def _normalize_field_name(value: str) -> str:
    return " ".join(value.split()).casefold()
