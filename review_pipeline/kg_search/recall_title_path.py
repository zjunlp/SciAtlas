from __future__ import annotations

import sys
import time

from .config import SearchConfig
from .models import CandidatePaper, QueryUnderstandingResult, TitleMatchEvidence
from .neo4j_repository import Neo4jSearchRepository
from .text_utils import normalize_title_exact, title_similarity


def recall_from_titles(
    understanding: QueryUnderstandingResult,
    config: SearchConfig,
    repository: Neo4jSearchRepository,
) -> dict[str, CandidatePaper]:
    def log_title_stage(phase: str, title_index: int, title_count: int, title: str, start_time: float | None = None, **fields: object) -> None:
        parts = [
            "[title_recall]",
            f"phase={phase}",
            f"title_index={title_index}/{title_count}",
            f"title={title}",
        ]
        if start_time is not None:
            parts.append(f"elapsed_ms={(time.perf_counter() - start_time) * 1000:.1f}")
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts), file=sys.stderr, flush=True)

    def upsert_title_match(
        extracted_title: str,
        confidence: float,
        row: dict[str, object],
        match_score: float,
        match_type: str,
        hit_source: str,
    ) -> None:
        matched_title = str(row.get("title") or "")
        score = confidence * match_score
        if match_type == "exact_normalized":
            score *= config.title_exact_boost

        candidate = results.get(str(row["paper_id"]))
        if candidate is None:
            candidate = CandidatePaper(
                paper_id=str(row["paper_id"]),
                title=matched_title,
                abstract=row.get("abstract"),
                publication_year=row.get("publication_year"),
                cited_by_count=int(row.get("cited_by_count") or 0),
            )
            results[str(row["paper_id"])] = candidate
        candidate.score_title_path = max(candidate.score_title_path, score)
        candidate.hit_sources.add(hit_source)
        candidate.title_evidence.append(
            TitleMatchEvidence(
                extracted_title=extracted_title,
                confidence=confidence,
                matched_title=matched_title,
                match_score=match_score,
                match_type=match_type,
            )
        )

    results: dict[str, CandidatePaper] = {}
    title_limit = (
        config.max_titles_from_pdf_references
        if understanding.title_source == "pdf_references"
        else config.max_titles_from_llm
    )
    selected_titles = sorted(
        understanding.titles,
        key=lambda item: (-item.confidence, normalize_title_exact(item.title), item.title),
    )[:title_limit]
    total_titles = len(selected_titles)
    total_rows = 0
    total_kept = 0
    exact_hit_count = 0
    ft_hit_count = 0
    overall_start = time.perf_counter()
    for index, extracted in enumerate(selected_titles, start=1):
        title_start = time.perf_counter()
        log_title_stage("start", index, total_titles, extracted.title)
        target_top_k = config.topk_papers_per_title * 4
        extracted_norm = normalize_title_exact(extracted.title)
        rows = repository.match_papers_by_normalized_title(
            normalized_title=extracted_norm,
            top_k=target_top_k,
            after_year=config.after_year,
            before_year=config.before_year,
        )
        hit_mode = "exact"
        if not rows and config.enable_title_ft:
            safe_query = extracted.title.strip().lstrip("/")
            rows = repository.fulltext_search_papers(
                index_name="paper_title_ft",
                query_text=safe_query,
                top_k=target_top_k,
                after_year=config.after_year,
                before_year=config.before_year,
            )
            hit_mode = "ft"
        elif not rows:
            hit_mode = "exact_only"
        total_rows += len(rows)
        kept = 0
        for row in rows:
            matched_title = row.get("title") or ""
            matched_norm = normalize_title_exact(matched_title)
            if not matched_norm:
                continue
            if hit_mode in {"exact", "exact_only"}:
                match_type = "exact_normalized"
                match_score = 1.0
            else:
                if extracted_norm == matched_norm:
                    match_type = "exact_normalized"
                    match_score = 1.0
                else:
                    match_type = "fuzzy"
                    match_score = title_similarity(extracted.title, matched_title)
                    if match_score < config.title_fuzzy_threshold:
                        continue

            upsert_title_match(
                extracted_title=extracted.title,
                confidence=extracted.confidence,
                row=row,
                match_score=match_score,
                match_type=match_type,
                hit_source="title_exact" if hit_mode in {"exact", "exact_only"} else "title_ft",
            )
            kept += 1
            if kept >= config.topk_papers_per_title:
                break

        if kept > 0:
            if hit_mode == "exact":
                exact_hit_count += 1
            else:
                ft_hit_count += 1
        total_kept += kept
        log_title_stage(
            "end",
            index,
            total_titles,
            extracted.title,
            title_start,
            hit_mode=hit_mode,
            row_count=len(rows),
            kept_count=kept,
            unique_candidate_count=len(results),
        )
    print(
        "[title_recall_summary] "
        f"title_count={total_titles} "
        f"elapsed_ms={(time.perf_counter() - overall_start) * 1000:.1f} "
        f"total_row_count={total_rows} "
        f"total_kept_count={total_kept} "
        f"exact_hit_count={exact_hit_count} "
        f"ft_hit_count={ft_hit_count} "
        f"unique_candidate_count={len(results)}",
        file=sys.stderr,
        flush=True,
    )
    return results
