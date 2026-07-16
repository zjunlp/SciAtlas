from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (
    LEGACY_SECTION_ALIASES,
    REVIEW_SECTIONS,
    RUBRIC_STANDARD_KEYS,
    citation_label,
    citation_markdown,
    normalize_whitespace,
    write_json,
)
from .figure_cards import generate_figure_cards


RUBRIC_SECTION_KEYS = dict(RUBRIC_STANDARD_KEYS)

SECTION_ID_PREFIX = {
    "Motivation": "motivation",
    "Method": "method",
    "Result": "result",
    "Discussion": "discussion",
}

FALLBACK_RUBRIC = {
    "Motivation": [
        {
            "dimension_name": "Problem-Specific Contribution Framing",
            "core_philosophy": "The idea should connect the claimed contribution to a concrete research gap rather than a broad field-level motivation.",
            "required_evidence": "A precise problem statement, named target setting, and explicit contrast with the closest mechanisms or assumptions in prior work.",
        },
        {
            "dimension_name": "Literature-Grounded Differentiation",
            "core_philosophy": "The motivation should make clear why the proposed mechanism is not a routine recombination of existing approaches.",
            "required_evidence": "Concrete overlap and difference points against representative related systems, datasets, or evaluation settings.",
        },
    ],
    "Method": [
        {
            "dimension_name": "Mechanism-Level Technical Consistency",
            "core_philosophy": "The central method should specify how its components interact and why those interactions support the claimed behavior.",
            "required_evidence": "Algorithmic steps, model components, assumptions, optimization objectives, and implementation constraints tied to the proposed mechanism.",
        },
        {
            "dimension_name": "Assumption and Resource Plausibility",
            "core_philosophy": "The proposal should identify the data, compute, privacy, safety, or deployment assumptions that make the method feasible.",
            "required_evidence": "Resource estimates, access requirements, threat or failure assumptions, and mitigation plans for the main technical risks.",
        },
    ],
    "Result": [
        {
            "dimension_name": "Claim-Aligned Evaluation Protocol",
            "core_philosophy": "The evaluation should directly test the strongest technical and empirical claims rather than only showing aggregate performance.",
            "required_evidence": "Datasets, splits, baselines, metrics, ablations, statistical checks, and failure analyses tied to each major claim.",
        }
    ],
    "Discussion": [
        {
            "dimension_name": "Boundary Conditions and Failure Modes",
            "core_philosophy": "The discussion should state where the idea is expected to fail or lose relevance, and what that implies for impact.",
            "required_evidence": "Limitations, generalization boundaries, deployment constraints, ethical or privacy risks, and concrete future validation steps.",
        }
    ],
}


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return normalize_whitespace("; ".join(normalize_whitespace(item) for item in value))
    if isinstance(value, dict):
        return normalize_whitespace(json.dumps(value, ensure_ascii=False))
    return normalize_whitespace(value)


def _normalize_standard(section: str, index: int, raw: dict[str, Any]) -> dict[str, Any]:
    prefix = SECTION_ID_PREFIX[section]
    payload = {
        "standard_id": f"{prefix}_{index:02d}",
        "dimension_name": normalize_whitespace(raw.get("dimension_name")) or f"{section} Standard {index}",
        "core_philosophy": normalize_whitespace(raw.get("core_philosophy") or raw.get("core_reason")),
        "required_evidence": _as_text(raw.get("required_evidence")),
    }
    source_tag = normalize_whitespace(raw.get("source_tag"))
    if source_tag:
        payload["source_tag"] = source_tag
    target_section = normalize_whitespace(raw.get("target_section"))
    if target_section:
        payload["target_section"] = target_section
    return payload


def _fallback_sections() -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section in REVIEW_SECTIONS:
        standards = FALLBACK_RUBRIC.get(section, [])
        sections.append(
            {
                "section": section,
                "standards": [
                    _normalize_standard(section, index, standard)
                    for index, standard in enumerate(standards, start=1)
                ],
            }
        )
    return sections


def _canonical_section_name(value: Any) -> str:
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


def _infer_standard_section(raw: dict[str, Any]) -> str:
    explicit = _canonical_section_name(raw.get("target_section"))
    if explicit:
        return explicit
    text = " ".join(
        normalize_whitespace(raw.get(key)).casefold()
        for key in ("dimension_name", "core_philosophy", "core_reason", "required_evidence")
    )
    if any(term in text for term in ("dataset", "baseline", "metric", "ablation", "benchmark", "experiment", "evaluation", "result")):
        return "Result"
    if any(term in text for term in ("algorithm", "model", "method", "mechanism", "optimization", "privacy", "protocol", "assumption")):
        return "Method"
    if any(term in text for term in ("limitation", "failure", "deployment", "generalization", "ethic", "risk", "future")):
        return "Discussion"
    return "Motivation"


def normalize_rubric(
    rubric_path: Path | str | None,
    *,
    idea_text: str = "",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    raw_payload: dict[str, Any] = {}
    source_path = normalize_whitespace(rubric_path)

    if source_path:
        try:
            path = Path(source_path).expanduser().resolve()
            if path.exists():
                raw_payload = read_json_object(path)
                source_path = str(path)
            else:
                errors.append({"path": source_path, "error": "rubric file does not exist"})
        except Exception as exc:
            errors.append({"path": source_path, "error": str(exc)})

    sections: list[dict[str, Any]] = []
    standard_count = 0
    standards_by_section: dict[str, list[dict[str, Any]]] = {section: [] for section in REVIEW_SECTIONS}
    for section, raw_key in RUBRIC_SECTION_KEYS.items():
        raw_standards = raw_payload.get(raw_key, [])
        if not isinstance(raw_standards, list):
            raw_standards = []
        standards_by_section[section].extend(standard for standard in raw_standards if isinstance(standard, dict))

    legacy_section_keys = {
        "Introduction": "Introduction_Standards",
        "Methods": "Methods_Standards",
        "Results": "Results_Standards",
    }
    for legacy_section, raw_key in legacy_section_keys.items():
        raw_standards = raw_payload.get(raw_key, [])
        canonical_section = _canonical_section_name(legacy_section)
        if not canonical_section or not isinstance(raw_standards, list):
            continue
        standards_by_section[canonical_section].extend(standard for standard in raw_standards if isinstance(standard, dict))

    raw_idea_specific = raw_payload.get("Idea_Specific_Standards", [])
    if isinstance(raw_idea_specific, list):
        for standard in raw_idea_specific:
            if not isinstance(standard, dict):
                continue
            standards_by_section[_infer_standard_section(standard)].append(standard)

    for section in REVIEW_SECTIONS:
        raw_standards = standards_by_section.get(section, [])
        standards = [
            _normalize_standard(section, index, standard)
            for index, standard in enumerate(raw_standards, start=1)
            if isinstance(standard, dict)
        ]
        standard_count += len(standards)
        sections.append({"section": section, "standards": standards})

    fallback_used = standard_count == 0
    if fallback_used:
        sections = _fallback_sections()

    payload = {
        "status": "fallback" if fallback_used else "ok",
        "source_path": source_path or None,
        "fallback_used": fallback_used,
        "errors": errors,
        "schema_version": raw_payload.get("schema_version"),
        "idea_summary": normalize_whitespace(raw_payload.get("Idea_Summary")) or normalize_whitespace(idea_text)[:500],
        "idea_breakdown": raw_payload.get("Idea_Breakdown") if isinstance(raw_payload.get("Idea_Breakdown"), dict) else None,
        "sections": sections,
    }
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def _load_reviewer_summary(result: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    summary_path = normalize_whitespace(result.get("summary_path"))
    if not summary_path:
        return {}, "", "missing summary_path"
    path = Path(summary_path).expanduser().resolve()
    if not path.exists():
        return {}, str(path), "summary_path does not exist"
    try:
        return read_json_object(path), str(path), ""
    except Exception as exc:
        return {}, str(path), str(exc)


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _paper_payload(item: dict[str, Any]) -> dict[str, Any]:
    paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
    return paper if isinstance(paper, dict) else {}


def _paper_url(item: dict[str, Any], paper: dict[str, Any]) -> str:
    return (
        normalize_whitespace(item.get("paper_url"))
        or normalize_whitespace(item.get("pdf_url"))
        or normalize_whitespace(paper.get("url"))
        or normalize_whitespace(paper.get("pdf_url"))
        or normalize_whitespace(item.get("doi"))
        or normalize_whitespace(paper.get("doi"))
        or normalize_whitespace(paper.get("id"))
    )


def _paper_year(item: dict[str, Any], paper: dict[str, Any]) -> Any:
    return (
        item.get("year")
        or item.get("publication_year")
        or paper.get("year")
        or paper.get("publication_year")
        or paper.get("publication_date")
    )


def _paper_publication_date(item: dict[str, Any], paper: dict[str, Any]) -> str | None:
    value = (
        normalize_whitespace(item.get("publication_date"))
        or normalize_whitespace(item.get("publicationDate"))
        or normalize_whitespace(paper.get("publication_date"))
        or normalize_whitespace(paper.get("publicationDate"))
    )
    return value or None


def _paper_authors(item: dict[str, Any], paper: dict[str, Any]) -> list[str]:
    raw_authors = item.get("authors") or paper.get("authors") or []
    if not isinstance(raw_authors, list):
        return []
    authors: list[str] = []
    for author in raw_authors:
        if isinstance(author, dict):
            name = normalize_whitespace(author.get("name") or author.get("display_name") or author.get("author_name"))
        else:
            name = normalize_whitespace(author)
        if name:
            authors.append(name)
    return authors


def _select_raw_papers(search_payload: dict[str, Any]) -> list[dict[str, Any]]:
    ranking_papers = search_payload.get("ranking", {}).get("papers") if isinstance(search_payload.get("ranking"), dict) else None
    if isinstance(ranking_papers, list) and ranking_papers:
        return [item for item in ranking_papers if isinstance(item, dict)]

    filtered_papers = search_payload.get("filtered", {}).get("unique_papers") if isinstance(search_payload.get("filtered"), dict) else None
    if isinstance(filtered_papers, list) and filtered_papers:
        return [item for item in filtered_papers if isinstance(item, dict)]

    combined_papers = search_payload.get("combined", {}).get("papers") if isinstance(search_payload.get("combined"), dict) else None
    if isinstance(combined_papers, list):
        return [item for item in combined_papers if isinstance(item, dict)]
    return []


def build_searched_papers(
    search_result_path: Path | str | None,
    *,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    source_path = normalize_whitespace(search_result_path)
    search_payload: dict[str, Any] = {}

    if source_path:
        try:
            path = Path(source_path).expanduser().resolve()
            if path.exists():
                search_payload = read_json_object(path)
                source_path = str(path)
            else:
                errors.append({"path": source_path, "error": "search result does not exist"})
        except Exception as exc:
            errors.append({"path": source_path, "error": str(exc)})

    papers: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for index, item in enumerate(_select_raw_papers(search_payload), start=1):
        paper = _paper_payload(item)
        title = normalize_whitespace(item.get("title") or paper.get("title"))
        if not title:
            continue
        title_key = title.casefold()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        authors = _paper_authors(item, paper)
        year = _paper_year(item, paper)
        publication_date = _paper_publication_date(item, paper)
        url = _paper_url(item, paper) or None
        papers.append(
            {
                "paper_index": len(papers) + 1,
                "paper_ref": f"[{len(papers) + 1}]",
                "source_rank": item.get("rank") or item.get("source_rank") or index,
                "source": normalize_whitespace(item.get("source")) or None,
                "title": title,
                "year": year,
                "publication_date": publication_date,
                "venue": normalize_whitespace(item.get("venue") or paper.get("venue") or paper.get("venue_source_display_name")) or None,
                "authors": authors,
                "citation_label": citation_label(authors, year=year, publication_date=publication_date),
                "citation_markdown": citation_markdown(
                    authors,
                    year=year,
                    publication_date=publication_date,
                    url=url,
                ),
                "abstract": normalize_whitespace(item.get("abstract") or paper.get("abstract")),
                "url": url,
                "pdf_url": normalize_whitespace(item.get("pdf_url") or paper.get("pdf_url")) or None,
                "doi": normalize_whitespace(item.get("doi") or paper.get("doi")) or None,
                "citation_count": item.get("citation_count") or paper.get("citationCount") or paper.get("cited_by_count"),
            }
        )

    payload = {
        "status": "ok" if papers else ("error" if errors else "missing"),
        "source_path": source_path or None,
        "paper_count": len(papers),
        "papers": papers,
        "errors": errors,
    }
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def _background_from_summary(summary: dict[str, Any]) -> str:
    profile = normalize_whitespace(summary.get("overall_academic_profile"))
    if profile:
        return profile
    summary_value = summary.get("summary")
    if isinstance(summary_value, list):
        return normalize_whitespace(" ".join(normalize_whitespace(item) for item in summary_value))
    return normalize_whitespace(summary_value)


def normalize_reviewers(
    reviewers_index_path: Path | str | None,
    *,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    raw_payload: dict[str, Any] = {}
    source_path = normalize_whitespace(reviewers_index_path)

    if source_path:
        try:
            path = Path(source_path).expanduser().resolve()
            if path.exists():
                raw_payload = read_json_object(path)
                source_path = str(path)
            else:
                errors.append({"path": source_path, "error": "reviewers index does not exist"})
        except Exception as exc:
            errors.append({"path": source_path, "error": str(exc)})

    raw_results = raw_payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    reviewers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_index, result in enumerate(raw_results, start=1):
        if not isinstance(result, dict):
            continue
        reviewer_label = f"reviewer_{raw_index:02d}"
        if normalize_whitespace(result.get("status")) != "ok":
            skipped.append(
                {
                    "reviewer_id": reviewer_label,
                    "author_name": normalize_whitespace(result.get("author_name")),
                    "error": normalize_whitespace(result.get("error")) or "reviewer worker did not finish ok",
                }
            )
            continue

        summary, summary_path, summary_error = _load_reviewer_summary(result)
        if summary_error:
            errors.append(
                {
                    "reviewer_id": reviewer_label,
                    "author_name": normalize_whitespace(result.get("author_name")),
                    "summary_path": summary_path or None,
                    "error": summary_error,
                }
            )
            continue

        author_name = normalize_whitespace(result.get("author_name") or summary.get("author_name"))
        author_id = normalize_whitespace(result.get("author_id") or summary.get("author_id"))
        reviewers.append(
            {
                "reviewer_id": f"reviewer_{len(reviewers) + 1:02d}",
                "source_reviewer_index": raw_index,
                "author_name": author_name or author_id or f"Reviewer {len(reviewers) + 1}",
                "author_id": author_id or None,
                "academic_background": _background_from_summary(summary),
                "research_trajectory": _normalize_list(summary.get("relevant_research_trajectory")),
                "technical_arsenal": _normalize_list(summary.get("technical_arsenal")),
                "persona": normalize_whitespace(summary.get("persona")),
                "summary_path": summary_path,
                "relevant_papers_path": normalize_whitespace(result.get("relevant_papers_path")) or None,
                "worker_result": {
                    "mode": normalize_whitespace(result.get("mode")),
                    "work_dir": normalize_whitespace(result.get("work_dir")) or None,
                },
            }
        )

    status = "ok" if reviewers else "error"
    if reviewers and (errors or skipped):
        status = "partial_error"
    payload = {
        "status": status,
        "source_path": source_path or None,
        "reviewer_count": len(reviewers),
        "reviewers": reviewers,
        "skipped": skipped,
        "errors": errors,
    }
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def _score_value(value: Any) -> float:
    try:
        if value is None:
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _extract_structured_idea(payload: dict[str, Any]) -> dict[str, Any]:
    query_generation = payload.get("query_generation")
    if isinstance(query_generation, dict):
        extraction = query_generation.get("extraction")
        if isinstance(extraction, dict):
            return extraction
    extraction = payload.get("structured_extraction")
    if isinstance(extraction, dict):
        return extraction
    return {}


def _extract_evidence_text(match: dict[str, Any]) -> tuple[str, str | None]:
    refined = match.get("refined_grounding")
    if isinstance(refined, dict):
        grounded_passage = normalize_whitespace(refined.get("grounded_passage"))
        if grounded_passage:
            return grounded_passage, normalize_whitespace(refined.get("coverage_label")) or None
    return normalize_whitespace(match.get("text")), None


def _paper_by_rank(searched_papers: dict[str, Any]) -> dict[int, dict[str, Any]]:
    papers = searched_papers.get("papers") if isinstance(searched_papers.get("papers"), list) else []
    by_rank: dict[int, dict[str, Any]] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        try:
            by_rank[int(paper.get("paper_index"))] = paper
        except (TypeError, ValueError):
            continue
    return by_rank


def _paper_by_title(searched_papers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    papers = searched_papers.get("papers") if isinstance(searched_papers.get("papers"), list) else []
    return {
        normalize_whitespace(paper.get("title")).casefold(): paper
        for paper in papers
        if isinstance(paper, dict) and normalize_whitespace(paper.get("title"))
    }


def _attach_paper_reference(
    card: dict[str, Any],
    *,
    by_rank: dict[int, dict[str, Any]],
    by_title: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paper: dict[str, Any] | None = None
    try:
        paper = by_rank.get(int(card.get("paper_rank")))
    except (TypeError, ValueError):
        paper = None
    if paper is None:
        paper = by_title.get(normalize_whitespace(card.get("paper_title")).casefold())
    if paper is None:
        return card
    citation = paper.get("citation_markdown") or citation_markdown(
        paper.get("authors"),
        year=paper.get("year"),
        publication_date=paper.get("publication_date"),
        url=paper.get("url"),
    )
    return {
        **card,
        "paper_index": paper.get("paper_index"),
        "paper_ref": paper.get("paper_ref"),
        "paper_title": paper.get("title") or card.get("paper_title"),
        "paper_url": paper.get("url"),
        "paper_year": paper.get("year"),
        "paper_venue": paper.get("venue"),
        "citation_label": paper.get("citation_label")
        or citation_label(paper.get("authors"), year=paper.get("year"), publication_date=paper.get("publication_date")),
        "citation_markdown": citation,
    }


def build_evidence_bank(
    grounding_result_path: Path | str | None,
    *,
    searched_papers: dict[str, Any] | None = None,
    figure_cards_payload: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    source_path = normalize_whitespace(grounding_result_path)
    grounding_payload: dict[str, Any] = {}

    if source_path:
        try:
            path = Path(source_path).expanduser().resolve()
            if path.exists():
                grounding_payload = read_json_object(path)
                source_path = str(path)
            else:
                errors.append({"path": source_path, "error": "grounding result does not exist"})
        except Exception as exc:
            errors.append({"path": source_path, "error": str(exc)})

    evidence_cards: list[dict[str, Any]] = []
    retrieval = grounding_payload.get("retrieval", {})
    raw_results = retrieval.get("results", []) if isinstance(retrieval, dict) else []
    if not isinstance(raw_results, list):
        raw_results = []
    searched_papers = searched_papers or {"papers": []}
    paper_rank_lookup = _paper_by_rank(searched_papers)
    paper_title_lookup = _paper_by_title(searched_papers)

    next_evidence_id = 1
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        raw_matches = result.get("matches", [])
        if not isinstance(raw_matches, list):
            continue
        for match in raw_matches:
            if not isinstance(match, dict):
                continue
            evidence_text, coverage_label = _extract_evidence_text(match)
            if not evidence_text:
                continue
            refined = match.get("refined_grounding") if isinstance(match.get("refined_grounding"), dict) else {}
            evidence_cards.append(
                _attach_paper_reference(
                    {
                    "evidence_id": f"E{next_evidence_id:03d}",
                    "query_id": normalize_whitespace(result.get("query_id")),
                    "idea_section": normalize_whitespace(result.get("section")),
                    "idea_sentence": normalize_whitespace(result.get("sentence")),
                    "query": normalize_whitespace(result.get("query")),
                    "paper_title": normalize_whitespace(match.get("paper_title")),
                    "paragraph_id": normalize_whitespace(match.get("paragraph_id")),
                    "paper_rank": match.get("paper_rank"),
                    "section_path": match.get("section_path") if isinstance(match.get("section_path"), list) else [],
                    "section_path_text": normalize_whitespace(match.get("section_path_text")),
                    "evidence_text": evidence_text,
                    "dense_score": match.get("dense_score"),
                    "rerank_score": match.get("rerank_score"),
                    "coverage_label": coverage_label,
                    "refined_status": normalize_whitespace(refined.get("status")),
                    "shared_points": refined.get("shared_points") if isinstance(refined.get("shared_points"), list) else [],
                    "different_points": (
                        refined.get("different_points") if isinstance(refined.get("different_points"), list) else []
                    ),
                    },
                    by_rank=paper_rank_lookup,
                    by_title=paper_title_lookup,
                )
            )
            next_evidence_id += 1

    evidence_cards.sort(
        key=lambda item: (
            _score_value(item.get("rerank_score")),
            _score_value(item.get("dense_score")),
            -int(item.get("paper_rank") or 999999),
        ),
        reverse=True,
    )

    experiment_cards: list[dict[str, Any]] = []
    experiment_grounding = grounding_payload.get("experiment_grounding", {})
    raw_experiment_results = (
        experiment_grounding.get("results", []) if isinstance(experiment_grounding, dict) else []
    )
    if not isinstance(raw_experiment_results, list):
        raw_experiment_results = []

    next_experiment_id = 1
    for result in raw_experiment_results:
        if not isinstance(result, dict) or normalize_whitespace(result.get("status")) != "ok":
            continue
        coverage = result.get("coverage_analysis")
        if not isinstance(coverage, dict):
            coverage = {}
        recommendation = result.get("paper_inspired_recommendation")
        raw_goals = recommendation.get("recommended_experimental_goals", []) if isinstance(recommendation, dict) else []
        if not isinstance(raw_goals, list) or not raw_goals:
            raw_goals = [{"goal": ""}]
        for raw_goal in raw_goals:
            if isinstance(raw_goal, dict):
                goal = normalize_whitespace(raw_goal.get("goal"))
                rationale = normalize_whitespace(raw_goal.get("rationale"))
                inspired_by = raw_goal.get("inspired_by") if isinstance(raw_goal.get("inspired_by"), list) else []
            else:
                goal = normalize_whitespace(raw_goal)
                rationale = ""
                inspired_by = []
            experiment_cards.append(
                _attach_paper_reference(
                    {
                    "evidence_id": f"X{next_experiment_id:03d}",
                    "paper_title": normalize_whitespace(result.get("paper_title")),
                    "paper_rank": result.get("paper_rank"),
                    "recommended_goal": goal,
                    "goal_rationale": rationale,
                    "inspired_by": inspired_by,
                    "coverage_label": normalize_whitespace(coverage.get("coverage_label")),
                    "coverage_score": coverage.get("coverage_score"),
                    "coverage_rationale": normalize_whitespace(coverage.get("coverage_rationale")),
                    "overlap": coverage.get("overlap") if isinstance(coverage.get("overlap"), list) else [],
                    "missing_or_undercovered": (
                        coverage.get("missing_or_undercovered")
                        if isinstance(coverage.get("missing_or_undercovered"), list)
                        else []
                    ),
                    "additional_focus_in_idea": (
                        coverage.get("additional_focus_in_idea")
                        if isinstance(coverage.get("additional_focus_in_idea"), list)
                        else []
                    ),
                    },
                    by_rank=paper_rank_lookup,
                    by_title=paper_title_lookup,
                )
            )
            next_experiment_id += 1

    experiment_cards.sort(
        key=lambda item: (_score_value(item.get("coverage_score")), -int(item.get("paper_rank") or 999999)),
        reverse=True,
    )

    status = "ok" if normalize_whitespace(grounding_payload.get("status")) == "ok" else "missing"
    if errors:
        status = "error"
    payload = {
        "status": status,
        "source_path": source_path or None,
        "structured_idea": _extract_structured_idea(grounding_payload),
        "searched_papers": searched_papers,
        "evidence_cards": evidence_cards,
        "experiment_cards": experiment_cards,
        "figure_cards": (
            figure_cards_payload.get("figure_cards", [])
            if isinstance(figure_cards_payload, dict) and isinstance(figure_cards_payload.get("figure_cards"), list)
            else []
        ),
        "stats": {
            "evidence_card_count": len(evidence_cards),
            "experiment_card_count": len(experiment_cards),
            "figure_card_count": (
                len(figure_cards_payload.get("figure_cards", []))
                if isinstance(figure_cards_payload, dict) and isinstance(figure_cards_payload.get("figure_cards"), list)
                else 0
            ),
            "grounding_status": normalize_whitespace(grounding_payload.get("status")),
            "retrieval_status": normalize_whitespace(retrieval.get("status")) if isinstance(retrieval, dict) else "",
            "experiment_grounding_status": (
                normalize_whitespace(experiment_grounding.get("status"))
                if isinstance(experiment_grounding, dict)
                else ""
            ),
            "figure_cards_status": (
                normalize_whitespace(figure_cards_payload.get("status"))
                if isinstance(figure_cards_payload, dict)
                else ""
            ),
        },
        "errors": errors,
        "figure_cards_errors": (
            figure_cards_payload.get("errors", [])
            if isinstance(figure_cards_payload, dict) and isinstance(figure_cards_payload.get("errors"), list)
            else []
        ),
    }
    if output_path is not None:
        write_json(Path(output_path), payload)
    return payload


def normalize_artifacts(
    *,
    idea_text: str,
    source_full_text: str = "",
    rubric_path: Path | str | None,
    reviewers_index_path: Path | str | None,
    grounding_result_path: Path | str | None,
    search_result_path: Path | str | None,
    enable_figure_cards: bool = False,
    figure_dir: Path | str | None = None,
    figure_card_base_url: str | None = None,
    figure_card_model_name: str | None = None,
    figure_card_api_key: str | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_rubric = normalize_rubric(
        rubric_path,
        idea_text=idea_text,
        output_path=output_dir / "rubric.normalized.json",
    )
    normalized_reviewers = normalize_reviewers(
        reviewers_index_path,
        output_path=output_dir / "reviewers.normalized.json",
    )
    searched_papers = build_searched_papers(
        search_result_path,
        output_path=output_dir / "searched_papers.json",
    )
    figure_cards_payload: dict[str, Any] = {
        "status": "skipped",
        "figure_dir": str(Path(str(figure_dir)).expanduser().resolve()) if figure_dir else None,
        "figure_card_count": 0,
        "figure_cards": [],
        "errors": [],
    }
    if enable_figure_cards:
        figure_cards_payload = generate_figure_cards(
            figure_dir=figure_dir,
            source_full_text=source_full_text,
            base_url=figure_card_base_url,
            model_name=figure_card_model_name,
            api_key=figure_card_api_key,
            output_path=output_dir / "figure_cards.json",
        )
    evidence_bank = build_evidence_bank(
        grounding_result_path,
        searched_papers=searched_papers,
        figure_cards_payload=figure_cards_payload,
        output_path=output_dir / "evidence_bank.json",
    )
    return {
        "status": "ok"
        if normalized_reviewers["status"] in {"ok", "partial_error"}
        else normalized_reviewers["status"],
        "rubric": normalized_rubric,
        "reviewers": normalized_reviewers,
        "evidence_bank": evidence_bank,
        "searched_papers": searched_papers,
        "figure_cards": figure_cards_payload,
        "paths": {
            "rubric": str((output_dir / "rubric.normalized.json").resolve()),
            "reviewers": str((output_dir / "reviewers.normalized.json").resolve()),
            "searched_papers": str((output_dir / "searched_papers.json").resolve()),
            "evidence_bank": str((output_dir / "evidence_bank.json").resolve()),
            "figure_cards": str((output_dir / "figure_cards.json").resolve()) if enable_figure_cards else None,
        },
    }
