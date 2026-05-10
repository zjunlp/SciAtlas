from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.api_client import SciNetApiClient, load_scinet_api_settings
from ..core.common import (
    dedupe_preserve_order,
    ensure_dir,
    get_env_value,
    load_env_values,
    normalize_whitespace,
    truncate_text,
    write_json,
)
from ..llm import call_llm_json, resolve_llm_settings
from ..llm.prompts import (
    IDEA_EVALUATION_SYSTEM_PROMPT,
    IDEA_EVALUATION_USER_PROMPT,
    build_author_profile_prompt,
    build_idea_generation_prompt,
    build_trend_prompt,
)
from ..search.planner import build_search_plan
from ..search.reranker import rerank_search_payload
from ..core.schemas import (
    SciNetRequest,
    TASK_AUTHOR_PROFILE,
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_RELATED_AUTHORS,
    TASK_TOPIC_TREND_REVIEW,
    merge_task_params,
)


def _paper_card_from_ranked_item(item: dict[str, Any], *, abstract_char_limit: int = 420) -> dict[str, Any]:
    return {
        "title": normalize_whitespace(item.get("title") or item.get("paper", {}).get("title")),
        "abstract": truncate_text(
            item.get("abstract") or item.get("paper", {}).get("abstract"),
            max_chars=abstract_char_limit,
        ),
        "year": item.get("year") or item.get("publication_year"),
        "source": normalize_whitespace(item.get("source") or item.get("match_type")),
        "rank": item.get("rank"),
        "score": item.get("llm_score") if item.get("llm_score") is not None else item.get("score"),
        "citation_count": item.get("citation_count") or item.get("cited_by_count") or item.get("citations"),
    }


def _search_options_from_params(params: dict[str, Any], *, top_k: int | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "top_k": int(top_k if top_k is not None else params.get("search_api_top_k") or params.get("search_final_top_k") or 20),
    }
    for key in ("target_field", "after", "before"):
        value = params.get(key)
        if value is not None and normalize_whitespace(value):
            options[key] = value
    return options


def _build_search_plan(
    *,
    input_payload: dict[str, Any],
    params: dict[str, Any],
    env_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    text = normalize_whitespace(input_payload.get("idea_text") or input_payload.get("topic_text")) or None
    pdf_path = normalize_whitespace(input_payload.get("pdf_path")) or None
    return build_search_plan(
        text=text,
        pdf_path=pdf_path,
        params=params,
        env_path=env_path,
        artifact_dir=artifact_dir,
    )


def _run_search_via_api(
    *,
    client: SciNetApiClient,
    plan: dict[str, Any],
    params: dict[str, Any],
    env_path: Path,
    artifact_dir: Path,
    api_top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    ensure_dir(artifact_dir)
    response = client.search(
        plan=plan,
        options=_search_options_from_params(params, top_k=api_top_k),
    )
    response_path = artifact_dir / "api_response.json"
    write_json(response_path, response)
    payload = response.get("result", {}) if isinstance(response.get("result"), dict) else {}
    payload = rerank_search_payload(
        search_payload=payload,
        plan=plan,
        env_path=env_path,
        params=params,
        final_top_k=rerank_top_k,
    )
    result_path = artifact_dir / "result.json"
    write_json(result_path, payload)
    return response, payload, result_path


def _run_manifest(
    *,
    search_result_path: Path,
    params: dict[str, Any],
    env_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, Any], Path]:
    from ..evidence import pdf_manifest

    ensure_dir(artifact_dir)
    env_values = load_env_values(env_path)
    grobid_base_url = normalize_whitespace(
        params.get("grobid_base_url")
        or get_env_value(env_values, "GROBID_BASE_URL")
        or "http://127.0.0.1:8070"
    )
    cli_args = [
        "--input",
        str(search_result_path),
        "--paper-list-path",
        "ranking.papers",
        "--top-k",
        str(int(params["manifest_top_k"])),
        "--env",
        str(env_path),
        "--output-root",
        str(artifact_dir),
        "--result-tag",
        "manifest",
        "--grobid-base-url",
        grobid_base_url,
    ]
    if params.get("use_env_proxy"):
        cli_args.append("--use-env-proxy")
    args = pdf_manifest.build_parser().parse_args(cli_args)
    manifest = pdf_manifest.run_pipeline(args)
    return manifest, Path(manifest["manifest_path"]).resolve()


def _default_query_provider(env_path: Path) -> str | None:
    return resolve_llm_settings(env_path, {}, require_api_key=False).provider


def _default_query_api_url(env_path: Path) -> str | None:
    settings = resolve_llm_settings(env_path, {}, require_api_key=False)
    return settings.base_url.rstrip("/") + "/chat/completions"


def _default_query_model(env_path: Path) -> str | None:
    return resolve_llm_settings(env_path, {}, require_api_key=False).model


def _apply_grounding_model_env_defaults(params: dict[str, Any], env_path: Path) -> None:
    env_values = load_env_values(env_path)
    if not normalize_whitespace(params.get("embedding_model")) and not normalize_whitespace(params.get("embedding_model_path")):
        embedding_model = get_env_value(
            env_values,
            "SCINET_EMBEDDING_MODEL_PATH",
            "SCINET_EMBEDDING_MODEL",
            "GROUNDING_EMBEDDING_MODEL_PATH",
            "GROUNDING_EMBEDDING_MODEL",
        )
        if embedding_model:
            params["embedding_model_path"] = embedding_model

    if not normalize_whitespace(params.get("reranker_model")) and not normalize_whitespace(params.get("reranker_model_path")):
        reranker_model = get_env_value(
            env_values,
            "SCINET_RERANKER_MODEL_PATH",
            "SCINET_RERANKER_MODEL",
            "GROUNDING_RERANKER_MODEL_PATH",
            "GROUNDING_RERANKER_MODEL",
        )
        if reranker_model:
            params["reranker_model_path"] = reranker_model


def _run_grounding(
    *,
    input_payload: dict[str, Any],
    manifest_path: Path,
    params: dict[str, Any],
    env_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, Any], Path]:
    from ..evidence import grounding

    ensure_dir(artifact_dir)
    output_path = artifact_dir / "result.json"
    cli_args: list[str] = []
    idea_text = normalize_whitespace(input_payload.get("idea_text"))
    pdf_path = normalize_whitespace(input_payload.get("pdf_path"))
    if idea_text:
        cli_args.extend(["--idea-text", idea_text])
    elif pdf_path:
        cli_args.extend(["--pdf-path", pdf_path])
    else:
        raise ValueError("Grounding task requires either idea_text or pdf_path.")

    cli_args.extend(["--manifest", str(manifest_path)])
    cli_args.extend(["--env", str(env_path)])
    cli_args.extend(["--top-k-papers", str(int(params["manifest_top_k"]))])
    cli_args.extend(["--dense-candidate-k", str(int(params["dense_candidate_k"]))])
    cli_args.extend(["--final-top-k", str(int(params["grounding_final_top_k"]))])
    cli_args.extend(["--max-paragraphs-per-paper", str(int(params["max_paragraphs_per_paper"]))])
    cli_args.extend(["--query-max-tokens", str(int(params.get("query_max_tokens") or 1000))])
    cli_args.extend(["--output", str(output_path)])

    query_provider = normalize_whitespace(params.get("query_provider")) or normalize_whitespace(_default_query_provider(env_path))
    if query_provider:
        cli_args.extend(["--query-provider", query_provider])
    query_model = normalize_whitespace(params.get("query_model")) or normalize_whitespace(_default_query_model(env_path))
    if query_model:
        cli_args.extend(["--query-model", query_model])
    query_api_url = normalize_whitespace(params.get("query_api_url")) or normalize_whitespace(_default_query_api_url(env_path))
    if query_api_url:
        cli_args.extend(["--query-api-url", query_api_url])

    embedding_model = normalize_whitespace(params.get("embedding_model")) or normalize_whitespace(
        params.get("embedding_model_path")
    )
    if embedding_model:
        cli_args.extend(["--embedding-model", embedding_model])
    reranker_model = normalize_whitespace(params.get("reranker_model")) or normalize_whitespace(
        params.get("reranker_model_path")
    )
    if reranker_model:
        cli_args.extend(["--reranker-model", reranker_model])
    if params.get("enable_grounding_refinement"):
        cli_args.append("--enable-grounding-refinement")
    if params.get("disable_experiment_grounding", True):
        cli_args.append("--disable-experiment-grounding")
    if params.get("grounding_device"):
        cli_args.extend(["--device", str(params["grounding_device"])])
    if params.get("use_env_proxy"):
        cli_args.append("--use-env-proxy")

    args = grounding.build_parser().parse_args(cli_args)
    payload = grounding.run_grounding(args)
    write_json(output_path, payload)
    return payload, output_path


def _resolve_query_text(search_payload: dict[str, Any]) -> str:
    query_text = normalize_whitespace(search_payload.get("query_text"))
    if query_text:
        return query_text
    plan = search_payload.get("plan")
    if isinstance(plan, dict):
        return normalize_whitespace(plan.get("query_text"))
    return ""


def _aggregate_grounding_matches(
    grounding_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    retrieval_results = grounding_payload.get("retrieval", {}).get("results", [])
    flattened: list[dict[str, Any]] = []
    per_paper_map: dict[str, dict[str, Any]] = {}

    for result in retrieval_results:
        if not isinstance(result, dict):
            continue
        query_sentence = normalize_whitespace(result.get("sentence"))
        query_text = normalize_whitespace(result.get("query"))
        for match in result.get("matches", []):
            if not isinstance(match, dict):
                continue
            refined = match.get("refined_grounding", {}) if isinstance(match.get("refined_grounding"), dict) else {}
            paper_title = normalize_whitespace(match.get("paper_title")) or "Unknown"
            entry = {
                "paper_title": paper_title,
                "query_sentence": query_sentence or query_text,
                "focus_aspect": normalize_whitespace(refined.get("focus_aspect")),
                "grounded_passage": normalize_whitespace(refined.get("grounded_passage")),
                "why_this_matches": normalize_whitespace(refined.get("why_this_matches")),
                "coverage_label": normalize_whitespace(refined.get("coverage_label")),
                "text": normalize_whitespace(match.get("text")),
            }
            flattened.append(entry)

            if paper_title not in per_paper_map:
                per_paper_map[paper_title] = {"matches": [], "similar_points": [], "different_points": []}
            per_paper_map[paper_title]["matches"].append(entry)
            for item in refined.get("shared_points", []) if isinstance(refined.get("shared_points"), list) else []:
                per_paper_map[paper_title]["similar_points"].append(normalize_whitespace(item))
            for item in refined.get("different_points", []) if isinstance(refined.get("different_points"), list) else []:
                per_paper_map[paper_title]["different_points"].append(normalize_whitespace(item))

    for paper_title in per_paper_map:
        per_paper_map[paper_title]["similar_points"] = dedupe_preserve_order(per_paper_map[paper_title]["similar_points"])
        per_paper_map[paper_title]["different_points"] = dedupe_preserve_order(per_paper_map[paper_title]["different_points"])

    flattened.sort(key=lambda item: (item.get("coverage_label") == "well_covered", item.get("paper_title") or ""), reverse=True)
    return flattened[:10], per_paper_map


def _build_grounding_summary(top_matches: list[dict[str, Any]], similar_points: list[str], different_points: list[str]) -> str:
    parts: list[str] = []
    if top_matches:
        titles = ", ".join(match["paper_title"] for match in top_matches[:3] if match.get("paper_title"))
        if titles:
            parts.append(f"Top evidence mainly comes from {titles}.")
    if similar_points:
        parts.append(f"Common overlap concentrates on {truncate_text('; '.join(similar_points[:3]), max_chars=220)}.")
    if different_points:
        parts.append(f"Main differences are {truncate_text('; '.join(different_points[:3]), max_chars=220)}.")
    return " ".join(parts).strip()


def _build_grounding_context_for_evaluation(top_matches: list[dict[str, Any]], per_paper: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for match in top_matches[:5]:
        title = match.get("paper_title", "Unknown")
        query = match.get("query_sentence", "")
        grounded = match.get("grounded_passage", "")
        original = match.get("text", "")
        coverage = match.get("coverage_label", "")
        parts.append(
            f"Paper: {title}\nQuery: {query}\nCoverage: {coverage}\nGrounded Passage: {grounded}\nOriginal Text: {original}"
        )
    for paper_title, pdata in list(per_paper.items())[:3]:
        similar = "; ".join(pdata.get("similar_points", [])[:5])
        different = "; ".join(pdata.get("different_points", [])[:5])
        parts.append(f"Paper: {paper_title}\nSimilar Points: {similar}\nDifferent Points: {different}")
    return "\n\n---\n\n".join(parts)


def _run_idea_evaluation(
    *,
    idea_text: str,
    grounding_payload: dict[str, Any],
    env_path: Path,
    params: dict[str, Any],
    artifact_dir: Path,
) -> tuple[dict[str, Any], Path]:
    top_matches, per_paper = _aggregate_grounding_matches(grounding_payload)
    grounding_context = _build_grounding_context_for_evaluation(top_matches, per_paper)
    result = call_llm_json(
        env_path=env_path,
        params=params,
        system_prompt=IDEA_EVALUATION_SYSTEM_PROMPT,
        user_prompt=IDEA_EVALUATION_USER_PROMPT.format(
            idea_text=idea_text,
            grounding_context=grounding_context,
        ),
        artifact_path=artifact_dir / "idea_evaluation.raw.json",
    )
    artifact_path = artifact_dir / "idea_evaluation.raw.json"
    return result, artifact_path


def _pick_representative_papers(
    papers: list[dict[str, Any]],
    representative_titles: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not representative_titles:
        return papers[:limit]

    selected: list[dict[str, Any]] = []
    for item in representative_titles:
        item_title = normalize_whitespace(item.get("title"))
        if not item_title:
            continue
        item_title_key = item_title.casefold()
        matched: dict[str, Any] | None = None

        for paper in papers:
            paper_title = normalize_whitespace(paper.get("title"))
            if paper_title and paper_title.casefold() == item_title_key:
                matched = dict(paper)
                break

        if not matched:
            item_tokens = set(item_title_key.split())
            best_score = 0.0
            best_paper: dict[str, Any] | None = None
            for paper in papers:
                paper_title = normalize_whitespace(paper.get("title")) or ""
                paper_key = paper_title.casefold()
                if not paper_key:
                    continue
                if item_title_key in paper_key or paper_key in item_title_key:
                    score = min(len(item_title_key), len(paper_key)) / max(len(item_title_key), len(paper_key))
                else:
                    paper_tokens = set(paper_key.split())
                    overlap = len(item_tokens & paper_tokens)
                    score = 2 * overlap / (len(item_tokens) + len(paper_tokens)) if (item_tokens or paper_tokens) else 0.0
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_paper = dict(paper)
            matched = best_paper

        if matched:
            matched["why_representative"] = normalize_whitespace(item.get("why_representative"))
            selected.append(matched)
        else:
            selected.append({"title": item.get("title"), "year": item.get("year")})

        if len(selected) >= limit:
            break
    return selected


def _merge_author_infos(author_infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not author_infos:
        return {}
    if len(author_infos) == 1:
        return dict(author_infos[0])
    return {
        "author_id": f"merged::{author_infos[0].get('author_id')}",
        "name": author_infos[0].get("name"),
        "h_index": sum(a.get("h_index") or 0 for a in author_infos),
        "total_works": sum(a.get("total_works") or 0 for a in author_infos),
        "total_citations": sum(a.get("total_citations") or 0 for a in author_infos),
    }


def _dedupe_author_papers(raw_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for paper in raw_papers:
        paper_id = normalize_whitespace(paper.get("paper_id"))
        title = normalize_whitespace(paper.get("title"))
        key = title.casefold()
        if paper_id and paper_id in seen_ids:
            continue
        if key and key in seen_titles:
            continue
        if paper_id:
            seen_ids.add(paper_id)
        if key:
            seen_titles.add(key)
        unique.append(dict(paper))
    return unique


def _select_author_profile_papers(
    raw_papers: list[dict[str, Any]],
    *,
    sample_size: int,
    recent_quota: int,
    top_cited_quota: int,
) -> list[dict[str, Any]]:
    unique_papers = _dedupe_author_papers(raw_papers)
    recent_sorted = sorted(
        unique_papers,
        key=lambda item: (
            int(item.get("year") or 0),
            int(item.get("citations") or 0),
            normalize_whitespace(item.get("title")).casefold(),
        ),
        reverse=True,
    )
    cited_sorted = sorted(
        unique_papers,
        key=lambda item: (
            int(item.get("citations") or 0),
            int(item.get("year") or 0),
            normalize_whitespace(item.get("title")).casefold(),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for pool, quota in ((recent_sorted, recent_quota), (cited_sorted, top_cited_quota)):
        added = 0
        for paper in pool:
            paper_id = normalize_whitespace(paper.get("paper_id")) or normalize_whitespace(paper.get("title")).casefold()
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            selected.append(dict(paper))
            added += 1
            if added >= quota:
                break
    return selected[:sample_size]


def _pick_ideas_references(ideas: list[dict[str, Any]], papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paper_titles = {normalize_whitespace(p.get("title") or "").casefold(): p for p in papers}
    result: list[dict[str, Any]] = []
    for idea in ideas:
        refs = idea.get("key_references") or []
        matched = []
        for ref in refs:
            ref_lower = normalize_whitespace(ref).casefold()
            if ref_lower in paper_titles:
                matched.append(paper_titles[ref_lower])
        result.append(
            {
                "title": normalize_whitespace(idea.get("title") or "Untitled"),
                "description": normalize_whitespace(idea.get("description") or ""),
                "novelty": normalize_whitespace(idea.get("novelty") or ""),
                "significance": normalize_whitespace(idea.get("significance") or ""),
                "related_papers": [_paper_card_from_ranked_item(p, abstract_char_limit=200) for p in matched],
            }
        )
    return result


def _merge_support_payload(
    authors: list[dict[str, Any]],
    support_authors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    support_by_id: dict[str, dict[str, Any]] = {}
    support_by_name: dict[str, dict[str, Any]] = {}
    for author in support_authors:
        author_id = normalize_whitespace(author.get("author_id"))
        author_name = normalize_whitespace(author.get("name"))
        if author_id:
            support_by_id[author_id] = author
        if author_name:
            support_by_name[author_name.casefold()] = author

    merged: list[dict[str, Any]] = []
    for author in authors:
        merged_author = dict(author)
        author_id = normalize_whitespace(author.get("author_id"))
        author_name = normalize_whitespace(author.get("name"))
        support = support_by_id.get(author_id)
        if not support and author_name:
            support = support_by_name.get(author_name.casefold())
        if support:
            for key in (
                "h_index",
                "total_works",
                "total_citations",
                "resolved_by",
                "support_papers",
                "support_error",
            ):
                if key in support:
                    merged_author[key] = support[key]
        else:
            merged_author.setdefault("support_papers", [])
        merged.append(merged_author)
    return merged


def _build_supporting_papers(authors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for author in authors:
        author_name = normalize_whitespace(author.get("name")) or "Unknown"
        for paper in author.get("support_papers", []) if isinstance(author.get("support_papers"), list) else []:
            title = normalize_whitespace(paper.get("title"))
            if not title:
                continue
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            papers.append(
                {
                    "title": title,
                    "year": paper.get("year"),
                    "source": f"support paper for {author_name}",
                    "summary_note": f"similarity={paper.get('similarity_score')}",
                }
            )
    return papers


def execute_grounded_review(request: SciNetRequest, run_dir: Path, client: SciNetApiClient) -> dict[str, Any]:
    params = merge_task_params(request.task_type, request.params)
    _apply_grounding_model_env_defaults(params, request.env_path)
    artifact_root = ensure_dir(run_dir / "artifacts")
    search_artifact_dir = ensure_dir(artifact_root / "search")
    plan = _build_search_plan(
        input_payload=request.input_payload,
        params=params,
        env_path=request.env_path,
        artifact_dir=search_artifact_dir,
    )
    _, search_payload, search_result_path = _run_search_via_api(
        client=client,
        plan=plan,
        params=params,
        env_path=request.env_path,
        artifact_dir=search_artifact_dir,
        api_top_k=int(params["search_api_top_k"]),
        rerank_top_k=int(params["search_final_top_k"]),
    )
    _, manifest_path = _run_manifest(
        search_result_path=search_result_path,
        params=params,
        env_path=request.env_path,
        artifact_dir=artifact_root / "manifest",
    )
    grounding_payload, grounding_path = _run_grounding(
        input_payload=request.input_payload,
        manifest_path=manifest_path,
        params=params,
        env_path=request.env_path,
        artifact_dir=artifact_root / "grounding",
    )

    retrieved_papers = [
        _paper_card_from_ranked_item(item, abstract_char_limit=420)
        for item in search_payload.get("ranking", {}).get("papers", [])[: int(params["search_final_top_k"])]
        if isinstance(item, dict)
    ]
    top_matches, per_paper = _aggregate_grounding_matches(grounding_payload)
    all_sp: list[str] = []
    all_dp: list[str] = []
    for pdata in per_paper.values():
        all_sp.extend(pdata["similar_points"])
        all_dp.extend(pdata["different_points"])

    idea_text = normalize_whitespace(request.input_payload.get("idea_text") or _resolve_query_text(search_payload))
    evaluation_artifact_dir = ensure_dir(artifact_root / "idea_evaluation")
    idea_evaluation, evaluation_artifact_path = _run_idea_evaluation(
        idea_text=idea_text,
        grounding_payload=grounding_payload,
        env_path=request.env_path,
        params=params,
        artifact_dir=evaluation_artifact_dir,
    )

    return {
        "status": "ok",
        "input_summary": {
            "input_mode": "pdf_path" if request.input_payload.get("pdf_path") else "idea_text",
            "idea_text": truncate_text(request.input_payload.get("idea_text"), max_chars=220),
            "pdf_path": normalize_whitespace(request.input_payload.get("pdf_path")),
        },
        "params_effective": params,
        "artifacts": {
            "search_result_path": str(search_result_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "grounding_result_path": str(grounding_path.resolve()),
            "idea_evaluation_path": str(evaluation_artifact_path.resolve()),
        },
        "result": {
            "query_text": _resolve_query_text(search_payload),
            "retrieved_papers": retrieved_papers,
            "grounding_queries": grounding_payload.get("query_generation", {}).get("queries", []),
            "top_matches": top_matches,
            "per_paper": per_paper,
            "summary": _build_grounding_summary(top_matches, all_sp, all_dp),
            "idea_evaluation": idea_evaluation,
        },
    }


def execute_topic_trend_review(request: SciNetRequest, run_dir: Path, client: SciNetApiClient) -> dict[str, Any]:
    params = merge_task_params(request.task_type, request.params)
    topic_text = normalize_whitespace(request.input_payload.get("topic_text") or request.input_payload.get("idea_text"))
    if not topic_text:
        raise ValueError(f"{TASK_TOPIC_TREND_REVIEW} requires topic_text.")

    search_artifact_dir = ensure_dir(run_dir / "artifacts" / "search")
    plan = _build_search_plan(
        input_payload={"idea_text": topic_text},
        params=params,
        env_path=request.env_path,
        artifact_dir=search_artifact_dir,
    )
    _, search_payload, search_result_path = _run_search_via_api(
        client=client,
        plan=plan,
        params=params,
        env_path=request.env_path,
        artifact_dir=search_artifact_dir,
        api_top_k=int(params["search_api_top_k"]),
        rerank_top_k=int(params["final_paper_count_for_summary"]),
    )

    ranked_papers = [
        _paper_card_from_ranked_item(item, abstract_char_limit=int(params["abstract_char_limit"]))
        for item in search_payload.get("ranking", {}).get("papers", [])
        if isinstance(item, dict)
    ]
    papers_by_year = sorted(
        ranked_papers,
        key=lambda item: (item.get("year") is None, item.get("year") or 9999, item.get("title") or ""),
    )
    llm_artifact_path = run_dir / "artifacts" / "trend_review.raw.json"
    trend_summary = call_llm_json(
        env_path=request.env_path,
        params=params,
        system_prompt="You analyze academic topic evolution and only return valid JSON objects.",
        user_prompt=build_trend_prompt(papers_by_year, int(params["abstract_char_limit"])),
        artifact_path=llm_artifact_path,
    )
    representative_papers = _pick_representative_papers(
        papers_by_year,
        trend_summary.get("representative_papers", []) if isinstance(trend_summary.get("representative_papers"), list) else [],
    )
    return {
        "status": "ok",
        "input_summary": {"topic_text": truncate_text(topic_text, max_chars=220)},
        "params_effective": params,
        "artifacts": {
            "search_result_path": str(search_result_path.resolve()),
            "trend_review_raw_path": str(llm_artifact_path.resolve()),
        },
        "result": {
            "query_text": topic_text,
            "papers_by_year": papers_by_year,
            "trend_summary": trend_summary,
            "representative_papers": representative_papers,
        },
    }


def execute_related_authors(request: SciNetRequest, run_dir: Path, client: SciNetApiClient) -> dict[str, Any]:
    params = merge_task_params(request.task_type, request.params)
    artifacts = ensure_dir(run_dir / "artifacts")
    plan_artifact_dir = ensure_dir(artifacts / "query_plan")
    plan = _build_search_plan(
        input_payload=request.input_payload,
        params=params,
        env_path=request.env_path,
        artifact_dir=plan_artifact_dir,
    )

    authors_response = client.authors_related(
        plan=plan,
        options={
            "top_k": int(params["author_top_k"]),
            **{
                key: params[key]
                for key in ("target_field", "after", "before")
                if params.get(key) is not None and normalize_whitespace(params.get(key))
            },
        },
    )
    write_json(artifacts / "authors_related.api_response.json", authors_response)
    authors_result = authors_response.get("result", {}) if isinstance(authors_response.get("result"), dict) else {}
    authors = [dict(item) for item in authors_result.get("authors", []) if isinstance(item, dict)]
    authors = authors[: int(params["author_top_k"])]
    query_text = normalize_whitespace(authors_result.get("resolved_query_text")) or normalize_whitespace(plan.get("query_text"))

    support_response_path: Path | None = None
    if authors and query_text:
        support_candidates = authors[: int(params["enrich_author_support_count"])]
        support_response = client.authors_support_papers(
            query_text=query_text,
            authors=[
                {
                    "author_id": author.get("author_id"),
                    "name": author.get("name"),
                    "score": author.get("score"),
                    "rank": author.get("rank"),
                }
                for author in support_candidates
            ],
            options={
                "top_k_per_author": int(params["author_support_top_k"]),
                "fetch_author_stats": bool(params.get("fetch_author_stats", True)),
                "author_search_fallback": params.get("author_search_fallback") or "id_then_name",
            },
        )
        support_response_path = artifacts / "authors_support_papers.api_response.json"
        write_json(support_response_path, support_response)
        support_result = support_response.get("result", {}) if isinstance(support_response.get("result"), dict) else {}
        authors = _merge_support_payload(
            authors,
            [dict(item) for item in support_result.get("authors", []) if isinstance(item, dict)],
        )

    supporting_papers = _build_supporting_papers(authors)
    summary = (
        f"Returned {len(authors)} related authors using SciScholar API author recall."
        if authors
        else "No related authors were returned."
    )
    artifact_payload = {
        "query_plan_path": str((plan_artifact_dir / "plan.json").resolve()),
    }
    if support_response_path is not None:
        artifact_payload["authors_support_papers_response_path"] = str(support_response_path.resolve())

    return {
        "status": "ok",
        "input_summary": {
            "input_mode": "pdf_path" if request.input_payload.get("pdf_path") else "idea_text",
            "idea_text": truncate_text(request.input_payload.get("idea_text"), max_chars=220),
            "pdf_path": normalize_whitespace(request.input_payload.get("pdf_path")),
        },
        "params_effective": params,
        "artifacts": artifact_payload,
        "result": {
            "query_text": query_text,
            "authors": authors,
            "author_count": len(authors),
            "supporting_papers": supporting_papers,
            "summary": summary,
        },
    }


def execute_author_profile(request: SciNetRequest, run_dir: Path, client: SciNetApiClient) -> dict[str, Any]:
    params = merge_task_params(request.task_type, request.params)
    author_name = normalize_whitespace(request.input_payload.get("author_name"))
    if not author_name:
        raise ValueError(f"{TASK_AUTHOR_PROFILE} requires author_name.")

    response = client.authors_papers(
        identifier=author_name,
        search_by="name",
        options={
            "limit": None,
            "merge_same_name_authors": bool(params.get("merge_same_name_authors", True)),
            "dedupe_papers": bool(params.get("dedupe_papers", True)),
            "include_abstract": bool(params.get("include_abstract", True)),
            "include_embeddings": False,
        },
    )
    artifact_root = ensure_dir(run_dir / "artifacts" / "author_profile")
    api_response_path = artifact_root / "authors_papers.api_response.json"
    write_json(api_response_path, response)
    result_payload = response.get("result", {}) if isinstance(response.get("result"), dict) else {}
    author_infos = [dict(item) for item in result_payload.get("matched_authors", []) if isinstance(item, dict)]
    raw_papers = [dict(item) for item in result_payload.get("papers", []) if isinstance(item, dict)]
    if not author_infos or not raw_papers:
        raise RuntimeError(f"No author papers found for {author_name!r}")

    raw_papers = _dedupe_author_papers(raw_papers)
    merged_stats = result_payload.get("merged_author_stats") if isinstance(result_payload.get("merged_author_stats"), dict) else _merge_author_infos(author_infos)

    author_metas_path = artifact_root / "author_metas.json"
    write_json(author_metas_path, {"authors": author_infos})
    selected_papers = _select_author_profile_papers(
        raw_papers,
        sample_size=int(params["author_paper_sample_size"]),
        recent_quota=int(params["recent_paper_quota"]),
        top_cited_quota=int(params["top_cited_quota"]),
    )
    raw_papers_path = artifact_root / "selected_papers.json"
    write_json(raw_papers_path, {"author_name": author_name, "papers": selected_papers})
    llm_raw_path = artifact_root / "author_profile.raw.json"
    summary_payload = call_llm_json(
        env_path=request.env_path,
        params=params,
        system_prompt="You summarize an author's research trajectory and only return valid JSON objects.",
        user_prompt=build_author_profile_prompt(author_name, selected_papers),
        artifact_path=llm_raw_path,
    )
    representative_papers = _pick_representative_papers(
        selected_papers,
        summary_payload.get("representative_papers", []) if isinstance(summary_payload.get("representative_papers"), list) else [],
        limit=int(params["representative_paper_top_k"]),
    )
    summary_payload["representative_papers"] = representative_papers
    summary_json_path = artifact_root / "author_profile.summary.json"
    write_json(summary_json_path, summary_payload)
    return {
        "status": "ok",
        "input_summary": {"author_name": author_name},
        "params_effective": params,
        "artifacts": {
            "authors_papers_response_path": str(api_response_path.resolve()),
            "author_metas_path": str(author_metas_path.resolve()),
            "selected_papers_path": str(raw_papers_path.resolve()),
            "author_profile_raw_path": str(llm_raw_path.resolve()),
            "author_profile_summary_path": str(summary_json_path.resolve()),
        },
        "result": {
            **summary_payload,
            "author_stats": merged_stats,
        },
    }


def execute_idea_generation(request: SciNetRequest, run_dir: Path, client: SciNetApiClient) -> dict[str, Any]:
    params = merge_task_params(request.task_type, request.params)
    topic_text = normalize_whitespace(request.input_payload.get("topic_text") or request.input_payload.get("idea_text"))
    if not topic_text:
        raise ValueError(f"{TASK_IDEA_GENERATION} requires topic_text or idea_text.")

    search_artifact_dir = ensure_dir(run_dir / "artifacts" / "search")
    plan = _build_search_plan(
        input_payload={"idea_text": topic_text},
        params=params,
        env_path=request.env_path,
        artifact_dir=search_artifact_dir,
    )
    _, search_payload, search_result_path = _run_search_via_api(
        client=client,
        plan=plan,
        params=params,
        env_path=request.env_path,
        artifact_dir=search_artifact_dir,
        api_top_k=int(params["search_api_top_k"]),
        rerank_top_k=int(params["final_paper_count_for_summary"]),
    )

    ranked_papers = [
        _paper_card_from_ranked_item(item, abstract_char_limit=int(params["abstract_char_limit"]))
        for item in search_payload.get("ranking", {}).get("papers", [])
        if isinstance(item, dict)
    ]
    llm_artifact_path = run_dir / "artifacts" / "idea_generation.raw.json"
    raw_ideas = call_llm_json(
        env_path=request.env_path,
        params=params,
        system_prompt="You are an expert research idea generator and only return valid JSON.",
        user_prompt=build_idea_generation_prompt(
            ranked_papers,
            idea_count=int(params["idea_count"]),
            abstract_char_limit=int(params["abstract_char_limit"]),
        ),
        artifact_path=llm_artifact_path,
    )
    ideas_list = raw_ideas.get("ideas", []) if isinstance(raw_ideas, dict) else []
    ideas = _pick_ideas_references(ideas_list, ranked_papers)
    return {
        "status": "ok",
        "input_summary": {"topic_text": truncate_text(topic_text, max_chars=220)},
        "params_effective": params,
        "artifacts": {
            "search_result_path": str(search_result_path.resolve()),
            "idea_generation_raw_path": str(llm_artifact_path.resolve()),
        },
        "result": {
            "query_text": topic_text,
            "ideas": ideas,
        },
    }


def execute_request(request: SciNetRequest, run_dir: Path) -> dict[str, Any]:
    settings = load_scinet_api_settings(request.env_path, request.params)
    with SciNetApiClient(settings) as client:
        if request.task_type == TASK_GROUNDED_REVIEW:
            response = execute_grounded_review(request, run_dir, client)
        elif request.task_type == TASK_TOPIC_TREND_REVIEW:
            response = execute_topic_trend_review(request, run_dir, client)
        elif request.task_type == TASK_RELATED_AUTHORS:
            response = execute_related_authors(request, run_dir, client)
        elif request.task_type == TASK_AUTHOR_PROFILE:
            response = execute_author_profile(request, run_dir, client)
        elif request.task_type == TASK_IDEA_GENERATION:
            response = execute_idea_generation(request, run_dir, client)
        else:
            raise ValueError(f"Unsupported task_type: {request.task_type}")

    response["task_type"] = request.task_type
    response["run_dir"] = str(run_dir.resolve())
    return response
