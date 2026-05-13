from __future__ import annotations

import hashlib
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..core.common import extract_json_object, normalize_whitespace
from ..llm.client import build_llm_client


LLM_SYSTEM_PROMPT = (
    "You are a rigorous academic literature relevance judge. "
    "Score each candidate paper independently and return strict JSON only."
)


def build_scoring_batches(
    *,
    paper_count: int,
    batch_size: int,
    paper_coverage: int,
    seed: int,
) -> list[dict[str, Any]]:
    if paper_count <= 0:
        return []
    if paper_count <= batch_size:
        return [{"batch_id": "batch-001", "round": 1, "paper_indices": list(range(paper_count))}]

    batch_count_per_round = math.ceil(paper_count / batch_size)
    rng = random.Random(seed)
    batches: list[dict[str, Any]] = []
    for round_index in range(paper_coverage):
        shuffled = list(range(paper_count))
        rng.shuffle(shuffled)
        round_batches: list[list[int]] = [[] for _ in range(batch_count_per_round)]
        for position, paper_index in enumerate(shuffled):
            round_batches[position % batch_count_per_round].append(paper_index)
        for paper_indices in round_batches:
            batches.append(
                {
                    "batch_id": f"batch-{len(batches) + 1:03d}",
                    "round": round_index + 1,
                    "paper_indices": paper_indices,
                }
            )
    return batches


def build_relevance_prompt(plan: dict[str, Any], batch_papers: list[dict[str, Any]]) -> str:
    query_lines = []
    if plan.get("source_type") == "pdf":
        query_lines.append("Query type: paper_abstract")
        if normalize_whitespace(plan.get("source_title")):
            query_lines.append(f"Query paper title: {normalize_whitespace(plan.get('source_title'))}")
        query_lines.append(f"Query paper abstract:\n{normalize_whitespace(plan.get('query_text'))}")
    else:
        query_lines.append("Query type: idea_text")
        query_lines.append(f"Query text:\n{normalize_whitespace(plan.get('query_text'))}")

    paper_blocks = []
    for slot, paper in enumerate(batch_papers, start=1):
        paper_blocks.append(
            f"[{slot}] Title: {normalize_whitespace(paper.get('title'))}\nAbstract: {normalize_whitespace(paper.get('abstract')) or '[missing abstract]'}"
        )

    return "\n\n".join(
        [
            "Evaluate how substantively relevant each candidate paper is to the query.",
            "",
            "Scoring target:",
            "- Judge each paper independently, not comparatively.",
            "- Focus on overlap in the core research problem, evaluation target, inputs/outputs, and methodological purpose.",
            "- A paper can score high even if the method differs, as long as it directly addresses the same problem.",
            "- Penalize papers that only share broad themes such as LLMs, scientific discovery, benchmarks, or evaluation.",
            "- Ignore citation counts, venue prestige, popularity, and writing polish.",
            "",
            "Rubric:",
            "- 9-10: Extremely close neighbor; essentially the same problem or an immediately adjacent paper you would definitely cite.",
            "- 7-8: Directly relevant; strong task or evaluation overlap.",
            "- 5-6: Meaningful partial overlap, but not the same core problem.",
            "- 3-4: Weak topical relation only.",
            "- 0-2: Effectively irrelevant.",
            "",
            'Return ONLY JSON in this format: {"papers":[{"paper_index":1,"score":8.7,"reason":"short reason"}]}',
            "",
            "Requirements for output:",
            "- Include every paper exactly once.",
            "- score must be a number in [0, 10].",
            "- reason must be concise and mention the main match or mismatch.",
            "",
            "\n".join(query_lines),
            "Candidate papers:\n" + "\n\n".join(paper_blocks),
        ]
    )


class PaperBatchScorer:
    def __init__(self, *, env_path: Path, params: dict[str, Any]) -> None:
        self.env_path = env_path
        self.params = params
        self.client = build_llm_client(env_path, params)
        self.model_name = normalize_whitespace(getattr(self.client.settings, "model", "")) or "unknown"

    def score_batch(self, *, plan: dict[str, Any], batch_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = build_relevance_prompt(plan, batch_papers)
        content = self.client.chat_text(
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=float(self.params.get("rerank_temperature") or 0.1),
            response_format={"type": "json_object"},
            max_tokens=int(self.params.get("rerank_max_tokens") or 900),
        )
        parsed = extract_json_object(content)
        return parse_batch_scores(parsed, expected_size=len(batch_papers), fallback_content=content)


def parse_batch_scores(payload: dict[str, Any], *, expected_size: int, fallback_content: str = "") -> list[dict[str, Any]]:
    values = payload.get("papers") or payload.get("scores")
    if not isinstance(values, list):
        raise ValueError(f"Missing paper list in batch response: {fallback_content or payload!r}")
    parsed: dict[int, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        paper_index = item.get("paper_index", item.get("index"))
        try:
            slot = int(paper_index)
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if slot < 1 or slot > expected_size:
            continue
        parsed[slot] = {
            "paper_index": slot,
            "score": max(0.0, min(10.0, score)),
            "reason": normalize_whitespace(item.get("reason")),
        }
    missing_slots = [slot for slot in range(1, expected_size + 1) if slot not in parsed]
    if missing_slots:
        raise ValueError(f"Batch response missing paper indices: {missing_slots}")
    return [parsed[slot] for slot in range(1, expected_size + 1)]


def compute_score_std(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return variance ** 0.5


def rerank_search_payload(
    *,
    search_payload: dict[str, Any],
    plan: dict[str, Any],
    env_path: Path,
    params: dict[str, Any],
    final_top_k: int | None = None,
) -> dict[str, Any]:
    ranking = search_payload.get("ranking", {}) if isinstance(search_payload.get("ranking"), dict) else {}
    papers = [dict(item) for item in ranking.get("papers", []) if isinstance(item, dict)]
    if not papers:
        search_payload["ranking"] = {
            "status": "skipped",
            "reason": "no_papers_to_rerank",
            "papers": [],
        }
        return search_payload

    scorer = PaperBatchScorer(env_path=env_path, params=params)
    seed_input = f"{normalize_whitespace(plan.get('source_type'))}::{normalize_whitespace(plan.get('source_title'))}::{normalize_whitespace(plan.get('query_text'))}"
    seed = int(hashlib.sha1(seed_input.encode("utf-8")).hexdigest()[:8], 16)
    batch_size = max(2, int(params.get("rerank_batch_size") or 4))
    paper_coverage = max(1, int(params.get("rerank_paper_coverage") or 2))
    batches = build_scoring_batches(
        paper_count=len(papers),
        batch_size=batch_size,
        paper_coverage=paper_coverage,
        seed=seed,
    )

    batch_results: list[dict[str, Any]] = []
    batch_errors: list[dict[str, Any]] = []
    max_parallel = max(1, min(int(params.get("rerank_max_parallel") or 8), len(batches)))
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {
            executor.submit(
                scorer.score_batch,
                plan=plan,
                batch_papers=[papers[index] for index in batch["paper_indices"]],
            ): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            batch = future_map[future]
            try:
                scores = future.result()
                batch_results.append(
                    {
                        "batch_id": batch["batch_id"],
                        "round": batch["round"],
                        "paper_indices": list(batch["paper_indices"]),
                        "scores": scores,
                    }
                )
            except Exception as exc:
                batch_errors.append(
                    {
                        "batch_id": batch["batch_id"],
                        "round": batch["round"],
                        "paper_indices": list(batch["paper_indices"]),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                )

    batch_results.sort(key=lambda item: item["batch_id"])
    paper_keys = [normalize_whitespace(item.get("paper_id")) or f"row-{index}" for index, item in enumerate(papers)]
    score_details_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in paper_keys}
    for batch_result in batch_results:
        batch_papers = [papers[index] for index in batch_result["paper_indices"]]
        batch_keys = [paper_keys[index] for index in batch_result["paper_indices"]]
        for score_item, paper_key in zip(batch_result["scores"], batch_keys):
            score_details_by_key[paper_key].append(
                {
                    "batch_id": batch_result["batch_id"],
                    "round": batch_result["round"],
                    "score": round(float(score_item["score"]), 4),
                    "reason": score_item["reason"],
                }
            )

    reranked: list[dict[str, Any]] = []
    unscored_keys: list[str] = []
    for index, paper in enumerate(papers):
        paper_key = paper_keys[index]
        details = score_details_by_key.get(paper_key, [])
        if not details:
            unscored_keys.append(paper_key)
            continue
        scores = [detail["score"] for detail in details]
        item = dict(paper)
        item["kg_score"] = item.get("kg_score", item.get("score"))
        item["kg_rank"] = item.get("rank")
        item["llm_score"] = round(sum(scores) / len(scores), 4)
        item["score_count"] = len(scores)
        item["score_std"] = round(compute_score_std(scores), 4)
        item["reasons"] = [detail["reason"] for detail in details]
        item["score_details"] = details
        reranked.append(item)

    reranked.sort(
        key=lambda item: (
            -float(item.get("llm_score") or 0.0),
            -int(item.get("score_count") or 0),
            -int(item.get("citation_count") or item.get("cited_by_count") or 0),
            normalize_whitespace(item.get("title")).casefold(),
        )
    )
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank

    returned = reranked[:final_top_k] if final_top_k is not None else reranked
    search_payload["query_text"] = normalize_whitespace(plan.get("query_text"))
    search_payload["plan"] = dict(plan)
    search_payload["ranking"] = {
        "status": "ok" if reranked and not batch_errors and not unscored_keys else ("partial_error" if reranked else "error"),
        "strategy": "client_llm_batch_mean",
        "model": scorer.model_name,
        "batch_size": batch_size,
        "paper_coverage": paper_coverage if len(papers) > batch_size else 1,
        "batch_count": len(batches),
        "full_paper_count": len(reranked),
        "returned_paper_count": len(returned),
        "papers": returned,
        "all_papers": reranked,
        "batches": batch_results,
        "batch_errors": batch_errors,
        "unscored_group_ids": unscored_keys,
    }
    return search_payload
