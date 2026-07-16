"""Step 8 — Gate LLM filters and ranks inspirations."""
from __future__ import annotations

import json
from typing import Any

from ..clients.llm_client import LLMClient
from ..models import Inspiration
from ..prompts import BEST_INSPIRATION_SELECTION
from ..utils import get_logger

log = get_logger("sciatlas.step8")


def _serialize_inspirations_for_selection(inspirations: list[Inspiration]) -> str:
    data = [
        {
            "candidate_id": item.candidate_id or index,
            "domain": item.domain,
            "radius": item.radius,
            "paper_title": item.paper_title,
            "combination_points": item.combination_points,
            "combination_plan": item.combination_plan,
            "justification": item.justification,
        }
        for index, item in enumerate(inspirations, start=1)
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def run(
    llm: LLMClient,
    rss: dict[str, Any],
    trend: str,
    inspirations: list[Inspiration],
) -> list[Inspiration]:
    log.info("Step 8: inspiration selection")
    if len(inspirations) <= 1:
        return inspirations
    prompt = BEST_INSPIRATION_SELECTION.format(
        rss_json=json.dumps(rss, ensure_ascii=False, indent=2),
        trend=trend,
        inspiration_list=_serialize_inspirations_for_selection(inspirations),
    )
    try:
        data = llm.chat_json(prompt, temperature=0.1)
    except Exception as exc:
        log.warning("  inspiration selection failed: %s", exc)
        return inspirations[:1]
    # Match back against the ids exposed to the LLM (`candidate_id or 1-based index`).
    id_map: dict[Any, Inspiration] = {}
    for index, item in enumerate(inspirations, start=1):
        id_map[item.candidate_id or index] = item
    selected_ids: list[Any] = []
    if isinstance(data, dict):
        raw = data.get("selected_candidate_ids")
        if isinstance(raw, list):
            selected_ids = list(raw)
        elif data.get("selected_candidate_id") is not None:  # backward-compat
            selected_ids = [data.get("selected_candidate_id")]
    chosen: list[Inspiration] = []
    seen: set[int] = set()
    for sid in selected_ids:
        item = id_map.get(sid)
        if item is not None and id(item) not in seen:
            seen.add(id(item))
            chosen.append(item)
    if chosen:
        log.info("  selected %d inspiration(s): %s", len(chosen),
                 [it.domain for it in chosen])
        return chosen
    log.warning("  invalid inspiration selection: %r", data)
    return inspirations[:1]
