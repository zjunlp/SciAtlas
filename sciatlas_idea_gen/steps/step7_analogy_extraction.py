"""Step 7 — analogy / mechanism extraction over retrieved inspirations."""
from __future__ import annotations

from ..clients.llm_client import LLMClient
from ..models import Inspiration
from ..prompts import COMBINATION_CHECK
from ..utils import get_logger

log = get_logger("sciatlas.step7")


def run(llm: LLMClient, problem_text: str, candidates: list[Inspiration]) -> list[Inspiration]:
    log.info("Step 7: analogy / mechanism extraction")
    inspirations: list[Inspiration] = []
    for candidate in candidates:
        prompt = COMBINATION_CHECK.format(
            problem_text=problem_text,
            title=candidate.paper_title,
            abstract=candidate.paper_abstract[:2000],
        )
        try:
            data = llm.chat_json(prompt, temperature=0.2)
        except Exception as exc:
            log.warning("  analogy extraction failed for %s: %s", candidate.paper_title, exc)
            continue
        # The LLM sometimes returns a JSON array instead of an object; pick the
        # first dict element, and skip the candidate if there's no usable dict
        # (rather than crashing the whole generation on data.get()).
        if isinstance(data, list):
            data = next((d for d in data if isinstance(d, dict)), None)
        if not isinstance(data, dict):
            log.warning("  analogy extraction returned non-dict for %s; skipping", candidate.paper_title)
            continue
        candidate.is_combination_plausible = bool(data.get("is_combination_plausible", False))
        candidate.combination_points = list(data.get("combination_points", []) or [])
        candidate.combination_plan = str(data.get("combination_plan", "") or "")
        candidate.justification = str(data.get("justification", "") or "")
        if candidate.is_combination_plausible:
            inspirations.append(candidate)
    log.info("  kept %d plausible inspirations out of %d candidates", len(inspirations), len(candidates))
    return inspirations
