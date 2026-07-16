"""Step 7 — naive novelty checking for generated ideas."""
from __future__ import annotations

import json

from ..clients.llm_client import LLMClient
from ..models import Idea
from ..prompts import NOVELTY_CHECK
from ..utils import get_logger

log = get_logger("sciatlas.step7")


def run(llm: LLMClient, ideas: list[Idea]) -> list[Idea]:
    """Annotate each idea with a novelty level and improvement suggestions."""
    log.info("Step 7: novelty checking")
    for idea in ideas:
        prompt = NOVELTY_CHECK.format(
            idea_json=json.dumps(
                {
                    "title": idea.title,
                    "motivation": idea.motivation,
                    "methods": idea.methods,
                },
                ensure_ascii=False,
                indent=2,
            ),
            reference_list=json.dumps(idea.key_references, ensure_ascii=False, indent=2),
        )
        try:
            data = llm.chat_json(prompt, temperature=0.1)
        except Exception as exc:
            log.warning("  novelty check failed for %s: %s", idea.title, exc)
            continue
        idea.novelty_level = str(data.get("novelty_level", "")).strip() or None
        idea.novelty_justification = (
            str(data.get("justification", "")).strip() or None
        )
        idea.improvement_suggestions = list(
            data.get("improvement_suggestions", []) or []
        )
    return ideas
