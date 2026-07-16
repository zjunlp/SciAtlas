"""Step 5 — Gate LLM decides the inspiration radius plan."""
from __future__ import annotations

import json
from typing import Any

from ..clients.llm_client import LLMClient
from ..config import PipelineConfig
from ..prompts import INSPIRATION_RADIUS_GATE
from ..utils import get_logger

log = get_logger("sciatlas.step5")


def _clamp_int(value: Any, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    out = max(minimum, out)
    if maximum is not None:
        out = min(maximum, out)
    return out


def run(
    llm: LLMClient,
    rss: dict[str, Any],
    trend: str,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    log.info("Step 5: inspiration radius gate")
    prompt = INSPIRATION_RADIUS_GATE.format(
        rss_json=json.dumps(rss, ensure_ascii=False, indent=2),
        trend=trend,
        same_field_top_k=cfg.inspiration_top_k_same_field,
        num_cross_domains=cfg.num_cross_domains,
        cross_domain_top_k_per_domain=cfg.inspiration_top_k_per_domain,
    )
    try:
        data = llm.chat_json(prompt, temperature=0.1)
    except Exception as exc:
        log.warning("  inspiration radius gate failed, falling back to defaults: %s", exc)
        data = {}

    use_same_field = bool(data.get("use_same_field", True))
    use_cross_domain = bool(data.get("use_cross_domain", True))
    same_field_top_k = _clamp_int(
        data.get("same_field_top_k"),
        cfg.inspiration_top_k_same_field,
        minimum=0,
        maximum=max(cfg.inspiration_top_k_same_field * 2, cfg.inspiration_top_k_same_field),
    )
    num_cross_domains = _clamp_int(
        data.get("num_cross_domains"),
        cfg.num_cross_domains,
        minimum=0,
        maximum=max(cfg.num_cross_domains * 2, cfg.num_cross_domains),
    )
    cross_domain_top_k_per_domain = _clamp_int(
        data.get("cross_domain_top_k_per_domain"),
        cfg.inspiration_top_k_per_domain,
        minimum=0,
        maximum=max(cfg.inspiration_top_k_per_domain * 2, cfg.inspiration_top_k_per_domain),
    )
    if not use_same_field:
        same_field_top_k = 0
    if not use_cross_domain:
        num_cross_domains = 0
        cross_domain_top_k_per_domain = 0
    if not use_same_field and not use_cross_domain:
        use_same_field = True
        same_field_top_k = cfg.inspiration_top_k_same_field

    preferred_radii: list[str] = []
    if use_same_field:
        preferred_radii.append("R1")
    if use_cross_domain:
        preferred_radii.append("R2")
    plan = {
        "use_same_field": use_same_field,
        "use_cross_domain": use_cross_domain,
        "preferred_radii": preferred_radii,
        "same_field_top_k": same_field_top_k,
        "num_cross_domains": num_cross_domains,
        "cross_domain_top_k_per_domain": cross_domain_top_k_per_domain,
        "rationale": str(data.get("rationale", "") or "default retrieval plan"),
    }
    log.info("  gate plan: %s", plan)
    return plan
