from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher

TITLE_EXACT_NORMALIZE_RE = re.compile(r"[^a-z]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "with",
    "we",
    "our",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title_exact(text: str) -> str:
    return TITLE_EXACT_NORMALIZE_RE.sub("", clean_text(text).lower())


def title_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    seq = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    token_overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return 0.65 * seq + 0.35 * token_overlap


def combine_embedding_scores(sim_title: float | None, sim_abstract: float | None) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    if sim_title is not None:
        weighted_sum += 0.40 * sim_title
        total_weight += 0.40
    if sim_abstract is not None:
        weighted_sum += 0.60 * sim_abstract
        total_weight += 0.60
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_importance(cited_by_count: int, citation_p95: float) -> float:
    cited = max(0, cited_by_count)
    denom = math.log1p(max(1.0, citation_p95))
    if denom <= 0:
        return 0.0
    return min(1.0, math.log1p(cited) / denom)


def resolve_importance(cited_by_count: int, citation_p95: float, uniform_importance: bool = False) -> float:
    if uniform_importance:
        return 1.0
    return compute_importance(cited_by_count, citation_p95)


def min_max_normalize(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return {key: 1.0 if value > 0 else 0.0 for key, value in score_map.items()}
    return {key: (value - lo) / (hi - lo) for key, value in score_map.items()}


def compact_abstract(text: str | None, limit: int = 320) -> str | None:
    if not text:
        return None
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
