#!/usr/bin/env python3
"""Declarative spec migration for the method-timeline case studies.

Given a target-outline yaml describing what method families you want and where
each lane should come from, this script produces a renderable spec yaml (same
shape as ``method_timeline_spec*.yaml``) with paper assignments resolved
automatically.

Per-lane source modes
---------------------

``inherit``
    Copy anchors verbatim from ``current_spec[from_lane]``. No LLM, no S2.
    Use when a lane is unchanged across migrations.

``reclassify``
    Pull paper_ids from one or more ``from_clusters`` (cluster ids inside
    ``clusters_recovered.json``) and LLM-classify each paper into the best
    matching reclassify-mode lane (or "neither"). Use for cluster splits /
    re-bucketing of corpus papers.

``discover``
    A brand-new family. Up to three stages run independently:

    - ``corpus_rescan`` — substring prefilter on ``family_keywords`` over all
      paper_cards, then LLM family-fit binary classifier.
    - ``named_fetch`` — for each ``seed_anchor.title``, hit the S2
      ``paper/search/match`` endpoint; matched papers become in_corpus seeds
      (with newly assigned ``corpus_paper_id`` if not already in corpus).
    - ``recommend_expansion`` — for each S2 paper id found in ``named_fetch``,
      hit ``recommendations/v1/papers/forpaper/{id}`` and LLM-classify each.

``manual``
    Pass anchors through verbatim from outline. Useful for hand-curated lanes
    or blank/divider rows.

Idempotency
-----------

The script is re-runnable. LLM batches and S2 calls are cached under
``<work_dir>/llm_cache/`` and ``<work_dir>/s2_cache/`` keyed by a hash of the
prompt or request payload plus the model id. Pass ``--resume`` to reuse
caches across runs.

CLI surface
-----------

    python3 migrate_method_spec.py \
        --outline   <path/to/target_outline.yaml> \
        --out-spec  <path/to/output_spec.yaml> \
        [--work-dir <dir>] [--resume] [--dry-run] \
        [--skip-stages STAGE [STAGE ...]] [--quiet]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LR_SEARCH_DIR = Path(__file__).resolve().parent
if str(LR_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(LR_SEARCH_DIR))

S2API_DIR = LR_SEARCH_DIR.parent.parent / "s2api"
if str(S2API_DIR) not in sys.path:
    sys.path.insert(0, str(S2API_DIR))

# Reused from existing lr_search infrastructure.
from literature_review_search import (  # noqa: E402
    DEFAULT_DMX_API_URL,
    DEFAULT_DMX_MODEL,
    DEFAULT_ENV_PATH,
    DmxJsonClient,
    SYSTEM_PROMPT,
    normalize_whitespace,
    parse_json_object,
    truncate_text,
)


# ============================================================================
# Constants & dataclasses
# ============================================================================


VALID_MODES = {"inherit", "reclassify", "discover", "manual"}
VALID_DISCOVER_STAGES = ("corpus_rescan", "named_fetch", "recommend_expansion")

# Default knobs (overridable by outline.defaults and CLI flags).
DEFAULT_LLM_CONFIDENCE = 0.65
DEFAULT_MAX_AUTO_EXTRAS = 3
DEFAULT_LLM_BATCH_SIZE = 25
DEFAULT_LLM_MAX_PARALLEL = 8
DEFAULT_LLM_TIMEOUT = 120
DEFAULT_LLM_MAX_TOKENS = 4000
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_ABSTRACT_CHAR_LIMIT = 750
DEFAULT_RECOMMEND_PER_SEED = 8

# Anchor fields we know about; everything else is preserved as-is when present.
ANCHOR_RENDER_FIELDS = (
    "label",
    "full_title",
    "year",
    "first_author",
    "citation_count",
    "venue",
    "in_corpus",
    "corpus_paper_id",
    "notes",
)


@dataclass
class OutlineLane:
    """One target lane parsed from the outline yaml."""

    id: str
    name: str
    color: str
    description: str = ""
    seed_anchors: list[dict[str, Any]] = field(default_factory=list)
    family_keywords: list[str] = field(default_factory=list)
    raw_anchors: list[dict[str, Any]] = field(default_factory=list)  # manual mode only
    source_mode: str = "manual"
    source_from_lane: str | None = None  # inherit
    source_from_clusters: list[str] = field(default_factory=list)  # reclassify
    source_stages: list[str] = field(default_factory=list)  # discover
    recommend_per_seed: int = DEFAULT_RECOMMEND_PER_SEED


@dataclass
class Outline:
    topic: dict[str, Any]
    artifacts: dict[str, Any]
    defaults: dict[str, Any]
    clusters: list[OutlineLane]


@dataclass
class Corpus:
    paper_cards: list[dict[str, Any]]
    cards_by_id: dict[str, dict[str, Any]]
    cluster_members: dict[str, list[str]]  # cluster_id -> [paper_id]
    cluster_meta: dict[str, dict[str, Any]]  # cluster_id -> cluster object
    paper_assignment_log: dict[str, dict[str, Any]]


@dataclass
class ResolvedAnchor:
    """An anchor row destined for the output spec."""

    label: str
    full_title: str
    year: int | None
    first_author: str = ""
    citation_count: int | None = None
    venue: str = ""
    in_corpus: bool = False
    corpus_paper_id: str | None = None
    notes: str = ""
    # Internal accounting (not emitted unless useful):
    is_seed: bool = True
    source_tag: str = ""  # e.g. "inherit", "manual", "auto:corpus_rescan", "auto:named_fetch"

    def to_yaml_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label}
        if self.full_title:
            out["full_title"] = self.full_title
        if self.first_author:
            out["first_author"] = self.first_author
        if self.year is not None:
            out["year"] = self.year
        if self.citation_count is not None:
            out["citation_count"] = self.citation_count
        if self.venue:
            out["venue"] = self.venue
        out["in_corpus"] = bool(self.in_corpus)
        if self.corpus_paper_id:
            out["corpus_paper_id"] = self.corpus_paper_id
        if self.notes:
            out["notes"] = self.notes
        return out


@dataclass
class ResolvedLane:
    id: str
    name: str
    color: str
    anchors: list[ResolvedAnchor]
    source_summary: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# IO helpers
# ============================================================================


def log(progress: bool, msg: str) -> None:
    if progress:
        print(msg, flush=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML required; pip install pyyaml") from exc
    payload = yaml.safe_load(read_text(path))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a top-level mapping")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML required") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10_000),
        encoding="utf-8",
    )


def resolve_under(workspace_root: Path, value: str | None) -> Path | None:
    """Resolve a path that may be absolute, repo-relative, or relative to outline file."""
    if not value:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    return p


def make_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ============================================================================
# Outline parsing
# ============================================================================


def parse_outline(path: Path) -> Outline:
    raw = load_yaml(path)

    topic = raw.get("topic") or {}
    if not isinstance(topic, dict):
        raise SystemExit("outline.topic must be a mapping")

    artifacts = raw.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise SystemExit("outline.artifacts must be a mapping")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SystemExit("outline.defaults must be a mapping")

    clusters_raw = raw.get("clusters") or []
    if not isinstance(clusters_raw, list) or not clusters_raw:
        raise SystemExit("outline.clusters must be a non-empty list")

    lanes: list[OutlineLane] = []
    for idx, item in enumerate(clusters_raw, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"outline.clusters[{idx-1}] must be a mapping")
        lane = _parse_lane(item, idx)
        lanes.append(lane)

    return Outline(topic=topic, artifacts=artifacts, defaults=defaults, clusters=lanes)


def _parse_lane(item: dict[str, Any], idx: int) -> OutlineLane:
    lane_id = normalize_whitespace(str(item.get("id") or f"L{idx}"))
    name = normalize_whitespace(str(item.get("name") or lane_id))
    color = normalize_whitespace(str(item.get("color") or "#444"))
    description = normalize_whitespace(str(item.get("description") or ""))

    source = item.get("source") or {}
    if not isinstance(source, dict):
        raise SystemExit(f"lane '{lane_id}': source must be a mapping")
    mode = normalize_whitespace(str(source.get("mode") or "manual")).casefold()
    if mode not in VALID_MODES:
        raise SystemExit(f"lane '{lane_id}': unknown source.mode='{mode}' (valid: {sorted(VALID_MODES)})")

    seed_anchors_raw = item.get("seed_anchors") or []
    if not isinstance(seed_anchors_raw, list):
        raise SystemExit(f"lane '{lane_id}': seed_anchors must be a list")
    seed_anchors = [_normalize_seed_anchor(a, lane_id) for a in seed_anchors_raw]

    raw_anchors = item.get("anchors") or []  # for manual mode
    if raw_anchors and not isinstance(raw_anchors, list):
        raise SystemExit(f"lane '{lane_id}': anchors must be a list")

    family_keywords = [
        normalize_whitespace(str(kw))
        for kw in (item.get("family_keywords") or [])
        if normalize_whitespace(str(kw))
    ]

    from_lane = normalize_whitespace(str(source.get("from_lane") or "")) or None
    from_clusters_raw = source.get("from_clusters") or source.get("from_cluster") or []
    if isinstance(from_clusters_raw, str):
        from_clusters_raw = [from_clusters_raw]
    from_clusters = [normalize_whitespace(str(c)) for c in from_clusters_raw if normalize_whitespace(str(c))]

    stages_raw = source.get("stages") or list(VALID_DISCOVER_STAGES)
    if isinstance(stages_raw, str):
        stages_raw = [stages_raw]
    stages = [s for s in stages_raw if s in VALID_DISCOVER_STAGES]

    # Per-mode validation.
    if mode == "inherit" and not from_lane:
        raise SystemExit(f"lane '{lane_id}': inherit mode requires source.from_lane")
    if mode == "reclassify" and not from_clusters:
        raise SystemExit(f"lane '{lane_id}': reclassify mode requires source.from_clusters")
    if mode == "reclassify" and not seed_anchors:
        raise SystemExit(
            f"lane '{lane_id}': reclassify mode requires seed_anchors so the LLM has a target definition"
        )
    if mode == "discover" and not seed_anchors:
        raise SystemExit(f"lane '{lane_id}': discover mode requires seed_anchors")
    if mode == "discover" and not stages:
        raise SystemExit(
            f"lane '{lane_id}': discover mode requires at least one valid stage in source.stages"
        )

    recommend_per_seed = int(source.get("recommend_per_seed") or DEFAULT_RECOMMEND_PER_SEED)

    return OutlineLane(
        id=lane_id,
        name=name,
        color=color,
        description=description,
        seed_anchors=seed_anchors,
        family_keywords=family_keywords,
        raw_anchors=list(raw_anchors),
        source_mode=mode,
        source_from_lane=from_lane,
        source_from_clusters=from_clusters,
        source_stages=stages,
        recommend_per_seed=recommend_per_seed,
    )


def _normalize_seed_anchor(item: Any, lane_id: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise SystemExit(f"lane '{lane_id}': each seed_anchor must be a mapping; got {type(item).__name__}")
    title = normalize_whitespace(str(item.get("title") or item.get("full_title") or ""))
    if not title:
        raise SystemExit(f"lane '{lane_id}': each seed_anchor must have a non-empty 'title'")
    out: dict[str, Any] = {
        "label": normalize_whitespace(str(item.get("label") or title[:32])),
        "title": title,
        "year": item.get("year"),
        "first_author": normalize_whitespace(str(item.get("first_author") or "")),
        "citation_count": item.get("citation_count"),
        "venue": normalize_whitespace(str(item.get("venue") or "")),
    }
    # Optional pre-known corpus paper id (lets user pin a match).
    if item.get("corpus_paper_id"):
        out["corpus_paper_id"] = normalize_whitespace(str(item["corpus_paper_id"]))
    if item.get("notes"):
        out["notes"] = normalize_whitespace(str(item["notes"]))
    return out


# ============================================================================
# Corpus loading
# ============================================================================


def load_corpus(search_result_path: Path, clusters_path: Path) -> Corpus:
    sr = read_json(search_result_path)
    if not isinstance(sr, dict) or "paper_cards" not in sr:
        raise SystemExit(f"{search_result_path}: missing 'paper_cards'")
    paper_cards = sr["paper_cards"] or []

    cards_by_id: dict[str, dict[str, Any]] = {}
    for card in paper_cards:
        pid = normalize_whitespace(str(card.get("paper_id") or ""))
        if pid:
            cards_by_id[pid] = card

    crj = read_json(clusters_path)
    if not isinstance(crj, dict) or "clusters" not in crj:
        raise SystemExit(f"{clusters_path}: missing 'clusters'")

    cluster_members: dict[str, list[str]] = {}
    cluster_meta: dict[str, dict[str, Any]] = {}
    for cl in crj["clusters"]:
        cid = normalize_whitespace(str(cl.get("cluster_id") or ""))
        if not cid:
            continue
        cluster_members[cid] = list(cl.get("paper_ids") or [])
        cluster_meta[cid] = cl

    log_data = ((crj.get("recluster_metadata") or {}).get("paper_assignment_log") or {})
    paper_assignment_log = log_data if isinstance(log_data, dict) else {}

    return Corpus(
        paper_cards=paper_cards,
        cards_by_id=cards_by_id,
        cluster_members=cluster_members,
        cluster_meta=cluster_meta,
        paper_assignment_log=paper_assignment_log,
    )


def load_current_spec(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"clusters": []}
    return load_yaml(path)


def index_spec_lanes(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cl in spec.get("clusters") or []:
        if not isinstance(cl, dict):
            continue
        lane_id = normalize_whitespace(str(cl.get("id") or ""))
        if lane_id:
            out[lane_id] = cl
    return out


# ============================================================================
# Fuzzy title matching (used by all modes to bind seeds to corpus papers)
# ============================================================================


_TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    return _TITLE_NORMALIZE_RE.sub(" ", title.lower()).strip()


def title_token_set(title: str) -> set[str]:
    return {tok for tok in normalize_title(title).split() if len(tok) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def fuzzy_title_match(
    target_title: str,
    target_year: int | None,
    candidates: Iterable[dict[str, Any]],
    *,
    min_jaccard: float = 0.55,
    year_tolerance: int = 3,
    min_substring_tokens: int = 4,
    excluded_paper_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, float]:
    """Find the best fuzzy match for ``target_title`` among ``candidates``.

    Substring shortcut only triggers when the shorter title is at least
    ``min_substring_tokens`` informative tokens — otherwise generic short titles
    like "Reinforcement learning" or "Deep learning" would slurp every seed.
    Pass ``excluded_paper_ids`` to skip papers already claimed by an earlier
    seed in the same lane.
    """
    target_tokens = title_token_set(target_title)
    if not target_tokens:
        return None, 0.0
    target_norm = normalize_title(target_title)
    excluded = excluded_paper_ids or set()
    best: tuple[dict[str, Any] | None, float] = (None, 0.0)
    for cand in candidates:
        cand_pid = str(cand.get("paper_id") or "")
        if cand_pid and cand_pid in excluded:
            continue
        ctitle = str(cand.get("title") or "")
        if not ctitle:
            continue
        cnorm = normalize_title(ctitle)
        ctokens = title_token_set(ctitle)
        # Substring shortcut, but only if the shorter title has enough content
        # to be meaningful — otherwise titles like "Reinforcement learning"
        # match every seed about reinforcement learning.
        shortest_tokens = min(len(target_tokens), len(ctokens))
        if (
            target_norm
            and cnorm
            and (target_norm in cnorm or cnorm in target_norm)
            and shortest_tokens >= min_substring_tokens
        ):
            score = 0.95
        else:
            score = jaccard(target_tokens, ctokens)
        if score < min_jaccard:
            continue
        # Year sanity: discard wildly off-year candidates if year is known.
        if target_year is not None:
            try:
                cyear = int(cand.get("year") or 0)
            except (TypeError, ValueError):
                cyear = 0
            if cyear and abs(cyear - int(target_year)) > year_tolerance:
                # Allow if score is very high (we know S2 sometimes mis-indexes years).
                if score < 0.9:
                    continue
        if score > best[1]:
            best = (cand, score)
    return best


# ============================================================================
# Inherit-mode resolver
# ============================================================================


def resolve_inherit(lane: OutlineLane, current_spec_by_id: dict[str, dict[str, Any]]) -> ResolvedLane:
    src_lane = current_spec_by_id.get(lane.source_from_lane or "")
    if src_lane is None:
        raise SystemExit(
            f"lane '{lane.id}': inherit from_lane='{lane.source_from_lane}' not found in current_spec"
        )
    anchors: list[ResolvedAnchor] = []
    for raw in src_lane.get("anchors") or []:
        if not isinstance(raw, dict):
            continue
        anchors.append(_anchor_from_spec_dict(raw, source_tag="inherit"))
    return ResolvedLane(
        id=lane.id,
        name=lane.name,
        color=lane.color,
        anchors=anchors,
        source_summary={
            "mode": "inherit",
            "from_lane": lane.source_from_lane,
            "anchor_count": len(anchors),
            "anchors_in_corpus": sum(1 for a in anchors if a.in_corpus),
        },
    )


def _anchor_from_spec_dict(raw: dict[str, Any], *, source_tag: str = "") -> ResolvedAnchor:
    return ResolvedAnchor(
        label=normalize_whitespace(str(raw.get("label") or "")),
        full_title=normalize_whitespace(str(raw.get("full_title") or "")),
        year=_safe_int(raw.get("year")),
        first_author=normalize_whitespace(str(raw.get("first_author") or "")),
        citation_count=_safe_int(raw.get("citation_count")),
        venue=normalize_whitespace(str(raw.get("venue") or "")),
        in_corpus=bool(raw.get("in_corpus") or False),
        corpus_paper_id=normalize_whitespace(str(raw.get("corpus_paper_id") or "")) or None,
        notes=normalize_whitespace(str(raw.get("notes") or "")),
        is_seed=True,
        source_tag=source_tag,
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================================
# Manual-mode resolver
# ============================================================================


def resolve_manual(lane: OutlineLane) -> ResolvedLane:
    """Pass anchors through, preferring the explicit ``anchors`` block if present,
    otherwise materializing ``seed_anchors`` as-is (in_corpus=false)."""
    anchors: list[ResolvedAnchor] = []
    if lane.raw_anchors:
        for raw in lane.raw_anchors:
            if isinstance(raw, dict):
                anchors.append(_anchor_from_spec_dict(raw, source_tag="manual"))
    else:
        for seed in lane.seed_anchors:
            anchors.append(_anchor_from_seed(seed, in_corpus=False, source_tag="manual"))
    return ResolvedLane(
        id=lane.id,
        name=lane.name,
        color=lane.color,
        anchors=anchors,
        source_summary={"mode": "manual", "anchor_count": len(anchors)},
    )


def _anchor_from_seed(
    seed: dict[str, Any],
    *,
    in_corpus: bool,
    corpus_paper_id: str | None = None,
    source_tag: str = "",
    notes: str = "",
) -> ResolvedAnchor:
    note = normalize_whitespace(seed.get("notes") or "")
    if notes:
        note = (note + " | " if note else "") + notes
    return ResolvedAnchor(
        label=normalize_whitespace(seed.get("label") or seed.get("title", "")[:32]),
        full_title=normalize_whitespace(seed.get("title") or ""),
        year=_safe_int(seed.get("year")),
        first_author=normalize_whitespace(seed.get("first_author") or ""),
        citation_count=_safe_int(seed.get("citation_count")),
        venue=normalize_whitespace(seed.get("venue") or ""),
        in_corpus=in_corpus,
        corpus_paper_id=corpus_paper_id,
        notes=note,
        is_seed=True,
        source_tag=source_tag,
    )


# ============================================================================
# LLM batch classifier (shared by reclassify + discover)
# ============================================================================


class LLMBatchClassifier:
    """Classifies a list of corpus papers against a set of candidate lanes.

    Each batch is one LLM call. Results are cached on disk keyed by a hash of
    the exact user prompt + model id, so reruns with ``--resume`` are free.
    """

    def __init__(
        self,
        *,
        client: DmxJsonClient | None,
        work_dir: Path,
        model: str,
        resume: bool,
        progress: bool,
        max_parallel: int,
        batch_size: int,
        abstract_char_limit: int,
    ) -> None:
        self.client = client
        self.model = model
        self.resume = resume
        self.progress = progress
        self.max_parallel = max(1, max_parallel)
        self.batch_size = max(1, batch_size)
        self.abstract_char_limit = abstract_char_limit
        self.cache_dir = work_dir / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def classify(
        self,
        papers: list[dict[str, Any]],
        candidate_lanes: list[dict[str, Any]],
        *,
        label_prefix: str,
    ) -> dict[str, tuple[str | None, float, str]]:
        """Return mapping ``paper_id -> (lane_id_or_None, confidence, justification)``.

        ``candidate_lanes`` items are dicts with keys ``id, name, description,
        seed_anchors`` (each seed_anchor is a dict with ``label, title, year,
        first_author``). A returned ``lane_id`` of ``None`` means the LLM judged
        the paper as not belonging to any candidate lane (i.e. NEITHER).
        """
        if not papers or not candidate_lanes:
            return {}
        if self.client is None:
            return {}

        # Stable batch order: sort by paper_id so the same call repeats identically.
        papers_sorted = sorted(papers, key=lambda p: str(p.get("paper_id") or ""))
        batches: list[list[dict[str, Any]]] = []
        for i in range(0, len(papers_sorted), self.batch_size):
            batches.append(papers_sorted[i : i + self.batch_size])

        results: dict[str, tuple[str | None, float, str]] = {}

        def run_one(idx: int, batch: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            prompt = _build_classifier_prompt(batch, candidate_lanes, self.abstract_char_limit)
            cache_key = make_hash(prompt, self.model)
            cache_path = self.cache_dir / f"{label_prefix}_{idx:02d}_{cache_key}.json"
            if self.resume and cache_path.exists():
                log(self.progress, f"  [llm] {label_prefix} batch {idx+1}/{len(batches)} CACHE HIT")
                return idx, read_json(cache_path)
            started = time.perf_counter()
            success = False
            try:
                payload = self.client.chat_json(  # type: ignore[union-attr]
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    label=f"{label_prefix}_b{idx+1}",
                )
                success = True
            except Exception as exc:  # pragma: no cover - degraded mode
                log(self.progress, f"  [llm] {label_prefix} batch {idx+1} FAILED: {exc}")
                payload = {"assignments": [], "_error": str(exc)[:300]}
            elapsed = time.perf_counter() - started
            # Only cache on success: a failed call should be retried next run.
            if success:
                write_json(cache_path, payload)
            log(
                self.progress,
                f"  [llm] {label_prefix} batch {idx+1}/{len(batches)} "
                f"({len(batch)} papers) in {elapsed:.1f}s"
                + (" [FAILED — not cached]" if not success else ""),
            )
            return idx, payload

        workers = max(1, min(self.max_parallel, len(batches)))
        if workers == 1:
            payloads = [run_one(i, b) for i, b in enumerate(batches)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(run_one, i, b) for i, b in enumerate(batches)]
                payloads = [f.result() for f in concurrent.futures.as_completed(futures)]

        valid_lane_ids = {str(c["id"]) for c in candidate_lanes}
        for _, payload in payloads:
            if not isinstance(payload, dict):
                continue
            for item in payload.get("assignments") or []:
                if not isinstance(item, dict):
                    continue
                pid = normalize_whitespace(str(item.get("paper_id") or ""))
                if not pid:
                    continue
                raw_lane = normalize_whitespace(str(item.get("family_id") or item.get("lane_id") or ""))
                lane_id: str | None
                if not raw_lane or raw_lane.casefold() in ("neither", "none", "n/a", ""):
                    lane_id = None
                elif raw_lane in valid_lane_ids:
                    lane_id = raw_lane
                else:
                    # LLM hallucinated a lane id; treat as NEITHER.
                    lane_id = None
                try:
                    conf = float(item.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                justification = normalize_whitespace(str(item.get("justification") or ""))
                # If a paper appears twice (shouldn't), keep the higher-confidence record.
                prev = results.get(pid)
                if prev is None or conf > prev[1]:
                    results[pid] = (lane_id, conf, justification)
        return results


def _build_classifier_prompt(
    papers: list[dict[str, Any]],
    candidate_lanes: list[dict[str, Any]],
    abstract_char_limit: int,
) -> str:
    """Compose the per-batch user prompt for the family classifier."""

    family_lines: list[str] = []
    for lane in candidate_lanes:
        seeds = lane.get("seed_anchors") or []
        seed_strs = []
        for s in seeds[:8]:
            label = s.get("label") or s.get("title", "")[:40]
            author = s.get("first_author") or ""
            year = s.get("year") or ""
            seed_strs.append(f"{label} ({author} {year})".strip())
        seed_blob = "; ".join(s for s in seed_strs if s)
        family_lines.append(
            f"  {lane['id']} · \"{lane['name']}\" — {lane.get('description') or ''}\n"
            f"        Example seeds: {seed_blob}"
        )
    family_lines.append(
        "  NEITHER — Paper does not clearly belong to any family above."
    )

    paper_lines: list[str] = []
    for p in papers:
        pid = p.get("paper_id") or ""
        title = truncate_text(str(p.get("title") or ""), 220)
        year = p.get("year") or "?"
        cites = p.get("citation_count") or 0
        abstract = truncate_text(str(p.get("abstract") or ""), abstract_char_limit)
        paper_lines.append(
            f"  {pid} · \"{title}\" · {year} · {cites} cites\n"
            f"        Abstract: {abstract}"
        )

    candidate_ids = [c["id"] for c in candidate_lanes] + ["NEITHER"]
    return (
        "You classify research papers into one of a fixed set of method families.\n"
        "Be conservative: prefer NEITHER when the paper is not a clear central member of any candidate.\n"
        "Cite the matching family by its short id only. Return strict JSON, no commentary.\n\n"
        "Target families:\n"
        + "\n".join(family_lines)
        + "\n\n"
        + "Schema:\n"
        "{\n"
        '  "assignments": [\n'
        f'    {{"paper_id": "Pxxx", "family_id": "{candidate_ids[0]}", "confidence": 0.0, '
        '"justification": "one short clause"}\n'
        "  ]\n"
        "}\n\n"
        f"Valid family_id values: {candidate_ids}\n\n"
        f"Papers ({len(papers)}):\n"
        + "\n".join(paper_lines)
        + "\n"
    )


# ============================================================================
# Reclassify-mode resolver
# ============================================================================


def resolve_reclassify_batch(
    lanes: list[OutlineLane],
    corpus: Corpus,
    llm_client: DmxJsonClient | None,
    work_dir: Path,
    *,
    defaults: dict[str, Any],
    resume: bool,
    dry_run: bool,
    progress: bool,
) -> dict[str, ResolvedLane]:
    """Resolve a group of reclassify lanes against a shared candidate set."""

    # 1) Bind each lane's seed_anchors to the corpus (fuzzy title match).
    bound_per_lane: dict[str, list[ResolvedAnchor]] = {}
    pinned_pids_per_lane: dict[str, set[str]] = {}
    for lane in lanes:
        bound = _bind_seeds_to_corpus(lane, corpus)
        bound_per_lane[lane.id] = bound
        pinned_pids_per_lane[lane.id] = {
            a.corpus_paper_id for a in bound if a.in_corpus and a.corpus_paper_id
        }

    all_pinned: set[str] = set().union(*pinned_pids_per_lane.values()) if pinned_pids_per_lane else set()

    # 2) Gather candidate paper_ids from the union of from_clusters across all lanes.
    candidate_pids: list[str] = []
    seen: set[str] = set()
    for lane in lanes:
        for cid in lane.source_from_clusters:
            for pid in corpus.cluster_members.get(cid, []):
                if pid in seen or pid in all_pinned:
                    continue
                if pid not in corpus.cards_by_id:
                    continue
                seen.add(pid)
                candidate_pids.append(pid)
    log(progress, f"  [reclassify] {len(candidate_pids)} candidate papers to classify "
                  f"(after removing {len(all_pinned)} seed-pinned)")

    # 3) Build the candidate-lanes structure expected by the classifier.
    candidate_lanes_blob: list[dict[str, Any]] = []
    for lane in lanes:
        candidate_lanes_blob.append({
            "id": lane.id,
            "name": lane.name,
            "description": lane.description,
            "seed_anchors": lane.seed_anchors,
        })

    # 4) Run the LLM classifier (skipped if dry_run or no LLM).
    classifier_results: dict[str, tuple[str | None, float, str]] = {}
    if not dry_run and llm_client is not None and candidate_pids:
        papers = [corpus.cards_by_id[pid] for pid in candidate_pids]
        classifier = LLMBatchClassifier(
            client=llm_client,
            work_dir=work_dir,
            model=llm_client.model,
            resume=resume,
            progress=progress,
            max_parallel=defaults.get("llm_max_parallel", DEFAULT_LLM_MAX_PARALLEL),
            batch_size=defaults.get("llm_batch_size", DEFAULT_LLM_BATCH_SIZE),
            abstract_char_limit=DEFAULT_ABSTRACT_CHAR_LIMIT,
        )
        classifier_results = classifier.classify(
            papers,
            candidate_lanes_blob,
            label_prefix="reclassify_" + "_".join(sorted(l.id for l in lanes))[:60],
        )

    # 5) Bucket papers per lane (respecting confidence threshold).
    confidence_threshold = float(defaults.get("llm_confidence_threshold", DEFAULT_LLM_CONFIDENCE))
    max_extras = int(defaults.get("max_auto_extras_per_lane", DEFAULT_MAX_AUTO_EXTRAS))
    extras_per_lane: dict[str, list[tuple[str, float, str]]] = {l.id: [] for l in lanes}
    for pid, (lane_id, conf, justification) in classifier_results.items():
        if lane_id is None or conf < confidence_threshold:
            continue
        if lane_id not in extras_per_lane:
            continue
        extras_per_lane[lane_id].append((pid, conf, justification))

    # 6) Build the per-lane anchor list: seeds first, then extras (capped).
    out: dict[str, ResolvedLane] = {}
    for lane in lanes:
        anchors: list[ResolvedAnchor] = list(bound_per_lane[lane.id])
        already_pids = {a.corpus_paper_id for a in anchors if a.corpus_paper_id}
        extras = sorted(
            extras_per_lane[lane.id],
            key=lambda t: (
                -int(corpus.cards_by_id.get(t[0], {}).get("citation_count") or 0),
                -int(corpus.cards_by_id.get(t[0], {}).get("year") or 0),
            ),
        )
        kept_extras = 0
        for pid, conf, justification in extras:
            if kept_extras >= max_extras:
                break
            if pid in already_pids:
                continue
            card = corpus.cards_by_id[pid]
            anchors.append(_anchor_from_corpus_card(
                card,
                source_tag=f"auto:reclassify(conf={conf:.2f})",
                notes=f"auto: reclassify from cluster, conf={conf:.2f}",
            ))
            already_pids.add(pid)
            kept_extras += 1

        out[lane.id] = ResolvedLane(
            id=lane.id,
            name=lane.name,
            color=lane.color,
            anchors=anchors,
            source_summary={
                "mode": "reclassify",
                "from_clusters": lane.source_from_clusters,
                "seed_count": len(lane.seed_anchors),
                "seeds_matched_in_corpus": sum(1 for a in bound_per_lane[lane.id] if a.in_corpus),
                "candidates_considered": len(candidate_pids),
                "candidates_assigned_to_this_lane": len(extras_per_lane[lane.id]),
                "auto_extras": kept_extras,
                "confidence_threshold": confidence_threshold,
            },
        )
    return out


def _anchor_from_corpus_card(card: dict[str, Any], *, source_tag: str, notes: str) -> ResolvedAnchor:
    """Build a fresh anchor from a corpus paper card (used for auto-extras)."""
    title = normalize_whitespace(str(card.get("title") or ""))
    authors = card.get("authors") or []
    first_author = ""
    if isinstance(authors, list) and authors:
        first_author = normalize_whitespace(str(authors[0]))
    # Short label: take first 3-4 informative words of the title.
    short = title.split(":")[0].split(",")[0]
    label = normalize_whitespace(" ".join(short.split()[:5]))[:42] or title[:32]
    return ResolvedAnchor(
        label=label,
        full_title=title,
        year=_safe_int(card.get("year")),
        first_author=first_author,
        citation_count=_safe_int(card.get("citation_count")),
        venue=normalize_whitespace(str(card.get("venue") or "")),
        in_corpus=True,
        corpus_paper_id=normalize_whitespace(str(card.get("paper_id") or "")) or None,
        notes=notes,
        is_seed=False,
        source_tag=source_tag,
    )


def _bind_seeds_to_corpus(lane: OutlineLane, corpus: Corpus) -> list[ResolvedAnchor]:
    """Match each seed_anchor against corpus paper_cards by fuzzy title.

    Tracks an in-lane ``claimed_pids`` set so the same corpus paper cannot be
    bound to two different seeds (which otherwise happens with generic-titled
    corpus papers like one literally titled "Reinforcement learning").
    """
    bound: list[ResolvedAnchor] = []
    claimed_pids: set[str] = set()
    for seed in lane.seed_anchors:
        # Honor explicit corpus_paper_id pin first.
        pinned = seed.get("corpus_paper_id")
        if pinned and pinned in corpus.cards_by_id and pinned not in claimed_pids:
            card = corpus.cards_by_id[pinned]
            bound.append(_anchor_from_seed_and_card(seed, card, source_tag="pinned"))
            claimed_pids.add(pinned)
            continue
        match, score = fuzzy_title_match(
            seed.get("title", ""),
            _safe_int(seed.get("year")),
            corpus.cards_by_id.values(),
            excluded_paper_ids=claimed_pids,
        )
        if match:
            pid = str(match.get("paper_id") or "")
            bound.append(_anchor_from_seed_and_card(
                seed, match, source_tag=f"seed_match(j={score:.2f})"
            ))
            if pid:
                claimed_pids.add(pid)
        else:
            bound.append(_anchor_from_seed(
                seed, in_corpus=False, source_tag="unmatched_seed",
                notes="auto: seed not found in corpus",
            ))
    return bound


def _anchor_from_seed_and_card(
    seed: dict[str, Any],
    card: dict[str, Any],
    *,
    source_tag: str = "",
) -> ResolvedAnchor:
    """Combine user-provided seed metadata with corpus card data. Seed wins for
    label/citation_count/year/venue/first_author (those are curated). Card
    contributes corpus_paper_id."""
    first_author = normalize_whitespace(seed.get("first_author") or "")
    if not first_author:
        authors = card.get("authors") or []
        if isinstance(authors, list) and authors:
            first_author = normalize_whitespace(str(authors[0]))
    return ResolvedAnchor(
        label=normalize_whitespace(seed.get("label") or ""),
        full_title=normalize_whitespace(seed.get("title") or card.get("title") or ""),
        year=_safe_int(seed.get("year") if seed.get("year") is not None else card.get("year")),
        first_author=first_author,
        citation_count=_safe_int(
            seed.get("citation_count") if seed.get("citation_count") is not None else card.get("citation_count")
        ),
        venue=normalize_whitespace(seed.get("venue") or card.get("venue") or ""),
        in_corpus=True,
        corpus_paper_id=normalize_whitespace(str(card.get("paper_id") or "")) or None,
        notes="",
        is_seed=True,
        source_tag=source_tag,
    )


# ============================================================================
# S2 fetch wrapper with on-disk cache
# ============================================================================


class S2FetchClient:
    """Thin caching wrapper around ``SemanticScholarSearchClient``.

    Caches every call to ``paper/search/match`` and ``recommendations`` so that
    re-runs with ``--resume`` are free. Missing-result lookups are also cached
    (as a small ``{"_miss": true}`` record) so we don't keep hammering S2 for
    seeds it doesn't know about.
    """

    def __init__(self, *, client: Any, work_dir: Path, resume: bool, progress: bool) -> None:
        self.client = client
        self.resume = resume
        self.progress = progress
        self.cache_dir = work_dir / "s2_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

    def match_title(self, title: str) -> dict[str, Any] | None:
        norm = normalize_title(title)
        if not norm:
            return None
        key = make_hash("title_match", norm)
        cache_path = self.cache_dir / f"title_{key}.json"
        if self.resume and cache_path.exists():
            cached = read_json(cache_path)
            if cached.get("_miss") or cached.get("_error"):
                return None
            return cached
        if self.client is None:
            return None
        self.call_count += 1
        try:
            result = self.client.paper_match_by_title(title)
            write_json(cache_path, result)
            return result
        except Exception as exc:
            log(self.progress, f"  [s2] title-match miss for '{title[:70]}': {type(exc).__name__}")
            write_json(cache_path, {"_miss": True, "_error": str(exc)[:200]})
            return None

    def recommend(self, seed_s2_id: str, limit: int) -> list[dict[str, Any]]:
        if not seed_s2_id:
            return []
        key = make_hash("recommend", seed_s2_id, str(limit))
        cache_path = self.cache_dir / f"rec_{key}.json"
        if self.resume and cache_path.exists():
            cached = read_json(cache_path)
            recs = cached.get("recommendedPapers")
            return recs if isinstance(recs, list) else []
        if self.client is None:
            return []
        self.call_count += 1
        try:
            result = self.client.recommend_papers(seed_s2_id, limit=limit)
            write_json(cache_path, result)
            recs = result.get("recommendedPapers")
            return recs if isinstance(recs, list) else []
        except Exception as exc:
            log(self.progress, f"  [s2] recommend miss for {seed_s2_id}: {type(exc).__name__}")
            write_json(cache_path, {"_error": str(exc)[:200]})
            return []


# Map S2 paper objects to the corpus paper_card schema we already use everywhere.
def s2_paper_to_card(s2_paper: dict[str, Any], synthetic_id: str) -> dict[str, Any]:
    s2_id = normalize_whitespace(str(s2_paper.get("paperId") or ""))
    authors_raw = s2_paper.get("authors") or []
    if isinstance(authors_raw, list):
        author_names = [
            normalize_whitespace(str(a.get("name"))) for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        ]
    else:
        author_names = []
    venue = normalize_whitespace(str(s2_paper.get("venue") or ""))
    if not venue:
        # Fall back to journal name / publicationVenue dict.
        pv = s2_paper.get("publicationVenue") or s2_paper.get("journal") or {}
        if isinstance(pv, dict):
            venue = normalize_whitespace(str(pv.get("name") or ""))
    external = s2_paper.get("externalIds") or {}
    doi = normalize_whitespace(str(external.get("DOI") or "")) if isinstance(external, dict) else ""
    return {
        "paper_id": synthetic_id,
        "stable_key": f"s2:{s2_id}" if s2_id else (f"doi:{doi}" if doi else ""),
        "title": normalize_whitespace(str(s2_paper.get("title") or "")),
        "abstract": str(s2_paper.get("abstract") or ""),
        "year": s2_paper.get("year"),
        "citation_count": s2_paper.get("citationCount"),
        "venue": venue,
        "authors": author_names,
        "doi": doi,
        "url": normalize_whitespace(str(s2_paper.get("url") or "")),
        "source": ["s2_named_fetch"],
        "source_actions": [],
        "retrieval_scores": {},
        "raw_group_id": "",
        "_s2_paper_id": s2_id,  # internal: used for recommend_expansion seed lookup
    }


def extract_s2_id_from_card(card: dict[str, Any]) -> str | None:
    """Best-effort S2 paper id from a corpus card.

    Corpus cards encode the id inside ``stable_key`` with a prefix:
    ``s2:abc123``, ``doi:10.x/y``, ``arxiv:2301.12345``, etc. S2's recommend
    endpoint accepts S2 paperIds, DOIs, and arXiv ids, so we return any of them.
    """
    inline = normalize_whitespace(str(card.get("_s2_paper_id") or ""))
    if inline:
        return inline
    stable_key = normalize_whitespace(str(card.get("stable_key") or ""))
    if ":" in stable_key:
        prefix, rest = stable_key.split(":", 1)
        rest = normalize_whitespace(rest)
        if not rest:
            return None
        if prefix == "s2":
            return rest
        if prefix == "doi":
            return f"DOI:{rest}"
        if prefix == "arxiv":
            return f"ARXIV:{rest}"
    doi = normalize_whitespace(str(card.get("doi") or ""))
    if doi:
        return f"DOI:{doi}"
    return None


# ============================================================================
# Discover-mode stages: named_fetch + recommend_expansion (real implementations)
# ============================================================================


# (resolve_discover is defined below; the helper stages live above so we can
# co-locate stage implementations near the S2/LLM glue they depend on.)


def resolve_discover(
    lane: OutlineLane,
    corpus: Corpus,
    llm_client: DmxJsonClient | None,
    s2_client: Any,
    work_dir: Path,
    *,
    defaults: dict[str, Any],
    resume: bool,
    dry_run: bool,
    skip_stages: set[str],
    progress: bool,
    added_papers: dict[str, dict[str, Any]],
) -> ResolvedLane:
    """Resolve a discover-mode lane by running its configured stages.

    Stages run in order and are additive — each can contribute additional
    auto-extras. Seed anchors are bound to the corpus once at the start (via
    fuzzy title match), so a seed that matches a corpus paper becomes a filled
    dot regardless of which stages run.
    """

    confidence_threshold = float(defaults.get("llm_confidence_threshold", DEFAULT_LLM_CONFIDENCE))
    max_extras = int(defaults.get("max_auto_extras_per_lane", DEFAULT_MAX_AUTO_EXTRAS))

    # Step A: bind seed_anchors to corpus by fuzzy title match (cheap, always run).
    bound_seeds = _bind_seeds_to_corpus(lane, corpus)
    claimed_pids: set[str] = {a.corpus_paper_id for a in bound_seeds if a.corpus_paper_id}

    # Build the shared classifier (used by corpus_rescan and recommend_expansion).
    classifier: LLMBatchClassifier | None = None
    if not dry_run and llm_client is not None:
        classifier = LLMBatchClassifier(
            client=llm_client,
            work_dir=work_dir,
            model=llm_client.model,
            resume=resume,
            progress=progress,
            max_parallel=defaults.get("llm_max_parallel", DEFAULT_LLM_MAX_PARALLEL),
            batch_size=defaults.get("llm_batch_size", DEFAULT_LLM_BATCH_SIZE),
            abstract_char_limit=DEFAULT_ABSTRACT_CHAR_LIMIT,
        )

    candidate_lane_blob = {
        "id": lane.id,
        "name": lane.name,
        "description": lane.description,
        "seed_anchors": lane.seed_anchors,
    }

    stage_results: dict[str, dict[str, Any]] = {}
    extras_pool: list[tuple[ResolvedAnchor, float]] = []

    # ----- Stage A: corpus_rescan -----
    if "corpus_rescan" in lane.source_stages and "corpus_rescan" not in skip_stages:
        rescan_extras, rescan_meta = _stage_corpus_rescan(
            lane=lane,
            corpus=corpus,
            classifier=classifier,
            candidate_lane_blob=candidate_lane_blob,
            claimed_pids=claimed_pids,
            confidence_threshold=confidence_threshold,
            dry_run=dry_run,
            progress=progress,
        )
        stage_results["corpus_rescan"] = rescan_meta
        for anchor, conf in rescan_extras:
            extras_pool.append((anchor, conf))
            if anchor.corpus_paper_id:
                claimed_pids.add(anchor.corpus_paper_id)

    # ----- Stage B + C: named_fetch + recommend_expansion (Step 4) -----
    if "named_fetch" in lane.source_stages and "named_fetch" not in skip_stages:
        named_meta = _stage_named_fetch(
            lane=lane,
            corpus=corpus,
            s2_client=s2_client,
            bound_seeds=bound_seeds,
            claimed_pids=claimed_pids,
            work_dir=work_dir,
            resume=resume,
            dry_run=dry_run,
            progress=progress,
            added_papers=added_papers,
        )
        stage_results["named_fetch"] = named_meta

    if "recommend_expansion" in lane.source_stages and "recommend_expansion" not in skip_stages:
        rec_extras, rec_meta = _stage_recommend_expansion(
            lane=lane,
            corpus=corpus,
            s2_client=s2_client,
            classifier=classifier,
            candidate_lane_blob=candidate_lane_blob,
            bound_seeds=bound_seeds,
            claimed_pids=claimed_pids,
            confidence_threshold=confidence_threshold,
            work_dir=work_dir,
            resume=resume,
            dry_run=dry_run,
            progress=progress,
            added_papers=added_papers,
        )
        stage_results["recommend_expansion"] = rec_meta
        for anchor, conf in rec_extras:
            extras_pool.append((anchor, conf))
            if anchor.corpus_paper_id:
                claimed_pids.add(anchor.corpus_paper_id)

    # ----- Pick top-N extras across stages, sorted by confidence then citations -----
    extras_pool.sort(key=lambda t: (-t[1], -(t[0].citation_count or 0), -(t[0].year or 0)))
    kept_extras: list[ResolvedAnchor] = []
    seen_pids: set[str] = set()
    for anchor, conf in extras_pool:
        if len(kept_extras) >= max_extras:
            break
        if anchor.corpus_paper_id and anchor.corpus_paper_id in seen_pids:
            continue
        kept_extras.append(anchor)
        if anchor.corpus_paper_id:
            seen_pids.add(anchor.corpus_paper_id)

    # Assemble final anchor list: seeds first, then capped extras.
    anchors: list[ResolvedAnchor] = list(bound_seeds) + kept_extras

    return ResolvedLane(
        id=lane.id,
        name=lane.name,
        color=lane.color,
        anchors=anchors,
        source_summary={
            "mode": "discover",
            "stages_planned": lane.source_stages,
            "stages_executed": [s for s in lane.source_stages if s not in skip_stages],
            "seed_count": len(lane.seed_anchors),
            "seeds_matched_in_corpus": sum(1 for a in bound_seeds if a.in_corpus),
            "auto_extras": len(kept_extras),
            "extras_pool_size": len(extras_pool),
            "stage_results": stage_results,
            "confidence_threshold": confidence_threshold,
        },
    )


# ----------------------------------------------------------------------------
# Stage A: corpus_rescan
# ----------------------------------------------------------------------------


def _stage_corpus_rescan(
    *,
    lane: OutlineLane,
    corpus: Corpus,
    classifier: LLMBatchClassifier | None,
    candidate_lane_blob: dict[str, Any],
    claimed_pids: set[str],
    confidence_threshold: float,
    dry_run: bool,
    progress: bool,
) -> tuple[list[tuple[ResolvedAnchor, float]], dict[str, Any]]:
    """Find corpus papers matching ``family_keywords`` (title/abstract substring),
    then LLM-classify each candidate as belonging to the lane or NEITHER."""

    if not lane.family_keywords:
        log(progress, f"  [corpus_rescan] {lane.id}: no family_keywords, skipped")
        return [], {"prefilter_hits": 0, "classified": 0, "kept": 0, "skipped": "no family_keywords"}

    needles = [k.casefold() for k in lane.family_keywords]
    prefilter_hits: list[dict[str, Any]] = []
    for card in corpus.paper_cards:
        pid = str(card.get("paper_id") or "")
        if not pid or pid in claimed_pids:
            continue
        haystack = (
            str(card.get("title") or "").casefold()
            + "\n"
            + str(card.get("abstract") or "").casefold()
        )
        if any(n in haystack for n in needles):
            prefilter_hits.append(card)

    log(progress, f"  [corpus_rescan] {lane.id}: {len(prefilter_hits)} prefilter hits "
                  f"on keywords {lane.family_keywords}")

    if not prefilter_hits or classifier is None:
        return [], {
            "prefilter_hits": len(prefilter_hits),
            "classified": 0,
            "kept": 0,
            "skipped": "dry_run or no LLM" if classifier is None else None,
        }

    if dry_run:
        return [], {"prefilter_hits": len(prefilter_hits), "classified": 0, "kept": 0, "skipped": "dry_run"}

    results = classifier.classify(
        prefilter_hits,
        [candidate_lane_blob],
        label_prefix=f"discover_rescan_{lane.id}",
    )

    extras: list[tuple[ResolvedAnchor, float]] = []
    for pid, (lane_id, conf, justification) in results.items():
        if lane_id != lane.id:
            continue
        if conf < confidence_threshold:
            continue
        if pid not in corpus.cards_by_id:
            continue
        card = corpus.cards_by_id[pid]
        anchor = _anchor_from_corpus_card(
            card,
            source_tag=f"auto:corpus_rescan(conf={conf:.2f})",
            notes=f"auto: corpus_rescan, conf={conf:.2f}",
        )
        extras.append((anchor, conf))

    log(progress, f"  [corpus_rescan] {lane.id}: classified {len(results)} → kept {len(extras)} above conf {confidence_threshold:.2f}")
    return extras, {
        "prefilter_hits": len(prefilter_hits),
        "classified": len(results),
        "kept": len(extras),
    }


# ----------------------------------------------------------------------------
# Stage B: named_fetch — hit S2 paper/search/match for unmatched seeds
# ----------------------------------------------------------------------------


def _stage_named_fetch(
    *,
    lane: OutlineLane,
    corpus: Corpus,
    s2_client: Any,
    bound_seeds: list[ResolvedAnchor],
    claimed_pids: set[str],
    work_dir: Path,
    resume: bool,
    dry_run: bool,
    progress: bool,
    added_papers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """For each seed that didn't match anything in the corpus, ask S2 for the
    closest title match. If S2 returns a usable record we upgrade the seed
    anchor in place: it becomes ``in_corpus=true`` with a synthetic
    ``corpus_paper_id`` and the fetched paper is recorded in ``added_papers``.

    Returns metadata (counts, fetched ids).
    """

    unmatched_indices = [i for i, a in enumerate(bound_seeds) if not a.in_corpus]
    if not unmatched_indices:
        log(progress, f"  [named_fetch] {lane.id}: all seeds already in corpus, nothing to fetch")
        return {"unmatched_seeds": 0, "fetched": 0, "skipped": "no unmatched seeds"}

    if dry_run or s2_client is None:
        log(progress, f"  [named_fetch] {lane.id}: {len(unmatched_indices)} unmatched seeds (dry_run or no S2)")
        return {
            "unmatched_seeds": len(unmatched_indices),
            "fetched": 0,
            "skipped": "dry_run or no S2",
        }

    s2 = S2FetchClient(client=s2_client, work_dir=work_dir, resume=resume, progress=progress)

    fetched_ids: list[str] = []
    for idx in unmatched_indices:
        anchor = bound_seeds[idx]
        seed = next(
            (s for s in lane.seed_anchors if normalize_whitespace(s.get("label") or "") == anchor.label),
            None,
        )
        title = anchor.full_title or (seed.get("title") if seed else "")
        target_year = anchor.year
        if not title:
            continue
        s2_paper = s2.match_title(title)
        if not s2_paper:
            log(progress, f"  [named_fetch] {lane.id}: '{title[:60]}' → MISS")
            continue
        # Sanity: year close enough.
        s2_year = s2_paper.get("year")
        if target_year and isinstance(s2_year, int) and abs(s2_year - target_year) > 4:
            log(progress, f"  [named_fetch] {lane.id}: '{title[:60]}' → year mismatch "
                          f"(want {target_year}, got {s2_year}) — SKIP")
            continue
        # Build a synthetic corpus card and update the bound anchor in place.
        synthetic_id = f"F_{lane.id}_{len(fetched_ids)+1:03d}"
        card = s2_paper_to_card(s2_paper, synthetic_id)
        added_papers[synthetic_id] = card

        # Refresh the anchor with the now-available metadata. Seed-supplied values
        # win where they exist (curated > fetched).
        new_anchor = _anchor_from_seed_and_card(
            seed or {"title": title, "label": anchor.label, "year": target_year,
                     "first_author": anchor.first_author, "citation_count": anchor.citation_count,
                     "venue": anchor.venue},
            card,
            source_tag=f"auto:named_fetch(s2:{card.get('_s2_paper_id','')[:10]})",
        )
        new_anchor.notes = (
            (anchor.notes + " | " if anchor.notes else "")
            + f"auto: named_fetch via S2 (paperId={card.get('_s2_paper_id','')[:16]})"
        )
        bound_seeds[idx] = new_anchor
        claimed_pids.add(synthetic_id)
        fetched_ids.append(synthetic_id)
        log(progress, f"  [named_fetch] {lane.id}: '{title[:60]}' → FETCHED {synthetic_id}")

    return {
        "unmatched_seeds": len(unmatched_indices),
        "fetched": len(fetched_ids),
        "fetched_ids": fetched_ids,
        "s2_calls": s2.call_count,
    }


# ----------------------------------------------------------------------------
# Stage C: recommend_expansion — S2 recommend + LLM family classifier
# ----------------------------------------------------------------------------


def _stage_recommend_expansion(
    *,
    lane: OutlineLane,
    corpus: Corpus,
    s2_client: Any,
    classifier: LLMBatchClassifier | None,
    candidate_lane_blob: dict[str, Any],
    bound_seeds: list[ResolvedAnchor],
    claimed_pids: set[str],
    confidence_threshold: float,
    work_dir: Path,
    resume: bool,
    dry_run: bool,
    progress: bool,
    added_papers: dict[str, dict[str, Any]],
) -> tuple[list[tuple[ResolvedAnchor, float]], dict[str, Any]]:
    """For each seed that resolves to a known S2 paperId (whether from corpus
    or named_fetch), ask S2 for ``recommend_per_seed`` similar papers, then
    run the LLM family classifier on the union. Above-threshold matches are
    returned as auto-extras.
    """

    # Collect seed paperIds suitable for S2 recommendations.
    seed_s2_ids: list[tuple[str, str]] = []  # (s2_id, seed_label)
    for anchor in bound_seeds:
        if not anchor.in_corpus:
            continue
        # Try the in-process added_papers first (named_fetch cards have _s2_paper_id),
        # then fall back to the existing corpus card.
        card = added_papers.get(anchor.corpus_paper_id or "") or corpus.cards_by_id.get(anchor.corpus_paper_id or "")
        if not card:
            continue
        s2_id = extract_s2_id_from_card(card)
        if s2_id:
            seed_s2_ids.append((s2_id, anchor.label))

    if not seed_s2_ids:
        log(progress, f"  [recommend_expansion] {lane.id}: no seeds with resolvable S2 id, skipped")
        return [], {"seed_count": 0, "fetched_recommendations": 0, "kept": 0,
                    "skipped": "no S2-resolvable seeds"}

    if dry_run or s2_client is None:
        log(progress, f"  [recommend_expansion] {lane.id}: {len(seed_s2_ids)} seeds (dry_run or no S2)")
        return [], {"seed_count": len(seed_s2_ids), "fetched_recommendations": 0, "kept": 0,
                    "skipped": "dry_run or no S2"}

    s2 = S2FetchClient(client=s2_client, work_dir=work_dir, resume=resume, progress=progress)
    limit_per_seed = max(1, int(lane.recommend_per_seed))

    # Dedupe recommendations by S2 paperId (or doi).
    seen_s2_keys: set[str] = set()
    recommendation_cards: list[dict[str, Any]] = []
    for s2_id, seed_label in seed_s2_ids:
        recs = s2.recommend(s2_id, limit=limit_per_seed)
        log(progress, f"  [recommend_expansion] {lane.id}: seed '{seed_label}' → {len(recs)} recs")
        for r in recs:
            r_s2_id = normalize_whitespace(str(r.get("paperId") or ""))
            dedupe_key = r_s2_id or normalize_whitespace(str((r.get("externalIds") or {}).get("DOI") or ""))
            if not dedupe_key or dedupe_key in seen_s2_keys:
                continue
            seen_s2_keys.add(dedupe_key)
            # Skip if this paper is already in the original corpus by DOI / title.
            doi = normalize_whitespace(str((r.get("externalIds") or {}).get("DOI") or ""))
            if doi and any(c.get("doi") == doi for c in corpus.paper_cards):
                continue
            synthetic_id = f"R_{lane.id}_{len(recommendation_cards)+1:03d}"
            card = s2_paper_to_card(r, synthetic_id)
            if not card["title"]:
                continue
            recommendation_cards.append(card)

    if not recommendation_cards:
        return [], {"seed_count": len(seed_s2_ids), "fetched_recommendations": 0, "kept": 0}

    log(progress, f"  [recommend_expansion] {lane.id}: {len(recommendation_cards)} unique recommendations to classify")

    if classifier is None:
        return [], {"seed_count": len(seed_s2_ids), "fetched_recommendations": len(recommendation_cards),
                    "kept": 0, "skipped": "no LLM classifier"}

    results = classifier.classify(
        recommendation_cards,
        [candidate_lane_blob],
        label_prefix=f"discover_recexp_{lane.id}",
    )

    extras: list[tuple[ResolvedAnchor, float]] = []
    cards_by_id = {c["paper_id"]: c for c in recommendation_cards}
    for pid, (lane_id, conf, justification) in results.items():
        if lane_id != lane.id:
            continue
        if conf < confidence_threshold:
            continue
        card = cards_by_id.get(pid)
        if not card:
            continue
        # Promote card into added_papers so it gets persisted.
        added_papers[pid] = card
        anchor = _anchor_from_corpus_card(
            card,
            source_tag=f"auto:recommend_expansion(conf={conf:.2f}, s2:{card.get('_s2_paper_id','')[:10]})",
            notes=f"auto: recommend_expansion, conf={conf:.2f}",
        )
        extras.append((anchor, conf))

    log(progress, f"  [recommend_expansion] {lane.id}: classified {len(results)} → kept {len(extras)}")
    return extras, {
        "seed_count": len(seed_s2_ids),
        "fetched_recommendations": len(recommendation_cards),
        "classified": len(results),
        "kept": len(extras),
        "s2_calls": s2.call_count,
    }


# ============================================================================
# Spec emit
# ============================================================================


def emit_spec(
    outline: Outline,
    resolved_lanes: list[ResolvedLane],
    out_path: Path,
) -> None:
    """Write a renderable spec yaml."""
    spec: dict[str, Any] = {"topic": dict(outline.topic), "clusters": []}
    for lane in resolved_lanes:
        cluster_dict: dict[str, Any] = {
            "id": lane.id,
            "name": lane.name,
            "color": lane.color,
            "anchors": [a.to_yaml_dict() for a in lane.anchors],
        }
        spec["clusters"].append(cluster_dict)
    dump_yaml(out_path, spec)


def emit_migration_report(
    outline: Outline,
    resolved_lanes: list[ResolvedLane],
    work_dir: Path,
    started_at: float,
) -> Path:
    elapsed = time.perf_counter() - started_at
    report = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 2),
        "topic": outline.topic.get("title"),
        "lane_count": len(resolved_lanes),
        "lanes": [
            {
                "id": lane.id,
                "name": lane.name,
                "color": lane.color,
                "anchor_count": len(lane.anchors),
                "anchors_in_corpus": sum(1 for a in lane.anchors if a.in_corpus),
                "anchors_injected": sum(1 for a in lane.anchors if not a.in_corpus),
                "summary": lane.source_summary,
                "anchors": [
                    {
                        "label": a.label,
                        "year": a.year,
                        "in_corpus": a.in_corpus,
                        "corpus_paper_id": a.corpus_paper_id,
                        "source_tag": a.source_tag,
                    }
                    for a in lane.anchors
                ],
            }
            for lane in resolved_lanes
        ],
    }
    out = work_dir / "migration_report.json"
    write_json(out, report)
    return out


def print_summary_table(resolved_lanes: list[ResolvedLane]) -> None:
    print("")
    print(f"{'lane_id':<8} {'mode':<11} {'in_corp':>7} {'injected':>9} {'extras':>7}  name")
    print("-" * 78)
    total_in_corpus = 0
    total_injected = 0
    for lane in resolved_lanes:
        mode = str(lane.source_summary.get("mode") or "?")[:11]
        in_corp = sum(1 for a in lane.anchors if a.in_corpus)
        injected = sum(1 for a in lane.anchors if not a.in_corpus)
        extras = int(lane.source_summary.get("auto_extras") or 0)
        total_in_corpus += in_corp
        total_injected += injected
        print(f"{lane.id:<8} {mode:<11} {in_corp:>7} {injected:>9} {extras:>7}  {lane.name}")
    print("-" * 78)
    print(f"{'TOTAL':<8} {'':<11} {total_in_corpus:>7} {total_injected:>9}")


# ============================================================================
# Main orchestration
# ============================================================================


def run(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    progress = not args.quiet

    outline_path = Path(args.outline).expanduser().resolve()
    if not outline_path.exists():
        raise SystemExit(f"--outline not found: {outline_path}")
    outline_dir = outline_path.parent

    log(progress, f"OUTLINE  {outline_path}")
    outline = parse_outline(outline_path)

    # Resolve artifact paths relative to the outline file location (or repo root).
    repo_root = LR_SEARCH_DIR.parent.parent
    artifacts = outline.artifacts

    def _resolve(name: str) -> Path | None:
        raw = artifacts.get(name)
        if not raw:
            return None
        # Try outline-relative first, then repo-relative.
        p = (outline_dir / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not p.exists():
            p2 = (repo_root / raw).resolve()
            if p2.exists():
                p = p2
        return p

    search_result_path = _resolve("search_result")
    clusters_path = _resolve("clusters_recovered")
    current_spec_path = _resolve("current_spec")

    if not search_result_path or not search_result_path.exists():
        raise SystemExit(f"artifacts.search_result not found: {artifacts.get('search_result')}")
    if not clusters_path or not clusters_path.exists():
        raise SystemExit(f"artifacts.clusters_recovered not found: {artifacts.get('clusters_recovered')}")

    log(progress, f"SEARCH   {search_result_path}")
    log(progress, f"CLUSTERS {clusters_path}")

    corpus = load_corpus(search_result_path, clusters_path)
    log(progress, f"LOADED   {len(corpus.cards_by_id)} papers, {len(corpus.cluster_members)} corpus clusters")

    current_spec = load_current_spec(current_spec_path)
    current_spec_by_id = index_spec_lanes(current_spec)
    if current_spec_path:
        log(progress, f"SPEC_IN  {current_spec_path}  ({len(current_spec_by_id)} lanes)")

    # Optional sanity check: every inherit's from_lane resolves; every reclassify's
    # from_clusters resolves; etc. Do this upfront so we fail loud before LLM/S2.
    _preflight(outline, current_spec_by_id, corpus)

    out_spec_path = Path(args.out_spec).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else out_spec_path.parent / "migration" / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log(progress, f"WORK_DIR {work_dir}")

    skip_stages = set(args.skip_stages or [])

    # Build LLM client lazily; only needed for reclassify or discover modes that
    # have unskipped stages requiring an LLM call.
    needs_llm = any(
        lane.source_mode == "reclassify"
        or (lane.source_mode == "discover" and any(s not in skip_stages for s in lane.source_stages))
        for lane in outline.clusters
    )
    needs_s2 = any(
        lane.source_mode == "discover"
        and any(s in lane.source_stages and s not in skip_stages for s in ("named_fetch", "recommend_expansion"))
        for lane in outline.clusters
    )
    llm_client: DmxJsonClient | None = None
    if needs_llm and not args.dry_run:
        llm_client = DmxJsonClient(
            env_path=DEFAULT_ENV_PATH,
            api_url=args.llm_api_url,
            model=args.llm_model,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature,
            use_env_proxy=False,
        )
        log(progress, f"LLM      {args.llm_model} via {args.llm_api_url}")

    s2_client: Any = None
    if needs_s2 and not args.dry_run:
        try:
            from search_s2 import SemanticScholarSearchClient  # noqa: E402
            s2_client = SemanticScholarSearchClient()
            log(progress, "S2       SemanticScholarSearchClient ready")
        except Exception as exc:
            log(progress, f"S2       FAILED to initialize: {exc} (S2 stages will be skipped)")
            s2_client = None

    # Accumulator for any papers fetched from S2 that weren't already in corpus —
    # persisted at end of run so future iterations can ingest them.
    added_papers: dict[str, dict[str, Any]] = {}

    defaults = {
        "llm_confidence_threshold": float(outline.defaults.get("llm_confidence_threshold") or DEFAULT_LLM_CONFIDENCE),
        "max_auto_extras_per_lane": int(outline.defaults.get("max_auto_extras_per_lane") or DEFAULT_MAX_AUTO_EXTRAS),
        "llm_batch_size": int(outline.defaults.get("llm_batch_size") or DEFAULT_LLM_BATCH_SIZE),
        "llm_max_parallel": int(outline.defaults.get("llm_max_parallel") or args.max_parallel),
    }

    # Resolve each lane in declaration order.
    resolved: list[ResolvedLane] = []
    reclassify_lanes = [l for l in outline.clusters if l.source_mode == "reclassify"]

    for lane in outline.clusters:
        if lane.source_mode == "inherit":
            log(progress, f"[{lane.id}] inherit ← {lane.source_from_lane}")
            resolved.append(resolve_inherit(lane, current_spec_by_id))
        elif lane.source_mode == "manual":
            log(progress, f"[{lane.id}] manual")
            resolved.append(resolve_manual(lane))
        elif lane.source_mode == "reclassify":
            # Handled in batch below.
            continue
        elif lane.source_mode == "discover":
            log(progress, f"[{lane.id}] discover ← stages={lane.source_stages}")
            resolved.append(
                resolve_discover(
                    lane, corpus, llm_client, s2_client, work_dir,
                    defaults=defaults, resume=args.resume,
                    dry_run=args.dry_run, skip_stages=skip_stages, progress=progress,
                    added_papers=added_papers,
                )
            )

    # Process reclassify lanes as one batch so the LLM sees all candidate lanes
    # in a single classification prompt per paper.
    if reclassify_lanes:
        log(progress, f"[reclassify] {len(reclassify_lanes)} lanes ← clusters={sorted({c for l in reclassify_lanes for c in l.source_from_clusters})}")
        rc_resolved = resolve_reclassify_batch(
            reclassify_lanes, corpus, llm_client, work_dir,
            defaults=defaults, resume=args.resume,
            dry_run=args.dry_run, progress=progress,
        )
        # Splice back in declaration order.
        rc_by_id = {r.id: r for r in rc_resolved.values()}
        # Walk outline again, inserting reclassify lanes at their original positions.
        ordered: list[ResolvedLane] = []
        rc_iter_idx = 0
        already_added = {r.id for r in resolved}
        for lane in outline.clusters:
            if lane.source_mode == "reclassify":
                if lane.id in rc_by_id:
                    ordered.append(rc_by_id[lane.id])
            else:
                # Pull from already-resolved list.
                for r in resolved:
                    if r.id == lane.id and r.id not in {o.id for o in ordered}:
                        ordered.append(r)
                        break
        resolved = ordered

    # Emit output.
    if args.dry_run:
        log(progress, "(dry-run) skipping spec emit")
    else:
        emit_spec(outline, resolved, out_spec_path)
        log(progress, f"SPEC_OUT {out_spec_path}")

    report_path = emit_migration_report(outline, resolved, work_dir, started_at)
    log(progress, f"REPORT   {report_path}")

    # Persist any papers fetched from S2 so subsequent runs / pipelines can ingest them.
    if added_papers:
        added_path = work_dir / "added_papers.json"
        # Strip internal-only field before writing.
        clean = []
        for pid, card in added_papers.items():
            c = {k: v for k, v in card.items() if not k.startswith("_")}
            c["_s2_paper_id"] = card.get("_s2_paper_id", "")  # keep this one — useful for re-runs
            clean.append(c)
        write_json(added_path, {"papers": clean, "count": len(clean)})
        log(progress, f"ADDED    {added_path}  ({len(clean)} S2-fetched papers)")

    if progress:
        print_summary_table(resolved)
        elapsed = time.perf_counter() - started_at
        print(f"\nDone in {elapsed:.1f}s. LLM calls: {(llm_client.call_count if llm_client else 0)}.")
    return 0


def _preflight(outline: Outline, current_spec_by_id: dict[str, dict[str, Any]], corpus: Corpus) -> None:
    """Fail fast on misreferenced lanes / clusters before doing any expensive work."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for lane in outline.clusters:
        if lane.id in seen_ids:
            errors.append(f"duplicate lane id '{lane.id}'")
        seen_ids.add(lane.id)
        if lane.source_mode == "inherit":
            if lane.source_from_lane not in current_spec_by_id:
                errors.append(
                    f"lane '{lane.id}': inherit from_lane='{lane.source_from_lane}' "
                    f"not found in current_spec (known: {sorted(current_spec_by_id)})"
                )
        elif lane.source_mode == "reclassify":
            for cid in lane.source_from_clusters:
                if cid not in corpus.cluster_members:
                    errors.append(
                        f"lane '{lane.id}': reclassify from_cluster='{cid}' "
                        f"not in clusters_recovered (known: {sorted(corpus.cluster_members)})"
                    )
    if errors:
        msg = "Outline preflight failed:\n  - " + "\n  - ".join(errors)
        raise SystemExit(msg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outline", required=True, help="Path to target-outline yaml.")
    p.add_argument("--out-spec", required=True, help="Path to write the new spec yaml.")
    p.add_argument("--work-dir", default=None, help="Where to put cache + reports. Default: <out_spec_dir>/migration/<ts>/")
    p.add_argument("--resume", action="store_true", help="Reuse cached LLM / S2 results from previous runs in the same work_dir.")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not call LLM / S2 and do not emit spec.")
    p.add_argument("--skip-stages", nargs="*", default=[], choices=list(VALID_DISCOVER_STAGES),
                   help="Global override: skip these discover stages everywhere.")
    p.add_argument("--llm-api-url", default=DEFAULT_DMX_API_URL)
    p.add_argument("--llm-model", default=DEFAULT_DMX_MODEL)
    p.add_argument("--llm-timeout", type=int, default=DEFAULT_LLM_TIMEOUT)
    p.add_argument("--llm-max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    p.add_argument("--llm-temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    p.add_argument("--max-parallel", type=int, default=DEFAULT_LLM_MAX_PARALLEL,
                   help="Max parallel LLM batches (default 8).")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
