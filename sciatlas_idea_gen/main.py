"""CLI entrypoint for the SciAtlas idea-generation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline
from .utils import get_logger, read_json
from .visualize_drawio import export_drawio

log = get_logger("sciatlas.main")


def apply_preset(args: argparse.Namespace, cfg) -> None:
    selected = getattr(args, "workflow", None) or getattr(args, "preset", "default")
    if selected == "flash":
        cfg.pipeline.workflow_mode = "flash"
        cfg.pipeline.k_step1 = 1
        cfg.pipeline.anchor_top_k = 3
        cfg.pipeline.seed_num_queries = 1
        cfg.pipeline.seed_per_query = 1
        cfg.pipeline.seed_request_multiplier = 3
        cfg.pipeline.seed_request_floor = 3
        cfg.pipeline.graph_budget_min = 5
        cfg.pipeline.graph_budget_max = 5
        cfg.pipeline.graph_budget_ratio = 1.0
        cfg.pipeline.graph_max_predecessors_per_paper = 1
        cfg.pipeline.graph_min_forward_papers = 1
        cfg.pipeline.graph_candidate_fetch_limit = 10
        cfg.pipeline.num_candidate_step2 = 1
        cfg.pipeline.num_expansion_step2 = 1
        cfg.pipeline.num_cross_domains = 1
        cfg.pipeline.inspiration_top_k_same_field = 1
        cfg.pipeline.inspiration_top_k_per_domain = 1
        cfg.pipeline.max_novelty_feedback_rounds = 0
        cfg.pipeline.idea_count = 1
    elif selected == "smoke":
        cfg.pipeline.workflow_mode = "smoke"
        cfg.pipeline.k_step1 = 4
        cfg.pipeline.anchor_top_k = 3
        cfg.pipeline.seed_num_queries = 2
        cfg.pipeline.seed_per_query = 2
        cfg.pipeline.seed_request_multiplier = 6
        cfg.pipeline.seed_request_floor = 12
        cfg.pipeline.graph_budget_min = 12
        cfg.pipeline.graph_budget_max = 12
        cfg.pipeline.graph_budget_ratio = 1.0
        cfg.pipeline.graph_max_predecessors_per_paper = 3
        cfg.pipeline.graph_min_forward_papers = 2
        cfg.pipeline.graph_candidate_fetch_limit = 40
        cfg.pipeline.num_candidate_step2 = 3
        cfg.pipeline.num_expansion_step2 = 1
        cfg.pipeline.num_cross_domains = 1
        cfg.pipeline.inspiration_top_k_same_field = 1
        cfg.pipeline.inspiration_top_k_per_domain = 1
        cfg.pipeline.max_novelty_feedback_rounds = 0
        cfg.pipeline.idea_count = 1
    elif selected == "full":
        cfg.pipeline.workflow_mode = "full"
        cfg.pipeline.k_step1 = 4
        cfg.pipeline.anchor_top_k = 5
        cfg.pipeline.seed_num_queries = 8
        cfg.pipeline.seed_per_query = 3
        cfg.pipeline.seed_request_multiplier = 6
        cfg.pipeline.seed_request_floor = 12
        cfg.pipeline.graph_budget_min = 12
        cfg.pipeline.graph_budget_max = 30
        cfg.pipeline.graph_budget_ratio = 0.35
        cfg.pipeline.graph_max_predecessors_per_paper = 3
        cfg.pipeline.graph_min_forward_papers = 3
        cfg.pipeline.graph_candidate_fetch_limit = 40
        cfg.pipeline.num_candidate_step2 = 3
        cfg.pipeline.num_expansion_step2 = 1
        cfg.pipeline.num_cross_domains = 2
        cfg.pipeline.inspiration_top_k_same_field = 2
        cfg.pipeline.inspiration_top_k_per_domain = 2
        cfg.pipeline.max_novelty_feedback_rounds = 2
        cfg.pipeline.idea_count = 1
    else:
        cfg.pipeline.workflow_mode = "default"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SciAtlas idea-generation pipeline.",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic, seed paper title, rough idea, or research problem.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional target field/domain constraint passed to SciAtlas.",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=None,
        metavar="PATH",
        help="Path to an input PDF paper. Repeatable. Each PDF is extracted (title/"
        "abstract), folded into query refinement + SciAtlas retrieval, and force-"
        "included as a seed so it is guaranteed to appear in the research graph.",
    )
    parser.add_argument(
        "--keywords-json",
        default=None,
        help='Optional seed keywords JSON, e.g. \'[{"text":"graph","score":9}]\'.',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to runs/<timestamp>/.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="SciAtlas KG request timeout in seconds. Defaults to SCIATLAS_TIMEOUT.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke mode: save intermediate outputs and return partial results on failure.",
    )
    parser.add_argument(
        "--workflow",
        choices=["flash", "full"],
        default=None,
        help=(
            "Choose the product workflow path: flash for fast interactive runs "
            "with compressed gate/selection stages, full for comprehensive runs."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["default", "smoke", "flash", "full"],
        default="default",
        help="Apply a built-in pipeline preset. Use --workflow for the main flash/full choice.",
    )
    parser.add_argument(
        "--seed-recent-years",
        type=int,
        default=None,
        help="Override Step 1 seed recency window (years). Use a large value (e.g. 1000) "
        "to effectively lift the recency restriction so older foundational seeds qualify.",
    )
    parser.add_argument(
        "--graph-budget-max",
        type=int,
        default=None,
        help="Override Step 2 max research-graph size (papers). Larger for literature-rich topics.",
    )
    parser.add_argument(
        "--graph-budget-min",
        type=int,
        default=None,
        help="Override Step 2 min research-graph size (papers).",
    )
    parser.add_argument(
        "--k-step1",
        type=int,
        default=None,
        help="Override the number of seed papers Step 1 returns. Larger for a "
        "broader, more literature-rich seed set.",
    )
    parser.add_argument(
        "--top-k",
        "--anchor-top-k",
        dest="anchor_top_k",
        type=int,
        default=None,
        help="Override the number of anchor papers used during Step 1 seed retrieval.",
    )
    parser.add_argument(
        "--seed-num-queries",
        type=int,
        default=None,
        help="Override the number of diverse Step 1 seed queries.",
    )
    parser.add_argument(
        "--seed-per-query",
        type=int,
        default=None,
        help="Override the number of seed papers kept per Step 1 query.",
    )
    parser.add_argument(
        "--idea-count",
        type=int,
        default=None,
        help="Override the number of final ideas to generate.",
    )
    parser.add_argument(
        "--resume-from-run-dir",
        default=None,
        help="Resume from an existing run directory with saved artifacts.",
    )
    parser.add_argument(
        "--no-intermediate-json",
        action="store_true",
        help="Do not save step1/2/4/5/6_7 intermediate JSON files or error JSON files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config()
    apply_preset(args, cfg)
    # Apply explicit overrides AFTER the preset so they win over preset budgets.
    if args.timeout is not None:
        cfg.sciatlas.timeout = max(1, args.timeout)
    if args.seed_recent_years is not None:
        cfg.pipeline.seed_recent_years = args.seed_recent_years
    if args.graph_budget_max is not None:
        cfg.pipeline.graph_budget_max = args.graph_budget_max
    if args.graph_budget_min is not None:
        cfg.pipeline.graph_budget_min = args.graph_budget_min
    if args.k_step1 is not None:
        cfg.pipeline.k_step1 = args.k_step1
    if args.anchor_top_k is not None:
        cfg.pipeline.anchor_top_k = max(1, args.anchor_top_k)
    if args.seed_num_queries is not None:
        cfg.pipeline.seed_num_queries = max(1, args.seed_num_queries)
    if args.seed_per_query is not None:
        cfg.pipeline.seed_per_query = max(1, args.seed_per_query)
    if args.idea_count is not None:
        cfg.pipeline.idea_count = max(1, args.idea_count)
    seed_keywords = json.loads(args.keywords_json) if args.keywords_json else None
    pdf_paths = args.pdf or None
    topic = args.topic
    if not topic and args.resume_from_run_dir:
        summary_path = Path(args.resume_from_run_dir) / "summary.json"
        if summary_path.exists():
            topic = read_json(summary_path).get("topic")
    if not topic and not pdf_paths:
        raise SystemExit(
            "topic is required unless --pdf is given or --resume-from-run-dir points to a run with summary.json"
        )
    summary = run_pipeline(
        topic,
        seed_keywords=seed_keywords,
        domain=args.domain,
        pdf_paths=pdf_paths,
        config=cfg,
        output_dir=args.output_dir,
        smoke_mode=args.smoke,
        resume_from_run_dir=args.resume_from_run_dir,
        save_intermediate_json=not args.no_intermediate_json,
    )
    run_dir = summary.get("run_dir")
    if run_dir:
        try:
            drawio_path = export_drawio(run_dir)
            summary["drawio_path"] = str(drawio_path)
        except Exception as exc:
            log.warning("draw.io export failed for %s: %s", run_dir, exc)
            summary["drawio_error"] = str(exc)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
