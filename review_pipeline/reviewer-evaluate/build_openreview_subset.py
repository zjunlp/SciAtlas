#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "dataset_runs" / "pairs_v2_final" / "summary_reviewers_20260529_130404.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a balanced subset of reviewer-evaluate papers for OpenReview-style baseline runs."
    )
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON), help="summary_reviewers_*.json path.")
    parser.add_argument("--subset-size", type=int, default=70, help="Number of papers to keep.")
    parser.add_argument(
        "--output-json",
        default=str(SCRIPT_DIR / "dataset_runs" / "pairs_v2_final" / "openreview_subset_70.json"),
        help="Output subset manifest JSON.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_paper_key(row: dict[str, Any]) -> str:
    paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
    return str(
        paper.get("source_key")
        or paper.get("pdf_path")
        or paper.get("title")
        or row.get("reviewer_lists_path")
        or ""
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary_json).expanduser().resolve()
    payload = read_json(summary_path)
    rows = [row for row in payload.get("results", []) if isinstance(row, dict) and row.get("status") == "ok"]

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
        domain = str(paper.get("domain") or "unknown")
        by_domain[domain].append(row)

    for domain_rows in by_domain.values():
        domain_rows.sort(key=stable_paper_key)

    ordered_domains = sorted(by_domain.keys(), key=lambda key: (-len(by_domain[key]), key))
    selected: list[dict[str, Any]] = []
    taken_keys: set[str] = set()

    while len(selected) < args.subset_size:
        progressed = False
        for domain in ordered_domains:
            domain_rows = by_domain[domain]
            while domain_rows:
                candidate = domain_rows.pop(0)
                key = stable_paper_key(candidate)
                if not key or key in taken_keys:
                    continue
                taken_keys.add(key)
                selected.append(candidate)
                progressed = True
                break
            if len(selected) >= args.subset_size:
                break
        if not progressed:
            break

    result = {
        "status": "ok",
        "summary_json": str(summary_path),
        "subset_size": len(selected),
        "requested_subset_size": args.subset_size,
        "results": selected,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
