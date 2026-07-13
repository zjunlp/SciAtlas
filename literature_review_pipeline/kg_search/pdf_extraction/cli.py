from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extractor import DEFAULT_GROBID_BASE_URL, extract_pdf_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract title, abstract, body and references from a PDF via GROBID.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("--base-url", default=DEFAULT_GROBID_BASE_URL, help="GROBID base URL.")
    parser.add_argument("--start-page", type=int, default=None, help="First page to process, 1-based.")
    parser.add_argument("--consolidate-citations", choices=["0", "1", "2"], default="0")
    parser.add_argument("--include-raw-citations", choices=["0", "1"], default="1")
    parser.add_argument(
        "--segment-sentences",
        choices=["0", "1"],
        default="0",
        help="Ask GROBID to segment body text into sentences so citation contexts can be returned at sentence level.",
    )
    parser.add_argument(
        "--preserve-bibr-refs",
        action="store_true",
        help="Preserve bibliography refs in body paragraphs as <ref type=\"bibr\">...</ref> instead of flattening them into plain text.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = extract_pdf_to_dict(
        args.pdf,
        base_url=args.base_url,
        start_page=args.start_page,
        consolidate_citations=args.consolidate_citations,
        include_raw_citations=args.include_raw_citations,
        segment_sentences=args.segment_sentences,
        preserve_bibr_refs=args.preserve_bibr_refs,
    )
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
