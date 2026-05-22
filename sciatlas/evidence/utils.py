"""Local input-parsing helpers for grounding."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def extract_text_from_pdf(pdf_path: str | Path) -> str | None:
    """Extract plain text from a local PDF file."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            parts: list[str] = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    parts.append(page_text)
            return "\n".join(parts)
    except Exception:
        return None
