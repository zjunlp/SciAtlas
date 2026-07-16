from __future__ import annotations

import urllib.parse
from pathlib import Path

from .grobid_client import post_pdf
from .models import ExtractedPdfDocument, PdfReference
from .parser import parse_tei_document


DEFAULT_GROBID_BASE_URL = "http://127.0.0.1:8070"


def extract_pdf(
    pdf_path: str | Path,
    *,
    base_url: str = DEFAULT_GROBID_BASE_URL,
    consolidate_citations: str = "0",
    include_raw_citations: str = "1",
    segment_sentences: str = "0",
    preserve_bibr_refs: bool = False,
    start_page: int | None = None,
) -> ExtractedPdfDocument:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    fulltext_fields = {
        "includeRawCitations": include_raw_citations,
        "consolidateCitations": consolidate_citations,
        "segmentSentences": segment_sentences,
        "start": str(start_page) if start_page else None,
    }
    refs_fields = {
        "includeRawCitations": include_raw_citations,
        "consolidateCitations": consolidate_citations,
    }

    fulltext_xml = post_pdf(
        urllib.parse.urljoin(base_url, "/api/processFulltextDocument"),
        path,
        fulltext_fields,
    )
    refs_xml = post_pdf(
        urllib.parse.urljoin(base_url, "/api/processReferences"),
        path,
        refs_fields,
    )

    document = parse_tei_document(fulltext_xml, preserve_bibr_refs=preserve_bibr_refs)
    reference_document = parse_tei_document(refs_xml)
    if reference_document.references:
        document.references = _merge_references(document.references, reference_document.references)
    document.references = _filter_references(document.references)
    return document


def extract_pdf_to_dict(pdf_path: str | Path, **kwargs: object) -> dict:
    return extract_pdf(pdf_path, **kwargs).to_dict()


def _merge_references(
    fulltext_references: list[PdfReference],
    process_references: list[PdfReference],
) -> list[PdfReference]:
    if not fulltext_references:
        return process_references

    merged: list[PdfReference] = []
    fulltext_by_id = {
        reference.ref_id: reference for reference in fulltext_references if reference.ref_id
    }

    for index, reference in enumerate(process_references):
        merged_reference = reference
        fulltext_reference = fulltext_by_id.get(reference.ref_id)
        if fulltext_reference is None and index < len(fulltext_references):
            fulltext_reference = fulltext_references[index]

        if fulltext_reference is not None:
            merged_reference.contexts = list(fulltext_reference.contexts)
            if not merged_reference.ref_id:
                merged_reference.ref_id = fulltext_reference.ref_id

        merged.append(merged_reference)

    if len(fulltext_references) > len(process_references):
        merged.extend(fulltext_references[len(process_references) :])

    return merged


def _filter_references(references: list[PdfReference]) -> list[PdfReference]:
    filtered: list[PdfReference] = []
    for reference in references:
        if not reference.authors:
            continue
        if not reference.contexts:
            continue
        filtered.append(reference)
    return filtered
