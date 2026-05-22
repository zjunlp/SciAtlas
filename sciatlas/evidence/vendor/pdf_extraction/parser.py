from __future__ import annotations

import xml.etree.ElementTree as ET

from .models import ExtractedPdfDocument, PdfReference, PdfSection


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def norm(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def text_content(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return norm("".join(node.itertext()))


def body_text_content(node: ET.Element | None, *, preserve_bibr_refs: bool = False) -> str:
    if node is None:
        return ""
    if not preserve_bibr_refs:
        return text_content(node)
    return norm(_serialize_body_node(node))


def parse_tei_document(
    tei_xml: str,
    *,
    preserve_bibr_refs: bool = False,
) -> ExtractedPdfDocument:
    root = ET.fromstring(tei_xml)
    title = text_content(root.find(".//tei:teiHeader//tei:titleStmt/tei:title", NS))
    abstract = _parse_abstract(root)
    body = _parse_body_sections(root, preserve_bibr_refs=preserve_bibr_refs)
    references = _parse_references(root)
    _attach_reference_contexts(root, references)
    return ExtractedPdfDocument(
        title=title,
        abstract=abstract,
        body=body,
        references=references,
    )


def _parse_abstract(root: ET.Element) -> str:
    abstract = root.find(".//tei:profileDesc/tei:abstract", NS)
    if abstract is None:
        return ""

    paragraphs: list[str] = []
    for paragraph in abstract.findall(".//tei:p", NS):
        text = text_content(paragraph)
        if text:
            paragraphs.append(text)

    if paragraphs:
        return "\n\n".join(paragraphs)

    return text_content(abstract)


def _parse_body_sections(
    root: ET.Element,
    *,
    preserve_bibr_refs: bool = False,
) -> list[PdfSection]:
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return []

    sections = [
        _parse_div(div, preserve_bibr_refs=preserve_bibr_refs)
        for div in body.findall("./tei:div", NS)
    ]
    sections = [section for section in sections if _section_has_content(section)]
    sections = _trim_sections_to_conclusion(sections)
    if sections:
        return sections

    paragraphs: list[str] = []
    for paragraph in body.findall(".//tei:p", NS):
        text = body_text_content(paragraph, preserve_bibr_refs=preserve_bibr_refs)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        return []
    return [PdfSection(heading="", paragraphs=paragraphs)]


def _parse_div(div: ET.Element, *, preserve_bibr_refs: bool = False) -> PdfSection:
    if _is_non_body_div(div):
        return PdfSection()

    heading = text_content(div.find("./tei:head", NS))
    paragraphs: list[str] = []
    subsections: list[PdfSection] = []

    for child in list(div):
        tag = _local_name(child.tag)
        if tag == "p":
            text = body_text_content(child, preserve_bibr_refs=preserve_bibr_refs)
            if text:
                paragraphs.append(text)
        elif tag == "div":
            subsection = _parse_div(child, preserve_bibr_refs=preserve_bibr_refs)
            if _section_has_content(subsection):
                subsections.append(subsection)

    return PdfSection(heading=heading, paragraphs=paragraphs, subsections=subsections)


def _parse_references(root: ET.Element) -> list[PdfReference]:
    references: list[PdfReference] = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", NS):
        ref_id = (
            bibl.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "")
            or bibl.attrib.get("xml:id", "")
        )
        title = text_content(bibl.find("./tei:analytic/tei:title", NS))
        if not title:
            title = text_content(bibl.find("./tei:monogr/tei:title", NS))

        authors: list[str] = []
        for author in bibl.findall(".//tei:author", NS):
            author_text = text_content(author)
            if author_text:
                authors.append(author_text)

        journal = text_content(bibl.find("./tei:monogr/tei:title", NS))
        date_node = bibl.find(".//tei:date", NS)
        date = ""
        if date_node is not None:
            date = date_node.attrib.get("when", "") or text_content(date_node)

        references.append(
            PdfReference(
                ref_id=ref_id,
                title=title,
                authors=authors,
                journal=journal,
                date=date,
            )
        )
    return references


def _attach_reference_contexts(root: ET.Element, references: list[PdfReference]) -> None:
    if not references:
        return

    reference_map = {reference.ref_id: reference for reference in references if reference.ref_id}
    if not reference_map:
        return

    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return

    for paragraph in body.findall(".//tei:p", NS):
        paragraph_text = text_content(paragraph)
        if not paragraph_text:
            continue

        contexts = _collect_contexts_from_paragraph(paragraph)
        for ref_id, snippets in contexts.items():
            reference = reference_map.get(ref_id)
            if reference is None:
                continue
            for snippet in snippets:
                if snippet not in reference.contexts:
                    reference.contexts.append(snippet)


def _collect_contexts_from_paragraph(paragraph: ET.Element) -> dict[str, list[str]]:
    contexts: dict[str, list[str]] = {}
    sentences = paragraph.findall("./tei:s", NS)
    units = sentences or [paragraph]

    for unit in units:
        unit_text = text_content(unit)
        if not unit_text:
            continue

        ref_ids: list[str] = []
        for ref in unit.findall(".//tei:ref[@type='bibr']", NS):
            target = (ref.attrib.get("target", "") or "").strip()
            if not target:
                continue
            normalized_target = target[1:] if target.startswith("#") else target
            if normalized_target:
                ref_ids.append(normalized_target)

        if not ref_ids:
            continue

        for ref_id in ref_ids:
            snippets = contexts.setdefault(ref_id, [])
            if unit_text not in snippets:
                snippets.append(unit_text)

    return contexts


def _section_has_content(section: PdfSection) -> bool:
    return bool(section.heading or section.paragraphs or section.subsections)


def _local_name(tag: str) -> str:
    if "}" not in tag:
        return tag
    return tag.rsplit("}", 1)[1]


def _serialize_body_node(node: ET.Element) -> str:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)

    for child in list(node):
        if _local_name(child.tag) == "ref" and child.attrib.get("type") == "bibr":
            parts.append(_serialize_bibr_ref(child))
        else:
            parts.append(_serialize_body_node(child))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def _serialize_bibr_ref(node: ET.Element) -> str:
    attrs = ['type="bibr"']
    target = (node.attrib.get("target", "") or "").strip()
    if target:
        attrs.append(f'target="{target}"')
    return f"<ref {' '.join(attrs)}>{text_content(node)}</ref>"


def _is_non_body_div(div: ET.Element) -> bool:
    div_type = (div.attrib.get("type", "") or "").strip().lower()
    return div_type in {"references", "annex", "acknowledgment", "availability", "funding"}


def _trim_sections_to_conclusion(sections: list[PdfSection]) -> list[PdfSection]:
    trimmed: list[PdfSection] = []
    for section in sections:
        trimmed.append(section)
        if _is_conclusion_heading(section.heading):
            break
    return trimmed


def _is_conclusion_heading(heading: str) -> bool:
    normalized = norm(heading).lower().rstrip(":")
    return normalized in {"conclusion", "conclusions"}
