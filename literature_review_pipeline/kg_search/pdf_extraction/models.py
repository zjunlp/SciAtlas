from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class PdfReference:
    ref_id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    date: str = ""
    contexts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PdfSection:
    heading: str = ""
    paragraphs: list[str] = field(default_factory=list)
    subsections: list["PdfSection"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "paragraphs": list(self.paragraphs),
            "subsections": [section.to_dict() for section in self.subsections],
        }


@dataclass(slots=True)
class ExtractedPdfDocument:
    title: str = ""
    abstract: str = ""
    body: list[PdfSection] = field(default_factory=list)
    references: list[PdfReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "body": [section.to_dict() for section in self.body],
            "references": [asdict(reference) for reference in self.references],
        }
