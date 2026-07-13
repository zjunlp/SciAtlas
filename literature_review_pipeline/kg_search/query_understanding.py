from __future__ import annotations

from .config import SearchConfig
from .llm_client import OpenAICompatibleClient
from .models import ExtractedKeyword, ExtractedTitle, QueryUnderstandingResult
from .text_utils import clean_text, normalize_text

KEYWORD_PROMPT = """You are an expert assistant that extracts high-level academic keywords for knowledge graph construction.

Goal:
From the following research idea text, extract a small set of canonical, high-level keywords that represent the main research topics, tasks, methods, or application areas.

Requirements:
- Extract only 3-8 keywords.
- Prefer reusable academic concepts suitable as knowledge graph entities.
- Use concise English noun phrases, usually 1-4 words.
- Avoid system names, long descriptive fragments, marketing wording, and paper-specific phrases.
- Also score each keyword's relevance to the idea text on a 1-10 integer scale.

Idea text:
{idea_text}

Return ONLY a JSON object:
{{"keywords": ["keyword1", "keyword2"], "scores": [8, 7]}}
"""

TITLE_PROMPT = """You are an expert assistant for academic paper retrieval.

Goal:
Extract only explicit or highly certain English paper titles mentioned in the idea text.

Requirements:
- Do not paraphrase.
- Do not guess missing titles.
- At most 5 titles.
- confidence must be a float from 0 to 1.

Idea text:
{idea_text}

Return ONLY a JSON object:
{{"titles": [{{"title": "Attention Is All You Need", "confidence": 0.96}}]}}
"""


class QueryUnderstandingService:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.llm_client = OpenAICompatibleClient(
            api_url=config.llm_api_url,
            model=config.llm_model,
            api_key=config.llm_api_key or "",
            timeout_s=config.llm_timeout_s,
        )

    def understand(self, idea_text: str) -> QueryUnderstandingResult:
        cleaned = clean_text(idea_text)
        titles, title_source = self.extract_titles(cleaned)
        if self.config.enable_keyword_nodes:
            keywords, keyword_source = self.extract_keywords(cleaned)
            keywords = self._filter_keywords(keywords, titles)
        else:
            keywords, keyword_source = [], "disabled"
        return QueryUnderstandingResult(
            cleaned_text=cleaned,
            keywords=keywords,
            titles=titles,
            keyword_source=keyword_source,
            title_source=title_source,
            source_type="idea_text",
        )

    def understand_pdf(
        self,
        *,
        pdf_title: str,
        abstract: str,
        reference_titles: list[str],
    ) -> QueryUnderstandingResult:
        cleaned = clean_text(abstract)
        if self.config.enable_keyword_nodes:
            keywords, keyword_source = self.extract_keywords(cleaned)
        else:
            keywords, keyword_source = [], "disabled"
        titles = self._build_reference_titles(reference_titles)
        keywords = self._filter_keywords(keywords, titles)
        return QueryUnderstandingResult(
            cleaned_text=cleaned,
            keywords=keywords,
            titles=titles,
            keyword_source=keyword_source,
            title_source="pdf_references",
            source_type="pdf",
            source_title=clean_text(pdf_title) or None,
            reference_titles=[item.title for item in titles],
        )

    def extract_keywords(self, idea_text: str) -> tuple[list[ExtractedKeyword], str]:
        parsed = self.llm_client.chat_json(KEYWORD_PROMPT.format(idea_text=idea_text))
        keywords = self._parse_keywords(parsed)
        if not keywords:
            raise RuntimeError("LLM keyword extraction returned no valid keywords.")
        return keywords[: self.config.max_keywords_from_llm], "llm"

    def extract_titles(self, idea_text: str) -> tuple[list[ExtractedTitle], str]:
        parsed = self.llm_client.chat_json(TITLE_PROMPT.format(idea_text=idea_text))
        titles = self._parse_titles(parsed)
        return titles[: self.config.max_titles_from_llm], "llm"

    def _parse_keywords(self, parsed: dict) -> list[ExtractedKeyword]:
        keywords = parsed.get("keywords")
        scores = parsed.get("scores")
        if not isinstance(keywords, list) or not isinstance(scores, list):
            raise RuntimeError("LLM keyword extraction payload must contain list fields: keywords, scores.")
        items: list[ExtractedKeyword] = []
        for keyword, score in zip(keywords, scores):
            if not isinstance(keyword, str):
                continue
            try:
                score_value = int(score)
            except (TypeError, ValueError):
                continue
            keyword = clean_text(keyword)
            if keyword:
                items.append(ExtractedKeyword(text=keyword, score=max(1, min(10, score_value))))
        return items

    def _parse_titles(self, parsed: dict) -> list[ExtractedTitle]:
        titles = parsed.get("titles")
        if not isinstance(titles, list):
            raise RuntimeError("LLM title extraction payload must contain a list field: titles.")
        items: list[ExtractedTitle] = []
        for item in titles:
            if not isinstance(item, dict):
                continue
            title = clean_text(str(item.get("title") or "")).strip().lstrip("/")
            if not title:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            items.append(ExtractedTitle(title=title, confidence=max(0.0, min(1.0, confidence))))
        return items

    def _filter_keywords(
        self,
        keywords: list[ExtractedKeyword],
        titles: list[ExtractedTitle],
    ) -> list[ExtractedKeyword]:
        if not keywords:
            return keywords
        title_norms = [normalize_text(item.title) for item in titles if normalize_text(item.title)]
        filtered: list[ExtractedKeyword] = []
        banned_tokens = {"i", "we", "you", "my", "our", "your", "want", "need"}
        for keyword in keywords:
            normalized = normalize_text(keyword.text)
            if not normalized:
                continue
            tokens = normalized.split()
            if any(token in banned_tokens for token in tokens):
                continue
            if len(tokens) >= 2 and any(normalized in title_norm for title_norm in title_norms):
                continue
            if any(set(tokens).issubset(set(title_norm.split())) and len(tokens) >= 2 for title_norm in title_norms):
                continue
            filtered.append(keyword)
        return filtered

    def _build_reference_titles(self, reference_titles: list[str]) -> list[ExtractedTitle]:
        items: list[ExtractedTitle] = []
        seen: set[str] = set()
        for raw_title in reference_titles:
            title = clean_text(raw_title)
            normalized = normalize_text(title)
            if not normalized or normalized in seen:
                continue
            if len(normalized.split()) < 2 or len(normalized) < 8:
                continue
            seen.add(normalized)
            items.append(ExtractedTitle(title=title, confidence=1.0))
            if len(items) >= self.config.max_titles_from_pdf_references:
                break
        return items
