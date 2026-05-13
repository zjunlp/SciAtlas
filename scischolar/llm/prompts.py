from __future__ import annotations

from ..core.common import normalize_whitespace, truncate_text


IDEA_EVALUATION_SYSTEM_PROMPT = (
    "You are an expert research idea evaluator. "
    "Evaluate the idea rigorously across multiple dimensions and return strict JSON only."
)

IDEA_EVALUATION_USER_PROMPT = """Evaluate this research idea using the grounding evidence below.

Research Idea:
{idea_text}

Grounding Evidence:
{grounding_context}

Requirements:
- Evaluate across these dimensions: Clarity, Novelty, Validity, Feasibility, Significance.
- Each dimension gets a score (0-10) and a brief reason grounded in the evidence.
- Also provide overall strengths, weaknesses, and suggestions.
- Output JSON with this schema:
{{
  "clarity": {{"score": float, "reason": str}},
  "novelty": {{"score": float, "reason": str}},
  "validity": {{"score": float, "reason": str}},
  "feasibility": {{"score": float, "reason": str}},
  "significance": {{"score": float, "reason": str}},
  "overall": {{
    "strengths": ["..."],
    "weaknesses": ["..."],
    "suggestions": ["..."],
    "summary": str,
    "recommendation": "Strong Accept | Accept | Borderline | Reject"
  }}
}}

Be critical and specific. Score honestly."""


def build_trend_prompt(papers_by_year: list[dict[str, object]], abstract_char_limit: int) -> str:
    paper_blocks: list[str] = []
    for index, paper in enumerate(papers_by_year, start=1):
        year = paper.get("year") or "Unknown"
        title = normalize_whitespace(paper.get("title")) or "Untitled"
        abstract = truncate_text(paper.get("abstract"), max_chars=abstract_char_limit)
        paper_blocks.append(f"{index}. [{year}] {title}\nAbstract: {abstract}")
    return (
        "You are an expert research trend analyst.\n"
        "Given chronologically ordered papers from one topic, summarize the research trend.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "one_sentence_summary": "...",\n'
        '  "trend_summary": "...",\n'
        '  "stage_summary": [{"period": "...", "theme": "...", "description": "..."}],\n'
        '  "methodological_shifts": ["..."],\n'
        '  "emerging_topics": ["..."],\n'
        '  "open_gaps": ["..."],\n'
        '  "future_directions": ["..."],\n'
        '  "representative_papers": [{"title": "...", "year": 2024, "why_representative": "..."}]\n'
        "}\n\n"
        "Papers:\n"
        + "\n\n".join(paper_blocks)
    )


def build_author_profile_prompt(author_name: str, papers: list[dict[str, object]]) -> str:
    paper_lines: list[str] = []
    for index, paper in enumerate(papers, start=1):
        year = paper.get("year") or "Unknown"
        citations = paper.get("citations") or 0
        title = normalize_whitespace(paper.get("title")) or "Untitled"
        abstract = truncate_text(paper.get("abstract"), max_chars=500)
        paper_lines.append(f"{index}. [{year}] {title} (citations={citations})\nAbstract: {abstract}")
    return (
        "You are an expert academic intelligence analyst.\n"
        "Summarize one researcher's major directions from their publication list.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        f'  "author_name": "{author_name}",\n'
        '  "overall_academic_profile": "...",\n'
        '  "main_research_directions": [{"theme": "...", "active_years": "...", "description": "..."}],\n'
        '  "technical_arsenal": ["..."],\n'
        '  "representative_papers": [{"title": "...", "year": 2024, "why_representative": "..."}]\n'
        "}\n\n"
        "Papers:\n"
        + "\n\n".join(paper_lines)
    )


def build_idea_generation_prompt(papers: list[dict[str, object]], idea_count: int, abstract_char_limit: int) -> str:
    paper_blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        title = normalize_whitespace(paper.get("title")) or "Untitled"
        year = paper.get("year") or "Unknown"
        abstract = truncate_text(paper.get("abstract") or "", max_chars=abstract_char_limit)
        paper_blocks.append(f"{index}. [{year}] {title}\nAbstract: {abstract}")
    return (
        "You are an expert research idea generator.\n"
        "Given the papers below, propose novel research ideas that extend, combine, or contrast this work.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "ideas": [\n'
        '    {\n'
        '      "title": "concise idea title",\n'
        '      "description": "2-3 sentence core description of the idea",\n'
        '      "novelty": "why this is novel compared to existing work",\n'
        '      "significance": "potential impact and importance",\n'
        '      "key_references": ["paper title 1", "paper title 2"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Generate exactly {idea_count} ideas.\n\n"
        "Papers:\n\n"
        + "\n\n".join(paper_blocks)
    )
