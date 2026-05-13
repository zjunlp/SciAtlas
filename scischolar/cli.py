#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

from .core.common import (
    DEFAULT_ENV_PATH,
    DEFAULT_RUN_ROOT,
    normalize_whitespace,
    read_json,
    resolve_run_dir,
    write_json,
    write_text,
)
from .core.schemas import (
    SUPPORTED_TASK_TYPES,
    TASK_AUTHOR_PROFILE,
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_RELATED_AUTHORS,
    TASK_TOPIC_TREND_REVIEW,
    SciScholarRequest,
)
from .renderers.markdown import render_response_markdown
from .tasks.dispatcher import execute_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SciScholar workflows and emit JSON + Markdown results.")
    parser.add_argument("--task-type", choices=SUPPORTED_TASK_TYPES, default=None, help="Task type to run.")
    parser.add_argument(
        "--idea-text",
        default=None,
        help=f"Idea text input for {TASK_GROUNDED_REVIEW} or {TASK_RELATED_AUTHORS}.",
    )
    parser.add_argument(
        "--pdf-path",
        default=None,
        help=f"PDF input for {TASK_GROUNDED_REVIEW} or {TASK_RELATED_AUTHORS}.",
    )
    parser.add_argument(
        "--topic-text",
        default=None,
        help=f"Topic text input for {TASK_TOPIC_TREND_REVIEW} or {TASK_IDEA_GENERATION}.",
    )
    parser.add_argument("--author-name", default=None, help=f"Author name input for {TASK_AUTHOR_PROFILE}.")
    parser.add_argument("--params-file", default=None, help="Path to a JSON file with task params overrides.")
    parser.add_argument("--params-json", default=None, help="Inline JSON object for task params overrides.")
    parser.add_argument("--api-timeout-default", type=float, default=None, help="Default SciScholar API read timeout in seconds.")
    parser.add_argument("--api-timeout-search", type=float, default=None, help="Read timeout in seconds for /v1/search.")
    parser.add_argument(
        "--api-timeout-authors-related",
        type=float,
        default=None,
        help="Read timeout in seconds for /v1/authors/related.",
    )
    parser.add_argument(
        "--api-timeout-authors-papers",
        type=float,
        default=None,
        help="Read timeout in seconds for /v1/authors/papers.",
    )
    parser.add_argument(
        "--api-timeout-support-papers",
        type=float,
        default=None,
        help="Read timeout in seconds for /v1/authors/support-papers.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_RUN_ROOT), help="Root folder for SciScholar runs.")
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to the SciScholar .env file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print final JSON to stdout.")
    return parser


def _load_optional_json(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    payload = read_json(Path(path_value).expanduser().resolve())
    return payload


def _parse_inline_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--params-json must decode to a JSON object")
    return payload


def _build_request_from_args(args: argparse.Namespace) -> SciScholarRequest:
    if not args.task_type:
        raise ValueError("--task-type is required")

    input_payload: dict[str, Any] = {}
    if args.task_type in {TASK_GROUNDED_REVIEW, TASK_RELATED_AUTHORS}:
        if bool(args.idea_text) == bool(args.pdf_path):
            raise ValueError(f"{TASK_GROUNDED_REVIEW} and {TASK_RELATED_AUTHORS} require exactly one of --idea-text or --pdf-path")
        if args.idea_text:
            input_payload["idea_text"] = args.idea_text
        else:
            input_payload["pdf_path"] = str(Path(args.pdf_path).expanduser().resolve())
    elif args.task_type == TASK_TOPIC_TREND_REVIEW:
        if not args.topic_text:
            raise ValueError(f"{TASK_TOPIC_TREND_REVIEW} requires --topic-text")
        input_payload["topic_text"] = args.topic_text
    elif args.task_type == TASK_AUTHOR_PROFILE:
        if not args.author_name:
            raise ValueError(f"{TASK_AUTHOR_PROFILE} requires --author-name")
        input_payload["author_name"] = args.author_name
    elif args.task_type == TASK_IDEA_GENERATION:
        if not args.topic_text:
            raise ValueError(f"{TASK_IDEA_GENERATION} requires --topic-text")
        input_payload["topic_text"] = args.topic_text

    params = {}
    params.update(_load_optional_json(args.params_file))
    params.update(_parse_inline_json(args.params_json))
    cli_timeout_params = {
        "api_timeout_default": "scischolar_api_timeout_default",
        "api_timeout_search": "scischolar_api_timeout_search",
        "api_timeout_authors_related": "scischolar_api_timeout_authors_related",
        "api_timeout_authors_papers": "scischolar_api_timeout_authors_papers",
        "api_timeout_support_papers": "scischolar_api_timeout_support_papers",
    }
    for arg_name, param_name in cli_timeout_params.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            params[param_name] = value
    return SciScholarRequest(
        task_type=args.task_type,
        input_payload=input_payload,
        params=params,
        output_root=Path(args.output_root).expanduser().resolve(),
        env_path=Path(args.env).expanduser().resolve(),
        run_id=args.run_id,
    )


def _build_input_summary_for_run_id(request: SciScholarRequest) -> str:
    for key in ("idea_text", "topic_text", "author_name", "pdf_path"):
        value = normalize_whitespace(request.input_payload.get(key))
        if value:
            return value
    return request.task_type



# ---------------------------------------------------------------------------
# Frontend snapshot extraction
# ---------------------------------------------------------------------------

RAW_REPORT_SECTION_KEYWORDS = (
    "raw json",
    "raw response",
    "full json",
    "request json",
    "response json",
    "debug",
    "metadata",
    "api response",
    "reproducibility record",
    "原始 json",
    "原始响应",
    "请求 json",
    "响应 json",
    "调试",
    "元数据",
    "复现实验",
    "复现记录",
)

CHANNEL_ALIASES = {
    "literature-review": "literature-review",
    "literature review": "literature-review",
    "review": "literature-review",
    "idea-grounding": "idea-grounding",
    "idea grounding": "idea-grounding",
    "grounded review": "idea-grounding",
    "grounded-review": "idea-grounding",
    "idea-evaluate": "idea-evaluate",
    "idea-evaluation": "idea-evaluate",
    "idea evaluation": "idea-evaluate",
    "idea-generate": "idea-generate",
    "idea-generation": "idea-generate",
    "idea generation": "idea-generate",
    "trend-report": "trend-report",
    "trend report": "trend-report",
    "research trend": "trend-report",
    "topic trend": "trend-report",
    "researcher-review": "researcher-review",
    "researcher review": "researcher-review",
    "author-profile": "researcher-review",
    "author profile": "researcher-review",
    "related-authors": "related-authors",
    "related authors": "related-authors",
    "paper-search": "paper-search",
    "search-papers": "paper-search",
    "paper search": "paper-search",
}

TASK_TYPE_TO_CHANNEL = {
    TASK_GROUNDED_REVIEW: "idea-grounding",
    TASK_RELATED_AUTHORS: "related-authors",
    TASK_TOPIC_TREND_REVIEW: "trend-report",
    TASK_AUTHOR_PROFILE: "researcher-review",
    TASK_IDEA_GENERATION: "idea-generate",
}

CHANNEL_PROFILES: dict[str, dict[str, Any]] = {
    "literature-review": {
        "title": "Literature Review Snapshot",
        "subtitle": "Review-ready structure extracted from the full Markdown report.",
        "purpose": "Organize the core paper pool into background, topic structure, timeline, representative papers, and writing suggestions.",
        "priority_sections": (
            ("Review Guide", ("review guide", "guide", "综述指南", "阅读指南"), 2),
            ("Topic Structure", ("topic structure", "keyword profile", "topic cues", "主题结构", "关键词"), 8),
            ("Timeline View", ("timeline view", "timeline", "时间线"), 8),
            ("Representative Papers", ("representative papers", "representative works", "代表论文", "代表工作"), 8),
            ("Writing Suggestions", ("writing suggestions", "suggestions", "写作建议", "建议"), 5),
            ("Related Authors", ("related authors", "auxiliary authors", "相关作者"), 5),
            ("Core Paper Pool", ("core paper pool", "paper pool", "核心论文池"), 8),
        ),
    },
    "idea-grounding": {
        "title": "Idea Grounding Snapshot",
        "subtitle": "Evidence, nearest prior work, and positioning signals for the submitted idea.",
        "purpose": "Ground the idea against retrieved literature and expose similarity, difference, and potential gaps.",
        "priority_sections": (
            ("Grounding Summary", ("grounding summary", "idea grounding", "grounding", "定位摘要", "idea 定位"), 4),
            ("Closest Prior Work", ("closest prior work", "prior work", "related work", "similar work", "相关工作", "相近工作"), 8),
            ("Grounding Evidence", ("grounding evidence", "evidence", "supporting evidence", "证据", "支撑证据"), 8),
            ("Potential Gap", ("potential gap", "research gap", "differentiation", "positioning", "差异", "研究空白", "定位"), 6),
            ("Risks and Weaknesses", ("risk", "weakness", "limitation", "风险", "不足", "局限"), 5),
            ("Suggested Revision", ("suggested revision", "recommendation", "revision", "建议", "修改建议"), 5),
        ),
    },
    "idea-evaluate": {
        "title": "Idea Evaluation Snapshot",
        "subtitle": "Novelty, feasibility, risk, and revision cues for idea assessment.",
        "purpose": "Evaluate whether the idea is novel, feasible, well-grounded, and worth developing.",
        "priority_sections": (
            ("Evaluation Summary", ("evaluation summary", "overall assessment", "verdict", "评价摘要", "总体评价"), 4),
            ("Novelty Signals", ("novelty", "novelty signals", "创新性"), 6),
            ("Feasibility Evidence", ("feasibility", "feasibility evidence", "可行性"), 6),
            ("Risks and Weaknesses", ("risk", "weakness", "limitation", "风险", "不足", "局限"), 6),
            ("Suggested Revision", ("suggested revision", "recommendation", "revision", "修改建议", "建议"), 5),
            ("Representative Evidence", ("representative evidence", "evidence", "supporting papers", "证据", "支撑论文"), 7),
        ),
    },
    "idea-generate": {
        "title": "Idea Generation Snapshot",
        "subtitle": "Generated idea seeds and evidence-backed research directions.",
        "purpose": "Convert retrieved literature and topic evidence into candidate research ideas.",
        "priority_sections": (
            ("Generated Idea Seeds", ("idea seeds", "generated ideas", "idea candidates", "research ideas", "生成想法", "候选 idea"), 8),
            ("Research Questions", ("research questions", "questions", "研究问题"), 6),
            ("Evidence Basis", ("evidence basis", "supporting evidence", "evidence", "证据基础", "支撑证据"), 7),
            ("Expected Contributions", ("contribution", "expected contribution", "预期贡献", "贡献"), 5),
            ("Risks and Feasibility", ("risk", "feasibility", "limitation", "风险", "可行性"), 5),
            ("Possible Directions", ("possible directions", "future directions", "opportunities", "可能方向", "机会"), 6),
        ),
    },
    "trend-report": {
        "title": "Research Trend Snapshot",
        "subtitle": "Temporal evolution, topic signals, and future opportunities.",
        "purpose": "Summarize how the field evolves and identify emerging research directions.",
        "priority_sections": (
            ("Trend Signals", ("trend signals", "trend", "趋势信号", "趋势"), 7),
            ("Timeline View", ("timeline view", "timeline", "时间线"), 8),
            ("Emerging Directions", ("emerging directions", "future directions", "opportunities", "新兴方向", "未来方向", "机会"), 7),
            ("Representative Papers", ("representative papers", "representative works", "代表论文", "代表工作"), 8),
            ("Topic Structure", ("topic structure", "keyword profile", "topic cues", "主题结构", "关键词"), 7),
            ("Limitations", ("limitations", "challenge", "gap", "局限", "挑战", "空白"), 5),
        ),
    },
    "researcher-review": {
        "title": "Researcher Review Snapshot",
        "subtitle": "Research profile, representative works, trajectory, and collaboration signals.",
        "purpose": "Profile a scholar's research direction and evidence from representative works.",
        "priority_sections": (
            ("Research Profile", ("research profile", "author profile", "researcher profile", "研究画像", "作者画像"), 5),
            ("Representative Works", ("representative works", "representative papers", "author papers", "代表工作", "代表论文"), 8),
            ("Research Trajectory", ("research trajectory", "timeline", "evolution", "研究轨迹", "时间线"), 7),
            ("Collaboration Signals", ("collaboration", "coauthor", "合作", "合作者"), 6),
            ("Support Papers", ("support papers", "supporting papers", "支撑论文"), 7),
            ("Research Topics", ("research topics", "topic profile", "关键词", "研究主题"), 6),
        ),
    },
    "related-authors": {
        "title": "Related Authors Snapshot",
        "subtitle": "Candidate collaborators/authors and evidence for relevance.",
        "purpose": "Find relevant authors based on seed idea or paper evidence.",
        "priority_sections": (
            ("Candidate Authors", ("candidate authors", "related authors", "recommended authors", "候选作者", "相关作者"), 8),
            ("Why They Match", ("why they match", "match rationale", "author signals", "匹配理由", "相关性"), 6),
            ("Representative Works", ("representative works", "author papers", "support papers", "代表工作", "支撑论文"), 7),
            ("Collaboration Signals", ("collaboration", "coauthor", "合作", "合作者"), 5),
            ("Related Authors", ("related authors", "相关作者"), 8),
        ),
    },
    "paper-search": {
        "title": "Paper Search Snapshot",
        "subtitle": "Top papers, retrieval settings, and representative evidence.",
        "purpose": "Return the most relevant paper pool for the query.",
        "priority_sections": (
            ("Core Paper Pool", ("core paper pool", "top papers", "paper pool", "核心论文池", "论文列表"), 10),
            ("Representative Papers", ("representative papers", "representative works", "代表论文"), 8),
            ("Topic Structure", ("topic structure", "keyword profile", "topic cues", "关键词"), 6),
            ("Retrieval Notes", ("retrieval notes", "retrieval options", "检索设置", "检索说明"), 5),
        ),
    },
}


def _canonical_channel(task_type: str, markdown: str) -> str:
    """Infer the downstream channel from task_type and report heading."""
    for line in markdown.splitlines()[:12]:
        match = re.search(r"Downstream Channel Report:\s*([A-Za-z0-9_\- ]+)", line, flags=re.I)
        if match:
            raw = match.group(1).strip().lower()
            return CHANNEL_ALIASES.get(raw, raw.replace(" ", "-"))

    task_l = str(task_type or "").strip().lower()
    if task_l in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[task_l]
    if task_type in TASK_TYPE_TO_CHANNEL:
        return TASK_TYPE_TO_CHANNEL[task_type]
    for alias, channel in CHANNEL_ALIASES.items():
        if alias in task_l:
            return channel
    return TASK_TYPE_TO_CHANNEL.get(task_type, "paper-search")


def _strip_heading_number(title: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*[\.\)]?\s*", "", title).strip()


def _remove_fenced_blocks(markdown: str) -> str:
    """Remove fenced code/JSON blocks before creating a terminal-friendly snapshot."""
    return re.sub(r"```[\s\S]*?```", "", markdown, flags=re.MULTILINE)


def _is_raw_report_section(title: str) -> bool:
    title_l = _strip_heading_number(title).lower()
    return any(keyword in title_l for keyword in RAW_REPORT_SECTION_KEYWORDS)


def _looks_like_json_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in {"{", "}", "[", "]", "},", "],"}:
        return True
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        return True
    if re.match(r'^"[^"]+"\s*:\s*', stripped):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:\s*(\{|\[)", stripped):
        return True
    return False


def _parse_markdown_sections(markdown: str) -> list[dict[str, Any]]:
    """Parse markdown into heading-based sections."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if current is not None:
                sections.append(current)
            raw_title = match.group(2).strip()
            current = {
                "level": len(match.group(1)),
                "title": _strip_heading_number(raw_title),
                "raw_title": raw_title,
                "body": [],
            }
        elif current is not None:
            current["body"].append(line)

    if current is not None:
        sections.append(current)

    return sections


def _split_markdown_table(body_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Return markdown table headers and rows if a table is present."""
    table_lines = [line.strip() for line in body_lines if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(table_lines) < 2:
        return [], []

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in table_lines[2:]:
        if re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", line):
            continue
        cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2:
            rows.append(cells)
    return header, rows


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = normalize_whitespace(text)
    return text


def _table_rows_to_items(
    title: str,
    body_lines: list[str],
    *,
    max_items: int,
) -> list[str]:
    headers, rows = _split_markdown_table(body_lines)
    if not headers or not rows:
        return []

    normalized_headers = [h.strip().lower() for h in headers]
    items: list[str] = []

    for row in rows[:max_items]:
        values = {normalized_headers[i]: row[i] for i in range(min(len(normalized_headers), len(row)))}

        rank = values.get("rank") or values.get("#") or values.get("序号") or values.get("排名")
        paper_title = values.get("title") or values.get("paper") or values.get("论文") or values.get("标题")
        author = values.get("author") or values.get("name") or values.get("作者")
        year = values.get("year") or values.get("年份")
        score = values.get("score") or values.get("分数")
        cites = values.get("cites") or values.get("citations") or values.get("引用")
        works = values.get("works") or values.get("作品")
        hindex = values.get("h-index") or values.get("h index")

        prefix = f"{rank}. " if rank else "• "
        if paper_title:
            details = []
            if year:
                details.append(str(year))
            if score:
                details.append(f"Score={score}")
            if cites:
                details.append(f"Cites={cites}")
            suffix = f" ({', '.join(details)})" if details else ""
            items.append("  " + prefix + _clean_inline_markdown(str(paper_title))[:180] + suffix)
        elif author:
            details = []
            if score:
                details.append(f"Score={score}")
            if works:
                details.append(f"Works={works}")
            if cites:
                details.append(f"Cites={cites}")
            if hindex:
                details.append(f"H-index={hindex}")
            suffix = f" ({', '.join(details)})" if details else ""
            items.append("  " + prefix + _clean_inline_markdown(str(author))[:160] + suffix)
        else:
            compact = " | ".join(_clean_inline_markdown(cell) for cell in row[:4] if cell.strip())
            if compact:
                items.append("  " + prefix + compact[:220])

    return items


def _extract_numbered_items(body_lines: list[str], *, max_items: int) -> list[str]:
    items: list[str] = []
    for raw in body_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        match = re.match(r"^(\d+)[\.\)]\s+(.*)$", stripped)
        if not match:
            continue
        content = _clean_inline_markdown(match.group(2))
        if content:
            items.append(f"  {match.group(1)}. {content[:240]}")
        if len(items) >= max_items:
            break
    return items


def _extract_bullet_items(body_lines: list[str], *, max_items: int) -> list[str]:
    items: list[str] = []
    for raw in body_lines:
        stripped = raw.strip()
        if not stripped or _looks_like_json_line(stripped):
            continue
        if stripped.startswith(("-", "*", "•")):
            content = _clean_inline_markdown(stripped.lstrip("-*• "))
            if content:
                items.append("  • " + content[:240])
        if len(items) >= max_items:
            break
    return items


def _extract_topic_profile(body_lines: list[str], *, max_items: int) -> list[str]:
    """Special compact rendering for Topic Structure and Keyword Profile."""
    items: list[str] = []
    joined = "\n".join(body_lines)

    cues_match = re.search(r"High-frequency topic cues.*?:\s*(.+)", joined, flags=re.I)
    if cues_match:
        cues = _clean_inline_markdown(cues_match.group(1))
        items.append("  • Topic cues: " + cues[:300])

    section_items = _extract_numbered_items(body_lines, max_items=max_items)
    for item in section_items:
        if len(items) >= max_items:
            break
        items.append(item)

    if not items:
        items = _extract_bullet_items(body_lines, max_items=max_items)

    return items[:max_items]


def _extract_timeline_items(body_lines: list[str], *, max_items: int) -> list[str]:
    headers, rows = _split_markdown_table(body_lines)
    if not headers or not rows:
        return _extract_bullet_items(body_lines, max_items=max_items)

    headers_l = [h.lower() for h in headers]
    items: list[str] = []
    for row in rows[:max_items]:
        year = row[0] if row else ""
        papers = row[1] if len(row) > 1 else ""
        if "year" in headers_l[0] or "年份" in headers_l[0]:
            item = f"  • {year}: {_clean_inline_markdown(papers)[:260]}"
        else:
            item = "  • " + " | ".join(_clean_inline_markdown(cell) for cell in row[:3])
        items.append(item)
    return items


def _compact_markdown_lines(
    title: str,
    body_lines: list[str],
    *,
    max_lines: int = 7,
) -> list[str]:
    """Keep task-relevant bullets/tables/plain text and remove raw dumps."""
    title_l = title.lower()

    if "topic structure" in title_l or "keyword profile" in title_l or "主题结构" in title_l:
        items = _extract_topic_profile(body_lines, max_items=max_lines)
        if items:
            return items

    if "timeline" in title_l or "时间线" in title_l:
        items = _extract_timeline_items(body_lines, max_items=max_lines)
        if items:
            return items

    table_items = _table_rows_to_items(title, body_lines, max_items=max_lines)
    if table_items:
        return table_items

    numbered_items = _extract_numbered_items(body_lines, max_items=max_lines)
    bullet_items = _extract_bullet_items(body_lines, max_items=max_lines)

    if numbered_items and len(numbered_items) >= max(2, len(bullet_items)):
        return numbered_items[:max_lines]
    if bullet_items:
        return bullet_items[:max_lines]

    compact: list[str] = []
    for raw in body_lines:
        stripped = raw.strip()
        if not stripped or _looks_like_json_line(stripped):
            continue
        if stripped.lower().startswith(("request.json", "response.json", "result.json", "metadata.json")):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        cleaned = _clean_inline_markdown(stripped)
        if cleaned and len(cleaned) <= 300:
            compact.append("  • " + cleaned[:260])
        if len(compact) >= max_lines:
            break

    return compact


def _section_match_score(section_title: str, aliases: tuple[str, ...], priority_index: int) -> int:
    title_l = section_title.lower()
    score = 0
    for alias in aliases:
        alias_l = alias.lower()
        if title_l == alias_l:
            score = max(score, 1000 - priority_index)
        elif alias_l in title_l:
            score = max(score, 900 - priority_index)
        elif title_l in alias_l:
            score = max(score, 700 - priority_index)
    return score


def _find_best_section(
    sections: list[dict[str, Any]],
    aliases: tuple[str, ...],
    used_indices: set[int],
    priority_index: int,
) -> tuple[int, dict[str, Any], int] | None:
    best: tuple[int, dict[str, Any], int] | None = None
    for idx, section in enumerate(sections):
        if idx in used_indices:
            continue
        title = str(section.get("title") or "")
        if not title or _is_raw_report_section(title):
            continue
        score = _section_match_score(title, aliases, priority_index)
        if score <= 0:
            body_l = "\n".join(str(line) for line in section.get("body", [])[:8]).lower()
            for alias_idx, alias in enumerate(aliases):
                if alias.lower() in body_l:
                    score = max(score, 300 - priority_index - alias_idx)
        if score <= 0:
            continue
        if best is None or score > best[2]:
            best = (idx, section, score)
    return best


def _build_metrics_card(markdown: str) -> dict[str, Any] | None:
    """Extract lightweight run metadata from Task Overview when available."""
    items: list[str] = []
    for pattern, label in (
        (r"Generated at:\s*`?([^`\n]+)`?", "Generated at"),
        (r"Query text:\s*([^\n]+)", "Query"),
        (r"Call status:\s*`?([^`;]+)`?.*?HTTP status:\s*`?([^`;]+)`?.*?elapsed:\s*`?([^`;]+)`?", "Call"),
        (r"Retrieval options:\s*`?([^`\n]+)`?", "Retrieval options"),
        (r"Keyword anchors:\s*([^\n]+)", "Keyword anchors"),
    ):
        match = re.search(pattern, markdown, flags=re.I)
        if not match:
            continue
        if label == "Call":
            items.append(f"  • Call: status={_clean_inline_markdown(match.group(1))}, HTTP={_clean_inline_markdown(match.group(2))}, elapsed={_clean_inline_markdown(match.group(3))}s")
        else:
            items.append(f"  • {label}: {_clean_inline_markdown(match.group(1))[:260]}")
    if not items:
        return None
    return {
        "title": "Run Context",
        "kind": "metadata",
        "items": items[:5],
        "source_heading": "Task Overview",
    }


def _fallback_report_sections(markdown: str, *, max_lines: int = 8) -> list[dict[str, Any]]:
    """Fallback when result.md lacks stable task-specific headings."""
    cleaned = _remove_fenced_blocks(markdown)
    items = _compact_markdown_lines("Report Highlights", cleaned.splitlines(), max_lines=max_lines)
    if not items:
        return []
    return [
        {
            "title": "Report Highlights",
            "kind": "fallback",
            "items": items,
            "source_heading": "report.md",
        }
    ]


def build_frontend_snapshot_from_markdown(
    *,
    task_type: str,
    markdown: str,
    markdown_path: Path,
    max_sections: int = 6,
    max_lines_per_section: int = 8,
) -> dict[str, Any]:
    """Build a structured frontend-facing summary from result.md.

    The snapshot is report-driven rather than JSON-driven. It removes raw JSON,
    recognizes downstream channels, extracts high-value task-specific sections,
    and returns both card data and a Markdown summary for frontend display.
    """
    channel = _canonical_channel(task_type, markdown)
    profile = CHANNEL_PROFILES.get(channel, CHANNEL_PROFILES["paper-search"])

    cleaned = _remove_fenced_blocks(markdown)
    sections = _parse_markdown_sections(cleaned)

    cards: list[dict[str, Any]] = []
    metrics_card = _build_metrics_card(cleaned)
    if metrics_card:
        cards.append(metrics_card)

    used_indices: set[int] = set()
    for priority_index, (display_title, aliases, max_items) in enumerate(profile["priority_sections"]):
        if len(cards) >= max_sections + (1 if metrics_card else 0):
            break

        found = _find_best_section(sections, tuple(aliases), used_indices, priority_index)
        if found is None:
            continue

        idx, section, _score = found
        body = list(section.get("body") or [])
        source_title = str(section.get("title") or display_title)
        items = _compact_markdown_lines(source_title, body, max_lines=max_items)
        if not items:
            continue

        cards.append(
            {
                "title": display_title,
                "kind": "section",
                "items": items,
                "source_heading": source_title,
            }
        )
        used_indices.add(idx)

    if not cards or (len(cards) == 1 and cards[0].get("kind") == "metadata"):
        cards.extend(_fallback_report_sections(cleaned, max_lines=max_lines_per_section))

    if metrics_card:
        final_cards = [cards[0]] + cards[1 : max_sections + 1]
    else:
        final_cards = cards[:max_sections]

    title = str(profile["title"])
    subtitle = str(profile["subtitle"])
    purpose = str(profile["purpose"])

    summary_lines = [f"## {title}", "", f"**Purpose.** {purpose}", "", subtitle]
    for card in final_cards:
        summary_lines.append("")
        summary_lines.append(f"### {card['title']}")
        summary_lines.extend(card["items"])

    return {
        "title": title,
        "channel": channel,
        "source": "result.md",
        "markdown_path": str(markdown_path.resolve()),
        "subtitle": subtitle,
        "purpose": purpose,
        "cards": final_cards,
        "sections": final_cards,
        "summary_markdown": "\n".join(summary_lines).strip(),
    }



def main() -> int:
    args = build_parser().parse_args()
    request = _build_request_from_args(args)
    run_dir = resolve_run_dir(request.output_root, request.task_type, request.run_id, _build_input_summary_for_run_id(request))
    request_path = run_dir / "request.json"
    result_path = run_dir / "result.json"
    markdown_path = run_dir / "result.md"

    request_payload = {
        "task_type": request.task_type,
        "input": request.input_payload,
        "params": request.params,
        "output_root": str(request.output_root),
        "env": str(request.env_path),
        "run_id": request.run_id,
    }
    write_json(request_path, request_payload)

    try:
        response = execute_request(request, run_dir)
        response.setdefault("artifacts", {})
        response["artifacts"]["json_path"] = str(result_path.resolve())
        response["artifacts"]["markdown_path"] = str(markdown_path.resolve())
        markdown = render_response_markdown(response)
        frontend_snapshot = build_frontend_snapshot_from_markdown(
            task_type=request.task_type,
            markdown=markdown,
            markdown_path=markdown_path,
        )
        if frontend_snapshot.get("sections"):
            response["frontend_snapshot"] = frontend_snapshot
            response["artifacts"]["frontend_snapshot_source"] = str(markdown_path.resolve())
        write_json(result_path, response)
        write_text(markdown_path, markdown)
    except Exception as exc:
        error_payload = {
            "status": "error",
            "task_type": request.task_type,
            "input_summary": request.input_payload,
            "params_effective": request.params,
            "artifacts": {
                "request_path": str(request_path.resolve()),
                "json_path": str(result_path.resolve()),
                "markdown_path": str(markdown_path.resolve()),
            },
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir.resolve()),
        }
        error_markdown = f"# Error\n\n`{exc.__class__.__name__}`: {exc}\n"
        error_payload["frontend_snapshot"] = build_frontend_snapshot_from_markdown(
            task_type=request.task_type,
            markdown=error_markdown,
            markdown_path=markdown_path,
        )
        write_json(result_path, error_payload)
        write_text(markdown_path, error_markdown)
        response = error_payload

    if args.pretty:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(response, ensure_ascii=False))
    return 0 if response.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
