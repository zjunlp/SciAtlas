#!/usr/bin/env python3

from __future__ import annotations



import argparse

import json

import itertools

import os

import re

import socket

import sys

import threading

import time

import uuid

import urllib.error

import urllib.request

from datetime import datetime

from pathlib import Path

from typing import Any





DEFAULT_BASE_URL = os.environ.get("SCISCHOLAR_API_BASE_URL") or os.environ.get("SCINET_API_BASE_URL") or os.environ.get("KG2API_BASE_URL", "http://scinet.openkg.cn")

DEFAULT_API_KEY = os.environ.get("SCISCHOLAR_API_KEY") or os.environ.get("SCINET_API_KEY") or os.environ.get("KG2API_API_KEY", "")

DEFAULT_RUNS_DIR = os.environ.get("SCISCHOLAR_RUNS_DIR") or os.environ.get("SCINET_RUNS_DIR") or os.environ.get("SCISCHOLAR_SKILL_RUNS_DIR") or os.environ.get("SCINET_SKILL_RUNS_DIR") or str(Path.cwd() / "runs")





# ============================================================

# 基础工具

# ============================================================



def print_json(obj: Any) -> None:

    print(json.dumps(obj, ensure_ascii=False, indent=2))





def normalize_text(text: str) -> str:

    return re.sub(r"\s+", " ", text.strip())





def compact_text(value: Any, max_len: int = 180) -> str:

    if value is None:

        return ""

    text = str(value).replace("\n", " ").strip()

    text = re.sub(r"\s+", " ", text)

    if len(text) > max_len:

        return text[:max_len - 3] + "..."

    return text





def read_text_input(text: str | None, text_file: str | None, query: str | None = None) -> str:

    # 专家模式下 --query 优先级最高，用于精确指定主查询文本。

    if query:

        return query.strip()



    if text:

        return text.strip()



    if text_file:

        with open(text_file, "r", encoding="utf-8") as f:

            return f.read().strip()



    raise SystemExit("Please provide --query, --text, or --text-file.")





def build_url(base_url: str, endpoint: str) -> str:

    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")





def ensure_run_dir(output_dir: str, run_id: str | None, prefix: str) -> Path:

    if not run_id:

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_id = f"{ts}_{prefix}_{uuid.uuid4().hex[:8]}"



    run_dir = Path(output_dir).expanduser().resolve() / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir





def write_json(path: Path, obj: Any) -> None:

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")





def read_json(path: str | Path) -> Any:

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)





# ============================================================
# 终端交互与美化输出
# ============================================================

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_MAGENTA = "\033[35m"


def _color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def color(text: Any, code: str) -> str:
    value = str(text)
    if not _color_enabled():
        return value
    return f"{code}{value}{ANSI_RESET}"


class Spinner:
    """Small stderr spinner for long blocking backend requests."""

    def __init__(self, message: str, *, enabled: bool = True) -> None:
        self.message = message
        self.enabled = enabled and sys.stderr.isatty() and not (
            os.environ.get("SCISCHOLAR_NO_SPINNER") or os.environ.get("SCINET_NO_SPINNER")
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()
        return False

    def _spin(self) -> None:
        frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        start = time.time()
        while not self._stop.is_set():
            elapsed = int(time.time() - start)
            sys.stderr.write(f"\r{next(frames)} {self.message} waiting for {elapsed}s")
            sys.stderr.flush()
            time.sleep(0.12)


def _plain_len(value: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", str(value)))


def _pad(value: Any, width: int) -> str:
    text = str(value)
    return text + " " * max(0, width - _plain_len(text))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    clean_rows = [[compact_text(cell, 120) for cell in row] for row in rows]
    widths = []
    for i, header in enumerate(headers):
        max_cell = max((_plain_len(row[i]) for row in clean_rows), default=0)
        widths.append(min(max(_plain_len(header), max_cell), 64))

    def crop(cell: Any, width: int) -> str:
        text = compact_text(cell, width)
        if _plain_len(text) > width:
            text = text[: max(0, width - 3)] + "..."
        return text

    header_line = " | ".join(_pad(color(headers[i], ANSI_BOLD + ANSI_CYAN), widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = []
    for row in clean_rows:
        body.append(" | ".join(_pad(crop(row[i], widths[i]), widths[i]) for i in range(len(headers))))
    return "\n".join([header_line, sep_line, *body])


def _format_score(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _format_elapsed(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return str(value)



# ============================================================
# Downstream-channel specific frontend cards
# ============================================================

DOWNSTREAM_FRONTEND_COMMANDS = {
    "literature-review",
    "idea-grounding",
    "idea-evaluate",
    "idea-generate",
    "trend-report",
    "researcher-review",
}

_FRONTEND_STOPWORDS = {
    "about", "after", "again", "against", "agent", "agents", "also", "among", "based",
    "being", "between", "could", "from", "have", "into", "large", "model", "models",
    "paper", "papers", "research", "study", "system", "systems", "their", "there",
    "these", "this", "through", "using", "with", "world", "would"
}


def _frontend_compact(value: Any, limit: int = 80) -> str:
    text = "" if value is None else str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _frontend_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-" if value in (None, "") else str(value)


def _safe_read_text(path: str | None, max_chars: int = 20000) -> str:
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def _strip_md(value: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", value)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("|", " ").strip()
    return _frontend_compact(text, 180)


def _section_excerpt(report_text: str, heading_keywords: list[str], *, max_lines: int = 5) -> list[str]:
    if not report_text:
        return []

    lines = report_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("##") and any(key.lower() in line.lower() for key in heading_keywords):
            start = idx + 1
            break

    if start is None:
        return []

    result: list[str] = []
    for line in lines[start:]:
        raw = line.strip()
        if raw.startswith("## "):
            break
        if not raw or raw.startswith("|") or raw.startswith("---"):
            continue
        if raw.startswith("- "):
            raw = raw[2:].strip()
        elif re.match(r"^\d+\.\s+", raw):
            raw = re.sub(r"^\d+\.\s+", "", raw)
        elif len(raw) > 220:
            continue

        cleaned = _strip_md(raw)
        if cleaned and cleaned not in result:
            result.append(cleaned)
        if len(result) >= max_lines:
            break

    return result


def _topic_terms_from_papers(papers: list[dict[str, Any]], *, max_terms: int = 8) -> list[str]:
    text = " ".join(str(p.get("title", "")) + " " + str(p.get("abstract", "")) for p in papers[:10])
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    counts: dict[str, int] = {}
    for t in tokens:
        if t in _FRONTEND_STOPWORDS or len(t) < 4:
            continue
        counts[t] = counts.get(t, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]]


def _timeline_rows_from_papers(papers: list[dict[str, Any]], *, max_rows: int = 6) -> list[list[Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for p in papers:
        year = str(p.get("year") or "-")
        if not year or year == "-":
            continue
        cur = buckets.setdefault(year, {"count": 0, "titles": []})
        cur["count"] += 1
        if len(cur["titles"]) < 2:
            cur["titles"].append(_frontend_compact(p.get("title", ""), 70))

    rows = []
    for year in sorted(buckets.keys()):
        rows.append([year, buckets[year]["count"], "; ".join(buckets[year]["titles"])])
    return rows[:max_rows]


def _top_paper_rows(papers: list[dict[str, Any]], *, max_rows: int = 4) -> list[list[Any]]:
    rows = []
    for p in papers[:max_rows]:
        rows.append([
            p.get("rank", ""),
            p.get("title", ""),
            p.get("year", "-"),
            _frontend_score(p.get("score")),
            p.get("citations", "-"),
        ])
    return rows


def build_downstream_channel_view(
    *,
    command: str,
    payload: dict[str, Any],
    report_path: str | None,
    max_items: int,
) -> dict[str, Any] | None:
    if command not in DOWNSTREAM_FRONTEND_COMMANDS:
        return None

    papers = payload.get("papers") or []
    report_text = _safe_read_text(report_path)
    topics = _topic_terms_from_papers(papers, max_terms=8)
    evidence_rows = _top_paper_rows(papers, max_rows=min(4, max_items))

    if command == "literature-review":
        bullets = _section_excerpt(report_text, ["主题脉络", "keyword", "写作建议", "review"], max_lines=4) or [
            "Use high-score papers to build the main technical storyline.",
            "Group representative works by task definition, method family, evaluation protocol, and limitations.",
            "Read the full report for timeline and representative-paper interpretation.",
        ]
        return {
            "title": "🧭 Literature Review Snapshot",
            "subtitle": "Core papers are reorganized into review-oriented reading and writing cues.",
            "sections": [
                {"title": "Review Focus", "items": bullets},
                {"title": "Representative Papers", "headers": ["#", "Title", "Year", "Score", "Cites"], "rows": evidence_rows},
                {"title": "Suggested Outline", "items": [
                    "Background and task definition",
                    "Method evolution and representative systems",
                    "Evaluation protocols and benchmarks",
                    "Limitations and future directions",
                ]},
            ],
        }

    if command == "idea-grounding":
        bullets = _section_excerpt(report_text, ["差异化", "相似", "grounding", "检查"], max_lines=4) or [
            "Check whether top papers solve the same problem or only share terminology.",
            "Compare motivation, method design, environment, and evaluation setting.",
            "Use report.md to record overlap points and differentiation opportunities.",
        ]
        return {
            "title": "🧭 Idea Grounding Card",
            "subtitle": "Use retrieved evidence to locate similar work and differentiation space.",
            "sections": [
                {"title": "Closest Evidence", "headers": ["#", "Title", "Year", "Score", "Cites"], "rows": evidence_rows},
                {"title": "Grounding Checklist", "items": bullets},
                {"title": "Differentiation Questions", "items": [
                    "What assumption does your idea change?",
                    "Which setting, data, tool, or evaluation differs from prior work?",
                    "Can the contribution be stated without merely renaming an existing method?",
                ]},
            ],
        }

    if command == "idea-evaluate":
        bullets = _section_excerpt(report_text, ["新颖", "可行", "可靠", "novelty", "feasibility", "soundness"], max_lines=4) or [
            "Novelty: compare against the most similar retrieved papers.",
            "Feasibility: check whether related methods, data, and evaluation protocols already exist.",
            "Soundness: identify whether assumptions can be validated with KG-backed literature.",
        ]
        return {
            "title": "🧪 Idea Evaluation Card",
            "subtitle": "Evidence is organized around novelty, feasibility, and soundness.",
            "sections": [
                {"title": "Evidence Papers", "headers": ["#", "Title", "Year", "Score", "Cites"], "rows": evidence_rows},
                {"title": "Evaluation Signals", "items": bullets},
                {"title": "Manual Review Rubric", "items": [
                    "Novelty: is there a clear gap beyond the closest papers?",
                    "Feasibility: are datasets, baselines, tools, or tasks available?",
                    "Soundness: can claims be verified through controlled experiments?",
                    "Risk: is the idea too broad, too incremental, or poorly scoped?",
                ]},
            ],
        }

    if command == "idea-generate":
        seed_terms = topics[:6] or ["retrieval", "evaluation", "agents", "knowledge graph"]
        seed_items = []
        for i in range(0, min(len(seed_terms), 6), 2):
            pair = seed_terms[i:i + 2]
            if len(pair) == 2:
                seed_items.append(f"Combine `{pair[0]}` with `{pair[1]}` and check whether the intersection is under-explored.")
        if not seed_items:
            seed_items = ["Increase `--bias-exploration` or add more high-quality keyword anchors to discover broader idea seeds."]
        return {
            "title": "💡 Idea Generation Seeds",
            "subtitle": "Exploratory KG retrieval is summarized as candidate research directions.",
            "sections": [
                {"title": "Topic Ingredients", "items": [f"`{t}`" for t in seed_terms[:8]]},
                {"title": "Candidate Combinations", "items": seed_items},
                {"title": "Seed Evidence", "headers": ["#", "Title", "Year", "Score", "Cites"], "rows": evidence_rows},
            ],
        }

    if command == "trend-report":
        timeline_rows = _timeline_rows_from_papers(papers, max_rows=8)
        bullets = _section_excerpt(report_text, ["趋势", "时间线", "trend", "timeline"], max_lines=4) or [
            "Use earlier high-citation papers as stable foundations.",
            "Use recent lower-citation papers as possible emerging signals.",
            "Compare year buckets to identify phase shifts in tasks and methods.",
        ]
        return {
            "title": "📈 Trend Timeline",
            "subtitle": "Results are reorganized by time and impact for trend analysis.",
            "sections": [
                {"title": "Year Buckets", "headers": ["Year", "Count", "Representative Titles"], "rows": timeline_rows},
                {"title": "Trend Reading Guide", "items": bullets},
                {"title": "High-value Papers", "headers": ["#", "Title", "Year", "Score", "Cites"], "rows": evidence_rows},
            ],
        }

    if command == "researcher-review":
        bullets = _section_excerpt(report_text, ["研究轨迹", "代表", "trajectory", "background"], max_lines=4) or [
            "Use representative works to infer the research trajectory.",
            "Group papers by topic terms, publication years, and citation strength.",
            "Read report.md to prepare a researcher background summary.",
        ]
        return {
            "title": "👤 Researcher Profile Snapshot",
            "subtitle": "Author-related papers are reorganized into trajectory and representative-work cues.",
            "sections": [
                {"title": "Topic Terms", "items": [f"`{t}`" for t in topics[:8]] or ["No stable topic terms extracted."]},
                {"title": "Representative Works", "headers": ["#", "Title", "Year", "Score", "Cites"], "rows": evidence_rows},
                {"title": "Profile Writing Hints", "items": bullets},
            ],
        }

    return None


def _render_channel_view(channel_view: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    title = channel_view.get("title")
    subtitle = channel_view.get("subtitle")

    if title:
        lines.append(color(str(title), ANSI_BOLD + ANSI_MAGENTA))
    if subtitle:
        lines.append(str(subtitle))

    for section in channel_view.get("sections", []) or []:
        section_title = section.get("title")
        if section_title:
            lines.append("")
            lines.append(color(str(section_title), ANSI_BOLD + ANSI_CYAN))

        if section.get("rows"):
            headers = section.get("headers") or []
            rows = section.get("rows") or []
            if headers and rows:
                lines.append(_table(headers, rows))

        items = section.get("items") or []
        for item in items:
            lines.append(f"  • {item}")

    return lines


def render_user_output(payload: dict[str, Any]) -> str:
    """Render concise user-facing output as colored text tables."""
    ok = bool(payload.get("ok"))
    command = payload.get("command", "scischolar")
    query = payload.get("query")
    elapsed = payload.get("elapsed_seconds")

    channel_titles = {
        "search-papers": "📚 Paper Search",
        "paper-search": "⚡ Low-Level Paper Search",
        "related-authors": "👥 Related Authors",
        "author-papers": "📄 Author Papers",
        "support-papers": "📌 Support Papers",
        "literature-review": "📚 Literature Review",
        "idea-grounding": "🧭 Idea Grounding",
        "idea-evaluate": "🧪 Idea Evaluation",
        "idea-generate": "💡 Idea Generation",
        "trend-report": "📈 Trend Report",
        "researcher-review": "👤 Researcher Review",
    }
    display_title = channel_titles.get(command, "SciScholar")

    lines: list[str] = []

    if not ok:
        lines.append(color("❌ Task failed", ANSI_BOLD + ANSI_RED) + f" · {display_title}")
        if query:
            lines.append(f"🔎 Query: {color(query, ANSI_YELLOW)}")
        if elapsed is not None:
            lines.append(f"⏱️  Elapsed: {_format_elapsed(elapsed)}")
        if payload.get("status_code") is not None:
            lines.append(f"HTTP status: {payload.get('status_code')}")
        if payload.get("error_type"):
            lines.append(f"Error type: {payload.get('error_type')}")
        if payload.get("message"):
            lines.append(f"Reason: {payload.get('message')}")
        if payload.get("response_json"):
            lines.append(f"🧩 Detailed response saved to: {color(payload.get('response_json'), ANSI_DIM)}")
        return "\n".join(lines)

    lines.append(color("✅ Task completed", ANSI_BOLD + ANSI_GREEN) + f" · {display_title}")
    if query:
        lines.append(f"🔎 Query: {color(query, ANSI_YELLOW)}")
    if elapsed is not None:
        lines.append(f"⏱️  Elapsed: {color(_format_elapsed(elapsed), ANSI_CYAN)}")

    channel_view = payload.get("channel_view")
    has_channel_view = isinstance(channel_view, dict) and bool(channel_view)
    if has_channel_view:
        lines.append("")
        lines.extend(_render_channel_view(channel_view))

    papers = payload.get("papers") or []
    authors = payload.get("authors") or []
    support = payload.get("supporting_papers") or []

    if papers and not has_channel_view:
        lines.append("")
        lines.append(color("📚 Papers", ANSI_BOLD + ANSI_MAGENTA))
        rows = []
        for item in papers:
            rows.append([
                item.get("rank", ""),
                item.get("title", ""),
                item.get("year", "-"),
                _format_score(item.get("score")),
                item.get("citations", "-"),
                item.get("url") or "-",
            ])
        lines.append(_table(["#", "Title", "Year", "Score", "Cites", "PDF"], rows))
    elif command in {"search-papers", "paper-search", "author-papers"}:
        lines.append("")
        lines.append(color("📭 No parsed paper results. Try semantic mode, reduce filters, or inspect the report.", ANSI_YELLOW))

    if command == "related-authors" and authors:
        lines.append("")
        lines.append(color("👥 Related Authors", ANSI_BOLD + ANSI_MAGENTA))
        rows = [[item.get("rank", ""), item.get("name", ""), _format_score(item.get("score")), item.get("id", "-")] for item in authors]
        lines.append(_table(["#", "Author", "Score", "ID"], rows))

    if command == "support-papers" and authors:
        lines.append("")
        lines.append(color("🧑‍🔬 Candidate Author Support Papers", ANSI_BOLD + ANSI_MAGENTA))
        for author in authors:
            lines.append(f"\n{color(str(author.get('rank', '-')) + '.', ANSI_CYAN)} {color(author.get('name', 'Unknown'), ANSI_BOLD)}")
            papers_for_author = author.get("support_papers") or []
            if papers_for_author:
                rows = [[p.get("rank", ""), p.get("title", ""), p.get("year", "-"), _format_score(p.get("score")), p.get("url") or "-"] for p in papers_for_author]
                lines.append(_table(["#", "Title", "Year", "Score", "PDF"], rows))
            else:
                lines.append("  No support papers found.")

    if support and command != "support-papers":
        lines.append("")
        lines.append(color("📌 Support Papers", ANSI_BOLD + ANSI_MAGENTA))
        rows = [[item.get("rank", ""), item.get("title", ""), item.get("year", "-"), _format_score(item.get("score")), item.get("url") or "-"] for item in support]
        lines.append(_table(["#", "Title", "Year", "Score", "PDF"], rows))

    if payload.get("message") and not (papers or authors or support):
        lines.append("")
        lines.append(str(payload["message"]))

    if payload.get("channel_hint") and not has_channel_view:
        lines.append("")
        lines.append(color("✨ Channel Hint", ANSI_BOLD + ANSI_CYAN))
        lines.append(str(payload["channel_hint"]))

    if payload.get("report"):
        lines.append("")
        lines.append(f"📝 Full Markdown report saved to: {color(payload['report'], ANSI_GREEN)}")
        lines.append(color("   Open it with less or VS Code to inspect full evidence and summaries.", ANSI_DIM))


    return "\n".join(lines)


# ============================================================

# HTTP 请求

# ============================================================



def request_json(

    *,

    method: str,

    base_url: str,

    endpoint: str,

    api_key: str | None,

    payload: dict[str, Any] | None = None,

    timeout: int = 600,

) -> dict[str, Any]:

    start_time = time.time()

    url = build_url(base_url, endpoint)



    headers = {

        "Accept": "application/json",

    }



    data = None

    if payload is not None:

        headers["Content-Type"] = "application/json"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")



    if endpoint.rstrip("/") != "/healthz":

        if not api_key:

            return {

                "ok": False,

                "status_code": None,

                "endpoint": endpoint,

                "url": url,

                "elapsed_seconds": round(time.time() - start_time, 3),

                "error_type": "MissingApiKey",

                "error": "This endpoint requires SCISCHOLAR_API_KEY or --api-key.",

            }

        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key



    req = urllib.request.Request(

        url=url,

        data=data,

        headers=headers,

        method=method.upper(),

    )



    spinner_message = "Contacting the SciScholar backend. Please wait; complex graph retrieval may take several seconds"
    show_spinner = payload is not None and endpoint.rstrip("/") != "/healthz"

    try:

        with Spinner(spinner_message, enabled=show_spinner):
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    body = json.loads(raw) if raw else None
                except json.JSONDecodeError:
                    body = raw

                return {
                    "ok": 200 <= resp.status < 300,
                    "status_code": resp.status,
                    "endpoint": endpoint,
                    "url": url,
                    "elapsed_seconds": round(time.time() - start_time, 3),
                    "data": body,
                }

    except urllib.error.HTTPError as exc:

        raw = exc.read().decode("utf-8", errors="replace")

        try:

            body = json.loads(raw)

        except json.JSONDecodeError:

            body = raw



        return {

            "ok": False,

            "status_code": exc.code,

            "endpoint": endpoint,

            "url": url,

            "elapsed_seconds": round(time.time() - start_time, 3),

            "error_type": "HTTPError",

            "error": body,

        }



    except TimeoutError:

        return {

            "ok": False,

            "status_code": None,

            "endpoint": endpoint,

            "url": url,

            "elapsed_seconds": round(time.time() - start_time, 3),

            "error_type": "TimeoutError",

            "error": f"Request timed out after {timeout} seconds.",

        }



    except socket.timeout:

        return {

            "ok": False,

            "status_code": None,

            "endpoint": endpoint,

            "url": url,

            "elapsed_seconds": round(time.time() - start_time, 3),

            "error_type": "SocketTimeout",

            "error": f"Request timed out after {timeout} seconds.",

        }



    except urllib.error.URLError as exc:

        return {

            "ok": False,

            "status_code": None,

            "endpoint": endpoint,

            "url": url,

            "elapsed_seconds": round(time.time() - start_time, 3),

            "error_type": "URLError",

            "error": str(exc),

        }





# ============================================================

# 自然语言 Plan Builder

# ============================================================



EN_STOPWORDS = {

    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",

    "by", "from", "as", "at", "is", "are", "was", "were", "be", "been",

    "using", "use", "used", "based", "via", "into", "over", "under",

    "between", "towards", "toward", "about", "this", "that", "these",

    "those", "their", "its", "our", "your", "can", "could", "should",

    "would", "may", "might", "research", "paper", "papers", "study",

    "studies", "method", "methods", "system", "systems", "approach",

    "approaches", "task", "tasks", "result", "results",

}



DOMAIN_BOOST_TERMS = {

    "knowledge graph": 2.0,

    "knowledge graphs": 2.0,

    "graph retrieval": 2.0,

    "retrieval augmented generation": 2.0,

    "retrieval-augmented generation": 2.0,

    "scientific discovery": 2.0,

    "large language model": 1.8,

    "large language models": 1.8,

    "llm": 1.6,

    "llms": 1.6,

    "multi agent": 1.6,

    "multi-agent": 1.6,

    "scientific idea": 1.6,

    "idea evaluation": 1.6,

    "semantic search": 1.5,

    "vector search": 1.5,

    "citation": 1.4,

    "author retrieval": 1.4,

}



# ============================================================
# 逐项相关度映射
# ============================================================
#
# 设计原则：
# - 不暴露后端内部 ranking 公式权重。
# - 只映射到 KG2API 当前公开支持的 plan 字段：
#   keywords[].score, titles[].confidence, reference_titles。
# - top_k / limit / after / before / target_field 不属于这里的“相关度倾向”
#   设定，它们仍由原来的 options 逻辑处理。

VALID_RELEVANCE_LEVELS = {"high", "middle", "low"}

KEYWORD_SCORE_RANGE = {
    "high": (8, 10),
    "middle": (6, 8),
    "low": (3, 5),
}

KEYWORD_SCORE_BASE = {
    "high": 9,
    "middle": 7,
    "low": 4,
}

TITLE_CONFIDENCE_BY_RELEVANCE = {
    "high": 0.95,
    "middle": 0.85,
    "low": 0.65,
}

REFERENCE_ANCHOR_CONFIDENCE_BY_RELEVANCE = {
    "high": 0.95,
    "middle": 0.80,
}

GENERIC_KEYWORD_TERMS = {
    "ai",
    "model",
    "models",
    "method",
    "methods",
    "system",
    "systems",
    "task",
    "tasks",
    "data",
    "paper",
    "papers",
    "research",
    "study",
    "analysis",
    "evaluation",
    "approach",
    "framework",
    "算法",
    "模型",
    "方法",
    "系统",
    "研究",
    "论文",
    "任务",
    "数据",
}

TECHNICAL_HINT_TERMS = [
    "graph",
    "retrieval",
    "knowledge",
    "scientific",
    "discovery",
    "citation",
    "agent",
    "llm",
    "language model",
    "rag",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "semantic",
    "vector",
    "neo4j",
    "知识图谱",
    "检索",
    "科学发现",
    "大语言模型",
    "引用",
    "向量",
]


def normalize_relevance_level(value: str | None) -> str | None:
    if value is None:
        return None

    raw = normalize_text(value).lower()

    mapping = {
        "high": "high",
        "h": "high",
        "高": "high",
        "强": "high",
        "高相关": "high",
        "强相关": "high",
        "核心": "high",
        "重要": "high",
        "main": "high",
        "core": "high",
        "important": "high",

        "middle": "middle",
        "medium": "middle",
        "mid": "middle",
        "m": "middle",
        "中": "middle",
        "中等": "middle",
        "中等相关": "middle",
        "辅助": "middle",
        "平衡": "middle",
        "balanced": "middle",
        "secondary": "middle",

        "low": "low",
        "l": "low",
        "低": "low",
        "弱": "low",
        "低相关": "low",
        "弱相关": "low",
        "背景": "low",
        "扩展": "low",
        "background": "low",
        "weak": "low",
        "loose": "low",
    }

    return mapping.get(raw)


def split_item_values(segment: str, *, bucket: str = "keywords") -> list[str]:
    segment = normalize_text(segment)
    if not segment:
        return []

    # keyword 通常可以安全地用逗号、顿号、and 分隔；
    # title/reference 中经常包含 comma 或 "and"，因此只用分号/顿号等强列表分隔符。
    if bucket == "keywords":
        pattern = r"\s*(?:[,，、；;]|\band\b|\|)\s*"
    else:
        pattern = r"\s*(?:[；;、]|\|)\s*"

    parts = re.split(pattern, segment)
    cleaned = []

    for p in parts:
        p = normalize_text(p)
        p = re.sub(r"^[\"'“”《》]+|[\"'“”《》]+$", "", p).strip()
        if p:
            cleaned.append(p)

    return dedupe_keep_order(cleaned)


def is_generic_keyword(text: str) -> bool:
    normalized = normalize_text(text).lower()
    if normalized in GENERIC_KEYWORD_TERMS:
        return True

    tokens = normalized.split()
    if len(tokens) == 1 and tokens[0] in GENERIC_KEYWORD_TERMS:
        return True

    # 中文短泛词。
    if normalized in GENERIC_KEYWORD_TERMS:
        return True

    return False


def is_technical_phrase(text: str) -> bool:
    normalized = normalize_text(text).lower()

    if any(hint in normalized for hint in TECHNICAL_HINT_TERMS):
        return True

    # 2-5 个英文词的短语通常比单词更有检索约束力。
    tokens = normalized.split()
    if 2 <= len(tokens) <= 5:
        return True

    # 4-16 个中文字符的技术片段一般比 2 字泛词更可靠。
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if 4 <= len(chinese_chars) <= 16:
        return True

    return False


def map_keyword_relevance_to_score(text: str, relevance: str) -> int:
    relevance = normalize_relevance_level(relevance) or "middle"

    low, high = KEYWORD_SCORE_RANGE[relevance]
    score = KEYWORD_SCORE_BASE[relevance]

    # 考虑引用项质量：明确技术短语提高一点，泛词降低一点。
    if is_technical_phrase(text):
        score += 1
    if is_generic_keyword(text):
        score -= 1

    return max(low, min(high, score))


def map_title_relevance_to_confidence(relevance: str) -> float:
    relevance = normalize_relevance_level(relevance) or "middle"
    return TITLE_CONFIDENCE_BY_RELEVANCE[relevance]


def map_reference_relevance_to_anchor_confidence(relevance: str) -> float | None:
    relevance = normalize_relevance_level(relevance) or "middle"
    return REFERENCE_ANCHOR_CONFIDENCE_BY_RELEVANCE.get(relevance)


def parse_relevance_tagged_items(text: str) -> dict[str, list[dict[str, str]]]:
    """解析用户显式标注的逐项相关度。

    支持示例：
      关键词[high]：retrieval augmented generation
      关键词[middle]：scientific discovery
      关键词[low]：knowledge graph

      keyword[high]: retrieval augmented generation
      title[middle]: Graph Neural Networks: A Review
      reference[low]: A Survey of Information Retrieval Models

    返回：
      {
        "keywords": [{"text": "...", "relevance": "high"}],
        "titles": [{"text": "...", "relevance": "middle"}],
        "references": [{"text": "...", "relevance": "low"}]
      }
    """

    result = {
        "keywords": [],
        "titles": [],
        "references": [],
    }

    type_aliases = {
        "关键词": "keywords",
        "关键字": "keywords",
        "keyword": "keywords",
        "keywords": "keywords",
        "kw": "keywords",

        "标题": "titles",
        "题名": "titles",
        "论文标题": "titles",
        "title": "titles",
        "titles": "titles",
        "paper": "titles",
        "paper title": "titles",

        "参考文献": "references",
        "引用": "references",
        "文献": "references",
        "reference": "references",
        "references": "references",
        "ref": "references",
        "refs": "references",
    }

    # 行级解析更稳定，避免把整段文本误切。
    line_pattern = re.compile(
        r"^\s*"
        r"(?P<kind>关键词|关键字|keyword|keywords|kw|标题|题名|论文标题|title|titles|paper title|paper|参考文献|引用|文献|reference|references|ref|refs)"
        r"\s*"
        r"(?:[\[\(（【]\s*(?P<rel1>high|middle|medium|mid|low|高|强|高相关|强相关|核心|重要|中|中等|中等相关|辅助|平衡|低|弱|低相关|弱相关|背景|扩展)\s*[\]\)）】])?"
        r"\s*[：:=]\s*"
        r"(?P<value>.+?)\s*$",
        flags=re.IGNORECASE,
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = line_pattern.match(line)
        if not m:
            continue

        kind_raw = normalize_text(m.group("kind")).lower()
        bucket = type_aliases.get(kind_raw)
        if not bucket:
            continue

        relevance = normalize_relevance_level(m.group("rel1")) or "middle"
        for value in split_item_values(m.group("value"), bucket=bucket):
            result[bucket].append({"text": value, "relevance": relevance})

    # 句内补充解析：例如 “keyword[high]: xxx；title[low]: yyy”
    inline_pattern = re.compile(
        r"(?P<kind>关键词|关键字|keyword|keywords|kw|标题|题名|论文标题|title|titles|paper title|paper|参考文献|引用|文献|reference|references|ref|refs)"
        r"\s*[\[\(（【]\s*(?P<rel>high|middle|medium|mid|low|高|强|高相关|强相关|核心|重要|中|中等|中等相关|辅助|平衡|低|弱|低相关|弱相关|背景|扩展)\s*[\]\)）】]"
        r"\s*[：:=]\s*"
        r"(?P<value>[^。\n；;]+)",
        flags=re.IGNORECASE,
    )

    for m in inline_pattern.finditer(text):
        kind_raw = normalize_text(m.group("kind")).lower()
        bucket = type_aliases.get(kind_raw)
        if not bucket:
            continue

        relevance = normalize_relevance_level(m.group("rel")) or "middle"
        for value in split_item_values(m.group("value"), bucket=bucket):
            result[bucket].append({"text": value, "relevance": relevance})

    for key in result:
        deduped = []
        seen = set()
        for item in result[key]:
            item_text = normalize_text(item["text"])
            item_rel = item["relevance"]
            dedupe_key = (item_text.lower(), item_rel)
            if not item_text or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append({"text": item_text, "relevance": item_rel})
        result[key] = deduped

    return result


def upsert_keyword(keywords: list[dict[str, Any]], text: str, score: int) -> None:
    key = normalize_text(text).lower()

    for item in keywords:
        if normalize_text(str(item.get("text", ""))).lower() == key:
            # 用户显式标注的 high/middle/low 应覆盖自动抽取得分。
            item["score"] = score
            return

    keywords.append({"text": normalize_text(text), "score": score})


def upsert_title(titles: list[dict[str, Any]], title: str, confidence: float) -> None:
    key = normalize_text(title).lower()

    for item in titles:
        if normalize_text(str(item.get("title", ""))).lower() == key:
            item["confidence"] = max(float(item.get("confidence", 0.0)), confidence)
            return

    titles.append({"title": normalize_text(title), "confidence": confidence})


def apply_relevance_items_to_plan(plan: dict[str, Any], text: str) -> dict[str, Any]:
    """将逐项 high/middle/low 映射到后端 plan 字段。

    keyword[rel]   -> keywords[].score
    title[rel]     -> titles[].confidence
    reference[rel] -> reference_titles；high/middle 额外转 titles 锚点
    """

    tagged = parse_relevance_tagged_items(text)

    keywords = list(plan.get("keywords", []))
    titles = list(plan.get("titles", []))
    reference_titles = list(plan.get("reference_titles", []))

    # 用户显式标注的关键词应优先于自动抽取结果。
    # 如果自动抽取出了显式多词关键词的子词，例如用户写了
    # “关键词[low]：knowledge graph”，自动候选中又有 “graph:8”，
    # 则删除该子词，避免违背用户对该引用项的相关度设定。
    explicit_keyword_texts = [normalize_text(item["text"]).lower() for item in tagged["keywords"]]
    if explicit_keyword_texts:
        pruned_keywords = []
        for kw in keywords:
            kw_text = normalize_text(str(kw.get("text", ""))).lower()
            should_drop = False

            for explicit in explicit_keyword_texts:
                if kw_text == explicit:
                    should_drop = True
                    break

                kw_tokens = kw_text.split()
                explicit_tokens = explicit.split()

                if kw_tokens and explicit_tokens and len(kw_tokens) < len(explicit_tokens):
                    if all(token in explicit_tokens for token in kw_tokens):
                        should_drop = True
                        break

                if len(kw_text) >= 2 and len(explicit) > len(kw_text) and kw_text in explicit:
                    should_drop = True
                    break

            if not should_drop:
                pruned_keywords.append(kw)

        keywords = pruned_keywords

    for item in tagged["keywords"]:
        score = map_keyword_relevance_to_score(item["text"], item["relevance"])
        upsert_keyword(keywords, item["text"], score)

    for item in tagged["titles"]:
        confidence = map_title_relevance_to_confidence(item["relevance"])
        upsert_title(titles, item["text"], confidence)

    for item in tagged["references"]:
        ref_title = normalize_text(item["text"])
        if ref_title and ref_title not in reference_titles:
            reference_titles.append(ref_title)

        anchor_confidence = map_reference_relevance_to_anchor_confidence(item["relevance"])
        if anchor_confidence is not None:
            upsert_title(titles, ref_title, anchor_confidence)

    plan["keywords"] = keywords
    plan["titles"] = titles
    plan["reference_titles"] = dedupe_keep_order(reference_titles)

    return plan





def dedupe_keep_order(items: list[str]) -> list[str]:

    seen = set()

    result = []



    for item in items:

        item = normalize_text(item)

        key = item.lower()

        if not key or key in seen:

            continue

        seen.add(key)

        result.append(item)



    return result





def extract_quoted_phrases(text: str, min_len: int = 2, max_len: int = 180) -> list[str]:

    patterns = [

        r'"([^"\n]{%d,%d})"' % (min_len, max_len),

        r"“([^”\n]{%d,%d})”" % (min_len, max_len),

        r"《([^》\n]{%d,%d})》" % (min_len, max_len),

        r"'([^'\n]{%d,%d})'" % (min_len, max_len),

    ]



    phrases: list[str] = []

    for pattern in patterns:

        phrases.extend(re.findall(pattern, text))



    return dedupe_keep_order([normalize_text(x) for x in phrases])





def extract_title_hints(text: str, max_titles: int = 5) -> list[dict[str, Any]]:

    phrases = extract_quoted_phrases(text, min_len=8, max_len=180)



    titles = []

    for phrase in phrases:

        if len(phrase) < 4:

            continue

        titles.append(phrase)



    titles = dedupe_keep_order(titles)[:max_titles]



    return [

        {

            "title": title,

            "confidence": 0.9,

        }

        for title in titles

    ]





def extract_reference_titles(text: str, max_refs: int = 10) -> list[str]:

    refs: list[str] = []



    for line in text.splitlines():

        raw = line.strip()

        if not raw:

            continue



        m = re.match(r"^(?:\[\d+\]|\d+\.|-)\s+(.{10,220})$", raw)

        if m:

            refs.append(normalize_text(m.group(1)))



    return dedupe_keep_order(refs)[:max_refs]





def tokenize_english(text: str) -> list[str]:

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())

    return [

        t for t in tokens

        if len(t) >= 3 and t not in EN_STOPWORDS

    ]





def generate_english_ngrams(tokens: list[str], min_n: int = 1, max_n: int = 4) -> list[str]:

    ngrams: list[str] = []



    for n in range(min_n, max_n + 1):

        for i in range(0, len(tokens) - n + 1):

            gram_tokens = tokens[i: i + n]

            if not gram_tokens:

                continue

            if gram_tokens[0] in EN_STOPWORDS or gram_tokens[-1] in EN_STOPWORDS:

                continue

            ngrams.append(" ".join(gram_tokens))



    return ngrams





def extract_chinese_candidates(text: str) -> list[str]:

    chunks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,40}", text)

    result: list[str] = []



    for chunk in chunks:

        chunk = chunk.strip()

        if len(chunk) < 2:

            continue

        if len(chunk) <= 18:

            result.append(chunk)



    return result





def score_keyword_candidates(candidates: list[str]) -> list[tuple[str, float]]:

    scores: dict[str, float] = {}



    for cand in candidates:

        c = normalize_text(cand.lower())

        if not c:

            continue



        token_count = len(c.split())

        base = 1.0 + min(token_count, 4) * 0.5



        if c in DOMAIN_BOOST_TERMS:

            base += DOMAIN_BOOST_TERMS[c]



        if any(

            x in c

            for x in [

                "graph",

                "retrieval",

                "knowledge",

                "scientific",

                "discovery",

                "evaluation",

                "citation",

                "agent",

                "llm",

                "language model",

                "database",

                "neo4j",

            ]

        ):

            base += 1.0



        scores[c] = scores.get(c, 0.0) + base



    return sorted(scores.items(), key=lambda x: x[1], reverse=True)





def score_to_1_10(raw_score: float, max_score: float) -> int:

    if max_score <= 0:

        return 8



    normalized = raw_score / max_score

    score = int(round(5 + normalized * 5))

    return max(1, min(10, score))





# ============================================================
# Plan Query 清洗：删除结构化控制行
# ============================================================

PLAN_CONTROL_LINE_RE = re.compile(
    r"^\s*"
    r"(?:关键词|关键字|keyword|keywords|kw|标题|题名|论文标题|title|titles|paper\s*title|paper|参考文献|引用|文献|reference|references|ref|refs)"
    r"\s*"
    r"(?:[\[\(（【]\s*"
    r"(?:high|middle|medium|mid|low|高|强|高相关|强相关|核心|重要|中|中等|中等相关|辅助|平衡|低|弱|低相关|弱相关|背景|扩展)"
    r"\s*[\]\)）】])?"
    r"\s*[：:=]\s*"
    r".+?\s*$",
    flags=re.IGNORECASE,
)


INLINE_PLAN_CONTROL_RE = re.compile(
    r"(?:关键词|关键字|keyword|keywords|kw|标题|题名|论文标题|title|titles|paper\s*title|paper|参考文献|引用|文献|reference|references|ref|refs)"
    r"\s*[\[\(（【]\s*"
    r"(?:high|middle|medium|mid|low|高|强|高相关|强相关|核心|重要|中|中等|中等相关|辅助|平衡|低|弱|低相关|弱相关|背景|扩展)"
    r"\s*[\]\)）】]\s*[：:=]\s*"
    r"[^。；;\n]+[。；;]?",
    flags=re.IGNORECASE,
)


def strip_plan_control_lines(text: str) -> str:
    """从 query_text 中删除关键词/标题/参考文献的控制行。

    这些控制行只用于生成结构化字段：
    - 关键词[...] -> keywords[].score
    - 标题[...] -> titles[].confidence
    - 参考文献[...] -> reference_titles / titles anchor

    它们不应该混入 plan.query_text，否则会污染语义编码和后端检索。
    """

    kept_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if PLAN_CONTROL_LINE_RE.match(line):
            continue

        # 兼容用户把控制标注写在同一行的情况。
        line = INLINE_PLAN_CONTROL_RE.sub("", line).strip()

        if line:
            kept_lines.append(line)

    cleaned = normalize_text(" ".join(kept_lines))

    return cleaned or normalize_text(text)




# ============================================================
# Optional frontend LLM keyword extraction
# ============================================================

def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return ""


def _missing_or_placeholder(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    placeholder_tokens = (
        "your-openai-compatible-endpoint",
        "your-chat-completions-endpoint",
        "your-llm-provider.example",
        "your-provider-or-gateway.example",
    )
    return (not normalized) or normalized in {
        "replace-me",
        "your-token",
        "your-api-key",
        "your-personal-scischolar-token",
        "your-personal-scinet-token",
        "your-model-name",
    } or any(token in normalized for token in placeholder_tokens)


def _chat_completions_url(base_url: str, full_url: str = "") -> str:
    url = (full_url or "").strip()
    if url:
        return url

    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""

    path_without_query = base.split("?", 1)[0].rstrip("/")
    if path_without_query.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _parse_llm_auth_header(value: str) -> tuple[str, str] | None:
    text = (value or "").strip()
    if _missing_or_placeholder(text):
        return None

    name, separator, header_value = text.partition(":")
    if separator and name.strip() and header_value.strip():
        return name.strip(), header_value.strip()
    return "Authorization", text


def _parse_llm_extra_headers(value: str) -> dict[str, str]:
    text = (value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(val) for key, val in parsed.items() if str(key).strip() and str(val).strip()}


def _optional_llm_config() -> dict[str, Any] | None:
    provider = (_env_first("LLM_PROVIDER", "SCISCHOLAR_LLM_PROVIDER", "SCINET_LLM_PROVIDER") or "chat_completions").lower()
    if provider in {"0", "false", "none", "off", "disabled"}:
        return None
    api_key = _env_first("LLM_API_KEY", "SCISCHOLAR_LLM_API_KEY", "SCINET_LLM_API_KEY", "OPENAI_API_KEY")
    base_url = _env_first("LLM_BASE_URL", "SCISCHOLAR_LLM_BASE_URL", "SCINET_LLM_BASE_URL", "OPENAI_BASE_URL")
    full_url = _env_first("LLM_CHAT_COMPLETIONS_URL", "SCISCHOLAR_LLM_CHAT_COMPLETIONS_URL", "SCINET_LLM_CHAT_COMPLETIONS_URL", "OPENAI_CHAT_COMPLETIONS_URL")
    auth_header = _env_first("LLM_AUTH_HEADER", "SCISCHOLAR_LLM_AUTH_HEADER", "SCINET_LLM_AUTH_HEADER")
    extra_headers = _env_first("LLM_HTTP_HEADERS", "SCISCHOLAR_LLM_HTTP_HEADERS", "SCINET_LLM_HTTP_HEADERS")
    model = _env_first("LLM_MODEL", "SCISCHOLAR_LLM_MODEL", "SCINET_LLM_MODEL", "OPENAI_MODEL")
    endpoint = _chat_completions_url(base_url, full_url)
    if _missing_or_placeholder(endpoint) or _missing_or_placeholder(model):
        return None
    headers = _parse_llm_extra_headers(extra_headers)
    parsed_auth_header = _parse_llm_auth_header(auth_header)
    if parsed_auth_header:
        headers[parsed_auth_header[0]] = parsed_auth_header[1]
    elif not _missing_or_placeholder(api_key):
        headers["Authorization"] = "Bearer " + api_key
    return {
        "provider": provider,
        "chat_completions_url": endpoint,
        "headers": headers,
        "model": model,
        "timeout": int(os.environ.get("SCISCHOLAR_LLM_TIMEOUT") or os.environ.get("SCINET_LLM_TIMEOUT") or os.environ.get("LLM_TIMEOUT") or 30),
        "temperature": float(os.environ.get("SCISCHOLAR_LLM_TEMPERATURE") or os.environ.get("SCINET_LLM_TEMPERATURE") or os.environ.get("LLM_TEMPERATURE") or 0),
        "max_tokens": int(os.environ.get("SCISCHOLAR_LLM_MAX_TOKENS") or os.environ.get("SCINET_LLM_MAX_TOKENS") or os.environ.get("LLM_MAX_TOKENS") or 512),
    }


def _parse_json_object_from_text(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_keywords_with_optional_llm(text: str, *, top_keywords: int = 8) -> list[dict[str, Any]]:
    if top_keywords <= 0:
        return []
    cfg = _optional_llm_config()
    if not cfg:
        return []

    prompt = (
        "Extract concise academic search keywords from the query. "
        "Return JSON only: {\"keywords\":[{\"text\":\"keyword phrase\",\"score\":1-10}]}. "
        f"Return at most {top_keywords} keywords. Prefer specific technical phrases."
    )
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You extract scientific retrieval keywords."},
            {"role": "user", "content": prompt + "\n\nQuery:\n" + text[:4000]},
        ],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }
    headers = {"Content-Type": "application/json", **cfg.get("headers", {})}
    request = urllib.request.Request(
        cfg["chat_completions_url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=cfg["timeout"]) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_json_object_from_text(content)
        if not parsed:
            return []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in parsed.get("keywords", []):
        if isinstance(item, str):
            kw, score = normalize_text(item), 8
        elif isinstance(item, dict):
            kw = normalize_text(str(item.get("text") or item.get("keyword") or ""))
            try:
                score = int(round(float(item.get("score", 8))))
            except Exception:
                score = 8
        else:
            continue
        if not kw or kw.lower() in seen:
            continue
        seen.add(kw.lower())
        out.append({"text": kw[:120], "score": max(1, min(10, score))})
        if len(out) >= top_keywords:
            break
    return out


def optional_provider_config_status() -> dict[str, Any]:
    return {
        "llm_keyword_extraction_enabled": _optional_llm_config() is not None,
        "openalex_configured": not _missing_or_placeholder(_env_first("OA_API_KEY", "OPENALEX_API_KEY", "SCISCHOLAR_OPENALEX_API_KEY", "SCINET_OPENALEX_API_KEY")),
        "openalex_mailto_configured": not _missing_or_placeholder(_env_first("OPENALEX_MAILTO", "OPENALEX_EMAIL", "SCISCHOLAR_OPENALEX_MAILTO", "SCINET_OPENALEX_MAILTO")),
        "grobid_configured": not _missing_or_placeholder(_env_first("GROBID_BASE_URL", "SCISCHOLAR_GROBID_BASE_URL", "SCINET_GROBID_BASE_URL")),
    }


def build_plan_from_text(

    *,

    text: str,

    source_type: str = "idea_text",

    source_title: str | None = None,

    top_keywords: int = 8,

    max_titles: int = 5,

    max_refs: int = 10,

) -> dict[str, Any]:

    query_text = strip_plan_control_lines(text)

    clean_text = normalize_text(query_text)



    titles = extract_title_hints(clean_text, max_titles=max_titles)

    reference_titles = extract_reference_titles(query_text, max_refs=max_refs)

    llm_keywords = extract_keywords_with_optional_llm(clean_text, top_keywords=top_keywords)
    if llm_keywords:
        plan = {
            "query_text": clean_text,
            "source_type": source_type,
            "source_title": source_title,
            "keywords": llm_keywords,
            "titles": titles,
            "reference_titles": reference_titles,
        }
        return plan




    english_tokens = tokenize_english(clean_text)

    english_ngrams = generate_english_ngrams(english_tokens, min_n=1, max_n=4)

    chinese_candidates = extract_chinese_candidates(clean_text)



    candidates = english_ngrams + chinese_candidates



    if not candidates:

        candidates = [clean_text[:120]]



    ranked = score_keyword_candidates(candidates)

    selected = ranked[:top_keywords]

    max_raw_score = selected[0][1] if selected else 1.0



    keywords = []

    for kw, raw_score in selected:

        keywords.append(

            {

                "text": kw,

                "score": score_to_1_10(raw_score, max_raw_score),

            }

        )



    if not keywords and not titles:

        keywords = [

            {

                "text": clean_text[:120],

                "score": 8,

            }

        ]



    plan = {

        "query_text": clean_text,

        "source_type": source_type,

        "source_title": source_title,

        "keywords": keywords,

        "titles": titles,

        "reference_titles": reference_titles,

    }



    # 将用户逐项标注的 high/middle/low 相关度映射到后端真实支持的字段。
    # 注意：不向 plan 添加任何后端 schema 不支持的字段。
    plan = apply_relevance_items_to_plan(plan, text)



    # 后端要求 keywords 或 titles 至少存在一个。若用户只提供了 low reference，
    # 其不会转成 title anchor，因此这里保留 query_text 兜底关键词。
    if not plan.get("keywords") and not plan.get("titles"):

        plan["keywords"] = [

            {

                "text": clean_text[:120],

                "score": 8,

            }

        ]



    return plan





# ============================================================
# 专家输入参数：--query / --keyword / --title / --reference / --time-range
# ============================================================

EXPERT_LEVELS = {"low", "middle", "medium", "mid", "high", "低", "中", "中等", "高"}


def normalize_expert_level(value: str | None) -> str:
    level = normalize_relevance_level(value) if value is not None else None
    return level or "middle"


def parse_expert_weighted_item(value: str) -> tuple[str, str]:
    """解析专家输入项。

    支持：
    - "open world agent:high"
    - "high:open world agent"
    - "open world agent=middle"
    - "open world agent"  # 默认 middle

    注意：标题本身常含冒号，例如 "Voyager: An Open-Ended ..."。
    因此只有当分隔符一侧能识别为 high/middle/low 时才把它当作挡位。
    """
    raw = normalize_text(value)
    if not raw:
        raise ValueError("empty expert item")

    for sep in ["=", "|", "@", ":", "："]:
        if sep not in raw:
            continue

        left, right = raw.split(sep, 1)
        left = normalize_text(left)
        right = normalize_text(right)
        if left.lower() in EXPERT_LEVELS or left in EXPERT_LEVELS:
            return right, normalize_expert_level(left)

        left, right = raw.rsplit(sep, 1)
        left = normalize_text(left)
        right = normalize_text(right)
        if right.lower() in EXPERT_LEVELS or right in EXPERT_LEVELS:
            return left, normalize_expert_level(right)

    return raw, "middle"


def merge_expert_plan_controls(args: argparse.Namespace, text: str) -> str:
    """把专家参数转成既有的结构化控制行，复用 plan builder。"""
    lines = [text.strip()] if text and text.strip() else []

    for raw in getattr(args, "expert_keywords", []) or []:
        item_text, level = parse_expert_weighted_item(raw)
        if item_text:
            lines.append(f"keyword[{level}]: {item_text}")

    for raw in getattr(args, "expert_titles", []) or []:
        item_text, level = parse_expert_weighted_item(raw)
        if item_text:
            lines.append(f"title[{level}]: {item_text}")

    for raw in getattr(args, "expert_references", []) or []:
        item_text, level = parse_expert_weighted_item(raw)
        if item_text:
            lines.append(f"reference[{level}]: {item_text}")

    merged = "\n".join(lines).strip()
    return merged or text

def append_soft_domain_to_query(text: str, domain: str | None) -> str:
    """把 --domain/--检索领域 作为软领域偏好拼入 query_text。

    该字段默认不再映射为 options.target_field，因此不会触发后端硬过滤。
    如果需要硬过滤，使用 --target-field。
    """
    domain = normalize_text(domain or "")
    if not domain:
        return text

    base = (text or "").strip()
    if not base:
        return f"Domain: {domain}"

    if domain.casefold() in normalize_text(base).casefold():
        return base

    return f"{base}\nDomain: {domain}"


def _year_to_date(year: str, *, start: bool) -> str:
    return f"{year}-01-01" if start else f"{year}-12-31"


def parse_time_range_arg(value: str | None) -> tuple[str | None, str | None]:
    """解析专家时间范围参数。

    支持示例：
    - 2020-2024
    - 2020..2024
    - 2020-01-01..2024-12-31
    - since 2020 / after 2020
    - before 2025
    - 2020年至今
    - 2020  # 表示 2020 全年
    """
    if not value:
        return None, None

    raw = normalize_text(value)
    lower = raw.lower()

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", raw)
    if len(dates) >= 2:
        return dates[0], dates[1]
    if len(dates) == 1:
        if re.search(r"before|until|to|截止|之前|以前", lower):
            return None, dates[0]
        return dates[0], None

    m = re.search(r"(?:since|after|from|自|从)\s*(\d{4})", lower)
    if m:
        return _year_to_date(m.group(1), start=True), None

    m = re.search(r"(?:before|until|截止|之前|以前)\s*(\d{4})", lower)
    if m:
        return None, _year_to_date(m.group(1), start=False)

    m = re.search(r"(\d{4})\s*(?:年至今|至今|-\s*present|\.\.\s*present)", lower)
    if m:
        return _year_to_date(m.group(1), start=True), None

    years = re.findall(r"\d{4}", raw)
    if len(years) >= 2:
        return _year_to_date(years[0], start=True), _year_to_date(years[1], start=False)
    if len(years) == 1:
        return _year_to_date(years[0], start=True), _year_to_date(years[0], start=False)

    raise SystemExit(f"Could not parse --time-range: {value}. Use 2020-2024 or 2020-01-01..2024-12-31.")


# ============================================================

# 从自然语言中抽取控制参数

# ============================================================



def extract_top_k(text: str, default: int) -> int:

    patterns = [

        r"top[-_\s]?k\s*[:=]?\s*(\d+)",

        r"返回\s*(\d+)\s*(?:篇|个|条|位)",

        r"取\s*(\d+)\s*(?:篇|个|条|位)",

        r"找\s*(\d+)\s*(?:篇|个|条|位)",

        r"(\d+)\s*(?:papers|authors|results|items)",

    ]



    for pattern in patterns:

        m = re.search(pattern, text, flags=re.IGNORECASE)

        if m:

            value = int(m.group(1))

            return max(1, min(value, 100))



    return default





def extract_target_field(text: str) -> str | None:

    patterns = [

        r"(?:target[-_\s]?field|field|domain)\s*[:=]\s*([A-Za-z][A-Za-z\s\-]{2,80})",

        r"(?:领域|方向|学科)\s*[：:]\s*([\u4e00-\u9fffA-Za-z\s\-]{2,80})",

    ]



    for pattern in patterns:

        m = re.search(pattern, text, flags=re.IGNORECASE)

        if m:

            value = normalize_text(m.group(1))

            value = re.split(r"[,，。.;；\n]", value)[0].strip()

            return value or None



    lower = text.lower()

    known_fields = [

        "artificial intelligence",

        "computer science",

        "machine learning",

        "natural language processing",

        "data mining",

        "information retrieval",

        "bioinformatics",

        "robotics",

    ]



    for field in known_fields:

        if field in lower:

            return field



    if "人工智能" in text:

        return "artificial intelligence"

    if "计算机" in text:

        return "computer science"

    if "自然语言处理" in text:

        return "natural language processing"

    if "信息检索" in text:

        return "information retrieval"



    return None





def extract_date_filters(text: str) -> tuple[str | None, str | None]:

    after = None

    before = None



    m = re.search(r"(?:after|since|from)\s+(\d{4})(?:-(\d{2})-(\d{2}))?", text, flags=re.IGNORECASE)

    if m:

        year = m.group(1)

        month = m.group(2) or "01"

        day = m.group(3) or "01"

        after = f"{year}-{month}-{day}"



    m = re.search(r"(?:before|until|to)\s+(\d{4})(?:-(\d{2})-(\d{2}))?", text, flags=re.IGNORECASE)

    if m:

        year = m.group(1)

        month = m.group(2) or "12"

        day = m.group(3) or "31"

        before = f"{year}-{month}-{day}"



    m = re.search(r"(?:从|自)\s*(\d{4})\s*年?", text)

    if m:

        after = f"{m.group(1)}-01-01"



    m = re.search(r"(?:到|至|截至)\s*(\d{4})\s*年?", text)

    if m:

        before = f"{m.group(1)}-12-31"



    all_dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)

    if len(all_dates) >= 2:

        after = all_dates[0]

        before = all_dates[1]

    elif len(all_dates) == 1 and after is None:

        after = all_dates[0]



    return after, before





def build_options_from_text(

    *,

    text: str,

    default_top_k: int,

    cli_top_k: int | None,

    cli_target_field: str | None,

    cli_after: str | None,

    cli_before: str | None,

    cli_time_range: str | None = None,

) -> dict[str, Any]:

    text_after, text_before = extract_date_filters(text)

    range_after, range_before = parse_time_range_arg(cli_time_range)



    top_k = cli_top_k if cli_top_k is not None else extract_top_k(text, default_top_k)

    # --domain/--检索领域 默认作为软领域偏好拼入 query_text；
    # 只有显式 --target-field 才进入后端 options.target_field 做硬过滤。
    target_field = cli_target_field

    after = cli_after or range_after or text_after

    before = cli_before or range_before or text_before



    options: dict[str, Any] = {

        "top_k": top_k,

    }



    if target_field:

        options["target_field"] = target_field

    if after:

        options["after"] = after

    if before:

        options["before"] = before



    return options



# ============================================================
# 检索偏向超参数：前端挡位 -> KG2API options.retrieval_bias
# ============================================================

def build_retrieval_bias_from_args(args: argparse.Namespace) -> dict[str, str] | None:
    values = {
        "keyword_association": getattr(args, "bias_keyword", None),
        "non_seed_keyword_expansion": getattr(args, "bias_non_seed_keyword", None),
        "citation": getattr(args, "bias_citation", None),
        "paper_relatedness": getattr(args, "bias_related", None),
        "authorship": getattr(args, "bias_authorship", None),
        "coauthorship": getattr(args, "bias_coauthorship", None),
        "keyword_cooccurrence": getattr(args, "bias_cooccurrence", None),
        "graph_exploration": getattr(args, "bias_exploration", None),
        "ranking_profile": getattr(args, "ranking_profile", None),
    }

    bias = {key: value for key, value in values.items() if value is not None}

    return bias or None




# ============================================================

# 自然语言作者抽取

# ============================================================



def clean_author_name(name: str) -> str:

    name = normalize_text(name)

    name = re.sub(r"^(author|researcher|scholar)\s*[:：]?\s*", "", name, flags=re.IGNORECASE)

    name = re.split(r"[,，。.;；\n]", name)[0].strip()

    return name





def extract_author_candidates(text: str, max_authors: int = 10) -> list[dict[str, str]]:

    candidates: list[str] = []



    # 1. 中文显式模式：作者：A, B

    patterns = [

        r"(?:候选作者|作者列表|作者|学者|研究者)\s*[：:]\s*([^\n。；;]+)",

        r"(?:authors?|researchers?|scholars?|candidates?)\s*[:=]\s*([^\n.;]+)",

    ]



    for pattern in patterns:

        for m in re.finditer(pattern, text, flags=re.IGNORECASE):

            segment = m.group(1)

            parts = re.split(r"[,，、/]| and ", segment)

            for p in parts:

                p = clean_author_name(p)

                if p:

                    candidates.append(p)



    # 2. 英文模式：papers by Geoffrey Hinton

    patterns = [

        r"(?:papers|works|publications)\s+(?:by|of|from)\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,4})",

        r"(?:find|retrieve|search)\s+(?:papers|works|publications)\s+(?:by|of|from)\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,4})",

        r"(?:author|researcher|scholar)\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,4})",

    ]



    for pattern in patterns:

        for m in re.finditer(pattern, text):

            candidates.append(clean_author_name(m.group(1)))



    # 3. 引号中的人名候选

    for phrase in extract_quoted_phrases(text, min_len=2, max_len=80):

        if re.match(r"^[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,4}$", phrase):

            candidates.append(phrase)

        elif re.match(r"^[\u4e00-\u9fff]{2,5}$", phrase):

            candidates.append(phrase)



    # 4. OpenAlex ID

    for m in re.finditer(r"https?://openalex\.org/[A-Za-z]\d+", text):

        candidates.append(m.group(0))



    cleaned = dedupe_keep_order([clean_author_name(c) for c in candidates if clean_author_name(c)])



    authors: list[dict[str, str]] = []

    for c in cleaned[:max_authors]:

        if "openalex.org" in c.lower() or re.match(r"^[A-Za-z]\d+$", c):

            authors.append({"author_id": c})

        else:

            authors.append({"name": c})



    return authors





def extract_single_author(text: str) -> str | None:

    authors = extract_author_candidates(text, max_authors=1)

    if not authors:

        return None



    first = authors[0]

    return first.get("author_id") or first.get("name")





# ============================================================

# 结果解析与报告生成

# ============================================================



def first_present(obj: dict[str, Any], keys: list[str]) -> Any:

    for key in keys:

        if key in obj and obj[key] not in [None, "", []]:

            return obj[key]

    return None






def _clean_pdf_url_value(value: Any) -> str:
    """Return a clean PDF URL only. Do not return OpenAlex landing pages."""
    if value is None:
        return ""

    if isinstance(value, dict):
        for key in (
            "pdf_url",
            "paper_pdf_url",
            "open_access_pdf_url",
            "best_oa_pdf_url",
            "oa_pdf_url",
            "fulltext_pdf_url",
            "full_text_pdf_url",
            "pdf",
            "pdfUrl",
            "url",
            "href",
        ):
            found = _clean_pdf_url_value(value.get(key))
            if found:
                return found
        return ""

    if isinstance(value, list):
        for item in value:
            found = _clean_pdf_url_value(item)
            if found:
                return found
        return ""

    url = str(value).strip()
    if not url:
        return ""

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return ""

    lower = url.lower()

    # 不再把 OpenAlex 页面作为用户可见论文链接。
    if "openalex.org/" in lower:
        return ""

    # 只保留 PDF 或明显 PDF 入口。
    if (
        ".pdf" in lower
        or "arxiv.org/pdf/" in lower
        or "openreview.net/pdf" in lower
        or lower.endswith("/pdf")
        or "pdf" in lower
    ):
        return url

    return ""


def select_pdf_url(obj: dict[str, Any]) -> str:
    """Select the best user-facing PDF URL from a paper-like object.

    This intentionally does not fall back to OpenAlex IDs or landing pages.
    """
    if not isinstance(obj, dict):
        return ""

    pdf_keys = [
        "pdf_url",
        "paper_pdf_url",
        "open_access_pdf_url",
        "best_oa_pdf_url",
        "oa_pdf_url",
        "fulltext_pdf_url",
        "full_text_pdf_url",
        "pdf",
        "pdfUrl",
    ]

    for key in pdf_keys:
        found = _clean_pdf_url_value(obj.get(key))
        if found:
            return found

    for nested_key in (
        "paper",
        "full_paper",
        "metadata",
        "open_access",
        "best_oa_location",
        "primary_location",
        "locations",
        "oa_locations",
    ):
        found = _clean_pdf_url_value(obj.get(nested_key))
        if found:
            return found

    return ""


def collect_dicts_recursive(obj: Any) -> list[dict[str, Any]]:

    found: list[dict[str, Any]] = []



    if isinstance(obj, dict):

        found.append(obj)

        for v in obj.values():

            found.extend(collect_dicts_recursive(v))

    elif isinstance(obj, list):

        for item in obj:

            found.extend(collect_dicts_recursive(item))



    return found





def normalize_authors(value: Any, max_authors: int = 4) -> str:

    if value is None:

        return ""



    if isinstance(value, str):

        return compact_text(value, 120)



    names = []

    if isinstance(value, list):

        for item in value:

            if isinstance(item, str):

                names.append(item)

            elif isinstance(item, dict):

                name = first_present(item, ["name", "author_name", "display_name", "full_name"])

                if name:

                    names.append(str(name))



    names = names[:max_authors]

    if len(names) == max_authors:

        return ", ".join(names) + "..."

    return ", ".join(names)





def as_paper_item(obj: dict[str, Any]) -> dict[str, Any] | None:

    title = first_present(obj, ["title", "paper_title", "work_title", "display_name", "name"])

    if not title:

        return None



    markers = [

        "abstract",

        "abstract_text",

        "publication_year",

        "year",

        "doi",

        "paper_id",

        "openalex_id",

        "venue",

        "journal",

        "authors",

        "author_names",

        "citation_count",

        "cited_by_count",

    ]



    if not any(k in obj for k in markers):

        return None



    return {

        "title": compact_text(title, 220),

        "year": first_present(obj, ["year", "publication_year", "pub_year"]),

        "score": first_present(obj, ["score", "rank_score", "kg_score", "similarity", "similarity_score", "final_score"]),

        "citations": first_present(obj, ["citation_count", "cited_by_count", "citations"]),

        "venue": compact_text(first_present(obj, ["venue", "journal", "conference", "source"]), 80),

        "authors": normalize_authors(first_present(obj, ["authors", "author_names"])),

        "id": compact_text(first_present(obj, ["id", "paper_id", "openalex_id", "doi", "url"]), 120),

        "url": compact_text(select_pdf_url(obj), 220),

        "abstract": compact_text(first_present(obj, ["abstract", "abstract_text"]), 300),

    }





def as_author_item(obj: dict[str, Any]) -> dict[str, Any] | None:

    name = first_present(obj, ["author_name", "name", "display_name", "full_name"])

    if not name:

        return None



    markers = [

        "author_id",

        "openalex_id",

        "works_count",

        "paper_count",

        "citation_count",

        "cited_by_count",

        "h_index",

        "score",

        "support_papers",

        "papers",

    ]



    if not any(k in obj for k in markers):

        return None



    return {

        "name": compact_text(name, 120),

        "id": compact_text(first_present(obj, ["author_id", "id", "openalex_id", "url"]), 120),

        "score": first_present(obj, ["score", "rank_score", "similarity", "final_score"]),

        "works_count": first_present(obj, ["works_count", "paper_count"]),

        "citation_count": first_present(obj, ["citation_count", "cited_by_count"]),

        "h_index": first_present(obj, ["h_index"]),

    }





def collect_papers(data: Any, max_items: int = 20) -> list[dict[str, Any]]:

    papers: list[dict[str, Any]] = []

    seen = set()



    for obj in collect_dicts_recursive(data):

        item = as_paper_item(obj)

        if not item:

            continue



        key = item.get("id") or item.get("title")

        if not key:

            continue



        key = str(key).lower()

        if key in seen:

            continue



        seen.add(key)

        papers.append(item)



        if len(papers) >= max_items:

            break



    return papers





def collect_authors(data: Any, max_items: int = 20) -> list[dict[str, Any]]:

    authors: list[dict[str, Any]] = []

    seen = set()



    for obj in collect_dicts_recursive(data):

        item = as_author_item(obj)

        if not item:

            continue



        key = item.get("id") or item.get("name")

        if not key:

            continue



        key = str(key).lower()

        if key in seen:

            continue



        seen.add(key)

        authors.append(item)



        if len(authors) >= max_items:

            break



    return authors





def md_escape(value: Any, max_len: int | None = None) -> str:

    return compact_text(value, 120).replace("|", "\\|")





def build_summary(command: str, response: dict[str, Any]) -> str:

    if not response.get("ok"):

        return (

            f"{command} failed, status code: {response.get('status_code')}, "

            f"error type: {response.get('error_type')}"

        )



    data = response.get("data")

    papers = collect_papers(data, max_items=10)

    authors = collect_authors(data, max_items=10)



    elapsed = response.get("elapsed_seconds")

    elapsed_text = f", elapsed about {elapsed} seconds" if elapsed is not None else ""



    if papers:

        return f"{command} succeeded{elapsed_text}; parsed {len(papers)} candidate papers. Top paper: {papers[0].get('title')}"



    if authors:

        return f"{command} succeeded{elapsed_text}; parsed {len(authors)} candidate authors. Top author: {authors[0].get('name')}"



    return f"{command} succeeded{elapsed_text}, but no standard paper or author entries were parsed. Please inspect response.json."







def _safe_number(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except Exception:
        return value


def _compact_optional(value: Any, max_len: int = 180) -> str | None:
    text = compact_text(value, max_len)
    return text or None


def format_paper_for_user(item: dict[str, Any], rank: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": rank,
        "title": item.get("title"),
    }

    optional_fields = {
        "year": item.get("year"),
        "score": _safe_number(item.get("score")),
        "citations": item.get("citations"),
        "authors": _compact_optional(item.get("authors"), 160),
        "venue": _compact_optional(item.get("venue"), 100),
        "id": _compact_optional(item.get("id"), 160),
        "url": _compact_optional(item.get("url"), 180),
        "abstract": _compact_optional(item.get("abstract"), 300),
    }

    for key, value in optional_fields.items():
        if value not in (None, ""):
            result[key] = value

    return result


def format_author_for_user(item: dict[str, Any], rank: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": rank,
        "name": item.get("name"),
    }

    optional_fields = {
        "score": _safe_number(item.get("score")),
        "works_count": item.get("works_count"),
        "citation_count": item.get("citation_count"),
        "h_index": item.get("h_index"),
        "id": _compact_optional(item.get("id"), 160),
    }

    for key, value in optional_fields.items():
        if value not in (None, ""):
            result[key] = value

    return result


def collect_support_author_items(data: Any, max_authors: int = 10, max_papers_per_author: int = 3) -> list[dict[str, Any]]:
    """Collect author-centric support-paper results for support-papers output."""
    authors: list[dict[str, Any]] = []
    seen = set()

    for obj in collect_dicts_recursive(data):
        if not isinstance(obj.get("support_papers"), list):
            continue

        name = first_present(obj, ["author_name", "name", "display_name", "full_name"])
        author_id = first_present(obj, ["author_id", "id", "openalex_id", "url"])
        key = str(author_id or name or len(authors)).lower()
        if key in seen:
            continue
        seen.add(key)

        support_papers: list[dict[str, Any]] = []
        for idx, paper_obj in enumerate(obj.get("support_papers") or [], start=1):
            if not isinstance(paper_obj, dict):
                continue
            paper_item = as_paper_item(paper_obj) or {
                "title": compact_text(first_present(paper_obj, ["title", "paper_title", "name"]), 220),
                "year": first_present(paper_obj, ["year", "publication_year", "pub_year"]),
                "score": first_present(paper_obj, ["score", "similarity_score", "kg_score", "final_score"]),
                "citations": first_present(paper_obj, ["citations", "citation_count", "cited_by_count"]),
                "id": compact_text(first_present(paper_obj, ["paper_id", "id", "openalex_id", "doi", "url"]), 120),
                "url": compact_text(select_pdf_url(paper_obj), 220),
                "abstract": compact_text(first_present(paper_obj, ["abstract", "abstract_text"]), 300),
            }
            if paper_item.get("title"):
                support_papers.append(format_paper_for_user(paper_item, idx))
            if len(support_papers) >= max_papers_per_author:
                break

        authors.append(
            {
                "rank": len(authors) + 1,
                "name": compact_text(name, 120),
                "id": compact_text(author_id, 160),
                "support_papers": support_papers,
            }
        )

        if len(authors) >= max_authors:
            break

    return authors


def build_user_results(command: str, response: dict[str, Any], max_items: int = 10) -> dict[str, Any] | None:
    """Extract compact ranked papers/authors/support papers for stdout.

    The raw backend response is still saved under runs/, but the default CLI
    output should only contain user-facing retrieval content.
    """
    if not response or not response.get("ok"):
        return None

    data = response.get("data")
    max_items = max(1, int(max_items or 10))

    papers = collect_papers(data, max_items=max_items)
    authors = collect_authors(data, max_items=max_items)

    result: dict[str, Any] = {}

    if command in {"search-papers", "paper-search", "author-papers"}:
        if papers:
            result["papers"] = [format_paper_for_user(item, idx) for idx, item in enumerate(papers, start=1)]

    elif command == "related-authors":
        if authors:
            result["authors"] = [format_author_for_user(item, idx) for idx, item in enumerate(authors, start=1)]
        if papers:
            result["supporting_papers"] = [
                format_paper_for_user(item, idx) for idx, item in enumerate(papers[: min(5, max_items)], start=1)
            ]

    elif command == "support-papers":
        support_authors = collect_support_author_items(data, max_authors=max_items, max_papers_per_author=3)
        if support_authors:
            result["authors"] = support_authors
        elif authors:
            result["authors"] = [format_author_for_user(item, idx) for idx, item in enumerate(authors, start=1)]
        if papers:
            result["supporting_papers"] = [format_paper_for_user(item, idx) for idx, item in enumerate(papers, start=1)]

    else:
        if papers:
            result["papers"] = [format_paper_for_user(item, idx) for idx, item in enumerate(papers, start=1)]
        if authors:
            result["authors"] = [format_author_for_user(item, idx) for idx, item in enumerate(authors, start=1)]

    return result or None


def _request_query_text(plan: dict[str, Any] | None, request: dict[str, Any] | None) -> str | None:
    if isinstance(plan, dict):
        value = plan.get("query_text")
        if value:
            return compact_text(value, 300)

    if isinstance(request, dict):
        if isinstance(request.get("plan"), dict):
            value = request["plan"].get("query_text")
            if value:
                return compact_text(value, 300)
        value = request.get("query_text")
        if value:
            return compact_text(value, 300)

    return None


def _compact_error_message(response: dict[str, Any] | None) -> Any:
    if not response:
        return "No response was produced."

    err = response.get("error")
    if isinstance(err, dict):
        for key in ("detail", "message", "error"):
            if key in err:
                return err[key]
        return err

    if err:
        return compact_text(err, 500)

    return response.get("error_type") or "Request failed."


def build_console_payload(
    *,
    command: str,
    plan: dict[str, Any] | None,
    request: dict[str, Any] | None,
    response: dict[str, Any] | None,
    artifacts: dict[str, str],
    max_items: int,
) -> dict[str, Any]:
    """Default stdout payload: concise, user-facing, no debug blocks."""
    if response is None:
        payload: dict[str, Any] = {
            "ok": True,
            "command": command,
        }

        query = _request_query_text(plan, request)
        if query:
            payload["query"] = query

        # build-plan is the only command where the plan itself is the user-facing result.
        if plan is not None:
            payload["plan"] = plan

        return payload

    ok = bool(response.get("ok"))
    payload = {
        "ok": ok,
        "command": command,
    }

    query = _request_query_text(plan, request)
    if query:
        payload["query"] = query

    if response.get("elapsed_seconds") is not None:
        payload["elapsed_seconds"] = response.get("elapsed_seconds")

    if ok:
        user_results = build_user_results(command, response, max_items=max_items)
        if user_results:
            payload.update(user_results)
        else:
            payload["message"] = build_summary(command, response)

        # One concise pointer to the readable report is useful but not noisy.
        if artifacts.get("report_md"):
            payload["report"] = artifacts["report_md"]
    else:
        payload.update(
            {
                "status_code": response.get("status_code"),
                "error_type": response.get("error_type"),
                "message": _compact_error_message(response),
            }
        )
        # On failure, keep a single pointer for diagnosis.
        if artifacts.get("response_json"):
            payload["response_json"] = artifacts["response_json"]

    return payload


def _extract_report_context(request: dict[str, Any], response: dict[str, Any], max_items: int) -> dict[str, Any]:
    data = response.get("data")
    papers = collect_papers(data, max_items=max_items)
    authors = collect_authors(data, max_items=max_items)

    if isinstance(request.get("plan"), dict):
        plan = request.get("plan") or {}
        query_text = plan.get("query_text", "")
        options = request.get("options", {})
        keywords = plan.get("keywords", []) or []
        titles = plan.get("titles", []) or []
        reference_titles = plan.get("reference_titles", []) or []
    else:
        plan = {}
        query_text = request.get("query_text", "")
        options = request.get("options", {})
        keywords = []
        titles = []
        reference_titles = []

    return {
        "data": data,
        "papers": papers,
        "authors": authors,
        "plan": plan,
        "query_text": query_text,
        "options": options if isinstance(options, dict) else {},
        "keywords": keywords,
        "titles": titles,
        "reference_titles": reference_titles,
    }


def _md_value(value: Any, max_len: int = 120) -> str:
    return md_escape(compact_text(value, max_len))


def _paper_markdown_table(papers: list[dict[str, Any]], *, include_abstract: bool = False) -> list[str]:
    if not papers:
        return ["No parsed paper results."]

    if include_abstract:
        lines = ["| Rank | Title | Year | Score | Cites | Abstract Hint |", "|---:|---|---:|---:|---:|---|"]
    else:
        lines = ["| Rank | Title | Year | Score | Cites | PDF |", "|---:|---|---:|---:|---:|---|"]

    for idx, p in enumerate(papers, start=1):
        if include_abstract:
            lines.append(
                f"| {idx} | {_md_value(p.get('title'), 160)} | {_md_value(p.get('year'))} | "
                f"{_md_value(p.get('score'))} | {_md_value(p.get('citations'))} | {_md_value(p.get('abstract'), 220)} |"
            )
        else:
            lines.append(
                f"| {idx} | {_md_value(p.get('title'), 180)} | {_md_value(p.get('year'))} | "
                f"{_md_value(p.get('score'))} | {_md_value(p.get('citations'))} | {_md_value(p.get('url') or '-', 180)} |"
            )
    return lines


def _author_markdown_table(authors: list[dict[str, Any]]) -> list[str]:
    if not authors:
        return ["No parsed author results."]
    lines = ["| Rank | Author | Score | Works | Citations | H-index | ID |", "|---:|---|---:|---:|---:|---:|---|"]
    for idx, a in enumerate(authors, start=1):
        lines.append(
            f"| {idx} | {_md_value(a.get('name'))} | {_md_value(a.get('score'))} | "
            f"{_md_value(a.get('works_count'))} | {_md_value(a.get('citation_count'))} | "
            f"{_md_value(a.get('h_index'))} | {_md_value(a.get('id'), 140)} |"
        )
    return lines


def _paper_year_timeline(papers: list[dict[str, Any]]) -> list[str]:
    year_map: dict[str, list[str]] = {}
    for p in papers:
        year = str(p.get("year") or "Unknown")
        year_map.setdefault(year, []).append(str(p.get("title") or "Untitled"))
    if not year_map:
        return ["No timeline can be formed from the parsed results."]
    lines = ["| Year | Representative Papers |", "|---:|---|"]
    for year in sorted(year_map.keys()):
        titles = "; ".join(compact_text(t, 90) for t in year_map[year][:3])
        lines.append(f"| {_md_value(year)} | {_md_value(titles, 260)} |")
    return lines


def _extract_topic_terms(papers: list[dict[str, Any]], max_terms: int = 12) -> list[str]:
    text = " ".join(str(p.get("title") or "") + " " + str(p.get("abstract") or "") for p in papers)
    tokens = tokenize_english(text)
    grams = generate_english_ngrams(tokens, min_n=1, max_n=3)
    scored = score_keyword_candidates(grams)
    terms: list[str] = []
    for term, _ in scored:
        if len(term) < 3:
            continue
        if term in EN_STOPWORDS:
            continue
        if any(term == existing or term in existing or existing in term for existing in terms):
            continue
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def _representative_insights(papers: list[dict[str, Any]], max_items: int = 5) -> list[str]:
    if not papers:
        return ["No representative papers are available for interpretation."]
    lines: list[str] = []
    for idx, p in enumerate(papers[:max_items], start=1):
        title = compact_text(p.get("title"), 180)
        abstract = compact_text(p.get("abstract"), 260)
        year = p.get("year") or "Unknown year"
        cites = p.get("citations") if p.get("citations") not in (None, "") else "-"
        if abstract:
            lines.append(f"{idx}. **{md_escape(title)}** ({md_escape(year)}, Cites={md_escape(cites)}): {md_escape(abstract, 260)}")
        else:
            lines.append(f"{idx}. **{md_escape(title)}** ({md_escape(year)}, Cites={md_escape(cites)}): inspect the full paper or abstract to assess its contribution.")
    return lines


def _reproducibility_section(request: dict[str, Any], response: dict[str, Any], section_index: int) -> list[str]:
    return [
        f"## {section_index}. Reproducibility Record",
        "",
        "### Request JSON",
        "",
        "```json",
        json.dumps(request, ensure_ascii=False, indent=2),
        "```",
        "",
        "### Response Metadata",
        "",
        "```json",
        json.dumps(
            {
                "ok": response.get("ok"),
                "status_code": response.get("status_code"),
                "endpoint": response.get("endpoint"),
                "url": response.get("url"),
                "elapsed_seconds": response.get("elapsed_seconds"),
                "error_type": response.get("error_type"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "Note: real API keys are not saved or displayed in this report.",
        "",
    ]


def _report_header(command: str, title: str, now: str, query_text: str, response: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    options = ctx.get("options") or {}
    keywords = ctx.get("keywords") or []
    titles = ctx.get("titles") or []
    reference_titles = ctx.get("reference_titles") or []

    lines = [
        f"# SciScholar Downstream Channel Report: {command}",
        "",
        f"> {title}",
        "",
        "## 1. Task Overview",
        "",
        f"- Generated at: `{now}`",
        f"- Query text: {md_escape(query_text, 400)}",
        f"- Call status: `{response.get('ok')}`; HTTP status: `{response.get('status_code')}`; elapsed: `{response.get('elapsed_seconds')}` seconds",
    ]
    if options:
        compact_options = {k: v for k, v in options.items() if k not in {"api_key"}}
        lines.append(f"- Retrieval options: `{compact_text(compact_options, 500)}`")
    if keywords:
        lines.append("- Keyword anchors: " + ", ".join(f"`{md_escape(k.get('text'))}`({md_escape(k.get('score'))})" for k in keywords[:8]))
    if titles:
        lines.append("- Title anchors: " + ", ".join(f"`{md_escape(t.get('title'), 80)}`({md_escape(t.get('confidence'))})" for t in titles[:5]))
    if reference_titles:
        lines.append("- Reference anchors: " + ", ".join(f"`{md_escape(t, 80)}`" for t in reference_titles[:5]))
    lines.append("")
    return lines



def render_literature_review_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int, now: str) -> str:
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    authors = ctx["authors"]
    query_text = ctx["query_text"]
    topics = _extract_topic_terms(papers, max_terms=10)

    lines = _report_header(command, "Literature Review: organize background, method lines, representative papers, and limitations from the core paper pool.", now, query_text, response, ctx)
    lines += [
        "## 2. Review Guide",
        "",
        "This report is not just a paper list. It organizes retrieval results into review-ready materials. A recommended writing order is: background problem -> method taxonomy -> representative papers -> limitations and opportunities.",
        "",
        "## 3. Core Paper Pool",
        "",
        *_paper_markdown_table(papers),
        "",
        "## 4. Topic Structure and Keyword Profile",
        "",
    ]
    if topics:
        lines += ["High-frequency topic cues extracted from titles and abstracts: " + ", ".join(f"`{md_escape(t)}`" for t in topics), ""]
    else:
        lines += ["The current results are insufficient for a stable topic profile.", ""]
    lines += [
        "Use these cues to structure the review into sections such as:",
        "",
        "1. **Problem Definition and Background**: what problem the direction addresses and why it matters.",
        "2. **Method Evolution**: organize by model architecture, memory mechanism, planning style, interaction environment, or retrieval path.",
        "3. **Representative Systems or Methods**: prioritize high-score, high-citation, and recent papers.",
        "4. **Evaluation Protocols and Environments**: compare benchmarks, task settings, baselines, and metrics.",
        "5. **Limitations and Future Directions**: summarize gaps in generalization, long-horizon planning, tool use, or real-environment transfer.",
        "",
        "## 5. Timeline View",
        "",
        *_paper_year_timeline(papers),
        "",
        "## 6. Representative Papers",
        "",
        *_representative_insights(papers, max_items=min(5, max_items)),
        "",
    ]
    if authors:
        lines += ["## 7. Related Authors (Auxiliary)", "", *_author_markdown_table(authors[: min(8, max_items)]), ""]
        next_index = 8
    else:
        next_index = 7
    lines += [
        f"## {next_index}. Writing Suggestions",
        "",
        "- Build the main thread from high-score papers, then use high-citation papers for historical context.",
        "- For each representative paper, record: problem definition, core method, experimental setting, main conclusion, and limitation.",
        "- If the papers span many years, add a section on stage-wise evolution.",
        "- If the results are too scattered, reduce `--bias-exploration` or switch to `--ranking-profile precision`.",
        "",
    ]
    lines += _reproducibility_section(request, response, next_index + 1)
    return "\n".join(lines)


def render_idea_grounding_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int, now: str) -> str:
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    query_text = ctx["query_text"]
    lines = _report_header(command, "Idea Grounding: locate similar work, supporting evidence, and differentiation space.", now, query_text, response, ctx)
    lines += [
        "## 2. Grounding Entry Point",
        "",
        "Treat the following papers as an evidence pool for the idea, not as a final judgment. Compare them across motivation, methodology, and evaluation settings.",
        "",
        "## 3. Similar or Supporting Papers",
        "",
        *_paper_markdown_table(papers, include_abstract=True),
        "",
        "## 4. Multi-Dimensional Comparison Template",
        "",
        "| Dimension | Question to Check | How to Use the Papers |",
        "|---|---|---|",
        "| Motivation | Has the target problem already been proposed? | Inspect problem definitions, application scenarios, and motivations in abstracts. |",
        "| Methodology | Is there an existing method with a similar structure? | Compare model structure, retrieval path, reasoning pipeline, and experiment workflow. |",
        "| Evaluation | Are there existing evaluation protocols? | Record benchmarks, metrics, baselines, and task settings. |",
        "| Difference | Where does your idea differ from prior work? | Specify whether the difference is task, assumption, method, data, or evaluation. |",
        "",
        "## 5. Initial Risk Notes",
        "",
        "- If top papers strongly overlap with the idea in title and abstract, inspect novelty first.",
        "- If only loosely related papers are retrieved, the idea may be new, or the query/anchors may be underspecified.",
        "- If evidence papers are old, add recent results to test whether the direction remains active.",
        "",
    ]
    lines += _reproducibility_section(request, response, 6)
    return "\n".join(lines)


def render_idea_evaluate_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int, now: str) -> str:
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    query_text = ctx["query_text"]
    recent = [p for p in papers if str(p.get("year") or "").isdigit() and int(str(p.get("year"))) >= 2020]
    highly_cited = []
    for p in papers:
        try:
            if float(p.get("citations") or 0) >= 50:
                highly_cited.append(p)
        except Exception:
            pass

    lines = _report_header(command, "Idea Evaluation: collect evidence for novelty, feasibility, soundness, and differentiation.", now, query_text, response, ctx)
    lines += [
        "## 2. Evaluation Summary",
        "",
        f"- Parsed papers: `{len(papers)}`",
        f"- Recent papers (>=2020): `{len(recent)}`",
        f"- Highly cited papers (>=50 cites): `{len(highly_cited)}`",
        "",
        "> Note: this report summarizes KG retrieval evidence. It is not a final LLM-as-a-judge score. Use it as input for manual review or a later reviewer module.",
        "",
        "## 3. Evidence Paper Pool",
        "",
        *_paper_markdown_table(papers, include_abstract=True),
        "",
        "## 4. Evaluation Rubric",
        "",
        "| Dimension | How the Evidence Helps | Suggested Human Check |",
        "|---|---|---|",
        "| Novelty | Highly similar titles/abstracts indicate higher novelty risk. | Check whether top-5 papers already cover the core claim. |",
        "| Feasibility | Existing methods or protocols strengthen feasibility. | Look for datasets, systems, experiments, or benchmarks in abstracts. |",
        "| Soundness | Highly cited or recent papers may support the assumptions. | Check whether support papers align with the idea's core assumptions. |",
        "| Differentiation | Similar papers help identify the unique variable. | Clarify whether the difference is task, scenario, method, data, or metric. |",
        "",
        "## 5. Initial Evaluation Notes",
        "",
    ]
    if not papers:
        lines.append("- No strongly related papers were found. The query may be too narrow, or the idea may be relatively novel; try adding more anchors.")
    else:
        lines.append("- If top-ranked papers strongly overlap with the core idea, inspect novelty risk first.")
        lines.append("- If the results cover multiple sub-directions, the idea may have cross-topic combination potential.")
        lines.append("- If papers concentrate in the last three years, the direction is likely active; if concentrated in earlier years, check whether mature conclusions already exist.")
    lines += ["", *_reproducibility_section(request, response, 6)]
    return "\n".join(lines)


def render_idea_generate_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int, now: str) -> str:
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    query_text = ctx["query_text"]
    topics = _extract_topic_terms(papers, max_terms=8)
    combos = []
    if topics:
        base = compact_text(query_text, 60)
        for t in topics[:5]:
            combos.append(f"{base} + {t}")

    lines = _report_header(command, "Idea Generation: use exploratory KG retrieval to find cross-topic combinations and possible research gaps.", now, query_text, response, ctx)
    lines += [
        "## 2. Idea Generation Evidence Pool",
        "",
        *_paper_markdown_table(papers, include_abstract=True),
        "",
        "## 3. Combinable Topic Seeds",
        "",
    ]
    if combos:
        for idx, combo in enumerate(combos, start=1):
            lines.append(f"{idx}. {md_escape(combo, 160)}")
    else:
        lines.append("No stable topic seeds were extracted.")
    lines += [
        "",
        "## 4. Candidate Idea Template",
        "",
        "Use the topic seeds above to form new ideas:",
        "",
        "| Idea Component | Prompt |",
        "|---|---|",
        "| Problem | Choose a recurring but insufficiently solved problem. |",
        "| New Angle | Use another retrieved topic as a new constraint, scenario, or tool. |",
        "| Method Sketch | Explain how the two topics connect into a testable method. |",
        "| Validation | Borrow benchmarks, metrics, or baselines from evidence papers. |",
        "| Risk | Mark similar existing work and implementation difficulties. |",
        "",
        "## 5. Generation Suggestions",
        "",
        "- For more novelty, increase `--bias-exploration` or use `--ranking-profile discovery`.",
        "- To reduce overly broad results, lower `--bias-exploration` and add high-level keyword anchors.",
        "- Run `idea-grounding` on each candidate idea to check similar work.",
        "",
    ]
    lines += _reproducibility_section(request, response, 6)
    return "\n".join(lines)


def render_trend_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int, now: str) -> str:
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    query_text = ctx["query_text"]
    lines = _report_header(command, "Research Trend Analysis: observe direction evolution through timeline and impact evidence.", now, query_text, response, ctx)
    lines += [
        "## 2. Trend Evidence Pool",
        "",
        *_paper_markdown_table(papers),
        "",
        "## 3. Timeline Overview",
        "",
        *_paper_year_timeline(papers),
        "",
        "## 4. Trend Interpretation Suggestions",
        "",
        "- Read papers by year and mark the representative problem and method in each stage.",
        "- Highly cited papers often indicate stable problems or classic methods.",
        "- Recent low-citation papers may indicate emerging directions; inspect abstracts to confirm.",
        "- If results concentrate in one year, expand `--time-range` or increase `--top-k`.",
        "",
        "## 5. Suggested Trend Report Structure",
        "",
        "1. Starting point and early motivation.",
        "2. Rapid method growth stage.",
        "3. Recent systematization and engineering trends.",
        "4. Open problems and future opportunities.",
        "",
    ]
    lines += _reproducibility_section(request, response, 6)
    return "\n".join(lines)


def render_researcher_review_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int, now: str) -> str:
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    query_text = ctx["query_text"]
    topics = _extract_topic_terms(papers, max_terms=10)
    lines = _report_header(command, "Researcher Background Review: organize author papers, representative work, and research trajectory.", now, query_text, response, ctx)
    lines += [
        "## 2. Author Paper Pool",
        "",
        *_paper_markdown_table(papers),
        "",
        "## 3. Research Topic Profile",
        "",
    ]
    if topics:
        lines.append(", ".join(f"`{md_escape(t)}`" for t in topics))
    else:
        lines.append("No stable topic profile was extracted.")
    lines += [
        "",
        "## 4. Research Trajectory Cues",
        "",
        *_paper_year_timeline(papers),
        "",
        "## 5. Background Review Suggestions",
        "",
        "- First inspect whether the author's themes shift over time.",
        "- Then identify representative works by citation count and relevance.",
        "- Finally cluster papers into 2-4 research stages or directions.",
        "",
    ]
    lines += _reproducibility_section(request, response, 6)
    return "\n".join(lines)


def render_generic_markdown_report(*, command: str, request: dict[str, Any], response: dict[str, Any], max_items: int = 10) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx = _extract_report_context(request, response, max_items)
    papers = ctx["papers"]
    authors = ctx["authors"]
    query_text = ctx["query_text"]
    options = ctx["options"]

    lines: list[str] = []
    lines.append("# SciScholar / KG2API Retrieval Report")
    lines.append("")
    lines.append("## 1. Basic Information")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Backend endpoint: `{response.get('endpoint')}`")
    lines.append(f"- Call status: `{response.get('ok')}`")
    lines.append(f"- HTTP status: `{response.get('status_code')}`")
    lines.append(f"- Elapsed: `{response.get('elapsed_seconds')}` seconds")
    lines.append("")
    lines.append("## 2. Search Input")
    lines.append("")
    lines.append(f"- Query text: {compact_text(query_text, 300)}")
    lines.append("")

    if isinstance(options, dict) and options:
        lines.append("### Retrieval Options")
        lines.append("")
        for k, v in options.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    lines.append("## 3. Brief Summary")
    lines.append("")
    lines.append(build_summary(command, response))
    lines.append("")

    section_index = 4
    if response.get("ok") and papers:
        lines.append(f"## {section_index}. Candidate Papers")
        lines.append("")
        lines.extend(_paper_markdown_table(papers))
        lines.append("")
        section_index += 1

    if response.get("ok") and authors:
        lines.append(f"## {section_index}. Related Authors")
        lines.append("")
        lines.extend(_author_markdown_table(authors))
        lines.append("")
        section_index += 1

    if not papers and not authors and response.get("ok"):
        lines.append(f"## {section_index}. Raw Result Notice")
        lines.append("")
        lines.append("No standard paper or author entries were parsed automatically. Please inspect `response.json`.")
        lines.append("")
        section_index += 1

    if not response.get("ok"):
        lines.append(f"## {section_index}. Error Information")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(response.get("error"), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        section_index += 1

    lines.extend(_reproducibility_section(request, response, section_index))
    return "\n".join(lines)


def render_markdown_report(
    *,
    command: str,
    request: dict[str, Any],
    response: dict[str, Any],
    max_items: int = 10,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if command == "literature-review":
        return render_literature_review_report(command=command, request=request, response=response, max_items=max_items, now=now)
    if command == "idea-grounding":
        return render_idea_grounding_report(command=command, request=request, response=response, max_items=max_items, now=now)
    if command == "idea-evaluate":
        return render_idea_evaluate_report(command=command, request=request, response=response, max_items=max_items, now=now)
    if command == "idea-generate":
        return render_idea_generate_report(command=command, request=request, response=response, max_items=max_items, now=now)
    if command == "trend-report":
        return render_trend_report(command=command, request=request, response=response, max_items=max_items, now=now)
    if command == "researcher-review":
        return render_researcher_review_report(command=command, request=request, response=response, max_items=max_items, now=now)

    return render_generic_markdown_report(command=command, request=request, response=response, max_items=max_items)
def save_artifacts(

    *,

    run_dir: Path,

    command: str,

    plan: dict[str, Any] | None,

    request: dict[str, Any] | None,

    response: dict[str, Any] | None,

    report_max_items: int,

) -> dict[str, str]:

    artifacts: dict[str, str] = {}



    if plan is not None:

        plan_path = run_dir / "plan.json"

        write_json(plan_path, plan)

        artifacts["plan_json"] = str(plan_path)



    if request is not None:

        request_path = run_dir / "request.json"

        write_json(request_path, request)

        artifacts["request_json"] = str(request_path)



    if response is not None:

        response_path = run_dir / "response.json"

        write_json(response_path, response)

        artifacts["response_json"] = str(response_path)



        summary = build_summary(command, response)

        summary_path = run_dir / "summary.txt"

        summary_path.write_text(summary + "\n", encoding="utf-8")

        artifacts["summary_txt"] = str(summary_path)



        if request is not None:

            report_path = run_dir / "report.md"

            report = render_markdown_report(

                command=command,

                request=request,

                response=response,

                max_items=report_max_items,

            )

            report_path.write_text(report, encoding="utf-8")

            artifacts["report_md"] = str(report_path)



    metadata = {

        "command": command,

        "created_at": datetime.now().isoformat(),

        "ok": response.get("ok") if response else True,

        "status_code": response.get("status_code") if response else None,

        "endpoint": response.get("endpoint") if response else None,

        "elapsed_seconds": response.get("elapsed_seconds") if response else None,

    }



    metadata_path = run_dir / "metadata.json"

    write_json(metadata_path, metadata)

    artifacts["metadata_json"] = str(metadata_path)

    artifacts["run_dir"] = str(run_dir)



    return artifacts





def finish_with_artifacts(

    *,

    args: argparse.Namespace,

    command: str,

    plan: dict[str, Any] | None,

    request: dict[str, Any] | None,

    response: dict[str, Any] | None,

    prefix: str,

) -> int:

    run_dir = ensure_run_dir(args.runs_dir, args.run_id, prefix)



    artifacts = save_artifacts(

        run_dir=run_dir,

        command=command,

        plan=plan,

        request=request,

        response=response,

        report_max_items=args.report_max_items,

    )



    payload = build_console_payload(

        command=command,

        plan=plan,

        request=request,

        response=response,

        artifacts=artifacts,

        max_items=args.report_max_items,

    )



    print(render_user_output(payload))



    if response is None:

        return 0

    return 0 if response.get("ok") else 1




# ============================================================

# 命令实现

# ============================================================



def cmd_health(args: argparse.Namespace) -> int:

    response = request_json(

        method="GET",

        base_url=args.base_url,

        endpoint="/healthz",

        api_key=None,

        timeout=args.timeout,

    )

    print_json(response)

    return 0 if response.get("ok") else 1





def cmd_config(args: argparse.Namespace) -> int:

    print_json(

        {

            "ok": True,

            "skill": "scischolar-cli-skill",

            "base_url": args.base_url,

            "api_key_configured": bool(args.api_key),
            "optional_provider_config": optional_provider_config_status(),

            "runs_dir": args.runs_dir,

            "commands": [

                "health",

                "config",

                "build-plan",

                "search-papers",

                "related-authors",

                "author-papers",

                "support-papers",

                "paper-search",

                "make-report",
                "literature-review",
                "idea-grounding",
                "idea-evaluate",
                "idea-generate",
                "trend-report",
                "researcher-review",

            ],

            "mode": "natural-language-parameter-extraction-with-downstream-channels",

        }

    )

    return 0





def cmd_build_plan(args: argparse.Namespace) -> int:

    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))
    text = append_soft_domain_to_query(text, getattr(args, "domain", None))
    text = merge_expert_plan_controls(args, text)



    plan = build_plan_from_text(

        text=text,

        source_type=args.source_type,

        source_title=args.source_title,

        top_keywords=args.top_keywords,

        max_titles=args.max_titles,

        max_refs=args.max_refs,

    )



    return finish_with_artifacts(

        args=args,

        command="build-plan",

        plan=plan,

        request=None,

        response=None,

        prefix="build_plan",

    )






# ============================================================
# 论文下游体验频道：复用现有检索接口的任务化封装
# ============================================================

DOWNSTREAM_CHANNEL_HINTS = {
    "literature-review": "Retrieved core papers for the Literature Review channel. Open report.md to organize background, method lines, representative works, and limitations.",
    "idea-grounding": "Retrieved related and supporting papers for the Idea Grounding channel. Compare motivation, methodology, and evaluation settings.",
    "idea-evaluate": "Collected KG evidence for the Idea Evaluation channel. Continue judging Novelty, Feasibility, and Soundness manually or with a reviewer module.",
    "idea-generate": "Enabled a more exploratory graph-discovery profile for Idea Generation. Look for cross-topic combinations and research gaps.",
    "trend-report": "Emphasized citation impact and timeline evidence for Research Trend Analysis. Read papers by year to summarize stage-wise evolution.",
    "researcher-review": "Collected researcher-related papers for Researcher Review. Use year, citation count, and topic clusters to analyze research trajectory.",
}


def _apply_channel_hint_to_report(report_path: str | None, command: str) -> None:
    if not report_path or command not in DOWNSTREAM_CHANNEL_HINTS:
        return
    path = Path(report_path)
    if not path.exists():
        return
    try:
        original = path.read_text(encoding="utf-8")
        hint = DOWNSTREAM_CHANNEL_HINTS[command]
        header = f"# SciScholar Downstream Channel Report: {command}\n\n> {hint}\n\n"
        if not (
            original.startswith("# SciScholar Downstream Channel Report")
            or original.startswith("# SciNet Downstream Channel Report")
        ):
            path.write_text(header + original, encoding="utf-8")
    except Exception:
        pass


def _run_plan_search_channel(
    *,
    args: argparse.Namespace,
    command: str,
    prefix: str,
    endpoint: str = "/v1/search",
    default_top_k: int = 5,
) -> int:
    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))
    text = append_soft_domain_to_query(text, getattr(args, "domain", None))
    text = merge_expert_plan_controls(args, text)

    plan = build_plan_from_text(
        text=text,
        source_type=args.source_type,
        source_title=args.source_title,
        top_keywords=args.top_keywords,
        max_titles=args.max_titles,
        max_refs=args.max_refs,
    )

    options = build_options_from_text(
        text=text,
        default_top_k=default_top_k,
        cli_top_k=args.top_k,
        cli_target_field=args.target_field,
        cli_after=args.after,
        cli_before=args.before,
        cli_time_range=getattr(args, "time_range", None),
    )

    retrieval_bias = build_retrieval_bias_from_args(args)
    if retrieval_bias:
        options["retrieval_bias"] = retrieval_bias

    retrieval_mode = getattr(args, "retrieval_mode", None)
    if retrieval_mode:
        options["retrieval_mode"] = retrieval_mode

    request = {"plan": plan, "options": options}

    response = request_json(
        method="POST",
        base_url=args.base_url,
        endpoint=endpoint,
        api_key=args.api_key,
        payload=request,
        timeout=args.timeout,
    )

    run_dir = ensure_run_dir(args.runs_dir, args.run_id, prefix)
    artifacts = save_artifacts(
        run_dir=run_dir,
        command=command,
        plan=plan,
        request=request,
        response=response,
        report_max_items=args.report_max_items,
    )
    _apply_channel_hint_to_report(artifacts.get("report_md"), command)

    payload = build_console_payload(
        command=command,
        plan=plan,
        request=request,
        response=response,
        artifacts=artifacts,
        max_items=args.report_max_items,
    )
    if response and response.get("ok") and command in DOWNSTREAM_CHANNEL_HINTS:
        payload["channel_hint"] = DOWNSTREAM_CHANNEL_HINTS[command]
        channel_view = build_downstream_channel_view(
            command=command,
            payload=payload,
            report_path=artifacts.get("report_md"),
            max_items=args.report_max_items,
        )
        if channel_view:
            payload["channel_view"] = channel_view

    print(render_user_output(payload))
    return 0 if response.get("ok") else 1


def _run_author_papers_channel(*, args: argparse.Namespace, command: str, prefix: str, default_limit: int = 20) -> int:
    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))

    # researcher-review 支持 --author "Yoshua Bengio" 这类专家式直填。
    # 由于 --author 当前复用 dest="query"，这里优先把 args.query 作为显式作者名处理；
    # 同时兼容 --query "papers by Yoshua Bengio" / "author: Yoshua Bengio"。
    author = None
    explicit_author = getattr(args, "query", None)
    if explicit_author:
        direct_author = normalize_text(explicit_author)

        m = re.match(
            r"^(?:papers|works|publications)\s+(?:by|of|from)\s+(.+)$",
            direct_author,
            flags=re.IGNORECASE,
        )
        if m:
            direct_author = normalize_text(m.group(1))

        direct_author = clean_author_name(direct_author)
        if direct_author:
            author = direct_author

    if not author:
        author = extract_single_author(text)

    if not author:
        print(render_user_output({
            "ok": False,
            "command": command,
            "query": text,
            "message": "Could not extract an author name. Try --author \"Geoffrey Hinton\" or --query \"papers by Geoffrey Hinton\".",
        }))
        return 2

    limit = args.limit if args.limit is not None else extract_top_k(text, default=default_limit)
    search_by = "id" if "openalex.org" in author.lower() or re.match(r"^[A-Za-z]\d+$", author) else "name"

    request = {
        "identifier": author,
        "search_by": search_by,
        "options": {
            "limit": limit,
            "merge_same_name_authors": not getattr(args, "no_merge_same_name_authors", False),
            "dedupe_papers": not getattr(args, "no_dedupe_papers", False),
            "include_abstract": not getattr(args, "no_abstract", False),
            "include_embeddings": getattr(args, "include_embeddings", False),
        },
    }

    response = request_json(
        method="POST",
        base_url=args.base_url,
        endpoint="/v1/authors/papers",
        api_key=args.api_key,
        payload=request,
        timeout=args.timeout,
    )

    run_dir = ensure_run_dir(args.runs_dir, args.run_id, prefix)
    artifacts = save_artifacts(
        run_dir=run_dir,
        command=command,
        plan=None,
        request=request,
        response=response,
        report_max_items=args.report_max_items,
    )
    _apply_channel_hint_to_report(artifacts.get("report_md"), command)

    payload = build_console_payload(
        command=command,
        plan=None,
        request={"query_text": author, **request},
        response=response,
        artifacts=artifacts,
        max_items=args.report_max_items,
    )
    if response and response.get("ok") and command in DOWNSTREAM_CHANNEL_HINTS:
        payload["channel_hint"] = DOWNSTREAM_CHANNEL_HINTS[command]
        channel_view = build_downstream_channel_view(
            command=command,
            payload=payload,
            report_path=artifacts.get("report_md"),
            max_items=args.report_max_items,
        )
        if channel_view:
            payload["channel_view"] = channel_view

    print(render_user_output(payload))
    return 0 if response.get("ok") else 1


def cmd_literature_review(args: argparse.Namespace) -> int:
    return _run_plan_search_channel(args=args, command="literature-review", prefix="literature_review", default_top_k=8)


def cmd_idea_grounding(args: argparse.Namespace) -> int:
    return _run_plan_search_channel(args=args, command="idea-grounding", prefix="idea_grounding", default_top_k=5)


def cmd_idea_evaluate(args: argparse.Namespace) -> int:
    return _run_plan_search_channel(args=args, command="idea-evaluate", prefix="idea_evaluate", default_top_k=5)


def cmd_idea_generate(args: argparse.Namespace) -> int:
    return _run_plan_search_channel(args=args, command="idea-generate", prefix="idea_generate", default_top_k=8)


def cmd_trend_report(args: argparse.Namespace) -> int:
    return _run_plan_search_channel(args=args, command="trend-report", prefix="trend_report", default_top_k=10)


def _extract_researcher_author_from_args(args: argparse.Namespace) -> tuple[str | None, str]:
    """Extract researcher name / OpenAlex Author ID from researcher-review args."""
    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))

    author = None
    explicit_author = getattr(args, "query", None)
    if explicit_author:
        direct_author = normalize_text(explicit_author)

        m = re.match(
            r"^(?:papers|works|publications)\s+(?:by|of|from)\s+(.+)$",
            direct_author,
            flags=re.IGNORECASE,
        )
        if m:
            direct_author = normalize_text(m.group(1))

        direct_author = clean_author_name(direct_author)
        if direct_author:
            author = direct_author

    if not author:
        author = extract_single_author(text)

    return author, text


def _run_researcher_review_full_kg(*, args: argparse.Namespace, default_limit: int = 20) -> int:
    """Run researcher-review through the full /v1/search KG retrieval path.

    Implementation strategy:
    1. Resolve the researcher name / author ID.
    2. Query /v1/authors/papers once as a lightweight seed step.
    3. Use the seed paper titles as high-confidence title anchors.
    4. Call /v1/search with hybrid retrieval, authorship/citation bias, and impact ranking.

    The author-papers command is intentionally unchanged and remains the fast
    /v1/authors/papers lookup.
    """
    command = "researcher-review"
    prefix = "researcher_review"

    author, original_text = _extract_researcher_author_from_args(args)

    if not author:
        print(render_user_output({
            "ok": False,
            "command": command,
            "query": original_text,
            "message": "Could not extract an author name. Try --author \"Geoffrey Hinton\" or --query \"papers by Geoffrey Hinton\".",
        }))
        return 2

    limit = args.limit if args.limit is not None else default_limit
    limit = max(1, min(int(limit), 100))

    search_by = "id" if "openalex.org" in author.lower() or re.match(r"^[A-Za-z]\d+$", author) else "name"

    # Lightweight seed lookup: used only to obtain title anchors for the KG search.
    seed_limit = max(1, min(limit, 5))
    seed_request = {
        "identifier": author,
        "search_by": search_by,
        "options": {
            "limit": seed_limit,
            "merge_same_name_authors": not getattr(args, "no_merge_same_name_authors", False),
            "dedupe_papers": not getattr(args, "no_dedupe_papers", False),
            "include_abstract": False,
            "include_embeddings": False,
        },
    }

    seed_response = request_json(
        method="POST",
        base_url=args.base_url,
        endpoint="/v1/authors/papers",
        api_key=args.api_key,
        payload=seed_request,
        timeout=args.timeout,
    )

    seed_papers = collect_papers(seed_response.get("data"), max_items=seed_limit) if seed_response.get("ok") else []

    title_anchors: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for paper in seed_papers:
        title = normalize_text(str(paper.get("title") or ""))
        if not title:
            continue
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        title_anchors.append({"title": title, "confidence": 0.95})

    query_text = f"Researcher background review for {author}"
    plan: dict[str, Any] = {
        "query_text": query_text,
        "source_type": "idea_text",
        "source_title": None,
        "keywords": [
            {"text": author, "score": 10},
            {"text": f"{author} research", "score": 8},
        ],
        "titles": title_anchors,
        "reference_titles": [],
    }

    # If seed titles are unavailable, keep a valid query-only plan with author keywords.
    if not plan["titles"] and not plan["keywords"]:
        plan["keywords"] = [{"text": query_text, "score": 8}]

    retrieval_bias = {
        "authorship": "high",
        "citation": "high",
        "paper_relatedness": "middle",
        "graph_exploration": "low",
        "ranking_profile": "impact",
    }

    request = {
        "plan": plan,
        "options": {
            "top_k": limit,
            "retrieval_mode": "hybrid",
            "retrieval_bias": retrieval_bias,
        },
    }

    response = request_json(
        method="POST",
        base_url=args.base_url,
        endpoint="/v1/search",
        api_key=args.api_key,
        payload=request,
        timeout=args.timeout,
    )

    run_dir = ensure_run_dir(args.runs_dir, args.run_id, prefix)
    seed_request_path = run_dir / "author_seed_request.json"
    seed_response_path = run_dir / "author_seed_response.json"
    write_json(seed_request_path, seed_request)
    write_json(seed_response_path, seed_response)

    artifacts = save_artifacts(
        run_dir=run_dir,
        command=command,
        plan=plan,
        request=request,
        response=response,
        report_max_items=args.report_max_items,
    )
    artifacts["author_seed_request_json"] = str(seed_request_path)
    artifacts["author_seed_response_json"] = str(seed_response_path)

    _apply_channel_hint_to_report(artifacts.get("report_md"), command)

    payload = build_console_payload(
        command=command,
        plan=plan,
        request=request,
        response=response,
        artifacts=artifacts,
        max_items=args.report_max_items,
    )
    if response and response.get("ok") and command in DOWNSTREAM_CHANNEL_HINTS:
        payload["channel_hint"] = DOWNSTREAM_CHANNEL_HINTS[command]
        channel_view = build_downstream_channel_view(
            command=command,
            payload=payload,
            report_path=artifacts.get("report_md"),
            max_items=args.report_max_items,
        )
        if channel_view:
            payload["channel_view"] = channel_view

    print(render_user_output(payload))
    return 0 if response.get("ok") else 1


def cmd_researcher_review(args: argparse.Namespace) -> int:
    return _run_researcher_review_full_kg(args=args, default_limit=20)


def cmd_search_papers(args: argparse.Namespace) -> int:

    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))
    text = append_soft_domain_to_query(text, getattr(args, "domain", None))
    text = merge_expert_plan_controls(args, text)



    plan = build_plan_from_text(

        text=text,

        source_type=args.source_type,

        source_title=args.source_title,

        top_keywords=args.top_keywords,

        max_titles=args.max_titles,

        max_refs=args.max_refs,

    )



    options = build_options_from_text(

        text=text,

        default_top_k=5,

        cli_top_k=args.top_k,

        cli_target_field=args.target_field,

        cli_after=args.after,

        cli_before=args.before,

        cli_time_range=getattr(args, "time_range", None),

    )

    retrieval_bias = build_retrieval_bias_from_args(args)
    if retrieval_bias:
        options["retrieval_bias"] = retrieval_bias

    retrieval_mode = getattr(args, "retrieval_mode", None)
    if retrieval_mode:
        options["retrieval_mode"] = retrieval_mode



    request = {

        "plan": plan,

        "options": options,

    }



    response = request_json(

        method="POST",

        base_url=args.base_url,

        endpoint="/v1/search",

        api_key=args.api_key,

        payload=request,

        timeout=args.timeout,

    )



    return finish_with_artifacts(

        args=args,

        command="search-papers",

        plan=plan,

        request=request,

        response=response,

        prefix="search_papers",

    )





def cmd_related_authors(args: argparse.Namespace) -> int:

    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))
    text = append_soft_domain_to_query(text, getattr(args, "domain", None))
    text = merge_expert_plan_controls(args, text)



    plan = build_plan_from_text(

        text=text,

        source_type=args.source_type,

        source_title=args.source_title,

        top_keywords=args.top_keywords,

        max_titles=args.max_titles,

        max_refs=args.max_refs,

    )



    options = build_options_from_text(

        text=text,

        default_top_k=10,

        cli_top_k=args.top_k,

        cli_target_field=args.target_field,

        cli_after=args.after,

        cli_before=args.before,

        cli_time_range=getattr(args, "time_range", None),

    )

    retrieval_bias = build_retrieval_bias_from_args(args)
    if retrieval_bias:
        options["retrieval_bias"] = retrieval_bias

    retrieval_mode = getattr(args, "retrieval_mode", None)
    if retrieval_mode:
        options["retrieval_mode"] = retrieval_mode



    request = {

        "plan": plan,

        "options": options,

    }



    response = request_json(

        method="POST",

        base_url=args.base_url,

        endpoint="/v1/authors/related",

        api_key=args.api_key,

        payload=request,

        timeout=args.timeout,

    )



    return finish_with_artifacts(

        args=args,

        command="related-authors",

        plan=plan,

        request=request,

        response=response,

        prefix="related_authors",

    )





def cmd_author_papers(args: argparse.Namespace) -> int:

    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))

    author = extract_single_author(text)



    if not author:

        print_json(

            {

                "ok": False,

                "skill_command": "author-papers",

                "error": "Could not extract an author name from natural language. Try author: Geoffrey Hinton or papers by Geoffrey Hinton.",

            }

        )

        return 2



    limit = args.limit if args.limit is not None else extract_top_k(text, default=20)



    search_by = "id" if "openalex.org" in author.lower() or re.match(r"^[A-Za-z]\d+$", author) else "name"



    request = {

        "identifier": author,

        "search_by": search_by,

        "options": {

            "limit": limit,

            "merge_same_name_authors": not args.no_merge_same_name_authors,

            "dedupe_papers": not args.no_dedupe_papers,

            "include_abstract": not args.no_abstract,

            "include_embeddings": args.include_embeddings,

        },

    }



    response = request_json(

        method="POST",

        base_url=args.base_url,

        endpoint="/v1/authors/papers",

        api_key=args.api_key,

        payload=request,

        timeout=args.timeout,

    )



    return finish_with_artifacts(

        args=args,

        command="author-papers",

        plan=None,

        request=request,

        response=response,

        prefix="author_papers",

    )





def cmd_support_papers(args: argparse.Namespace) -> int:

    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))

    authors = extract_author_candidates(text, max_authors=20)



    if not authors:

        print_json(

            {

                "ok": False,

                "skill_command": "support-papers",

                "error": "Could not extract candidate authors from natural language. Try: Candidate authors: Alice Smith, Bob Lee. Query topic: ...",

            }

        )

        return 2



    top_k_per_author = (

        args.top_k_per_author

        if args.top_k_per_author is not None

        else extract_top_k(text, default=3)

    )



    request = {

        "query_text": normalize_text(text),

        "authors": authors,

        "options": {

            "top_k_per_author": top_k_per_author,

            "fetch_author_stats": not args.no_author_stats,

            "author_search_fallback": args.author_search_fallback,

        },

    }



    response = request_json(

        method="POST",

        base_url=args.base_url,

        endpoint="/v1/authors/support-papers",

        api_key=args.api_key,

        payload=request,

        timeout=args.timeout,

    )



    return finish_with_artifacts(

        args=args,

        command="support-papers",

        plan=None,

        request=request,

        response=response,

        prefix="support_papers",

    )





def cmd_paper_search(args: argparse.Namespace) -> int:

    text = read_text_input(args.text, args.text_file, getattr(args, "query", None))
    text = append_soft_domain_to_query(text, getattr(args, "domain", None))



    top_k = args.top_k if args.top_k is not None else extract_top_k(text, default=5)



    request = {

        "query_text": normalize_text(text),

        "mode": args.mode,

        "field": args.field,

        "options": {

            "top_k": top_k,

        },

    }



    response = request_json(

        method="POST",

        base_url=args.base_url,

        endpoint="/v1/papers/search",

        api_key=args.api_key,

        payload=request,

        timeout=args.timeout,

    )



    return finish_with_artifacts(

        args=args,

        command="paper-search",

        plan=None,

        request=request,

        response=response,

        prefix="paper_search",

    )





def cmd_make_report(args: argparse.Namespace) -> int:

    request_obj = read_json(args.request_json)

    response_obj = read_json(args.response_json)



    response = response_obj.get("response", response_obj) if isinstance(response_obj, dict) else response_obj

    request = request_obj.get("request", request_obj) if isinstance(request_obj, dict) else request_obj



    run_dir = ensure_run_dir(args.runs_dir, args.run_id, "make_report")



    report = render_markdown_report(

        command=args.command_name,

        request=request,

        response=response,

        max_items=args.report_max_items,

    )



    report_path = run_dir / "report.md"

    report_path.write_text(report, encoding="utf-8")



    metadata = {

        "command": "make-report",

        "source_request_json": str(Path(args.request_json).resolve()),

        "source_response_json": str(Path(args.response_json).resolve()),

        "created_at": datetime.now().isoformat(),

    }



    metadata_path = run_dir / "metadata.json"

    write_json(metadata_path, metadata)



    print_json(

        {

            "ok": True,

            "skill_command": "make-report",

            "artifacts": {

                "run_dir": str(run_dir),

                "report_md": str(report_path),

                "metadata_json": str(metadata_path),

            },

        }

    )



    return 0





# ============================================================

# 参数定义

# ============================================================



def add_text_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--query", dest="query", default=None, help="Expert mode: primary query text. Takes precedence over --text/--text-file.")
    parser.add_argument("--查询", dest="query", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--domain", dest="domain", default=None, help="Soft domain hint appended to query_text; use --target-field for strict backend filtering.")
    parser.add_argument("--检索领域", dest="domain", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--text", default=None, help="Natural-language input.")

    parser.add_argument("--text-file", default=None, help="Read natural-language input from a file.")





def add_plan_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(

        "--source-type",

        default="idea_text",

        choices=["idea_text", "pdf"],

        help="Input source type.",

    )

    parser.add_argument(

        "--source-title",

        default=None,

        help="Source title. Recommended when source-type=pdf.",

    )

    parser.add_argument(

        "--top-keywords",

        type=int,

        default=8,

        help="Number of automatically extracted keywords.",

    )

    parser.add_argument(

        "--max-titles",

        type=int,

        default=5,

        help="Maximum number of title hints to extract.",

    )

    parser.add_argument(

        "--max-refs",

        type=int,

        default=10,

        help="Maximum number of reference titles to extract.",

    )

    parser.add_argument(

        "--keyword",

        dest="expert_keywords",

        action="append",

        default=[],

        help="Expert mode: keyword anchor; repeatable. Format: text:high or high:text; levels: low/middle/high.",

    )

    parser.add_argument(

        "--title",

        dest="expert_titles",

        action="append",

        default=[],

        help="Expert mode: title anchor; repeatable. Format: title:high or high:title.",

    )

    parser.add_argument(

        "--reference",

        "--ref",

        dest="expert_references",

        action="append",

        default=[],

        help="Expert mode: reference-title anchor; repeatable. Format: title:high or high:title.",

    )





def add_retrieval_control_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--top-k", type=int, default=None, help="Number of returned items. If omitted, it is inferred from text or defaults.")

    parser.add_argument("--target-field", dest="target_field", default=None, help="Strict backend field filter. Usually prefer --domain as a soft query hint.")

    parser.add_argument("--after", "--start-date", dest="after", default=None, help="Start date, YYYY-MM-DD. If omitted, inferred from text or --time-range.")

    parser.add_argument("--before", "--end-date", dest="before", default=None, help="End date, YYYY-MM-DD. If omitted, inferred from text or --time-range.")

    parser.add_argument("--time-range", dest="time_range", default=None, help="Expert mode: time range, e.g. 2020-2024, 2020-01-01..2024-12-31, or since 2020.")
    parser.add_argument("--时间范围", dest="time_range", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--retrieval-mode", choices=["keyword", "semantic", "title", "hybrid"], default="hybrid", help="Retrieval mode: keyword, semantic, title, or hybrid. Default: hybrid.")

    parser.add_argument("--bias-keyword", choices=["low", "middle", "high"], default=None, help="Keyword-association strength mapped to HAS_KEYWORD edge weights.")

    parser.add_argument("--bias-non-seed-keyword", choices=["low", "middle", "high"], default=None, help="Non-seed keyword expansion strength mapped to graph keyword smoothing.")

    parser.add_argument("--bias-citation", choices=["low", "middle", "high"], default=None, help="Citation-relation strength mapped to CITES edge weights.")

    parser.add_argument("--bias-related", choices=["low", "middle", "high"], default=None, help="Paper RELATED_TO relation strength.")

    parser.add_argument("--bias-authorship", choices=["low", "middle", "high"], default=None, help="Author-paper AUTHORED relation strength.")

    parser.add_argument("--bias-coauthorship", choices=["low", "middle", "high"], default=None, help="COAUTHOR network strength.")

    parser.add_argument("--bias-cooccurrence", choices=["low", "middle", "high"], default=None, help="Keyword COOCCUR co-occurrence network strength.")

    parser.add_argument("--bias-exploration", choices=["low", "middle", "high"], default=None, help="Graph exploration level; high is broader, low stays closer to seeds.")

    parser.add_argument("--ranking-profile", choices=["precision", "balanced", "discovery", "impact"], default=None, help="Final ranking profile.")





def add_output_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--run-id", default=None, help="Custom run ID.")

    parser.add_argument("--report-max-items", type=int, default=10, help="Maximum number of papers/authors shown in the report.")







def add_downstream_search_args(parser: argparse.ArgumentParser) -> None:
    add_text_args(parser)
    add_plan_args(parser)
    add_retrieval_control_args(parser)
    add_output_args(parser)


def add_researcher_review_args(parser: argparse.ArgumentParser) -> None:
    add_text_args(parser)
    add_output_args(parser)
    parser.add_argument("--author", dest="query", default=None, help="Researcher name or OpenAlex Author ID.")
    parser.add_argument("--limit", type=int, default=None, help="Number of returned papers.")
    parser.add_argument("--no-merge-same-name-authors", action="store_true", help="Do not merge same-name authors.")
    parser.add_argument("--no-dedupe-papers", action="store_true", help="Do not deduplicate papers.")
    parser.add_argument("--no-abstract", action="store_true", help="Do not return abstracts for seed author-paper lookup; researcher-review still uses the full KG retrieval path.")
    parser.add_argument("--include-embeddings", action="store_true", help="Return embeddings if supported by the backend.")

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="scischolar",
        usage="scischolar [global-options] <command> [command-options]",
        description="SciScholar CLI: scientific knowledge-graph retrieval and downstream research workflows.",
        epilog="""commands:
  health                 Check KG2API backend health
  config                 Show current CLI configuration
  build-plan             Build a structured search plan without backend calls
  search-papers          Search related papers
  related-authors        Retrieve related authors
  author-papers          Query papers by author
  support-papers         Retrieve support papers for candidate authors
  paper-search           Run lightweight low-level paper search
  make-report            Regenerate a Markdown report from saved artifacts

Skill system:\n  skill list             List editable downstream skills\n  skill run <name>       Run a skill preset\n  skill init <name>      Create a local editable skill\n\nDownstream channels:
  literature-review      Literature review material generation
  idea-grounding         Research idea grounding and related-work search
  idea-evaluate          Evidence collection for idea evaluation
  idea-generate          Research idea seed discovery
  trend-report           Research trend analysis
  researcher-review      Researcher background review

examples:
  scischolar health
  scischolar config
  scischolar paper-search --text "open world agent" --mode vector --field title --top-k 3
  scischolar search-papers --query "open world agent" --domain "artificial intelligence" --time-range 2020-2024 --keyword "high:open world agent" --top-k 3
  scischolar literature-review --query "open world agent" --domain "artificial intelligence" --time-range 2020-2024 --keyword "high:open world agent" --top-k 3
  scischolar idea-evaluate --idea "LLM-based multi-perspective evaluation for scientific research ideas" --keyword "high:idea evaluation" --top-k 3

input:
  --query TEXT                      expert query text
  --text TEXT                       natural-language input
  --domain FIELD                    soft domain hint appended to query_text
  --time-range RANGE                time range, e.g. 2020-2024
  --keyword "high:TERM"             keyword anchor, repeatable
  --title "middle:TITLE"            title anchor, repeatable
  --reference "low:TITLE"           reference-title anchor, repeatable
  --retrieval-mode MODE             keyword | semantic | title | hybrid
  --bias-*                          graph retrieval bias controls
  --ranking-profile PROFILE         precision | balanced | discovery | impact

tips:
  scischolar <command> -h               show command-specific help
  export KG2API_BASE_URL="http://127.0.0.1:8000" for local server testing
  reports are saved under runs/<run_id>/report.md""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )



    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="KG2API backend URL.")

    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="KG2API API key; environment variables are recommended.")

    parser.add_argument("--timeout", type=int, default=600, help="HTTP request timeout in seconds.")

    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory for saved run artifacts.")



    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="<command>",
        title="commands",
        description="Use 'scischolar <command> -h' for command-specific options.",
    )



    p = sub.add_parser("health", help="Check KG2API backend health.")

    p.set_defaults(func=cmd_health)



    p = sub.add_parser("config", help="Show base_url, API key status, and runs directory.")

    p.set_defaults(func=cmd_config)



    p = sub.add_parser("build-plan", help="Build a structured search plan without backend retrieval.")

    add_text_args(p)

    add_plan_args(p)

    add_output_args(p)

    p.set_defaults(func=cmd_build_plan)



    p = sub.add_parser("search-papers", help="Search related papers; supports keyword/semantic/title/hybrid.")

    add_text_args(p)

    add_plan_args(p)

    add_retrieval_control_args(p)

    add_output_args(p)

    p.set_defaults(func=cmd_search_papers)



    p = sub.add_parser("related-authors", help="Retrieve authors related to a research direction.")

    add_text_args(p)

    add_plan_args(p)

    add_retrieval_control_args(p)

    add_output_args(p)

    p.set_defaults(func=cmd_related_authors)



    p = sub.add_parser("author-papers", help="Query papers by a specified author.")

    add_text_args(p)

    add_output_args(p)

    p.add_argument("--limit", type=int, default=None, help="Number of returned papers. If omitted, inferred from text or defaults to 20.")

    p.add_argument("--no-merge-same-name-authors", action="store_true", help="Do not merge same-name authors.")

    p.add_argument("--no-dedupe-papers", action="store_true", help="Do not deduplicate papers.")

    p.add_argument("--no-abstract", action="store_true", help="Do not return abstracts.")

    p.add_argument("--include-embeddings", action="store_true", help="Return embeddings if supported by the backend.")

    p.set_defaults(func=cmd_author_papers)



    p = sub.add_parser("support-papers", help="Find topic-relevant support papers for candidate authors.")

    add_text_args(p)

    add_output_args(p)

    p.add_argument("--top-k-per-author", type=int, default=None, help="Number of support papers returned per author.")

    p.add_argument("--no-author-stats", action="store_true", help="Do not return author statistics.")

    p.add_argument(

        "--author-search-fallback",

        default="id_then_name",

        choices=["id_then_name", "name_then_id", "none"],

        help="Author lookup fallback strategy.",

    )

    p.set_defaults(func=cmd_support_papers)



    # Downstream user channels from the SciScholar paper.
    p = sub.add_parser("literature-review", help="Downstream channel: literature review material generation.")
    add_downstream_search_args(p)
    p.set_defaults(
        func=cmd_literature_review,
        retrieval_mode="hybrid",
        top_keywords=0,
        max_titles=0,
        max_refs=0,
        bias_exploration="low",
        ranking_profile="balanced",
        report_max_items=8,
    )

    p = sub.add_parser("idea-grounding", help="Downstream channel: research idea grounding and related-work retrieval.")
    add_downstream_search_args(p)
    p.add_argument("--idea", dest="query", default=None, help="Research idea text; equivalent to --query.")
    p.set_defaults(
        func=cmd_idea_grounding,
        retrieval_mode="hybrid",
        top_keywords=0,
        max_titles=0,
        max_refs=0,
        bias_keyword="high",
        bias_related="high",
        bias_citation="low",
        bias_exploration="low",
        ranking_profile="precision",
        report_max_items=5,
    )

    p = sub.add_parser("idea-evaluate", help="Downstream channel: evidence collection for idea evaluation.")
    add_downstream_search_args(p)
    p.add_argument("--idea", dest="query", default=None, help="Research idea text; equivalent to --query.")
    p.set_defaults(
        func=cmd_idea_evaluate,
        retrieval_mode="hybrid",
        top_keywords=0,
        max_titles=0,
        max_refs=0,
        bias_keyword="high",
        bias_related="high",
        bias_citation="low",
        bias_exploration="low",
        ranking_profile="precision",
        report_max_items=5,
    )

    p = sub.add_parser("idea-generate", help="Downstream channel: research idea seed discovery.")
    add_downstream_search_args(p)
    p.set_defaults(
        func=cmd_idea_generate,
        retrieval_mode="hybrid",
        top_keywords=0,
        max_titles=0,
        max_refs=0,
        bias_related="high",
        bias_cooccurrence="high",
        bias_exploration="high",
        ranking_profile="discovery",
        report_max_items=8,
    )

    p = sub.add_parser("trend-report", help="Downstream channel: research trend analysis.")
    add_downstream_search_args(p)
    p.set_defaults(
        func=cmd_trend_report,
        retrieval_mode="hybrid",
        top_keywords=0,
        max_titles=0,
        max_refs=0,
        bias_citation="high",
        bias_exploration="middle",
        ranking_profile="impact",
        report_max_items=10,
    )

    p = sub.add_parser("researcher-review", help="Downstream channel: researcher background review.")
    add_researcher_review_args(p)
    p.set_defaults(func=cmd_researcher_review, report_max_items=10)

    p = sub.add_parser("paper-search", help="Lightweight low-level paper search for quick testing.")

    add_text_args(p)

    add_output_args(p)

    p.add_argument("--mode", default="vector", help="Low-level search mode.")

    p.add_argument("--field", default="title", help="Low-level search field.")

    p.add_argument("--top-k", type=int, default=None, help="Number of returned items.")

    p.set_defaults(func=cmd_paper_search)



    p = sub.add_parser("make-report", help="Regenerate a Markdown report from saved request/response files.")

    p.add_argument("--request-json", required=True, help="Path to request.json.")

    p.add_argument("--response-json", required=True, help="Path to response.json.")

    p.add_argument("--command-name", default="manual-report", help="Command name displayed in the report.")

    add_output_args(p)

    p.set_defaults(func=cmd_make_report)



    # ============================================================
    # Final help polish
    # ============================================================
    # Keep top-level help concise, conda/python style.
    parser.epilog = """examples:
  scischolar health
  scischolar config
  scischolar paper-search --text "open world agent" --mode vector --field title --top-k 3
  scischolar search-papers --retrieval-mode hybrid --query "open world agent" --keyword "high:open world agent" --top-k 3
  scischolar literature-review --query "open world agent" --keyword "high:open world agent" --top-k 3
  scischolar idea-evaluate --idea "LLM-based multi-perspective evaluation for scientific research ideas" --keyword "high:idea evaluation" --top-k 3

input:
  --query TEXT                         expert query text
  --text TEXT                          natural-language input
  --domain FIELD                       soft domain hint
  --time-range RANGE                   time range, e.g. 2020-2024
  --keyword "high:TERM"                keyword anchor
  --title "middle:TITLE"               title anchor
  --reference "low:TITLE"              reference anchor

retrieval:
  --retrieval-mode MODE                keyword | semantic | title | hybrid, default: hybrid
  --ranking-profile PROFILE            precision | balanced | discovery | impact
  --bias-*                             graph retrieval bias controls

tips:
  scischolar <command> -h                  show command-specific help
  export KG2API_BASE_URL="http://127.0.0.1:8000" for local server testing
  reports are saved under runs/<run_id>/report.md"""

    # Make sub-command help look like mainstream CLIs:
    #   usage: scischolar literature-review [options]
    # rather than:
    #   usage: scischolar [global-options] <command> [command-options] literature-review ...
    for _cmd_name, _sub_parser in sub.choices.items():
        _sub_parser.prog = f"scischolar {_cmd_name}"
        _sub_parser.usage = f"scischolar {_cmd_name} [options]"
        _sub_parser.formatter_class = argparse.RawDescriptionHelpFormatter

    _examples = {
        "search-papers": """examples:
  scischolar search-papers --retrieval-mode hybrid --query "open world agent" --keyword "high:open world agent" --top-k 3
  scischolar search-papers --retrieval-mode semantic --query "idea evaluation" --top-k 3
  scischolar search-papers --retrieval-mode title --title "high:Voyager: An Open-Ended Embodied Agent with Large Language Models" --top-k 3

purpose:
  Retrieve papers with keyword, semantic, title, or hybrid KG retrieval.""",

        "literature-review": """examples:
  scischolar literature-review --query "open world agent" --domain "artificial intelligence" --time-range 2020-2024 --keyword "high:open world agent" --top-k 3
  scischolar literature-review --query "retrieval augmented generation for scientific discovery" --keyword "high:retrieval augmented generation" --keyword "middle:knowledge graph" --top-k 5

purpose:
  Retrieve core papers and generate a review-oriented report with paper pool,
  timeline view, representative works, and writing suggestions.""",

        "idea-grounding": """examples:
  scischolar idea-grounding --idea "Communication-efficient multi-agent collaboration for long-horizon Minecraft construction tasks" --keyword "high:multi-agent collaboration" --top-k 3
  scischolar idea-grounding --idea "LLM-based multi-perspective evaluation for scientific research ideas" --keyword "high:idea evaluation" --top-k 3

purpose:
  Ground a research idea against related papers and collect evidence for
  similarity, difference, motivation, methodology, and evaluation design.""",

        "idea-evaluate": """examples:
  scischolar idea-evaluate --idea "LLM-based multi-perspective evaluation for scientific research ideas" --keyword "high:idea evaluation" --top-k 3
  scischolar idea-evaluate --idea "Federated and privacy-preserving knowledge editing for large language models" --keyword "high:knowledge editing" --keyword "middle:federated learning" --top-k 3

purpose:
  Collect KG evidence for novelty, feasibility, soundness, and differentiation.
  Current version does not use LLM judging.""",

        "idea-generate": """examples:
  scischolar idea-generate --query "knowledge editing for large language models" --keyword "high:knowledge editing" --top-k 5
  scischolar idea-generate --query "scientific idea evaluation and automated research" --keyword "high:idea evaluation" --keyword "middle:LLM as a judge" --top-k 5

purpose:
  Use exploratory KG retrieval to discover possible idea seeds and topic combinations.""",

        "trend-report": """examples:
  scischolar trend-report --query "open world agent" --time-range 2018-2025 --keyword "high:open world agent" --top-k 5
  scischolar trend-report --query "retrieval augmented generation" --keyword "high:retrieval augmented generation" --top-k 5

purpose:
  Retrieve influential papers and organize evidence for research trend analysis.""",

        "researcher-review": """examples:
  scischolar researcher-review --author "Yoshua Bengio" --limit 10 --no-abstract
  scischolar researcher-review --query "papers by Geoffrey Hinton" --limit 10 --no-abstract

purpose:
  Run a full KG retrieval path for researcher background review. The command first
  collects seed papers for the author, then uses them as title anchors in hybrid
  KG retrieval to produce a richer background-review oriented report.""",

        "related-authors": """examples:
  scischolar related-authors --query "open world agent" --keyword "high:open world agent" --top-k 5
  scischolar related-authors --query "idea evaluation" --keyword "high:idea evaluation" --top-k 5

purpose:
  Find authors related to a research direction.""",

        "author-papers": """examples:
  scischolar author-papers --text "author: Yoshua Bengio" --limit 10 --no-abstract
  scischolar author-papers --text "papers by Geoffrey Hinton" --limit 10 --no-abstract

purpose:
  Retrieve papers written by a specified author.""",

        "support-papers": """examples:
  scischolar support-papers --text "Query topic: open world agents and embodied AI. Candidate authors: Yoshua Bengio, Pieter Abbeel." --top-k-per-author 1 --no-author-stats

purpose:
  For each candidate author, find supporting papers related to the query topic.""",

        "paper-search": """examples:
  scischolar paper-search --text "open world agent" --mode vector --field title --top-k 3
  scischolar paper-search --text "idea evaluation" --mode vector --field abstract --top-k 3

purpose:
  Lightweight low-level paper search for fast testing.""",
    }

    for _cmd_name, _epilog in _examples.items():
        if _cmd_name in sub.choices:
            sub.choices[_cmd_name].epilog = _epilog


    return parser





def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "skill":
        from .skills import dispatch_skill_cli
        skill_result = dispatch_skill_cli(sys.argv[2:])
        if isinstance(skill_result, int):
            return int(skill_result)
        sys.argv = [sys.argv[0], *skill_result]

    parser = build_parser()

    args = parser.parse_args()

    return int(args.func(args))





if __name__ == "__main__":

    raise SystemExit(main())

