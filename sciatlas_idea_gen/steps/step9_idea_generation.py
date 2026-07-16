"""Step 9 — generate candidate research ideas from RSS and selected inspirations."""
from __future__ import annotations

import json
import re

import requests

from ..clients.llm_client import LLMClient
from ..config import PipelineConfig
from ..models import Idea, Inspiration, Paper, ResearchGraph
from ..prompts import IDEA_GENERATION
from ..utils import get_logger, read_json, save_json, stable_hash
from .step3_trend_summarizer import render_graph_markdown

log = get_logger("sciatlas.step9")

# Markdown inline link, e.g. [Karpov et al., 2019](https://doi.org/10.1007/...).
_CITATION_RE = re.compile(r"\[([^\]\n]+?)\]\((https?://[^)\s]+)\)")
# Bare inline citation with no URL, e.g. [Hoover et al., 2011] — the model often
# drops the (URL) part in motivation/method prose, so resolve these by token too.
_BARE_CITATION_RE = re.compile(r"\[([^\]\n]+?)\](?!\()")


def _norm_citation(citation: str) -> str:
    """Normalize an inline-citation token for token-based lookup (case/space-insensitive)."""
    return re.sub(r"\s+", " ", (citation or "").strip().lower())


def _serialize_inspirations(inspirations: list[Inspiration]) -> str:
    data = [
        {
            "candidate_id": item.candidate_id,
            "domain": item.domain,
            "radius": item.radius,
            "paper_title": item.paper_title,
            "combination_points": item.combination_points,
            "combination_plan": item.combination_plan,
            "justification": item.justification,
        }
        for item in inspirations
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def _graph_reference_titles(graph: ResearchGraph) -> set[str]:
    return {paper.title for paper in graph.papers.values() if paper.title}


def _ranked_reference_papers(graph: ResearchGraph, limit: int = 8) -> list[Paper]:
    """Most citation-worthy graph papers: seed first, then by citations/recency."""
    scored: list[tuple[bool, int, int, str, Paper]] = []
    for paper in graph.papers.values():
        if not paper.title:
            continue
        cite_count = 0
        if isinstance(paper.raw, dict):
            value = paper.raw.get("cited_by_count")
            if isinstance(value, (int, float)):
                cite_count = int(value)
        scored.append(
            (paper.paper_id == graph.seed_paper_id, cite_count, paper.year or 0, paper.title, paper)
        )
    scored.sort(key=lambda row: row[:4], reverse=True)
    return [paper for *_rest, paper in scored[:limit]]


def _graph_key_references(graph: ResearchGraph, limit: int = 8) -> list[str]:
    return [paper.title for paper in _ranked_reference_papers(graph, limit)]


def _surname(display_name: str) -> str:
    name = (display_name or "").strip()
    if not name:
        return ""
    return name.split()[-1]


# Semantic Scholar paper id (40-hex SHA) inside a paper URL / identifier.
_S2_ID_RE = re.compile(r"(?:semanticscholar\.org/paper/|s2:|corpusid:)\s*([0-9a-fA-F]{40}|\d+)", re.IGNORECASE)


def _s2_paper_id(paper: Paper) -> str:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    haystacks = [str(raw.get(k) or "") for k in ("paper_url", "pdf_url", "url", "id")]
    haystacks.append(str(paper.paper_id or ""))
    ids = raw.get("identifiers")
    if isinstance(ids, list):
        haystacks.extend(str(x) for x in ids)
    for text in haystacks:
        match = _S2_ID_RE.search(text)
        if match:
            return match.group(1)
    return ""


def _empty_meta() -> dict:
    return {"surname": "", "authors": [], "doi": None, "venue": None, "year": None}


def _s2_record_to_meta(record: dict) -> dict:
    """First-author surname, full author list, DOI, venue, year from an S2 record."""
    authors = [
        a.get("name", "").strip()
        for a in (record.get("authors") or [])
        if isinstance(a, dict) and a.get("name")
    ]
    ext = record.get("externalIds") or {}
    doi = ext.get("DOI") if isinstance(ext, dict) else None
    if not doi and isinstance(ext, dict) and ext.get("ArXiv"):
        doi = f"10.48550/arxiv.{ext['ArXiv']}"
    return {
        "surname": _surname(authors[0]) if authors else "",
        "authors": authors,
        "doi": doi if isinstance(doi, str) else None,
        "venue": (str(record.get("venue") or "").strip() or None),
        "year": record.get("year") if isinstance(record.get("year"), int) else None,
    }


def _s2_enrich_batch(sciatlas, s2_ids: list[str]) -> dict[str, dict]:
    """Resolve many Semantic Scholar ids in one batched POST (avoids 429s).

    Returns ``{s2_id: meta}`` for those that resolve. Cached by id set.
    """
    from .step2_graph_construction import _openalex_cache_path  # generic cache path

    if not s2_ids:
        return {}
    cache_path = _openalex_cache_path(sciatlas, f"s2batch:{stable_hash({'ids': sorted(s2_ids)})}")
    data = None
    if sciatlas.cfg.use_cache:
        try:
            data = read_json(cache_path)
        except Exception:
            data = None
    if data is None:
        try:
            response = requests.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "authors,externalIds,venue,year"},
                json={"ids": s2_ids},
                headers={"User-Agent": "sciatlas-idea-gen/1.0"},
                timeout=min(max(int(sciatlas.cfg.timeout), 30), 60),
            )
            response.raise_for_status()
            data = response.json()
            if sciatlas.cfg.use_cache:
                save_json(data, cache_path)
        except Exception as exc:
            log.warning("  semantic scholar batch enrichment failed (%d ids): %s", len(s2_ids), exc)
            return {}
    out: dict[str, dict] = {}
    if isinstance(data, list):
        for s2_id, record in zip(s2_ids, data):
            if isinstance(record, dict):
                out[s2_id] = _s2_record_to_meta(record)
    return out


def _best_url(paper: Paper, doi: str | None = None) -> str:
    """Prefer a resolvable DOI, then any stored URL, then the OpenAlex id."""
    if doi:
        doi = doi.strip()
        return doi if doi.startswith("http") else f"https://doi.org/{doi.lstrip('/').replace('doi.org/', '')}"
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    for key in ("paper_url", "pdf_url", "url"):
        value = raw.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value.strip()
    pid = (paper.paper_id or "").strip()
    if pid.startswith("http"):
        return pid
    oa_id = raw.get("id")
    if isinstance(oa_id, str) and oa_id.startswith("http"):
        return oa_id.strip()
    return ""


def _enrich_references(papers: list[Paper], sciatlas) -> dict[str, dict]:
    """Fetch bibliographic metadata (authors, DOI, venue, year) per paper.

    Returns ``{paper_id: meta}`` where meta = ``{surname, authors, doi, venue,
    year}``. OpenAlex resolves papers carrying a ``W…`` id; papers from the local
    KG (only a Semantic Scholar URL) fall back to the Semantic Scholar Graph API.
    Both endpoints are international (keep them proxied); calls are cached.
    """
    # Imported lazily so the module's import cost stays in Step 2's orbit.
    from .step2_graph_construction import _normalize_openalex_id, _openalex_get_json

    out: dict[str, dict] = {}
    if sciatlas is None:
        return out

    short_to_pid: dict[str, str] = {}
    for paper in papers:
        raw = paper.raw if isinstance(paper.raw, dict) else {}
        oa_id = _normalize_openalex_id(str(raw.get("id") or paper.paper_id or ""))
        if oa_id:
            short_to_pid[oa_id.rsplit("/", 1)[-1]] = paper.paper_id
    short_ids = list(short_to_pid)
    select = "id,doi,publication_year,authorships,primary_location"
    for start in range(0, len(short_ids), 40):
        chunk = short_ids[start : start + 40]
        try:
            data = _openalex_get_json(
                sciatlas,
                url=(
                    "https://api.openalex.org/works"
                    f"?filter=openalex_id:{'|'.join(chunk)}&per-page={len(chunk)}&select={select}"
                ),
                cache_key=f"refmeta2:{','.join(sorted(chunk))}",
            )
        except Exception as exc:
            log.warning("  reference enrichment failed (%d ids): %s", len(chunk), exc)
            continue
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_openalex_id(str(item.get("id") or ""))
            if not normalized:
                continue
            pid = short_to_pid.get(normalized.rsplit("/", 1)[-1])
            if not pid:
                continue
            authors = [
                str((a.get("author") or {}).get("display_name", "")).strip()
                for a in (item.get("authorships") or [])
                if isinstance(a, dict)
            ]
            authors = [a for a in authors if a]
            venue = None
            ploc = item.get("primary_location") or {}
            source = (ploc.get("source") or {}) if isinstance(ploc, dict) else {}
            if isinstance(source, dict):
                venue = str(source.get("display_name") or "").strip() or None
            year = item.get("publication_year")
            out[pid] = {
                "surname": _surname(authors[0]) if authors else "",
                "authors": authors,
                "doi": item.get("doi") if isinstance(item.get("doi"), str) else None,
                "venue": venue,
                "year": year if isinstance(year, int) else None,
            }

    # Semantic Scholar fallback (one batched call) for papers OpenAlex could not name.
    pending: dict[str, str] = {}  # s2_id -> paper_id
    for paper in papers:
        if out.get(paper.paper_id, {}).get("surname"):
            continue
        s2_id = _s2_paper_id(paper)
        if s2_id:
            pending[s2_id] = paper.paper_id
    if pending:
        resolved = _s2_enrich_batch(sciatlas, list(pending))
        for s2_id, meta in resolved.items():
            pid = pending[s2_id]
            if meta.get("surname") or meta.get("authors") or meta.get("doi"):
                out[pid] = meta
    return out


def _abbrev_author(name: str, *, first: bool) -> str:
    """First author keeps its full name; co-authors get abbreviated given names."""
    tokens = name.split()
    if first or len(tokens) < 2:
        return name
    initials = " ".join(f"{t[0]}." for t in tokens[:-1] if t)
    return f"{initials} {tokens[-1]}".strip()


def _format_authors(authors: list[str], *, max_authors: int = 8) -> str:
    names = [a for a in authors if a]
    if not names:
        return ""
    shown = names[:max_authors]
    formatted = [_abbrev_author(name, first=(i == 0)) for i, name in enumerate(shown)]
    tail = ", et al." if len(names) > max_authors else ""
    return ", ".join(formatted) + tail


def _full_citation(meta: dict, *, title: str, year, url: str) -> str:
    """Classic bibliographic entry: ``Authors. Title. Year. Venue. URL``."""
    authors = _format_authors(meta.get("authors") or [])
    if not authors and meta.get("surname"):
        authors = f"{meta['surname']} et al."
    parts: list[str] = []
    if authors:
        parts.append(authors.rstrip(".") + ".")
    if title:
        parts.append(title.rstrip(".") + ".")
    if year:
        parts.append(f"{year}.")
    venue = meta.get("venue")
    if venue:
        parts.append(venue.rstrip(".") + ".")
    if url:
        parts.append(url)
    return " ".join(parts).strip()


def _norm_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _catalog_entries_for_papers(
    papers: list[Paper], sciatlas, *, tag: str = ""
) -> tuple[list[str], dict, dict]:
    """Render catalog lines + ``{normalized_url: ref}`` and ``{normalized_citation: ref}`` lookups.

    ``tag`` (e.g. "inspiration") is appended to each line as ``type: <tag>`` so the
    LLM can tell graph references from cross-domain inspiration source papers. The
    citation-keyed lookup lets us recover references when the model writes a bare
    ``[Author et al., Year]`` token without the ``(URL)`` part.
    """
    enrichment = _enrich_references(papers, sciatlas)
    lines: list[str] = []
    lookup: dict[str, dict] = {}
    cite_lookup: dict[str, dict] = {}
    for paper in papers:
        meta = enrichment.get(paper.paper_id, _empty_meta())
        url = _best_url(paper, meta.get("doi"))
        if not url:
            continue
        if _norm_url(url) in lookup:
            continue
        year = paper.year or meta.get("year") or "n.d."
        author = meta.get("surname") or "Author"
        citation = f"{author} et al., {year}"
        suffix = f" | type: {tag}" if tag else ""
        lines.append(f'- citation: "{citation}" | url: {url} | title: {paper.title}{suffix}')
        ref = {
            "citation": citation,
            "url": url,
            "full": _full_citation(meta, title=paper.title, year=year, url=url),
        }
        lookup[_norm_url(url)] = ref
        cite_lookup.setdefault(_norm_citation(citation), ref)
    return lines, lookup, cite_lookup


def _inspiration_source_papers(inspirations: list[Inspiration]) -> list[Paper]:
    """De-duplicated source papers behind the selected cross-domain inspirations."""
    papers: list[Paper] = []
    seen: set[str] = set()
    for item in inspirations:
        paper = item.source_paper
        if not isinstance(paper, Paper) or not paper.title:
            continue
        key = paper.paper_id or paper.title.lower()
        if key in seen:
            continue
        seen.add(key)
        papers.append(paper)
    return papers


def _build_reference_catalog(
    graph: ResearchGraph,
    inspirations: list[Inspiration],
    sciatlas,
    limit: int = 12,
) -> tuple[str, dict, dict]:
    """Render the citable-works catalog plus ``{url: ref}`` and ``{citation: ref}`` lookups.

    The catalog (fed to the LLM) carries the short inline-citation token; the
    lookups let us expand each cited work into a full bibliographic entry for the
    rendered References section — by URL for Markdown-link citations and by the
    citation token for bare ``[Author et al., Year]`` citations the model emits
    without a URL. Cross-domain inspiration source papers are added (tagged
    ``inspiration``) so any analogy the idea uses can be cited inline.
    """
    graph_lines, lookup, cite_lookup = _catalog_entries_for_papers(
        _ranked_reference_papers(graph, limit), sciatlas
    )
    insp_lines, insp_lookup, insp_cite_lookup = _catalog_entries_for_papers(
        _inspiration_source_papers(inspirations), sciatlas, tag="inspiration"
    )
    # Graph lookup wins on collisions (keeps the un-tagged citation), but any
    # inspiration-only entry is added so its analogy is citable.
    for key, value in insp_lookup.items():
        lookup.setdefault(key, value)
    for key, value in insp_cite_lookup.items():
        cite_lookup.setdefault(key, value)
    lines = graph_lines + insp_lines
    if not lines:
        return "None available — write the idea without inline citations.", lookup, cite_lookup
    return "\n".join(lines), lookup, cite_lookup


def _extract_references(texts: list[str], lookup: dict, cite_lookup: dict | None = None) -> list[dict]:
    """Pull the inline citations actually used, deduped by URL, each expanded to a
    full bibliographic entry via ``lookup`` (falls back to the short form).

    Handles both Markdown-link citations ``[Author et al., Year](URL)`` and bare
    ``[Author et al., Year]`` tokens (resolved via ``cite_lookup`` by citation
    token), since the model frequently drops the ``(URL)`` part in prose.
    """
    cite_lookup = cite_lookup or {}
    refs: list[dict] = []
    seen: set[str] = set()
    for text in texts:
        text = text or ""
        for match in _CITATION_RE.finditer(text):
            citation, url = match.group(1).strip(), match.group(2).strip()
            key = _norm_url(url)
            if key in seen:
                continue
            seen.add(key)
            entry = lookup.get(key)
            if entry:
                refs.append({"citation": entry["citation"], "url": entry["url"], "full": entry["full"]})
            else:
                refs.append({"citation": citation, "url": url, "full": ""})
        # Bare [Author et al., Year] tokens without a URL: resolve by citation token.
        for match in _BARE_CITATION_RE.finditer(text):
            entry = cite_lookup.get(_norm_citation(match.group(1)))
            if not entry:
                continue
            key = _norm_url(entry["url"])
            if key in seen:
                continue
            seen.add(key)
            refs.append({"citation": entry["citation"], "url": entry["url"], "full": entry["full"]})
    return refs


def _combine_text_sections(*, motivation: str, methods: str) -> str:
    sections: list[str] = []
    if motivation:
        sections.append(f"Motivation:\n{motivation}")
    if methods:
        sections.append(f"Methods:\n{methods}")
    return "\n\n".join(sections)


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _parse_method_components(value) -> list[dict]:
    """Normalize the model's ``proposed_method`` into ``[{"name","description"}]``."""
    components: list[dict] = []
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict):
                name = str(entry.get("name", "")).strip()
                description = str(entry.get("description", "")).strip()
            else:
                name, description = "", str(entry or "").strip()
            if name or description:
                components.append({"name": name, "description": description})
    elif str(value or "").strip():
        components.append({"name": "", "description": str(value).strip()})
    return components


def _parse_analogy_mapping(value) -> list[dict]:
    mapping: list[dict] = []
    if not isinstance(value, list):
        return mapping
    for entry in value:
        if not isinstance(entry, dict):
            continue
        inspiration = str(entry.get("inspiration", "")).strip()
        concept = str(entry.get("concept", "")).strip()
        adaptation = str(entry.get("adaptation", "")).strip()
        citation = str(entry.get("citation", "")).strip()
        if inspiration or concept or adaptation:
            mapping.append(
                {
                    "inspiration": inspiration,
                    "concept": concept,
                    "adaptation": adaptation,
                    "citation": citation,
                }
            )
    return mapping


def _synthesize_methods_text(method_components: list[dict], analogy_mapping: list[dict], experiment_plan: list[str]) -> str:
    """Flatten the structured method into prose for the novelty check + back-compat.

    Keeps every inline citation intact (so reference extraction still sees them)
    while giving plain-text consumers a readable Methods blob.
    """
    parts: list[str] = []
    for index, component in enumerate(method_components, start=1):
        name = component.get("name") or f"Component {index}"
        description = component.get("description", "")
        parts.append(f"{index}. {name}: {description}".strip())
    for entry in analogy_mapping:
        inspiration = entry.get("inspiration", "")
        concept = entry.get("concept", "")
        adaptation = entry.get("adaptation", "")
        citation = entry.get("citation", "")
        line = f"Analogy — borrows {concept} from {inspiration}".strip(" —")
        if adaptation:
            line += f"; adapted as {adaptation}"
        if citation:
            line += f" {citation}"
        parts.append(line.strip())
    if experiment_plan:
        parts.append("Experiment plan: " + "; ".join(experiment_plan))
    return "\n".join(p for p in parts if p)


def run(
    llm: LLMClient,
    graph: ResearchGraph,
    rss: dict,
    trend: str,
    inspirations: list[Inspiration],
    cfg: PipelineConfig,
    original_query: str,
    novelty_feedback: str = "",
    sciatlas=None,
) -> list[Idea]:
    """Generate a small set of grounded candidate ideas with inline citations."""
    log.info("Step 9: idea generation")
    serialized_graph = render_graph_markdown(graph)
    reference_catalog, reference_lookup, reference_cite_lookup = _build_reference_catalog(
        graph, inspirations, sciatlas
    )
    prompt = IDEA_GENERATION.format(
        original_query=original_query,
        serialized_graph=serialized_graph,
        rss_json=json.dumps(rss, ensure_ascii=False, indent=2),
        trend=trend,
        inspiration_list=_serialize_inspirations(inspirations),
        novelty_feedback=novelty_feedback or "None.",
        reference_catalog=reference_catalog,
        idea_count=cfg.idea_count,
    )
    data = llm.chat_json(prompt, temperature=0.5)
    raw_ideas = data.get("ideas", []) if isinstance(data, dict) else []

    valid_refs = _graph_reference_titles(graph)
    graph_key_refs = _graph_key_references(graph)
    selected_titles = [item.paper_title for item in inspirations if item.paper_title]
    ideas: list[Idea] = []
    for item in raw_ideas:
        if not isinstance(item, dict):
            continue
        motivation = str(item.get("motivation", "")).strip()
        method_overview = str(item.get("method_overview", "")).strip()
        # Structured method body; fall back to a legacy `methods` string if present.
        method_components = _parse_method_components(item.get("proposed_method"))
        analogy_mapping = _parse_analogy_mapping(item.get("analogy_mapping"))
        experiment_plan = _coerce_str_list(item.get("experiment_plan"))
        methods = _synthesize_methods_text(method_components, analogy_mapping, experiment_plan)
        if not methods:
            methods = str(item.get("methods", "")).strip()
        description = (
            _combine_text_sections(motivation=motivation, methods=methods)
            or str(item.get("description", "")).strip()
        )
        # References = the inline citations actually used across the prose (motivation,
        # method-component descriptions, analogy citations), expanded to full
        # bibliographic entries. Fall back to the model's declared `references` array.
        citation_texts = (
            [motivation]
            + [c.get("description", "") for c in method_components]
            + [a.get("citation", "") for a in analogy_mapping]
        )
        references = _extract_references(citation_texts, reference_lookup, reference_cite_lookup)
        if not references and isinstance(item.get("references"), list):
            seen: set[str] = set()
            for ref in item["references"]:
                if not (isinstance(ref, dict) and ref.get("url")):
                    continue
                url = str(ref["url"]).strip()
                key = _norm_url(url)
                if key in seen:
                    continue
                seen.add(key)
                entry = reference_lookup.get(key)
                if entry:
                    references.append({"citation": entry["citation"], "url": entry["url"], "full": entry["full"]})
                else:
                    references.append(
                        {"citation": str(ref.get("citation", "")).strip(), "url": url, "full": ""}
                    )
        selected_refs = [title for title in graph_key_refs if title in valid_refs]
        # inspiration_sources = the inspirations the idea ACTUALLY used (named in the
        # analogy mapping), so a listed inspiration always corresponds to a cited
        # analogy. Fall back to all selected inspirations only if none were mapped.
        used_inspirations = [
            entry["inspiration"] for entry in analogy_mapping if entry.get("inspiration")
        ]
        inspiration_sources = used_inspirations or selected_titles
        ideas.append(
            Idea(
                title=str(item.get("title", "")).strip(),
                description=description,
                motivation=motivation,
                methods=methods,
                method_overview=method_overview,
                proposed_method=method_components,
                analogy_mapping=analogy_mapping,
                experiment_plan=experiment_plan,
                inspiration_sources=inspiration_sources,
                key_references=selected_refs,
                references=references,
            )
        )

    log.info("  generated %d candidate ideas", len(ideas))
    return [idea for idea in ideas if idea.title and idea.description]
