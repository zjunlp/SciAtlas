#!/usr/bin/env python3
"""Download PDFs for ranked search results and convert them into GROBID TEI XML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_INPUT_PATH = REPO_ROOT / "runs" / "grounded_review_search_result.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "pdf_manifest"
DEFAULT_GROBID_BASE_URL = "http://127.0.0.1:8070"
OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_CONTENT_BASE = "https://content.openalex.org"
USER_AGENT = "sciatlas-pdf-manifest/1.0"
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "the",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "using",
    "via",
    "with",
}

from .vendor.pdf_extraction.grobid_client import post_pdf
from .vendor.pdf_extraction.models import ExtractedPdfDocument, PdfReference
from .vendor.pdf_extraction.parser import parse_tei_document


class PipelineError(RuntimeError):
    """Raised when the PDF/XML pipeline cannot continue."""


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    paper_list_path: str
    top_k: int | None
    env_path: Path
    output_dir: Path
    grobid_base_url: str
    grobid_start_page: int | None
    timeout: int
    use_env_proxy: bool
    overwrite: bool
    consolidate_citations: str
    include_raw_citations: str
    segment_sentences: str
    preserve_bibr_refs: bool
    openalex_api_key: str
    openalex_mailto: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve ranked search results into PDFs, fall back to OpenAlex content downloads, "
            "and convert each PDF into GROBID TEI XML + parsed JSON."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input search result JSON.")
    parser.add_argument(
        "--paper-list-path",
        default="ranking.papers",
        help="Dot path inside the input JSON that points to the ranked paper list.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of ranked papers to process.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to the SciAtlas .env file.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory under which the run folder will be created.",
    )
    parser.add_argument(
        "--result-tag",
        default=None,
        help="Optional run folder name. Defaults to a timestamped tag.",
    )
    parser.add_argument("--grobid-base-url", default=DEFAULT_GROBID_BASE_URL, help="GROBID base URL.")
    parser.add_argument("--grobid-start-page", type=int, default=None, help="First page sent to GROBID.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the environment for OpenAlex and publisher downloads.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redownload and re-extract existing artifacts.")
    parser.add_argument("--consolidate-citations", choices=["0", "1", "2"], default="0")
    parser.add_argument("--include-raw-citations", choices=["0", "1"], default="1")
    parser.add_argument("--segment-sentences", choices=["0", "1"], default="0")
    parser.add_argument("--preserve-bibr-refs", action="store_true")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the final manifest JSON.")
    return parser


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def normalize_title(text: str | None) -> str:
    cleaned = normalize_whitespace(text).casefold()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def tokenize_title(text: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_whitespace(text).casefold())
        if token and token not in TITLE_STOPWORDS
    }


def parse_year(value: Any) -> int | None:
    text = normalize_whitespace(str(value) if value is not None else "")
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.match(r"^([0-9]{4})", text)
    if match:
        return int(match.group(1))
    return None


def slugify(text: str, *, limit: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_whitespace(text).casefold()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        return "untitled"
    return slug[:limit].rstrip("-") or "untitled"


def load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        raise FileNotFoundError(f".env not found: {env_path}")

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_opener(env_values: dict[str, str], *, use_env_proxy: bool) -> urllib.request.OpenerDirector:
    if not use_env_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    proxy_map: dict[str, str] = {}
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        value = env_values.get(name) or os.environ.get(name)
        if not value:
            continue
        scheme = name.split("_", 1)[0].lower()
        proxy_map[scheme] = value

    if proxy_map:
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxy_map))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def resolve_dot_path(payload: Any, dot_path: str) -> Any:
    current = payload
    for raw_part in dot_path.split("."):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"Invalid empty path segment in {dot_path!r}")
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Missing key {part!r} while resolving {dot_path!r}")
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index]
            continue
        raise KeyError(f"Cannot resolve {part!r} inside {type(current).__name__} for path {dot_path!r}")
    return current


def load_paper_records(input_path: Path, paper_list_path: str) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = resolve_dot_path(payload, paper_list_path)
    if not isinstance(records, list):
        raise TypeError(f"Resolved paper list at {paper_list_path!r} is not a list")
    if not all(isinstance(item, dict) for item in records):
        raise TypeError(f"Resolved paper list at {paper_list_path!r} must contain dict items")
    return records


def normalize_openalex_work_id(value: Any) -> str:
    text = normalize_whitespace(str(value) if value is not None else "")
    if not text:
        return ""
    if text.startswith("openalex:"):
        text = text.split(":", 1)[1]
    match = re.search(r"(?:https?://openalex\.org/)?(W\d+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def extract_record_title(record: dict[str, Any]) -> str:
    title = normalize_whitespace(record.get("title"))
    if title:
        return title
    paper = record.get("paper")
    if isinstance(paper, dict):
        return normalize_whitespace(paper.get("title"))
    return ""


def extract_record_year(record: dict[str, Any]) -> int | None:
    year = parse_year(record.get("year"))
    if year is not None:
        return year
    paper = record.get("paper")
    if not isinstance(paper, dict):
        return None
    return parse_year(
        paper.get("year") or paper.get("publication_year") or paper.get("publication_date")
    )


def extract_record_pdf_url(record: dict[str, Any]) -> str:
    url = normalize_whitespace(record.get("pdf_url"))
    if url:
        return url

    paper = record.get("paper")
    if not isinstance(paper, dict):
        return ""

    open_access_pdf = paper.get("openAccessPdf")
    if isinstance(open_access_pdf, dict):
        url = normalize_whitespace(open_access_pdf.get("url"))
        if url:
            return url

    for key in ("pdf_url", "pdfUrl"):
        url = normalize_whitespace(paper.get(key))
        if url:
            return url
    return ""


def extract_openalex_id_from_record(record: dict[str, Any]) -> str:
    source = normalize_whitespace(record.get("source"))
    paper = record.get("paper")
    if not isinstance(paper, dict):
        paper = {}

    candidates: list[Any] = []
    if source == "kg":
        candidates.append(paper.get("id"))
        candidates.append(record.get("paper_url"))

    identifiers = record.get("identifiers")
    if isinstance(identifiers, list):
        candidates.extend(identifiers)

    for candidate in candidates:
        work_id = normalize_openalex_work_id(candidate)
        if work_id:
            return work_id
    return ""


def iter_equivalent_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for key in ("variants", "removed_variants_without_pdf"):
        items = record.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                variants.append(item)
    variants.sort(
        key=lambda item: (
            0 if normalize_whitespace(item.get("source")) == "kg" else 1,
            normalize_whitespace(item.get("title")).casefold(),
        )
    )
    return variants


def extract_openalex_id_from_equivalent_records(record: dict[str, Any]) -> dict[str, Any] | None:
    for variant in iter_equivalent_records(record):
        openalex_id = extract_openalex_id_from_record(variant)
        if not openalex_id:
            continue
        return {
            "openalex_id": openalex_id,
            "variant_source": normalize_whitespace(variant.get("source")),
            "variant_title": extract_record_title(variant),
        }
    return None


def score_openalex_title_match(
    query_title: str,
    query_year: int | None,
    candidate_title: str,
    candidate_year: int | None,
) -> tuple[float, bool, float, float]:
    query_title_norm = normalize_title(query_title)
    candidate_title_norm = normalize_title(candidate_title)
    exact = bool(query_title_norm) and query_title_norm == candidate_title_norm

    sequence_ratio = SequenceMatcher(
        None,
        normalize_whitespace(query_title).casefold(),
        normalize_whitespace(candidate_title).casefold(),
    ).ratio()
    query_tokens = tokenize_title(query_title)
    candidate_tokens = tokenize_title(candidate_title)
    token_overlap = (
        len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
        if query_tokens and candidate_tokens
        else 0.0
    )

    score = sequence_ratio + token_overlap
    if exact:
        score += 2.0
    if query_year is not None and candidate_year is not None:
        year_gap = abs(query_year - candidate_year)
        score += max(0.0, 0.4 - 0.1 * year_gap)
        if year_gap > 4:
            score -= 0.5
    return score, exact, sequence_ratio, token_overlap


def choose_best_openalex_match(
    title: str,
    expected_year: int | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        candidate_id = normalize_openalex_work_id(candidate.get("id"))
        candidate_title = normalize_whitespace(candidate.get("title") or candidate.get("display_name"))
        if not candidate_id or not candidate_title:
            continue
        candidate_year = parse_year(candidate.get("publication_year"))
        score, exact, sequence_ratio, token_overlap = score_openalex_title_match(
            title,
            expected_year,
            candidate_title,
            candidate_year,
        )
        scored.append(
            (
                score,
                {
                    "openalex_id": candidate_id,
                    "title": candidate_title,
                    "publication_year": candidate_year,
                    "match_score": round(score, 4),
                    "exact_title_match": exact,
                    "sequence_ratio": round(sequence_ratio, 4),
                    "token_overlap": round(token_overlap, 4),
                },
            )
        )

    if not scored:
        return None

    scored.sort(
        key=lambda item: (
            item[0],
            item[1]["exact_title_match"],
            item[1]["sequence_ratio"],
            item[1]["token_overlap"],
            -(item[1]["publication_year"] or 0),
        ),
        reverse=True,
    )
    best = scored[0][1]
    if best["exact_title_match"]:
        return best
    if best["sequence_ratio"] >= 0.9:
        return best
    if best["token_overlap"] >= 0.8 and best["sequence_ratio"] >= 0.75:
        return best
    return None


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def sanitize_url_for_output(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return url
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", parsed.fragment))


def append_trace(
    trace: list[dict[str, Any]],
    *,
    step: str,
    status: str,
    message: str = "",
    url: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {
        "step": step,
        "status": status,
    }
    if message:
        item["message"] = message
    if url:
        item["url"] = sanitize_url_for_output(url)
    if extra:
        item.update(extra)
    trace.append(item)


def build_run_dir(output_root: Path, result_tag: str | None) -> Path:
    tag = result_tag or datetime.now().strftime("%Y%m%d_%H%M%S_pdf_manifest")
    return output_root / tag


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


class OpenAlexClient:
    def __init__(
        self,
        *,
        opener: urllib.request.OpenerDirector,
        api_key: str,
        mailto: str,
        timeout: int,
    ) -> None:
        self.opener = opener
        self.api_key = api_key
        self.mailto = mailto
        self.timeout = timeout

    def _build_url(self, base_url: str, params: dict[str, Any]) -> str:
        query_params = {key: value for key, value in params.items() if value not in {None, ""}}
        query_params["api_key"] = self.api_key
        if self.mailto:
            query_params["mailto"] = self.mailto
        return f"{base_url}?{urllib.parse.urlencode(query_params)}"

    def _request_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PipelineError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f"Failed to reach {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Invalid JSON returned from {url}") from exc

    def search_work_by_title(self, title: str, expected_year: int | None) -> dict[str, Any] | None:
        title_text = normalize_whitespace(title)
        if not title_text:
            return None

        url = self._build_url(
            f"{OPENALEX_API_BASE}/works",
            {
                "filter": f"title.search:{title_text}",
                "per-page": 10,
            },
        )
        payload = self._request_json(url)
        results = payload.get("results")
        if not isinstance(results, list):
            return None
        return choose_best_openalex_match(title_text, expected_year, results)

    def build_pdf_url(self, work_id: str) -> str:
        return self._build_url(f"{OPENALEX_CONTENT_BASE}/works/{work_id}.pdf", {})


class KgTitleLookupClient:
    def search_work_by_title(self, title: str, expected_year: int | None) -> dict[str, Any] | None:
        return None


class PdfXmlPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        opener: urllib.request.OpenerDirector,
        openalex_client: OpenAlexClient,
        kg_title_lookup_client: KgTitleLookupClient | None = None,
    ) -> None:
        self.config = config
        self.opener = opener
        self.openalex_client = openalex_client
        self.kg_title_lookup_client = kg_title_lookup_client or KgTitleLookupClient()

    def run(self) -> dict[str, Any]:
        papers = load_paper_records(self.config.input_path, self.config.paper_list_path)
        if self.config.top_k is not None:
            papers = papers[: self.config.top_k]

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        success_count = 0
        for rank, record in enumerate(papers, start=1):
            entry = self.process_record(rank, record)
            entries.append(entry)
            if entry["status"] == "ok":
                success_count += 1

        manifest = {
            "status": "ok" if success_count == len(entries) else ("partial_error" if success_count else "error"),
            "input_path": str(self.config.input_path.resolve()),
            "paper_list_path": self.config.paper_list_path,
            "requested_top_k": self.config.top_k,
            "processed_count": len(entries),
            "success_count": success_count,
            "failed_count": len(entries) - success_count,
            "output_dir": str(self.config.output_dir.resolve()),
            "papers": entries,
        }
        return manifest

    def process_record(self, rank: int, record: dict[str, Any]) -> dict[str, Any]:
        title = extract_record_title(record)
        if not title:
            return {
                "rank": rank,
                "status": "error",
                "error": "Paper title is missing.",
            }

        paper_dir = self.config.output_dir / "papers" / f"{rank:02d}_{slugify(title)}"
        pdf_path = paper_dir / "paper.pdf"
        tei_xml_path = paper_dir / "fulltext.tei.xml"
        parsed_json_path = paper_dir / "parsed.json"
        trace_path = paper_dir / "acquisition_log.json"
        paper_dir.mkdir(parents=True, exist_ok=True)
        trace: list[dict[str, Any]] = []

        entry: dict[str, Any] = {
            "rank": rank,
            "status": "error",
            "group_id": normalize_whitespace(record.get("group_id")),
            "title": title,
            "source": normalize_whitespace(record.get("source")),
            "year": extract_record_year(record),
            "source_rank": record.get("source_rank"),
            "original_pdf_url": extract_record_pdf_url(record),
            "paper_dir": maybe_relative(paper_dir, self.config.output_dir),
            "trace_path": maybe_relative(trace_path, self.config.output_dir),
        }

        try:
            append_trace(
                trace,
                step="start",
                status="ok",
                message="Starting paper processing.",
                extra={
                    "title": title,
                    "source": entry["source"],
                    "year": entry["year"],
                },
            )
            fetch_result = self.ensure_pdf(record, pdf_path, trace)
            entry["pdf"] = fetch_result
            tei_result = self.ensure_tei(pdf_path, tei_xml_path, parsed_json_path, trace)
            entry["tei"] = tei_result
            entry["status"] = "ok"
        except Exception as exc:
            entry["error_type"] = exc.__class__.__name__
            entry["error"] = str(exc)
            append_trace(
                trace,
                step="paper",
                status="error",
                message=str(exc),
                extra={"error_type": exc.__class__.__name__},
            )
        finally:
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            entry["trace"] = trace
        return entry

    def ensure_pdf(self, record: dict[str, Any], pdf_path: Path, trace: list[dict[str, Any]]) -> dict[str, Any]:
        if pdf_path.exists() and not self.config.overwrite:
            append_trace(
                trace,
                step="pdf_cache",
                status="ok",
                message="Reusing existing PDF artifact.",
                url=extract_record_pdf_url(record),
                extra={"path": maybe_relative(pdf_path, self.config.output_dir)},
            )
            return self.describe_pdf_artifact(
                pdf_path,
                method="cache",
                source_url=extract_record_pdf_url(record),
                final_url="",
                openalex_id=(
                    extract_openalex_id_from_record(record)
                    or (extract_openalex_id_from_equivalent_records(record) or {}).get("openalex_id", "")
                ),
            )

        initial_pdf_url = extract_record_pdf_url(record)
        attempts: list[dict[str, Any]] = []
        if initial_pdf_url:
            append_trace(
                trace,
                step="pdf_direct",
                status="running",
                message="Trying original PDF URL.",
                url=initial_pdf_url,
            )
            try:
                result = self.download_pdf(initial_pdf_url, pdf_path)
                result.update({"method": "original_pdf_url", "openalex_id": ""})
                append_trace(
                    trace,
                    step="pdf_direct",
                    status="ok",
                    message="Downloaded PDF from original URL.",
                    url=initial_pdf_url,
                    extra={
                        "final_url": sanitize_url_for_output(result.get("final_url", "")),
                        "size_bytes": result.get("size_bytes"),
                    },
                )
                return result
            except Exception as exc:
                attempts.append({"method": "original_pdf_url", "url": initial_pdf_url, "error": str(exc)})
                append_trace(
                    trace,
                    step="pdf_direct",
                    status="error",
                    message=str(exc),
                    url=initial_pdf_url,
                    extra={"error_type": exc.__class__.__name__},
                )
                if pdf_path.exists():
                    pdf_path.unlink()

        openalex_resolution = self.resolve_openalex(record, trace)
        openalex_id = normalize_whitespace(openalex_resolution.get("openalex_id"))
        if openalex_id:
            openalex_pdf_url = self.openalex_client.build_pdf_url(openalex_id)
            append_trace(
                trace,
                step="pdf_openalex",
                status="running",
                message="Trying OpenAlex content API.",
                url=openalex_pdf_url,
                extra={"openalex_id": openalex_id},
            )
            try:
                result = self.download_pdf(openalex_pdf_url, pdf_path)
                result.update(
                    {
                        "method": "openalex_content_api",
                        "openalex_id": openalex_id,
                        "title_match": openalex_resolution.get("title_match"),
                    }
                )
                append_trace(
                    trace,
                    step="pdf_openalex",
                    status="ok",
                    message="Downloaded PDF from OpenAlex content API.",
                    url=openalex_pdf_url,
                    extra={
                        "openalex_id": openalex_id,
                        "final_url": sanitize_url_for_output(result.get("final_url", "")),
                        "size_bytes": result.get("size_bytes"),
                    },
                )
                return result
            except Exception as exc:
                attempts.append(
                    {
                        "method": "openalex_content_api",
                        "url": openalex_pdf_url,
                        "openalex_id": openalex_id,
                        "error": str(exc),
                    }
                )
                append_trace(
                    trace,
                    step="pdf_openalex",
                    status="error",
                    message=str(exc),
                    url=openalex_pdf_url,
                    extra={
                        "openalex_id": openalex_id,
                        "error_type": exc.__class__.__name__,
                    },
                )
                if pdf_path.exists():
                    pdf_path.unlink()

        raise PipelineError(f"Unable to download PDF. Attempts: {json.dumps(attempts, ensure_ascii=False)}")

    def resolve_openalex(self, record: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
        existing_id = extract_openalex_id_from_record(record)
        if existing_id:
            append_trace(
                trace,
                step="openalex_resolve",
                status="ok",
                message="Resolved OpenAlex ID from record metadata.",
                extra={"openalex_id": existing_id},
            )
            return {"openalex_id": existing_id, "title_match": None}

        variant_match = extract_openalex_id_from_equivalent_records(record)
        if variant_match is not None:
            append_trace(
                trace,
                step="openalex_variants",
                status="ok",
                message="Resolved OpenAlex ID from an equivalent variant.",
                extra=variant_match,
            )
            return {"openalex_id": variant_match["openalex_id"], "title_match": None}

        title = extract_record_title(record)
        expected_year = extract_record_year(record)
        append_trace(
            trace,
            step="kg_title_lookup",
            status="running",
            message="Skipping local KG title lookup in the open-source SciAtlas client.",
            extra={"title": title, "expected_year": expected_year},
        )
        kg_match = self.kg_title_lookup_client.search_work_by_title(title, expected_year)
        if kg_match is not None:
            append_trace(
                trace,
                step="kg_title_lookup",
                status="ok",
                message="Resolved OpenAlex ID from local title lookup.",
                extra=kg_match,
            )
            return {"openalex_id": kg_match["openalex_id"], "title_match": kg_match}

        append_trace(
            trace,
            step="openalex_search",
            status="running",
            message="Searching OpenAlex by title.",
            extra={"title": title, "expected_year": expected_year},
        )
        match = self.openalex_client.search_work_by_title(title, expected_year)
        if match is None:
            append_trace(
                trace,
                step="openalex_search",
                status="error",
                message="OpenAlex title search returned no acceptable match.",
                extra={"title": title, "expected_year": expected_year},
            )
            return {"openalex_id": "", "title_match": None}
        append_trace(
            trace,
            step="openalex_search",
            status="ok",
            message="Resolved OpenAlex ID from title search.",
            extra=match,
        )
        return {"openalex_id": match["openalex_id"], "title_match": match}

    def download_pdf(self, url: str, destination: Path) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                "User-Agent": USER_AGENT,
            },
            method="GET",
        )
        try:
            with self.opener.open(request, timeout=self.config.timeout) as response:
                content = response.read()
                final_url = response.geturl()
                content_type = normalize_whitespace(response.headers.get("Content-Type"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PipelineError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f"Failed to reach {url}: {exc}") from exc

        if not self.is_pdf_payload(content, content_type):
            snippet = content[:200].decode("utf-8", errors="replace")
            raise PipelineError(
                f"Response is not a PDF for {url}. content_type={content_type!r}, body_prefix={snippet!r}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return self.describe_pdf_artifact(
            destination,
            method="download",
            source_url=url,
            final_url=final_url,
            openalex_id="",
            content_type=content_type,
        )

    def is_pdf_payload(self, content: bytes, content_type: str) -> bool:
        if content.lstrip().startswith(b"%PDF-"):
            return True
        lowered = content_type.casefold()
        return "application/pdf" in lowered and content.lstrip().startswith(b"%PDF-")

    def describe_pdf_artifact(
        self,
        pdf_path: Path,
        *,
        method: str,
        source_url: str,
        final_url: str,
        openalex_id: str,
        content_type: str = "",
    ) -> dict[str, Any]:
        pdf_bytes = pdf_path.read_bytes()
        return {
            "method": method,
            "path": maybe_relative(pdf_path, self.config.output_dir),
            "source_url": sanitize_url_for_output(source_url),
            "final_url": sanitize_url_for_output(final_url),
            "openalex_id": openalex_id,
            "content_type": content_type,
            "size_bytes": len(pdf_bytes),
            "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        }

    def ensure_tei(
        self,
        pdf_path: Path,
        tei_xml_path: Path,
        parsed_json_path: Path,
        trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if tei_xml_path.exists() and parsed_json_path.exists() and not self.config.overwrite:
            parsed_payload = json.loads(parsed_json_path.read_text(encoding="utf-8"))
            append_trace(
                trace,
                step="tei_cache",
                status="ok",
                message="Reusing existing GROBID TEI and parsed JSON artifacts.",
                extra={
                    "tei_xml_path": maybe_relative(tei_xml_path, self.config.output_dir),
                    "parsed_json_path": maybe_relative(parsed_json_path, self.config.output_dir),
                },
            )
            return {
                "method": "cache",
                "tei_xml_path": maybe_relative(tei_xml_path, self.config.output_dir),
                "parsed_json_path": maybe_relative(parsed_json_path, self.config.output_dir),
                "title": normalize_whitespace(parsed_payload.get("title")),
                "reference_count": len(parsed_payload.get("references") or []),
                "body_section_count": len(parsed_payload.get("body") or []),
            }

        append_trace(
            trace,
            step="grobid",
            status="running",
            message="Submitting PDF to GROBID.",
            extra={"pdf_path": maybe_relative(pdf_path, self.config.output_dir)},
        )
        document, fulltext_xml = self.extract_tei(pdf_path)
        tei_xml_path.write_text(fulltext_xml, encoding="utf-8")
        parsed_payload = document.to_dict()
        parsed_json_path.write_text(
            json.dumps(parsed_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        append_trace(
            trace,
            step="grobid",
            status="ok",
            message="GROBID extraction completed.",
            extra={
                "tei_xml_path": maybe_relative(tei_xml_path, self.config.output_dir),
                "parsed_json_path": maybe_relative(parsed_json_path, self.config.output_dir),
                "reference_count": len(parsed_payload.get("references") or []),
                "body_section_count": len(parsed_payload.get("body") or []),
            },
        )
        return {
            "method": "grobid",
            "tei_xml_path": maybe_relative(tei_xml_path, self.config.output_dir),
            "parsed_json_path": maybe_relative(parsed_json_path, self.config.output_dir),
            "title": normalize_whitespace(parsed_payload.get("title")),
            "reference_count": len(parsed_payload.get("references") or []),
            "body_section_count": len(parsed_payload.get("body") or []),
        }

    def extract_tei(self, pdf_path: Path) -> tuple[ExtractedPdfDocument, str]:
        fulltext_fields = {
            "includeRawCitations": self.config.include_raw_citations,
            "consolidateCitations": self.config.consolidate_citations,
            "segmentSentences": self.config.segment_sentences,
            "start": str(self.config.grobid_start_page) if self.config.grobid_start_page else None,
        }
        refs_fields = {
            "includeRawCitations": self.config.include_raw_citations,
            "consolidateCitations": self.config.consolidate_citations,
        }
        fulltext_xml = post_pdf(
            urllib.parse.urljoin(self.config.grobid_base_url, "/api/processFulltextDocument"),
            pdf_path,
            fulltext_fields,
            timeout_s=self.config.timeout,
        )
        refs_xml = post_pdf(
            urllib.parse.urljoin(self.config.grobid_base_url, "/api/processReferences"),
            pdf_path,
            refs_fields,
            timeout_s=self.config.timeout,
        )
        document = parse_tei_document(fulltext_xml, preserve_bibr_refs=self.config.preserve_bibr_refs)
        reference_document = parse_tei_document(refs_xml)
        if reference_document.references:
            document.references = _merge_references(document.references, reference_document.references)
        document.references = _filter_references(document.references)
        return document, fulltext_xml


def build_config(args: argparse.Namespace) -> tuple[PipelineConfig, urllib.request.OpenerDirector]:
    env_path = Path(args.env).resolve()
    env_values = load_env_values(env_path)
    openalex_api_key = env_values.get("OA-API-KEY") or env_values.get("OA_API_KEY") or ""

    output_root = Path(args.output_root).resolve()
    output_dir = build_run_dir(output_root, args.result_tag).resolve()
    opener = build_opener(env_values, use_env_proxy=args.use_env_proxy)

    config = PipelineConfig(
        input_path=Path(args.input).resolve(),
        paper_list_path=args.paper_list_path,
        top_k=args.top_k,
        env_path=env_path,
        output_dir=output_dir,
        grobid_base_url=args.grobid_base_url,
        grobid_start_page=args.grobid_start_page,
        timeout=args.timeout,
        use_env_proxy=args.use_env_proxy,
        overwrite=args.overwrite,
        consolidate_citations=args.consolidate_citations,
        include_raw_citations=args.include_raw_citations,
        segment_sentences=args.segment_sentences,
        preserve_bibr_refs=args.preserve_bibr_refs,
        openalex_api_key=openalex_api_key,
        openalex_mailto=env_values.get("OPENALEX_MAILTO") or env_values.get("OA_MAILTO") or "",
    )
    return config, opener


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config, opener = build_config(args)
    client = OpenAlexClient(
        opener=opener,
        api_key=config.openalex_api_key,
        mailto=config.openalex_mailto,
        timeout=config.timeout,
    )
    pipeline = PdfXmlPipeline(
        config,
        opener=opener,
        openalex_client=client,
    )
    manifest = pipeline.run()
    manifest["manifest_path"] = str((config.output_dir / "manifest.json").resolve())
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest = run_pipeline(args)
    if args.pretty:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
