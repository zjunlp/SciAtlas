from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sciatlas.evidence import grounding as sciatlas_grounding


def _normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _read_idea_context_text(path_value: str | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    target = payload.get("evaluation_target")
    if isinstance(target, dict):
        text = _normalize_whitespace(target.get("text"))
        if text:
            return text
    seed = payload.get("retrieval_seed")
    if isinstance(seed, dict):
        text = _normalize_whitespace(seed.get("text"))
        if text:
            return text
    return ""


def _structured_text(structured: dict[str, Any], fallback: str) -> str:
    lines: list[str] = []
    for key in ("basic_idea", "motivation", "method", "experimental_focus"):
        values = structured.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            text = _normalize_whitespace(item)
            if text:
                lines.append(text)
    return "\n".join(dict.fromkeys(lines)) or _normalize_whitespace(fallback)


def extract_structured_idea(
    source_text: str,
    *,
    env_path: Path,
    debug_dir: Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    source_text = _normalize_whitespace(source_text)
    try:
        client = sciatlas_grounding.GroundingLlmClient(
            sciatlas_grounding.QueryGeneratorConfig(
                env_path=Path(env_path).expanduser().resolve(),
                timeout=120,
                max_tokens=1600,
            )
        )
        extraction = client.extract_structure(source_text)
        structured = extraction.to_dict()
        payload = {
            "status": "ok",
            "text": _structured_text(structured, source_text),
            "structured": structured,
            "model": client.model_name,
            "source_text_chars": len(source_text),
            "error": None,
        }
    except Exception as exc:
        structured = {
            "basic_idea": [source_text] if source_text else [],
            "motivation": [],
            "method": [source_text] if source_text else [],
            "experimental_focus": [],
        }
        payload = {
            "status": "fallback",
            "text": source_text,
            "structured": structured,
            "model": None,
            "source_text_chars": len(source_text),
            "error": str(exc),
        }

    if debug_dir is not None:
        try:
            debug_path = Path(debug_dir).expanduser().resolve() / "structured_idea.json"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
    if logger is not None:
        try:
            logger.info("Structured idea extraction status=%s chars=%s", payload["status"], payload["source_text_chars"])
        except Exception:
            pass
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SciAtlas grounding compatibility wrapper for review_pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--env", default=str(sciatlas_grounding.DEFAULT_ENV_PATH))
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--target-dir", default=str(sciatlas_grounding.DEFAULT_TARGET_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--idea-context", default=None)
    parser.add_argument("--idea-text", default=None)
    parser.add_argument("--pdf-path", default=None)
    parser.add_argument("--enable-grounding-refinement", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--disable-experiment-grounding", action="store_true")
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--dense-candidate-k", type=int, default=40)
    return parser


def run_grounding(args: argparse.Namespace) -> dict[str, Any]:
    idea_text = _normalize_whitespace(getattr(args, "idea_text", None))
    pdf_path = _normalize_whitespace(getattr(args, "pdf_path", None))
    if not idea_text and not pdf_path:
        idea_text = _read_idea_context_text(getattr(args, "idea_context", None))
    if not idea_text and not pdf_path:
        raise ValueError("--idea-context, --idea-text, or --pdf-path must provide grounding input.")

    cli_args = [
        "--manifest",
        str(Path(args.manifest).expanduser().resolve()),
        "--env",
        str(Path(args.env).expanduser().resolve()),
        "--final-top-k",
        str(args.final_top_k),
        "--target-dir",
        str(Path(args.target_dir).expanduser().resolve()),
        "--max-queries",
        str(args.max_queries),
        "--dense-candidate-k",
        str(args.dense_candidate_k),
    ]
    if pdf_path:
        cli_args.extend(["--pdf-path", pdf_path])
    else:
        cli_args.extend(["--idea-text", idea_text])
    if getattr(args, "device", None):
        cli_args.extend(["--device", str(args.device)])
    if getattr(args, "enable_grounding_refinement", False):
        cli_args.append("--enable-grounding-refinement")
    if getattr(args, "disable_reranker", False):
        cli_args.append("--disable-reranker")
    if getattr(args, "disable_experiment_grounding", False):
        cli_args.append("--disable-experiment-grounding")

    grounding_args = sciatlas_grounding.build_parser().parse_args(cli_args)
    return sciatlas_grounding.run_grounding(grounding_args)
