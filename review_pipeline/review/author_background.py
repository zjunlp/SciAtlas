from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from cuda_devices import default_torch_device
from sklearn.metrics.pairwise import cosine_similarity

from .common import ensure_directory, first_non_empty, load_env_values, normalize_whitespace, write_json
from .persona import DEFAULT_PERSONA_JSON_PATH, DEFAULT_PERSONA_SUBJECT, select_random_persona


DEFAULT_NEO4J_URI = "neo4j://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "weiyunxiang"
DEFAULT_EMBEDDING_MODEL_NAME = "/data1/bge-model/AI-ModelScope/bge-large-en-v1.5"
DEFAULT_TOP_K_RELEVANT = 40
DEFAULT_LLM_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_LLM_MODEL_NAME = "DeepSeek-V3.2"
DEFAULT_DEVICE = default_torch_device()


@dataclass(slots=True)
class SharedSentenceTransformer:
    model: SentenceTransformer
    encode_lock: threading.Lock

    def encode(self, inputs: list[str]) -> np.ndarray:
        with self.encode_lock:
            return self.model.encode(inputs)


_SHARED_MODEL_CACHE: dict[tuple[str, str], SharedSentenceTransformer] = {}
_SHARED_MODEL_CACHE_LOCK = threading.Lock()


@dataclass(slots=True)
class AuthorBackgroundConfig:
    target_author: str
    target_idea: str
    output_root: Path
    search_method: str = "name"
    env_path: Path | None = None
    neo4j_uri: str = DEFAULT_NEO4J_URI
    neo4j_user: str = DEFAULT_NEO4J_USER
    neo4j_password: str = DEFAULT_NEO4J_PASSWORD
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    top_k_relevant: int = DEFAULT_TOP_K_RELEVANT
    llm_api_key: str | None = None
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    device: str = DEFAULT_DEVICE
    persona_subject: str = DEFAULT_PERSONA_SUBJECT
    persona_prompts_path: Path | None = None

    @property
    def target_folder(self) -> Path:
        safe_author_name = str(self.target_author).replace(" ", "_")
        return Path(self.output_root).expanduser().resolve() / f"merged_{safe_author_name}"

    @property
    def output_summary_path(self) -> Path:
        return self.target_folder / "author_summary.json"


class AcademicDataExtractor:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def get_author_academic_profile(
        self,
        *,
        identifier: str,
        search_by: str = "id",
        limit: int | None = None,
        fetch_embedding: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if search_by == "id":
            match_clause = """
            MATCH (a:Author {id: $identifier})
            USING INDEX a:Author(id)
            WITH a
            """
        elif search_by == "name":
            match_clause = """
            MATCH (a:Author {display_name: $identifier})
            WITH a
            """
        else:
            raise ValueError("search_by must be 'id' or 'name'")

        query = f"""
        {match_clause}
        MATCH (a)-[r:AUTHORED]->(p:Paper)
        WHERE p.has_abstract = true AND p.abstract IS NOT NULL
        RETURN
            a.id AS author_id,
            a.display_name AS author_name,
            a.h_index AS h_index,
            a.works_count AS total_works,
            a.cited_by_count AS total_citations,
            r.position AS author_position,
            r.is_corresponding AS is_corresponding,
            p.id AS paper_id,
            p.title AS title,
            p.abstract AS abstract,
            p.publication_year AS year,
            coalesce(p.cited_by_count, 0) AS paper_citations
        ORDER BY p.publication_year DESC, coalesce(p.cited_by_count, 0) DESC
        """

        if fetch_embedding:
            query = query.replace(
                "coalesce(p.cited_by_count, 0) AS paper_citations",
                "coalesce(p.cited_by_count, 0) AS paper_citations,\n            p.abstract_embedding AS embedding",
            )

        if limit:
            query += f"\nLIMIT {limit}"

        with self.driver.session() as session:
            result = session.run(query, identifier=identifier)
            author_metas_dict: dict[str, dict[str, Any]] = {}
            papers: list[dict[str, Any]] = []

            for record in result:
                author_id = record["author_id"]
                if author_id not in author_metas_dict:
                    author_metas_dict[author_id] = {
                        "author_id": author_id,
                        "name": record["author_name"],
                        "h_index": record["h_index"],
                        "total_works": record["total_works"],
                        "total_citations": record["total_citations"],
                    }

                paper_data = {
                    "author_id": author_id,
                    "paper_id": record["paper_id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "year": record["year"],
                    "citations": record["paper_citations"],
                    "role": {
                        "position": record["author_position"],
                        "is_corresponding": record["is_corresponding"],
                    },
                }
                if fetch_embedding:
                    paper_data["embedding"] = record["embedding"]
                papers.append(paper_data)

        return list(author_metas_dict.values()), papers

    def save_data_to_disk(
        self,
        author_metas: list[dict[str, Any]],
        papers: list[dict[str, Any]],
        *,
        target_dir: Path,
    ) -> None:
        ensure_directory(target_dir)
        write_json(target_dir / "author_metas.json", {"authors": author_metas})
        write_json(target_dir / "papers.json", {"papers": papers})

    def get_author_graph_context(
        self,
        *,
        identifier: str,
        search_by: str = "id",
        coauthor_limit: int = 8,
        institution_limit: int = 5,
        keyword_limit: int = 12,
    ) -> dict[str, list[dict[str, Any]]]:
        if search_by == "id":
            match_clause = """
            MATCH (a:Author {id: $identifier})
            USING INDEX a:Author(id)
            WITH a
            """
        elif search_by == "name":
            match_clause = """
            MATCH (a:Author {display_name: $identifier})
            WITH a
            """
        else:
            raise ValueError("search_by must be 'id' or 'name'")

        coauthor_query = f"""
        {match_clause}
        MATCH (a)-[r:COAUTHOR]-(co:Author)
        RETURN
            co.id AS author_id,
            co.display_name AS name,
            co.h_index AS h_index,
            co.works_count AS total_works,
            co.cited_by_count AS total_citations,
            coalesce(r.count, 0) AS coauthor_count
        ORDER BY coauthor_count DESC, total_citations DESC, total_works DESC, name ASC
        LIMIT $limit
        """

        institution_query = f"""
        {match_clause}
        MATCH (a)-[r:AFFILIATED_WITH]->(i:Institution)
        RETURN
            i.id AS institution_id,
            i.display_name AS name,
            i.country AS country,
            i.city AS city,
            i.type AS type,
            i.works_count AS works_count,
            i.cited_by_count AS cited_by_count,
            i.h_index AS h_index,
            coalesce(r.is_current, false) AS is_current
        ORDER BY is_current DESC, cited_by_count DESC, works_count DESC, name ASC
        LIMIT $limit
        """

        keyword_query = f"""
        {match_clause}
        MATCH (a)-[:AUTHORED]->(p:Paper)-[r:HAS_KEYWORD]->(k:Keyword)
        WITH
            k,
            count(DISTINCT p) AS paper_count,
            avg(coalesce(r.relevance_score, 0.0)) AS avg_relevance,
            sum(coalesce(r.relevance_score, 0.0)) AS total_relevance,
            max(coalesce(p.cited_by_count, 0)) AS max_paper_citations
        RETURN
            k.id AS keyword_id,
            k.text AS keyword,
            k.text_normalized AS keyword_normalized,
            k.frequency AS global_frequency,
            paper_count,
            avg_relevance,
            total_relevance,
            max_paper_citations
        ORDER BY paper_count DESC, total_relevance DESC, max_paper_citations DESC, keyword ASC
        LIMIT $limit
        """

        with self.driver.session() as session:
            coauthors = [dict(record) for record in session.run(coauthor_query, identifier=identifier, limit=coauthor_limit)]
            institutions = [dict(record) for record in session.run(institution_query, identifier=identifier, limit=institution_limit)]
            keywords = [dict(record) for record in session.run(keyword_query, identifier=identifier, limit=keyword_limit)]
        return {
            "coauthors": coauthors,
            "institutions": institutions,
            "keywords": keywords,
        }


class AcademicSummarizer:
    def __init__(self, *, api_key: str, base_url: str, model_name: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _build_prompt(self, papers: list[dict[str, Any]], target_idea: str) -> str:
        paper_lines = []
        for paper in papers:
            year = paper.get("year", "Unknown")
            citations = paper.get("citations", 0)
            title = paper.get("title", "Unknown Title")
            similarity = float(paper.get("similarity_score", 0.0))
            paper_lines.append(
                f"- [{year}] {title} (Citations: {citations}) [Relevance: {similarity:.4f}]"
            )

        papers_context = "\n".join(paper_lines)
        return f"""
You are an expert academic intelligence analyst.
A user is exploring the following research idea:
"{target_idea}"

Below is a list of publications authored by a specific researcher that are relevant to this idea.
Your task is to analyze these titles and output a professional academic background summary.

[Publication Data List]
{papers_context}

CRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object. Do not include markdown code blocks.
The JSON must strictly follow this structure:
{{
  "target_idea": "{target_idea}",
  "overall_academic_profile": "<Write a detailed 3-5 sentence summary.>",
  "relevant_research_trajectory": [
    {{
      "research_theme": "<Clustered research area>",
      "active_years": "<e.g., 2020-2023>"
    }}
  ],
  "technical_arsenal": [
    "<algorithm/model/framework 1>",
    "<algorithm/model/framework 2>"
  ]
}}
""".strip()

    def generate_summary(self, papers: list[dict[str, Any]], target_idea: str) -> str:
        if not papers:
            return '{"error": "No papers provided."}'

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict AI assistant that only outputs valid JSON objects.",
                },
                {"role": "user", "content": self._build_prompt(papers, target_idea)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


def get_shared_sentence_transformer(model_name: str, device: str) -> SharedSentenceTransformer:
    key = (model_name, device)
    with _SHARED_MODEL_CACHE_LOCK:
        cached = _SHARED_MODEL_CACHE.get(key)
        if cached is None:
            cached = SharedSentenceTransformer(
                model=SentenceTransformer(model_name, device=device),
                encode_lock=threading.Lock(),
            )
            _SHARED_MODEL_CACHE[key] = cached
        return cached


def _resolve_config(config: AuthorBackgroundConfig) -> AuthorBackgroundConfig:
    env_values = load_env_values(config.env_path)
    return AuthorBackgroundConfig(
        target_author=normalize_whitespace(config.target_author),
        target_idea=normalize_whitespace(config.target_idea),
        output_root=Path(config.output_root).expanduser().resolve(),
        search_method=normalize_whitespace(config.search_method) or "name",
        env_path=Path(config.env_path).expanduser().resolve() if config.env_path else None,
        neo4j_uri=first_non_empty(config.neo4j_uri, env_values.get("NEO4J_URI"), DEFAULT_NEO4J_URI),
        neo4j_user=first_non_empty(config.neo4j_user, env_values.get("NEO4J_USER"), DEFAULT_NEO4J_USER),
        neo4j_password=first_non_empty(
            config.neo4j_password,
            env_values.get("NEO4J_PASSWORD"),
            DEFAULT_NEO4J_PASSWORD,
        ),
        embedding_model_name=first_non_empty(config.embedding_model_name, DEFAULT_EMBEDDING_MODEL_NAME),
        top_k_relevant=int(config.top_k_relevant),
        llm_api_key=first_non_empty(
            config.llm_api_key,
            env_values.get("DMX-API-KEY"),
            env_values.get("DMX_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
        )
        or None,
        llm_base_url=first_non_empty(
            config.llm_base_url,
            env_values.get("OPENAI_BASE_URL"),
            DEFAULT_LLM_BASE_URL,
        ),
        llm_model_name=first_non_empty(config.llm_model_name, DEFAULT_LLM_MODEL_NAME),
        device=first_non_empty(config.device, DEFAULT_DEVICE),
        persona_subject=first_non_empty(
            config.persona_subject,
            env_values.get("REVIEWER_PERSONA_SUBJECT"),
            DEFAULT_PERSONA_SUBJECT,
        ),
        persona_prompts_path=(
            Path(config.persona_prompts_path).expanduser().resolve()
            if config.persona_prompts_path
            else DEFAULT_PERSONA_JSON_PATH
        ),
    )


def _deduplicate_papers(raw_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()

    for paper in raw_papers:
        title = normalize_whitespace(paper.get("title"))
        embedding = paper.get("embedding")
        if not title or embedding is None:
            continue

        paper_id = normalize_whitespace(paper.get("paper_id"))
        normalized_title = title.casefold()
        if (paper_id and paper_id in seen_ids) or normalized_title in seen_titles:
            continue

        if paper_id:
            seen_ids.add(paper_id)
        seen_titles.add(normalized_title)
        unique_papers.append(dict(paper))
    return unique_papers


def run_author_background(config: AuthorBackgroundConfig) -> dict[str, Any]:
    resolved = _resolve_config(config)
    if not resolved.target_author:
        raise ValueError("target_author must not be empty")
    if not resolved.target_idea:
        raise ValueError("target_idea must not be empty")
    if not resolved.llm_api_key:
        raise ValueError("LLM API key is required for author background generation")

    extractor = AcademicDataExtractor(
        resolved.neo4j_uri,
        resolved.neo4j_user,
        resolved.neo4j_password,
    )
    try:
        author_infos, raw_papers = extractor.get_author_academic_profile(
            identifier=resolved.target_author,
            search_by=resolved.search_method,
            limit=None,
            fetch_embedding=True,
        )
        if not author_infos or not raw_papers:
            raise RuntimeError(f"No author papers found for {resolved.target_author!r}")

        extractor.save_data_to_disk(author_infos, raw_papers, target_dir=resolved.target_folder)
    finally:
        extractor.close()

    unique_papers = _deduplicate_papers(raw_papers)
    if not unique_papers:
        raise RuntimeError("No usable author papers remained after deduplication")

    model = get_shared_sentence_transformer(resolved.embedding_model_name, resolved.device)
    idea_embedding = model.encode([resolved.target_idea])[0]
    paper_embeddings = np.array([paper["embedding"] for paper in unique_papers])
    similarities = cosine_similarity([idea_embedding], paper_embeddings)[0]

    for index, paper in enumerate(unique_papers):
        paper["similarity_score"] = float(similarities[index])

    unique_papers.sort(key=lambda item: item["similarity_score"], reverse=True)
    selected_papers = unique_papers[: min(resolved.top_k_relevant, len(unique_papers))]
    write_json(resolved.target_folder / "relevant_papers.json", {"papers": selected_papers})

    summarizer = AcademicSummarizer(
        api_key=resolved.llm_api_key,
        base_url=resolved.llm_base_url,
        model_name=resolved.llm_model_name,
    )
    summary_text = summarizer.generate_summary(selected_papers, resolved.target_idea)
    persona = select_random_persona(
        persona_json_path=resolved.persona_prompts_path or DEFAULT_PERSONA_JSON_PATH,
        subject=resolved.persona_subject,
    )

    summary_payload: dict[str, Any]
    raw_output_path = resolved.target_folder / "author_summary.raw.txt"
    try:
        parsed_payload = json.loads(summary_text)
        if not isinstance(parsed_payload, dict):
            raise ValueError("Author background summary is not a JSON object")
        summary_payload = parsed_payload
        summary_payload["persona"] = persona
        if raw_output_path.exists():
            raw_output_path.unlink()
    except Exception:
        raw_output_path.write_text(summary_text, encoding="utf-8")
        summary_payload = {
            "error": "LLM output could not be parsed as JSON",
            "raw_output_path": str(raw_output_path.resolve()),
            "persona": persona,
        }

    write_json(resolved.output_summary_path, summary_payload)
    return {
        "status": "ok",
        "author_count": len(author_infos),
        "raw_paper_count": len(raw_papers),
        "relevant_paper_count": len(selected_papers),
        "target_folder": str(resolved.target_folder.resolve()),
        "summary_path": str(resolved.output_summary_path.resolve()),
        "relevant_papers_path": str((resolved.target_folder / "relevant_papers.json").resolve()),
        "persona_subject": resolved.persona_subject,
        "persona_prompts_path": str((resolved.persona_prompts_path or DEFAULT_PERSONA_JSON_PATH).resolve()),
    }
