from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from neo4j import GraphDatabase

from .models import CandidatePaper, GraphEdge, GraphNode


_LUCENE_SPECIAL_CHARS = frozenset('+-&|!(){}[]^"~*?:\\/')


def escape_lucene_query_text(query_text: str) -> str:
    """Escape user/PDF-derived text before passing it to Neo4j Lucene fulltext."""
    cleaned = " ".join(str(query_text or "").split())
    if not cleaned:
        return ""
    escaped: list[str] = []
    for char in cleaned:
        if char in _LUCENE_SPECIAL_CHARS:
            escaped.append("\\")
        escaped.append(char)
    return "".join(escaped)


def _paper_year_filter(alias: str, after_year: int | None, before_year: int | None) -> str:
    conditions: list[str] = []
    if after_year is not None:
        conditions.append(f"{alias}.publication_year IS NOT NULL")
        conditions.append(f"{alias}.publication_year >= $after_year")
    if before_year is not None:
        conditions.append(f"{alias}.publication_year IS NOT NULL")
        conditions.append(f"{alias}.publication_year <= $before_year")
    if not conditions:
        return ""
    return f"WHERE {' AND '.join(conditions)}"


def _vector_search_year_filter(alias: str, after_year: int | None, before_year: int | None) -> str:
    conditions: list[str] = []
    if after_year is not None:
        conditions.append(f"{alias}.publication_year >= $after_year")
    if before_year is not None:
        conditions.append(f"{alias}.publication_year <= $before_year")
    if not conditions:
        return ""
    return f"WHERE {' AND '.join(conditions)}"


class Neo4jSearchRepository(AbstractContextManager):
    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def run(self, cypher: str, **params: Any):
        with self._driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(cypher, **params)]

    def wait_for_indexes(self) -> None:
        cypher = """
        SHOW INDEXES
        YIELD name, state
        WHERE name IN [
          'paper_title_embedding_idx',
          'paper_abstract_embedding_idx',
          'paper_specter2_embedding_idx',
          'paper_title_normalized_idx',
          'keyword_text_embedding_idx',
          'paper_title_ft',
          'keyword_text_ft'
        ]
        RETURN name, state
        """
        rows = self.run(cypher)
        states = {row["name"]: row["state"] for row in rows}
        expected = {
            "paper_title_embedding_idx",
            "paper_abstract_embedding_idx",
            "paper_specter2_embedding_idx",
            "paper_title_normalized_idx",
            "keyword_text_embedding_idx",
            "paper_title_ft",
            "keyword_text_ft",
        }
        missing = expected - set(states)
        if missing:
            raise RuntimeError(f"Neo4j indexes missing: {sorted(missing)}")
        not_online = {name: state for name, state in states.items() if state != "ONLINE"}
        if not_online:
            raise RuntimeError(f"Neo4j indexes not ONLINE: {not_online}")

    def match_keywords_exact(self, normalized_texts: list[str]) -> list[dict[str, Any]]:
        if not normalized_texts:
            return []
        cypher = """
        MATCH (k:Keyword)
        WHERE k.text_normalized IN $normalized_texts
        RETURN
          k.id AS id,
          k.text AS text,
          k.text_normalized AS text_normalized,
          k.frequency AS frequency
        """
        return self.run(cypher, normalized_texts=normalized_texts)

    def vector_search_keywords(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        cypher = """
        CALL db.index.vector.queryNodes('keyword_text_embedding_idx', $top_k, $query_vector)
        YIELD node, score
        RETURN
          node.id AS id,
          node.text AS text,
          node.text_normalized AS text_normalized,
          node.frequency AS frequency,
          score AS score
        ORDER BY score DESC
        """
        return self.run(cypher, top_k=top_k, query_vector=query_vector)

    def papers_from_keyword_matches(
        self,
        matches: list[dict[str, Any]],
        top_k: int = 500,
        after_year: int | None = None,
        before_year: int | None = None,
    ) -> list[dict[str, Any]]:
        if not matches:
            return []
        paper_year_filter = _paper_year_filter("p", after_year, before_year)
        cypher = """
        UNWIND $matches AS match
        MATCH (p:Paper)-[rel:HAS_KEYWORD]->(k:Keyword {id: match.kg_keyword_id})
        __PAPER_YEAR_FILTER__
        WITH p, rel, k, match,
             (match.input_score * match.match_score * coalesce(rel.relevance_score, 0.0)) AS contrib
        RETURN
          p.id AS paper_id,
          p.title AS title,
          p.abstract AS abstract,
          p.publication_year AS publication_year,
          coalesce(p.cited_by_count, 0) AS cited_by_count,
          sum(contrib) AS score_kw_path,
          collect({
            input_keyword: match.input_keyword,
            input_score: match.input_score,
            kg_keyword_id: k.id,
            kg_keyword_text: k.text,
            match_score: match.match_score,
            match_type: match.match_type,
            edge_relevance_score: coalesce(rel.relevance_score, 0.0)
          }) AS keyword_evidence
        ORDER BY score_kw_path DESC
        LIMIT $top_k
        """
        cypher = cypher.replace("__PAPER_YEAR_FILTER__", paper_year_filter)
        return self.run(
            cypher,
            matches=matches,
            top_k=top_k,
            after_year=after_year,
            before_year=before_year,
        )

    def vector_search_papers(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int,
        after_year: int | None = None,
        before_year: int | None = None,
    ) -> list[dict[str, Any]]:
        allowed_indexes = {
            "paper_title_embedding_idx",
            "paper_abstract_embedding_idx",
            "paper_specter2_embedding_idx",
        }
        if index_name not in allowed_indexes:
            raise ValueError(f"Unsupported paper vector index: {index_name}")
        paper_year_filter = _vector_search_year_filter("node", after_year, before_year)
        cypher = """
        MATCH (node:Paper)
        SEARCH node IN (
          VECTOR INDEX __INDEX_NAME__
          FOR $query_vector
          __PAPER_YEAR_FILTER__
          LIMIT $top_k
        ) SCORE AS score
        RETURN
          node.id AS paper_id,
          node.title AS title,
          node.abstract AS abstract,
          node.publication_year AS publication_year,
          coalesce(node.cited_by_count, 0) AS cited_by_count,
          score AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        cypher = cypher.replace("__INDEX_NAME__", index_name)
        cypher = cypher.replace("__PAPER_YEAR_FILTER__", paper_year_filter)
        return self.run(
            cypher,
            top_k=top_k,
            query_vector=query_vector,
            after_year=after_year,
            before_year=before_year,
        )

    def fulltext_search_papers(
        self,
        index_name: str,
        query_text: str,
        top_k: int,
        after_year: int | None = None,
        before_year: int | None = None,
    ) -> list[dict[str, Any]]:
        allowed_indexes = {
            "paper_title_ft",
            "paper_abstract_ft",
        }
        if index_name not in allowed_indexes:
            raise ValueError(f"Unsupported paper fulltext index: {index_name}")
        safe_query_text = escape_lucene_query_text(query_text)
        if not safe_query_text:
            return []
        paper_year_filter = _paper_year_filter("node", after_year, before_year)
        cypher = """
        CALL db.index.fulltext.queryNodes('__INDEX_NAME__', $query_text)
        YIELD node, score
        WITH node, score
        __PAPER_YEAR_FILTER__
        RETURN
          node.id AS paper_id,
          node.title AS title,
          node.abstract AS abstract,
          node.publication_year AS publication_year,
          coalesce(node.cited_by_count, 0) AS cited_by_count,
          score AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        cypher = cypher.replace("__INDEX_NAME__", index_name)
        cypher = cypher.replace("__PAPER_YEAR_FILTER__", paper_year_filter)
        return self.run(
            cypher,
            query_text=safe_query_text,
            top_k=top_k,
            after_year=after_year,
            before_year=before_year,
        )

    def match_papers_by_normalized_title(
        self,
        normalized_title: str,
        top_k: int,
        after_year: int | None = None,
        before_year: int | None = None,
    ) -> list[dict[str, Any]]:
        if not normalized_title:
            return []
        conditions = ["p.title_normalized = $normalized_title"]
        if after_year is not None:
            conditions.append("p.publication_year IS NOT NULL")
            conditions.append("p.publication_year >= $after_year")
        if before_year is not None:
            conditions.append("p.publication_year IS NOT NULL")
            conditions.append("p.publication_year <= $before_year")
        cypher = """
        MATCH (p:Paper)
        WHERE __CONDITIONS__
        RETURN
          p.id AS paper_id,
          p.title AS title,
          p.abstract AS abstract,
          p.publication_year AS publication_year,
          coalesce(p.cited_by_count, 0) AS cited_by_count,
          1.0 AS score
        ORDER BY cited_by_count DESC, publication_year DESC, title ASC
        LIMIT $top_k
        """
        cypher = cypher.replace("__CONDITIONS__", " AND ".join(conditions))
        return self.run(
            cypher,
            normalized_title=normalized_title,
            top_k=top_k,
            after_year=after_year,
            before_year=before_year,
        )

    def fetch_papers_by_ids(self, paper_ids: list[str]) -> list[CandidatePaper]:
        if not paper_ids:
            return []
        cypher = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        RETURN
          p.id AS paper_id,
          p.title AS title,
          p.abstract AS abstract,
          p.publication_year AS publication_year,
          coalesce(p.cited_by_count, 0) AS cited_by_count
        """
        rows = self.run(cypher, paper_ids=paper_ids)
        return [
            CandidatePaper(
                paper_id=row["paper_id"],
                title=row.get("title") or "",
                abstract=row.get("abstract"),
                publication_year=row.get("publication_year"),
                cited_by_count=int(row.get("cited_by_count") or 0),
            )
            for row in rows
        ]

    def fetch_full_paper_records(self, paper_ids: list[str]) -> list[dict[str, Any]]:
        if not paper_ids:
            return []
        cypher = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        CALL (p) {
          OPTIONAL MATCH (a:Author)-[r:AUTHORED]->(p)
          WITH a, r
          ORDER BY
            CASE WHEN r.position IS NULL THEN 1 ELSE 0 END,
            r.position ASC,
            coalesce(a.display_name, a.label, r.raw_name, a.id) ASC
          RETURN collect(coalesce(a.display_name, a.label, r.raw_name, a.id)) AS authors
        }
        RETURN
          p.id AS paper_id,
          properties(p) AS paper,
          authors
        """
        rows = self.run(cypher, paper_ids=paper_ids)
        paper_by_id = {
            row["paper_id"]: dict(
                {
                    key: value
                    for key, value in dict(row.get("paper") or {}).items()
                    if "embedding" not in key.casefold()
                },
                authors=[author for author in row.get("authors", []) if author],
            )
            for row in rows
            if row.get("paper_id")
        }
        return [paper_by_id[paper_id] for paper_id in paper_ids if paper_id in paper_by_id]

    def fetch_paper_embeddings(self, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not paper_ids:
            return {}
        cypher = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        RETURN
          p.id AS paper_id,
          p.title_embedding AS title_embedding,
          p.abstract_embedding AS abstract_embedding
        """
        rows = self.run(cypher, paper_ids=paper_ids)
        return {
            row["paper_id"]: {
                "title_embedding": row.get("title_embedding"),
                "abstract_embedding": row.get("abstract_embedding"),
            }
            for row in rows
        }

    def fetch_demo_metadata(self, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not paper_ids:
            return {}
        cypher = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(topic:Topic)
        OPTIONAL MATCH (topic)-[:SUBFIELD_OF]->(subfield:Subfield)
        OPTIONAL MATCH (subfield)-[:FIELD_OF]->(field:Field)
        OPTIONAL MATCH (field)-[:DOMAIN_OF]->(domain:Domain)
        RETURN
          p.id AS paper_id,
          p.type AS paper_type,
          [item IN collect(DISTINCT coalesce(topic.display_name, topic.label)) WHERE item IS NOT NULL] AS topics,
          [item IN collect(DISTINCT coalesce(subfield.display_name, subfield.label)) WHERE item IS NOT NULL] AS subfields,
          [item IN collect(DISTINCT coalesce(field.display_name, field.label)) WHERE item IS NOT NULL] AS fields,
          [item IN collect(DISTINCT coalesce(domain.display_name, domain.label)) WHERE item IS NOT NULL] AS domains
        """
        rows = self.run(cypher, paper_ids=paper_ids)
        metadata: dict[str, dict[str, Any]] = {}
        for row in rows:
            metadata[row["paper_id"]] = {
                "type": row.get("paper_type"),
                "topics": sorted(row.get("topics") or []),
                "subfields": sorted(row.get("subfields") or []),
                "fields": sorted(row.get("fields") or []),
                "domains": sorted(row.get("domains") or []),
            }
        return metadata

    def fetch_paper_fields(self, paper_ids: list[str]) -> dict[str, list[str]]:
        if not paper_ids:
            return {}
        cypher = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(:Topic)-[:SUBFIELD_OF]->(:Subfield)-[:FIELD_OF]->(field:Field)
        RETURN
          p.id AS paper_id,
          [item IN collect(DISTINCT coalesce(field.display_name, field.label)) WHERE item IS NOT NULL] AS fields
        """
        rows = self.run(cypher, paper_ids=paper_ids)
        return {
            row["paper_id"]: sorted(row.get("fields") or [])
            for row in rows
        }

    def fetch_neighbors(
        self,
        paper_ids: list[str],
        keyword_ids: list[str],
        author_ids: list[str],
        per_query_limit: int,
        after_year: int | None = None,
        before_year: int | None = None,
    ) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        def add_records(records: list[dict[str, Any]]) -> None:
            for row in records:
                source_id = row["source_id"]
                target_id = row["target_id"]
                nodes[source_id] = GraphNode(
                    node_id=source_id,
                    node_type=row["source_type"],
                    title=row.get("source_title"),
                    text=row.get("source_text"),
                    abstract=row.get("source_abstract"),
                    publication_year=row.get("source_year"),
                    cited_by_count=int(row.get("source_cited") or 0),
                )
                nodes[target_id] = GraphNode(
                    node_id=target_id,
                    node_type=row["target_type"],
                    title=row.get("target_title"),
                    text=row.get("target_text"),
                    abstract=row.get("target_abstract"),
                    publication_year=row.get("target_year"),
                    cited_by_count=int(row.get("target_cited") or 0),
                )
                edges.append(
                    GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=row["edge_type"],
                        properties=row.get("edge_props") or {},
                    )
                )

        if paper_ids:
            paper_neighbor_filter = _paper_year_filter("o", after_year, before_year)
            cypher = """
            CALL () {
              WITH $paper_ids AS ids
              MATCH (p:Paper) WHERE p.id IN ids
              MATCH (p)-[r:HAS_KEYWORD]->(k:Keyword)
              RETURN
                p.id AS source_id, 'Paper' AS source_type, p.title AS source_title, p.abstract AS source_abstract,
                p.publication_year AS source_year, coalesce(p.cited_by_count, 0) AS source_cited, NULL AS source_text,
                k.id AS target_id, 'Keyword' AS target_type, NULL AS target_title, NULL AS target_abstract,
                NULL AS target_year, 0 AS target_cited, k.text AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            UNION ALL
            CALL () {
              WITH $paper_ids AS ids
              MATCH (p:Paper) WHERE p.id IN ids
              MATCH (p)-[r:CITES]-(o:Paper)
              __PAPER_NEIGHBOR_FILTER__
              RETURN
                p.id AS source_id, 'Paper' AS source_type, p.title AS source_title, p.abstract AS source_abstract,
                p.publication_year AS source_year, coalesce(p.cited_by_count, 0) AS source_cited, NULL AS source_text,
                o.id AS target_id, 'Paper' AS target_type, o.title AS target_title, o.abstract AS target_abstract,
                o.publication_year AS target_year, coalesce(o.cited_by_count, 0) AS target_cited, NULL AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            UNION ALL
            CALL () {
              WITH $paper_ids AS ids
              MATCH (p:Paper) WHERE p.id IN ids
              MATCH (p)-[r:RELATED_TO]-(o:Paper)
              __PAPER_NEIGHBOR_FILTER__
              RETURN
                p.id AS source_id, 'Paper' AS source_type, p.title AS source_title, p.abstract AS source_abstract,
                p.publication_year AS source_year, coalesce(p.cited_by_count, 0) AS source_cited, NULL AS source_text,
                o.id AS target_id, 'Paper' AS target_type, o.title AS target_title, o.abstract AS target_abstract,
                o.publication_year AS target_year, coalesce(o.cited_by_count, 0) AS target_cited, NULL AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            UNION ALL
            CALL () {
              WITH $paper_ids AS ids
              MATCH (p:Paper) WHERE p.id IN ids
              MATCH (a:Author)-[r:AUTHORED]->(p)
              RETURN
                p.id AS source_id, 'Paper' AS source_type, p.title AS source_title, p.abstract AS source_abstract,
                p.publication_year AS source_year, coalesce(p.cited_by_count, 0) AS source_cited, NULL AS source_text,
                a.id AS target_id, 'Author' AS target_type, NULL AS target_title, NULL AS target_abstract,
                NULL AS target_year, 0 AS target_cited, coalesce(a.display_name, a.label) AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            """
            cypher = cypher.replace("__PAPER_NEIGHBOR_FILTER__", paper_neighbor_filter)
            add_records(
                self.run(
                    cypher,
                    paper_ids=paper_ids,
                    limit=per_query_limit,
                    after_year=after_year,
                    before_year=before_year,
                )
            )

        if keyword_ids:
            paper_neighbor_filter = _paper_year_filter("p", after_year, before_year)
            cypher = """
            CALL () {
              WITH $keyword_ids AS ids
              MATCH (k:Keyword) WHERE k.id IN ids
              MATCH (p:Paper)-[r:HAS_KEYWORD]->(k)
              __PAPER_NEIGHBOR_FILTER__
              RETURN
                k.id AS source_id, 'Keyword' AS source_type, NULL AS source_title, NULL AS source_abstract,
                NULL AS source_year, 0 AS source_cited, k.text AS source_text,
                p.id AS target_id, 'Paper' AS target_type, p.title AS target_title, p.abstract AS target_abstract,
                p.publication_year AS target_year, coalesce(p.cited_by_count, 0) AS target_cited, NULL AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            UNION ALL
            CALL () {
              WITH $keyword_ids AS ids
              MATCH (k:Keyword) WHERE k.id IN ids
              MATCH (k)-[r:COOCCUR]-(o:Keyword)
              RETURN
                k.id AS source_id, 'Keyword' AS source_type, NULL AS source_title, NULL AS source_abstract,
                NULL AS source_year, 0 AS source_cited, k.text AS source_text,
                o.id AS target_id, 'Keyword' AS target_type, NULL AS target_title, NULL AS target_abstract,
                NULL AS target_year, 0 AS target_cited, o.text AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            """
            cypher = cypher.replace("__PAPER_NEIGHBOR_FILTER__", paper_neighbor_filter)
            add_records(
                self.run(
                    cypher,
                    keyword_ids=keyword_ids,
                    limit=per_query_limit,
                    after_year=after_year,
                    before_year=before_year,
                )
            )

        if author_ids:
            paper_neighbor_filter = _paper_year_filter("p", after_year, before_year)
            cypher = """
            CALL () {
              WITH $author_ids AS ids
              MATCH (a:Author) WHERE a.id IN ids
              MATCH (a)-[r:AUTHORED]->(p:Paper)
              __PAPER_NEIGHBOR_FILTER__
              RETURN
                a.id AS source_id, 'Author' AS source_type, NULL AS source_title, NULL AS source_abstract,
                NULL AS source_year, 0 AS source_cited, coalesce(a.display_name, a.label) AS source_text,
                p.id AS target_id, 'Paper' AS target_type, p.title AS target_title, p.abstract AS target_abstract,
                p.publication_year AS target_year, coalesce(p.cited_by_count, 0) AS target_cited, NULL AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            UNION ALL
            CALL () {
              WITH $author_ids AS ids
              MATCH (a:Author) WHERE a.id IN ids
              MATCH (a)-[r:COAUTHOR]-(o:Author)
              RETURN
                a.id AS source_id, 'Author' AS source_type, NULL AS source_title, NULL AS source_abstract,
                NULL AS source_year, 0 AS source_cited, coalesce(a.display_name, a.label) AS source_text,
                o.id AS target_id, 'Author' AS target_type, NULL AS target_title, NULL AS target_abstract,
                NULL AS target_year, 0 AS target_cited, coalesce(o.display_name, o.label) AS target_text,
                type(r) AS edge_type, properties(r) AS edge_props
              LIMIT $limit
            }
            RETURN *
            """
            cypher = cypher.replace("__PAPER_NEIGHBOR_FILTER__", paper_neighbor_filter)
            add_records(
                self.run(
                    cypher,
                    author_ids=author_ids,
                    limit=per_query_limit,
                    after_year=after_year,
                    before_year=before_year,
                )
            )

        return nodes, edges
