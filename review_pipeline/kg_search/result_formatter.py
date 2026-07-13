from __future__ import annotations

from .models import CandidatePaper
from .text_utils import compact_abstract


def format_results(results: list[CandidatePaper]) -> list[dict]:
    formatted = []
    for item in results:
        formatted.append(
            {
                "paper_id": item.paper_id,
                "title": item.title,
                "abstract": compact_abstract(item.abstract),
                "publication_year": item.publication_year,
                "cited_by_count": item.cited_by_count,
                "final_score": item.final_score,
                "pre_graph_score": item.pre_graph_score,
                "graph_score": item.graph_score,
                "importance": item.importance,
                "hit_sources": sorted(item.hit_sources),
                "evidence": {
                    "keywords": [
                        {
                            "input_keyword": evidence.input_keyword,
                            "input_score": evidence.input_score,
                            "kg_keyword_id": evidence.kg_keyword_id,
                            "kg_keyword_text": evidence.kg_keyword_text,
                            "match_score": evidence.match_score,
                            "match_type": evidence.match_type,
                            "edge_relevance_score": evidence.edge_relevance_score,
                        }
                        for evidence in item.keyword_evidence
                    ],
                    "titles": [
                        {
                            "extracted_title": evidence.extracted_title,
                            "confidence": evidence.confidence,
                            "matched_title": evidence.matched_title,
                            "match_score": evidence.match_score,
                            "match_type": evidence.match_type,
                        }
                        for evidence in item.title_evidence
                    ],
                    "graph_paths": item.graph_evidence,
                },
                "score_breakdown": item.score_breakdown,
            }
        )
    return formatted
