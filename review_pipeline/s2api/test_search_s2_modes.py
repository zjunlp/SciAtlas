from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent / "search_s2.py"
SPEC = importlib.util.spec_from_file_location("search_s2_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
search_s2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search_s2
SPEC.loader.exec_module(search_s2)


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        idea_text=None,
        pdf_path="/tmp/demo.pdf",
        env="/tmp/test.env",
        mode="hybrid",
        top_k=20,
        search_top_k=None,
        recommend_top_k=None,
        per_keyword_limit=10,
        include_per_keyword_results=False,
        keyword_model="test-model",
        keyword_api_url="http://example.test",
        keyword_timeout=10,
        search_timeout=11,
        search_retries=1,
        recommendation_limit=9,
        grobid_base_url="http://127.0.0.1:8070",
        grobid_start_page=None,
        use_env_proxy=False,
        pretty=False,
    )


class SearchS2ModeTests(unittest.TestCase):
    def test_search_mode_returns_only_search_payload(self) -> None:
        args = make_args()
        args.mode = "search"
        args.top_k = 1
        args.search_top_k = 2

        search_papers = [
            {"paperId": "s1", "title": "Search 1", "_retrieval": {"matched_keywords": ["q1"], "keyword_hit_count": 1, "best_keyword_rank": 1}},
            {"paperId": "s2", "title": "Search 2", "_retrieval": {"matched_keywords": ["q2"], "keyword_hit_count": 1, "best_keyword_rank": 2}},
            {"paperId": "s3", "title": "Search 3", "_retrieval": {"matched_keywords": ["q3"], "keyword_hit_count": 1, "best_keyword_rank": 3}},
        ]
        per_query = [{"keyword": "q1", "papers": []}]
        extractor_instance = type("Extractor", (), {"extract_keywords": lambda self, _: ["seed"]})()

        with patch.object(search_s2, "extract_pdf_payload", return_value={"title": "Demo", "abstract": "Abstract", "body": ""}), patch.object(
            search_s2, "KeywordExtractor", return_value=extractor_instance
        ), patch.object(search_s2, "compose_search_queries", return_value=["q1", "q2"]), patch.object(
            search_s2, "aggregate_keyword_searches", return_value=(search_papers, per_query)
        ), patch.object(search_s2, "SemanticScholarSearchClient"):
            payload = search_s2.run_pipeline(args)

        self.assertEqual(payload["mode"], "search")
        self.assertEqual(payload["paper_count"], 1)
        self.assertEqual(payload["papers"][0]["paperId"], "s1")
        self.assertEqual(payload["search_queries"], ["q1", "q2"])
        self.assertNotIn("recommendation", payload)

    def test_recommend_mode_returns_only_recommend_payload(self) -> None:
        args = make_args()
        args.mode = "recommend"
        args.top_k = 2
        args.recommend_top_k = 3

        recommendation_result = {
            "seed_title": "Demo",
            "seed_paper": {"paperId": "seed-1", "title": "Demo"},
            "recommended_papers": [
                {"paperId": "r1", "title": "Reco 1"},
                {"paperId": "r2", "title": "Reco 2"},
                {"paperId": "r3", "title": "Reco 3"},
                {"paperId": "r4", "title": "Reco 4"},
            ],
        }

        with patch.object(search_s2, "extract_pdf_payload", return_value={"title": "Demo", "abstract": "Abstract", "body": ""}), patch.object(
            search_s2, "recommendation_search_by_pdf_title", return_value=recommendation_result
        ), patch.object(search_s2, "SemanticScholarSearchClient"):
            payload = search_s2.run_pipeline(args)

        self.assertEqual(payload["mode"], "recommend")
        self.assertEqual(payload["paper_count"], 2)
        self.assertEqual([paper["paperId"] for paper in payload["papers"]], ["r1", "r2"])
        self.assertEqual(payload["recommendation"]["seed_paper"]["paperId"], "seed-1")
        self.assertNotIn("search_queries", payload)

    def test_hybrid_mode_contains_search_and_recommend_sections(self) -> None:
        args = make_args()
        args.mode = "hybrid"
        args.top_k = 10
        args.search_top_k = 1
        args.recommend_top_k = 1

        search_papers = [
            {"paperId": "shared", "title": "Shared", "_retrieval": {"matched_keywords": ["q1"], "keyword_hit_count": 1, "best_keyword_rank": 2}},
            {"paperId": "search-only", "title": "Search Only", "_retrieval": {"matched_keywords": ["q2"], "keyword_hit_count": 1, "best_keyword_rank": 1}},
        ]
        extractor_instance = type("Extractor", (), {"extract_keywords": lambda self, _: ["seed"]})()
        recommendation_result = {
            "seed_title": "Demo",
            "seed_paper": {"paperId": "seed-1", "title": "Demo"},
            "recommended_papers": [
                {"paperId": "shared", "title": "Shared"},
                {"paperId": "reco-only", "title": "Reco Only"},
            ],
        }

        with patch.object(search_s2, "extract_pdf_payload", return_value={"title": "Demo", "abstract": "Abstract", "body": ""}), patch.object(
            search_s2, "KeywordExtractor", return_value=extractor_instance
        ), patch.object(search_s2, "compose_search_queries", return_value=["q1"]), patch.object(
            search_s2, "aggregate_keyword_searches", return_value=(search_papers, [])
        ), patch.object(
            search_s2, "recommendation_search_by_pdf_title", return_value=recommendation_result
        ), patch.object(search_s2, "SemanticScholarSearchClient"):
            payload = search_s2.run_pipeline(args)

        self.assertEqual(payload["mode"], "hybrid")
        self.assertIn("search", payload)
        self.assertIn("recommendation", payload)
        self.assertEqual(payload["search"]["paper_count"], 1)
        self.assertEqual(payload["recommendation"]["count"], 1)
        self.assertEqual(payload["paper_count"], 1)
        self.assertEqual(payload["papers"][0]["paperId"], "shared")


if __name__ == "__main__":
    unittest.main()
