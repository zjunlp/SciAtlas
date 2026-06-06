from __future__ import annotations

from sciatlas.cli import compact_response_for_artifact


def test_compact_response_prunes_source_authors_and_preserves_papers():
    paper = {"paper_id": "W1", "title": "Open World Agent", "score": 0.9}
    response = {
        "ok": True,
        "data": {
            "result": {
                "ranking": {"papers": [paper]},
                "sources": {
                    "kg": {
                        "status": "ok",
                        "paper_count": 1,
                        "papers": [paper],
                        "author_count": 5,
                        "authors": [
                            {"name": "A", "score": 0.4, "rank": 1},
                            {"name": "B", "score": 0.0, "rank": 2},
                            {"name": "C", "score": 0.3, "rank": 3},
                            {"name": "D", "score": None, "rank": 4},
                            {"name": "E", "score": -0.1, "rank": 5},
                        ],
                    }
                },
            }
        },
    }

    compacted = compact_response_for_artifact(response, author_limit=2, min_author_score=0.0)

    kg_source = compacted["data"]["result"]["sources"]["kg"]
    assert kg_source["papers"] == [paper]
    assert compacted["data"]["result"]["ranking"]["papers"] == [paper]
    assert [author["name"] for author in kg_source["authors"]] == ["A", "C"]
    assert kg_source["author_count"] == 2
    assert kg_source["author_count_full"] == 5
    assert kg_source["authors_pruned"] is True

    pruning = compacted["artifact_compaction"]["source_author_pruning"][0]
    assert pruning["original_count"] == 5
    assert pruning["kept_count"] == 2
    assert pruning["dropped_score_lte_min"] == 2
    assert pruning["dropped_unscored"] == 1

    assert len(response["data"]["result"]["sources"]["kg"]["authors"]) == 5
