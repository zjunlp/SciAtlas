#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
RECO_BASE = "https://api.semanticscholar.org/recommendations/v1"
RICH_PAPER_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "publicationVenue",
        "year",
        "referenceCount",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
        "publicationDate",
        "journal",
        "citationStyles",
        "authors",
    ]
)


def load_api_key(env_path: str) -> str:
    candidates = ["S2-API-KEY", "S2_API_KEY", "S2-API-KEY1", "S2-API-KEY2"]
    values: Dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("'").strip('"')

    for key_name in candidates:
        if values.get(key_name):
            return values[key_name]
    raise RuntimeError(
        f"Cannot find Semantic Scholar key in {env_path}. "
        f"Tried: {', '.join(candidates)}"
    )


def request_json(url: str, api_key: str, retries: int = 3, timeout: int = 20) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url,
            headers={
                "x-api-key": api_key,
                "accept": "application/json",
                "user-agent": "s2api-smoke-test/1.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {e.code} for {url}: {body}")
            if e.code == 429 and attempt < retries:
                retry_after = e.headers.get("Retry-After")
                sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
                time.sleep(sleep_s)
                continue
            raise last_error
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Request failed: {last_error}")


def test_search_api(api_key: str, query: str, limit: int) -> Dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": limit,
            "fields": RICH_PAPER_FIELDS,
        }
    )
    url = f"{GRAPH_BASE}/paper/search?{params}"
    data = request_json(url, api_key)
    papers = data.get("data", [])
    assert isinstance(papers, list), "search api: data is not a list"
    assert len(papers) > 0, "search api: no papers returned"
    return data


def test_snippet_api(api_key: str, query: str, limit: int) -> Dict[str, Any]:
    params = urllib.parse.urlencode({"query": query, "limit": limit})
    url = f"{GRAPH_BASE}/snippet/search?{params}"
    data = request_json(url, api_key)
    snippets = data.get("data", [])
    assert isinstance(snippets, list), "snippet api: data is not a list"
    assert len(snippets) > 0, "snippet api: no snippets returned"
    return data


def paper_match_by_title(api_key: str, title: str, fields: str = RICH_PAPER_FIELDS) -> Dict[str, Any]:
    params = urllib.parse.urlencode({"query": title, "fields": fields})
    match_url = f"{GRAPH_BASE}/paper/search/match?{params}"
    match = request_json(match_url, api_key)
    paper_id = match.get("paperId")
    matched_paper: Dict[str, Any] = match
    if not paper_id and isinstance(match.get("data"), list) and match["data"]:
        matched_paper = match["data"][0]
        paper_id = matched_paper.get("paperId")
    if paper_id and "paperId" not in matched_paper:
        matched_paper["paperId"] = paper_id
    return matched_paper


def test_snippet_to_papers_by_title(api_key: str, query: str, limit: int) -> Dict[str, Any]:
    snippet_resp = test_snippet_api(api_key, query, limit)
    snippets = snippet_resp.get("data", [])
    titles: List[str] = []
    for item in snippets:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper", {})
        title = paper.get("title") if isinstance(paper, dict) else None
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())

    unique_titles = list(dict.fromkeys(titles))
    matched_papers: List[Dict[str, Any]] = []
    failed_titles: List[str] = []
    for t in unique_titles:
        try:
            matched = paper_match_by_title(api_key, t, fields=RICH_PAPER_FIELDS)
            if matched.get("paperId"):
                matched_papers.append(matched)
            else:
                failed_titles.append(t)
        except Exception:
            failed_titles.append(t)

    assert len(matched_papers) > 0, "snippet->title->paper: no matched papers returned"
    return {
        "retrievalVersion": snippet_resp.get("retrievalVersion"),
        "snippetCount": len(snippets),
        "matchedPaperCount": len(matched_papers),
        "papers": matched_papers,
        "failedTitles": failed_titles,
    }


def test_recommendation_api_by_title(api_key: str, title: str, limit: int) -> Dict[str, Any]:
    matched_paper = paper_match_by_title(api_key, title, fields=RICH_PAPER_FIELDS)
    paper_id = matched_paper.get("paperId")
    if not paper_id:
        raise RuntimeError(f"recommendation api: cannot find paper by title '{title}'")

    reco_params = urllib.parse.urlencode(
        {
            "from": "all-cs",
            "limit": limit,
            "fields": RICH_PAPER_FIELDS,
        }
    )
    reco_url = f"{RECO_BASE}/papers/forpaper/{paper_id}?{reco_params}"
    reco = request_json(reco_url, api_key)
    recs = reco.get("recommendedPapers", [])
    assert isinstance(recs, list), "recommendation api: recommendedPapers is not a list"
    return {"matchedPaper": matched_paper, "recommendationCount": len(recs), "recommendedPapers": recs}


def pretty_preview_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    papers = payload.get("data", [])
    return {"total": payload.get("total"), "count": len(payload.get("data", [])), "papers": papers}


def pretty_preview_snippet_to_papers(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "retrievalVersion": payload.get("retrievalVersion"),
        "snippetCount": payload.get("snippetCount"),
        "matchedPaperCount": payload.get("matchedPaperCount"),
        "failedTitles": payload.get("failedTitles", []),
        "papers": payload.get("papers", []),
    }


def pretty_preview_reco(payload: Dict[str, Any]) -> Dict[str, Any]:
    matched = payload.get("matchedPaper", {})
    return {
        "matchedPaper": matched,
        "recommendationCount": payload.get("recommendationCount"),
        "recommendedPapers": payload.get("recommendedPapers", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for Semantic Scholar Search/Snippet/Recommendation APIs."
    )
    parser.add_argument("--env", default="/home/weiyunxiang/yunx/.env", help="Path to .env file.")
    parser.add_argument("--query", default="transformer attention", help="Query for search/snippet API.")
    parser.add_argument(
        "--title",
        default="Attention Is All You Need",
        help="Paper title used to find a seed paper for recommendation API.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of items to request from each API.")
    args = parser.parse_args()

    api_key = load_api_key(args.env)
    print(f"Using key from: {args.env}")

    print("\n[1/3] Testing search API: /graph/v1/paper/search")
    search_resp = test_search_api(api_key, args.query, args.limit)
    print(json.dumps(pretty_preview_search(search_resp), ensure_ascii=False, indent=2))

    print("\n[2/3] Testing snippet -> title -> paper flow:")
    print("     /graph/v1/snippet/search -> /graph/v1/paper/search/match")
    snippet_papers_resp = test_snippet_to_papers_by_title(api_key, args.query, args.limit)
    print(json.dumps(pretty_preview_snippet_to_papers(snippet_papers_resp), ensure_ascii=False, indent=2))

    print("\n[3/3] Testing recommendation API by title:")
    print("     /graph/v1/paper/search/match -> /recommendations/v1/papers/forpaper/{paper_id}")
    reco_resp = test_recommendation_api_by_title(api_key, args.title, args.limit)
    print(json.dumps(pretty_preview_reco(reco_resp), ensure_ascii=False, indent=2))

    print("\nAll API smoke tests passed.")


if __name__ == "__main__":
    main()

#  1. query -> papers 的 search（/graph/v1/paper/search）是 relevance search，不是“只查 title”或“只查 abstract”的精确匹配模式。官方没有公开具体排序算法细节，所以“是否语义+关键词混合”只能推断为“相关性检索”，不能当成严格公开承诺。
#  2. query -> snippets 的 snippet（/graph/v1/snippet/search）明确是从论文 title/abstract/body 中抽约 500 词片段并按“closest match”排序返回，所以不是只基于 title。
#  3. title -> paperId 如果用 search/match（/graph/v1/paper/search/match），是 closest title match，不是 exact match。允许有差异（如大小写、轻微措辞差异）；找不到会返回 404 Title match not found。它只返回最高的 1 条。

#   不支持高级搜索语法。/graph/v1/paper/search 和 /graph/v1/snippet/search 的 query 都是 plain-text，官方明确写了 “No special query syntax is supported”，所以像 AND/OR、引号精确短语、字段限定（title:）这类都不应依赖。另一个明确限制是：带连字符词可能匹配不到，建议把 graph-based 改成 graph based。

#   最佳实践（基于官方说明 + 我的推断）：

#   1. paper/search：优先用关键词短语（2-8 个词）而不是长句子。
#   2. snippet/search：可用句子或片段，但核心术语要保留；它会在 title/abstract/body 找最相关片段。
#   3. 不要用高级语法，直接自然语言或关键词。
#   4. 避免连字符写法，统一改空格。
#   5. 先用关键词拿召回，再逐步加限定词（年份、领域、venue 等参数）提高精度。

#   短结论：对 API 的 paper/search，优先用“提炼后的关键词短语（kw1 kw2 kw3）”，通常比完整问句更稳。

#   依据与说明：

#   - 官方对 /graph/v1/paper/search 写的是 plain-text query，且 不支持高级语法（no special query syntax）。
#   - 同一份官方文档里还写了连字符词可能匹配不到（如 graph-based 要改成 graph based）。
#   - /graph/v1/paper/search/bulk 才支持 + | - " * 等高级语法，并且明确是匹配 title + abstract。

#   “问句 vs 关键词”怎么选：

#   - paper/search：可以输入问句，但这是我的推断：问句里停用词多、意图噪声更大，通常不如关键词短语精准。
#   - 实践建议：先用 3-8 个核心术语（空格分隔），再配合 year/fieldsOfStudy/venue 等过滤逐步收敛。
#   - 如果你确实要复杂布尔/短语控制，改用 paper/search/bulk。
#   - snippet/search 由于是找文本片段，句子/片段输入通常可行。

# • 可以，ICLR 直接用就行。我刚联网查了官方 Swagger 并实测了 venue=ICLR 与 venue=International Conference on Learning Representations，两者返回结果一致（同样的 total=415，示例查询 query=transformer&year=2024-2026）。

#   /graph/v1/paper/search 支持的“限定词过滤器”是这些：

#   1. publicationTypes
#      示例：publicationTypes=Conference 或 publicationTypes=Review,JournalArticle
#   2. openAccessPdf（无值开关）
#      示例：...&openAccessPdf&...
#      含义：只要有公开 PDF 的论文
#   3. minCitationCount
#      示例：minCitationCount=200
#   4. publicationDateOrYear（日期/年份范围，格式最灵活）
#      示例：publicationDateOrYear=2024:2026
#      示例：publicationDateOrYear=2025-01-01:2025-12-31
#   5. year（年份范围）
#      示例：year=2026
#      示例：year=2024-2026
#   6. venue（会场/期刊，逗号分隔）
#      示例：venue=ICLR
#      示例：venue=International Conference on Learning Representations
#      示例：venue=Nature,Science
#   7. fieldsOfStudy（学科，逗号分隔）
#      示例：fieldsOfStudy=Computer Science
#      示例：fieldsOfStudy=Physics,Mathematics

#   常用完整写法示例：

#   /graph/v1/paper/search?query=diffusion+transformer&year=2024-2026&venue=ICLR&fieldsOfStudy=Computer%20Science&publicationTypes=Conference&minCitationCount=20&openAccessPdf&limit=20

#   补充：

#   - fields 是返回字段选择，不是过滤器。
#   - limit/offset 是分页，不是过滤器。
