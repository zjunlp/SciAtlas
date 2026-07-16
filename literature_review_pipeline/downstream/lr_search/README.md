# Literature Review Search MVP

Standalone MVP for producing a structured literature-review search evidence map.

```bash
python3 downstream/lr_search/literature_review_search.py \
  --topic "LLM reasoning" \
  --output-dir downstream/runs/lr_search_llm_reasoning/artifacts/lr_search \
  --llm-model deepseek-v3.2 \
  --probe-kg-top-k 30 --probe-s2-top-k 30 \
  --round-kg-top-k 20 --round-s2-top-k 20 \
  --round1-action-limit 7 --round2-action-limit 4 \
  --llm-paper-limit 100
```

By default, the search path stays close to the original MVP behavior: one probe,
Round 1 actions, no automatic KG scoring policy, no query cleaning, no relevance
guard, and no Round 2 refinement. The planner still records action-level
`query_style`, `evidence_role`, and empty `kg_policy_args` for auditability.

Experimental knobs are opt-in:

```bash
python3 downstream/lr_search/literature_review_search.py \
  --topic "LLM reasoning" \
  --output-dir downstream/runs/lr_search_experimental/artifacts/lr_search \
  --llm-model deepseek-v3.2 \
  --enable-round2 \
  --enable-kg-policy \
  --enable-query-cleaning \
  --enable-relevance-guard
```

For repeated ablations, reuse a shared per-action merge-search cache:

```bash
python3 downstream/lr_search/literature_review_search.py \
  --topic "LLM reasoning" \
  --output-dir downstream/runs/lr_search_default/artifacts/lr_search \
  --search-cache-dir downstream/runs/lr_search_shared_cache \
  --llm-model deepseek-v3.2
```

For a cheap schema smoke test without external search or LLM calls:

```bash
python3 downstream/lr_search/literature_review_search.py \
  --topic "LLM reasoning" \
  --output-dir /tmp/lr_search_smoke \
  --mock-backend
```

After a search run, organize every retrieved paper into method clusters, time
buckets, paper roles, and a method-by-time matrix:

```bash
python3 downstream/lr_search/organize_search_result.py \
  --search-result downstream/runs/lr_search_real_llm_reasoning_full/artifacts/lr_search/search_result.json
```

Generate a polished Markdown literature-review report from the organized
evidence map:

```bash
python3 downstream/lr_search/generate_literature_review_report.py \
  --evidence-map downstream/runs/lr_search_real_llm_reasoning_full/artifacts/lr_search/organized_search_result.json \
  --llm-model deepseek-v3.2
```
