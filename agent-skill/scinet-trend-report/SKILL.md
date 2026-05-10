---
name: scinet-trend-report
description: Run SciNet's trend-report workflow for timeline-oriented, citation-aware analysis of a research topic. Use when the user asks for research trends, field evolution, timeline reports, representative papers over time, topic history, or emerging directions through SciNet.
---

# SciNet Trend Report

Use this skill to trace how a research topic changes over time with SciNet's citation-aware retrieval.

## Workflow

1. Capture the topic as `--query`. Add `--time-range` whenever the user gives dates or asks for recent history.
2. Add `--domain`, `--keyword`, `--title`, or `--reference` anchors when available.
3. Check SciNet access with `scinet health` if needed. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
4. Run the preset:

```bash
scinet skill run trend-report --query "retrieval augmented generation" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:retrieval augmented generation"
```

5. Inspect `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
6. Build the answer around phases, representative papers, citation influence, and shifts in terminology or methods.

## Preset

- SciNet skill: `trend-report`
- Alias: `trend`
- Underlying command: `trend-report`
- Defaults: `retrieval_mode=hybrid`, `top_k=8`, `top_keywords=0`, `max_titles=0`, `max_refs=0`, `bias_citation=high`, `bias_exploration=middle`, `ranking_profile=impact`, `report_max_items=8`

## Parameter Hints

- Use `--time-range "YYYY-YYYY"` for trend questions.
- Use `--ranking-profile impact` unless the user asks for frontier discovery instead of influential work.
- Use a higher `--top-k` for broad fields, but preserve runtime by keeping `top_keywords`, `max_titles`, and `max_refs` low unless needed.

## Output Style

Return a timeline or phase-based report. Keep claims tied to returned papers and state where evidence is sparse.
