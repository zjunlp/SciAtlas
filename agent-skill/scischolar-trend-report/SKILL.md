---
name: scischolar-trend-report
description: Use SciScholar search-papers as the retrieval base to build a timeline-oriented research trend report. Trigger when the user asks for field evolution, topic history, recent trends, representative papers over time, citation-aware development, or emerging directions.
---

# SciScholar Trend Report

This skill migrates `search-papers` into a trend-analysis workflow. The agent must retrieve representative papers, organize them over time, and explain how the topic evolved.

## End-to-End Workflow

1. Capture the trend topic as `--query`.
2. Use `--time-range` when the user gives years. If not, choose a sensible recent window and state it.
3. Add core terms as `--keyword "high:<term>"` and known anchor papers as `--title`.
4. Run the trend-report channel:

```bash
scischolar trend-report --query "retrieval augmented generation" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:retrieval augmented generation" --top-k 10
```

5. If the downstream command is unavailable, use base `search-papers`:

```bash
scischolar search-papers --retrieval-mode hybrid --query "retrieval augmented generation" --time-range "2020-2025" --keyword "high:retrieval augmented generation" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-citation high --bias-exploration middle --ranking-profile impact --report-max-items 10
```

6. Read `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
7. Convert retrieved evidence into phases, not just a ranked paper list.

## Deliverable

Return:

- Search window and framing.
- Phase-by-phase timeline.
- Representative papers in each phase.
- Shifts in problems, methods, evaluation, or datasets.
- Current frontier and uncertain areas.
- Suggested follow-up searches for missing periods or subtopics.

## Evidence Rules

- Keep year ordering explicit.
- Do not infer a trend from one paper unless you label it as weak evidence.
- If the retrieval lacks older or newer papers, state the coverage gap.
