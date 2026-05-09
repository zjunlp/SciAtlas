---
name: scinet-literature-review
description: Run SciNet's literature-review workflow for graph-aware paper pools, reading lists, related-work evidence, and literature-review reports. Use when the user asks for a literature review, survey preparation, related work summary, reading list, or evidence-backed paper retrieval for a research topic through SciNet.
---

# SciNet Literature Review

Use this skill to drive the SciNet `literature-review` preset instead of doing an ad hoc web search. It collects a core paper pool and produces terminal summaries plus saved artifacts.

## Workflow

1. Collect the research topic as `--query`. Add `--domain`, `--time-range`, `--keyword`, `--title`, or `--reference` when the user gives them.
2. Check that the SciNet CLI can run. Use `scinet health` if credentials or connectivity are uncertain. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
3. Prefer explicit expert parameters. Use natural-language `--text` only for quick trials or when the user has not provided structured fields.
4. Run the preset:

```bash
scinet skill run literature-review --query "retrieval augmented generation" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:retrieval augmented generation"
```

5. Read the generated run directory under `runs/<run_id>/`, especially `summary.txt`, `report.md`, `request.json`, and `response.json`.
6. Summarize only what SciNet returned. Preserve paper titles, years, scores, evidence snippets, and uncertainty from the artifacts.

## Preset

- SciNet skill: `literature-review`
- Aliases: `review`, `lit-review`
- Underlying command: `literature-review`
- Defaults: `retrieval_mode=hybrid`, `top_k=5`, `top_keywords=0`, `max_titles=0`, `max_refs=0`, `bias_exploration=low`, `ranking_profile=balanced`, `report_max_items=5`

## Parameter Hints

- Use `--keyword "high:<term>"` for the main concept.
- Use `--keyword "middle:<term>"` for secondary concepts.
- Use `--title "middle:<paper title>"` when the user names a representative paper.
- Use `--reference "low:<paper title>"` when a cited work should guide retrieval.
- Increase `--top-k` only when the user asks for a broader review or survey.

## Output Style

Return a concise review-oriented answer: topic framing, representative papers, clusters or themes, and next reading steps. Mention the run directory so the user can inspect the full artifacts.
