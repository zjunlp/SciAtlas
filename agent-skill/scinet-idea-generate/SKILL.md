---
name: scinet-idea-generate
description: Run SciNet's idea-generate workflow for broad knowledge-graph exploration and research idea seed discovery. Use when the user wants new research directions, topic combinations, brainstorming grounded in literature, project ideas, or hypotheses derived from SciNet retrieval.
---

# SciNet Idea Generate

Use this skill when the user wants literature-grounded idea seeds rather than a freeform brainstorm.

## Workflow

1. Capture the broad topic as `--query`. Use `--keyword` anchors for core concepts and constraints.
2. Add `--domain` and `--time-range` when the user gives a field or publication window.
3. Check SciNet access with `scinet health` if needed. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
4. Run the preset:

```bash
scinet skill run idea-generate --query "knowledge editing for large language models" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:knowledge editing" --keyword "middle:large language models"
```

5. Inspect `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
6. Turn returned clusters, papers, and related concepts into candidate ideas. Keep the evidence trail visible.

## Preset

- SciNet skill: `idea-generate`
- Alias: `ideate`
- Underlying command: `idea-generate`
- Defaults: `retrieval_mode=hybrid`, `top_k=8`, `top_keywords=0`, `max_titles=0`, `max_refs=0`, `bias_related=high`, `bias_cooccurrence=high`, `bias_exploration=high`, `ranking_profile=discovery`, `report_max_items=8`

## Generation Rules

- Favor ideas that combine at least two retrieved concepts or paper clusters.
- Include a short evidence note for each idea.
- Mark speculative leaps clearly.
- Suggest a follow-up `scinet-idea-evaluate` run for the strongest ideas.

## Output Style

Return 3-8 idea seeds with names, core hypothesis, related evidence, why it might be interesting, and a quick validation plan.
