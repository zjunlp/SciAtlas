---
name: scinet-idea-grounding
description: Run SciNet's idea-grounding workflow to compare a research idea with related work and gather graph-backed differentiation evidence. Use when the user gives a research idea and wants similar papers, prior-art grounding, novelty context, support papers, or evidence for positioning the idea.
---

# SciNet Idea Grounding

Use this skill when the user wants to ground a research idea in existing literature before writing, pitching, or refining it.

## Workflow

1. Capture the idea as `--idea`. Add `--domain`, `--time-range`, and one or more `--keyword` anchors when available.
2. Check SciNet access with `scinet health` if the API token or CLI state is uncertain. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
3. Run the preset:

```bash
scinet skill run idea-grounding --idea "LLM-based idea evaluation for scientific research" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:idea evaluation" --keyword "middle:LLM as a judge"
```

4. Inspect `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
5. Report related work, overlaps, gaps, and differentiation opportunities. Do not claim novelty beyond what the returned evidence supports.

## Preset

- SciNet skill: `idea-grounding`
- Alias: `ground`
- Underlying command: `idea-grounding`
- Defaults: `retrieval_mode=hybrid`, `top_k=5`, `top_keywords=0`, `max_titles=0`, `max_refs=0`, `bias_keyword=high`, `bias_related=high`, `bias_citation=low`, `bias_exploration=low`, `ranking_profile=precision`, `report_max_items=5`

## Parameter Hints

- Use `--idea` for the full proposal, not just a short query.
- Add `--keyword "high:<core concept>"` for the central method or task.
- Add `--keyword "middle:<neighbor concept>"` for evaluation criteria, model family, domain, or data type.
- Keep defaults when the user wants precise prior-art grounding.

## Output Style

Structure the response around evidence: closest prior work, what the idea shares with it, possible differentiators, and follow-up searches worth running.
