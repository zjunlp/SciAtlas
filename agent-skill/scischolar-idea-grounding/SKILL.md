---
name: scischolar-idea-grounding
description: Use SciScholar search-papers as the retrieval base to ground a research idea against prior work. Trigger when the user gives an idea and asks for similar work, differentiation evidence, prior-art positioning, motivation support, or literature-grounded refinement.
---

# SciScholar Idea Grounding

This skill turns `search-papers` retrieval into an idea-grounding workflow. The goal is to move from a proposed idea to prior-work evidence, overlap analysis, and concrete positioning.

## End-to-End Workflow

1. Rewrite the idea into retrieval anchors.
   - Full idea -> `--idea` or `--query`
   - Core method/task -> `--keyword "high:<term>"`
   - Application, evaluation, or setting -> `--keyword "middle:<term>"`
   - Known baselines -> `--title` or `--reference`
2. Run the idea-grounding channel:

```bash
scischolar idea-grounding --idea "LLM-based multi-perspective evaluation for scientific research ideas" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:idea evaluation" --keyword "middle:LLM as a judge" --top-k 5
```

3. If the downstream command is unavailable, use `search-papers` with the idea as the query:

```bash
scischolar search-papers --retrieval-mode hybrid --query "LLM-based multi-perspective evaluation for scientific research ideas" --keyword "high:idea evaluation" --keyword "middle:LLM as a judge" --top-k 5 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-keyword high --bias-related high --bias-citation low --bias-exploration low --ranking-profile precision --report-max-items 5
```

4. Read `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
5. Compare the user's idea with retrieved works and finish with a positioning answer.

## Deliverable

Return:

- Closest related works and what they appear to cover.
- Overlap between the user's idea and the evidence.
- Differentiation opportunities that remain plausible.
- Missing evidence or weak assumptions.
- Suggested revised idea statement.

## Evidence Rules

- Do not claim novelty just because no exact match was retrieved.
- Treat retrieved papers as grounding evidence, not a complete patent-style prior-art search.
- If the idea has multiple components, run or recommend separate searches for components with weak coverage.
