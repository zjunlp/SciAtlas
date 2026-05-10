---
name: scischolar-quick-paper-search
description: Use SciScholar search-papers as a fast evidence-seeding workflow before heavier downstream tasks. Trigger when the user wants a quick paper check, small candidate list, retrieval smoke test, or first-pass evidence pool that can later feed literature review, idea grounding, trend analysis, or researcher review.
---

# SciScholar Quick Paper Search

This skill is the lightweight entry point for the Agent Skill pack. It uses `search-papers` directly to create a small evidence seed that can be expanded into a downstream task.

## End-to-End Workflow

1. Convert the user's phrase into a precise `--query`.
2. Add one `--keyword "high:<term>"` when the main concept is clear.
3. Keep the search small unless the user asks for breadth.
4. Run base retrieval:

```bash
scischolar search-papers --retrieval-mode hybrid --query "open world agent" --keyword "high:open world agent" --top-k 3 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-exploration low --ranking-profile precision --report-max-items 3
```

5. Read `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
6. Return the candidate papers and recommend which downstream Agent Skill should handle the next step.

## Deliverable

Return:

- 3-5 candidate papers or evidence items.
- One-line relevance note for each.
- Whether the evidence is enough for the user's current purpose.
- Suggested next skill: `scischolar-literature-review`, `scischolar-idea-grounding`, `scischolar-idea-evaluate`, `scischolar-idea-generate`, `scischolar-trend-report`, or `scischolar-researcher-review`.

## Evidence Rules

- Keep the answer short.
- Do not turn this into a full survey.
- If no strong results appear, propose a refined query or additional anchor terms.
