---
name: scischolar-literature-review
description: Use SciScholar search-papers as the retrieval base to complete an end-to-end literature review workflow. Trigger when the user asks for a literature review, related work section, survey outline, reading list, paper map, or evidence-backed overview of a research topic.
---

# SciScholar Literature Review

This skill is a downstream task playbook built on SciScholar's `search-papers` retrieval capability. Do not stop after listing papers: retrieve evidence, read the artifacts, then synthesize a literature-review deliverable.

## End-to-End Workflow

1. Translate the user's topic into a retrieval plan.
   - Main topic -> `--query`
   - Field constraint -> `--domain`
   - Year window -> `--time-range`
   - Core terms -> `--keyword "high:<term>"`
   - Secondary terms -> `--keyword "middle:<term>"`
   - Known representative papers -> `--title "middle:<title>"`
2. Run the literature-review channel, which is a task-oriented migration of `search-papers`:

```bash
scischolar literature-review --query "retrieval augmented generation" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:retrieval augmented generation" --top-k 8
```

3. If the downstream command is unavailable, fall back to the base retrieval primitive:

```bash
scischolar search-papers --retrieval-mode hybrid --query "retrieval augmented generation" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:retrieval augmented generation" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-exploration low --ranking-profile balanced --report-max-items 8
```

4. Locate the generated `runs/<run_id>/` directory and read `summary.txt`, `report.md`, `request.json`, and `response.json`.
5. Build the final review from retrieved evidence, not from memory alone.

## Deliverable

Produce a concise literature review with:

- Scope and search framing.
- Core paper clusters or methodological lines.
- Representative papers with years and why they matter.
- Gaps, limitations, or unresolved questions visible from the evidence.
- A short reading order or next-search plan.

## Evidence Rules

- Preserve paper titles, years, scores, venues, and evidence snippets when present.
- Separate "SciScholar returned" from your own synthesis.
- If evidence is thin, say what additional `search-papers` query should be run.
- Mention the run directory so the user can inspect the full artifacts.
