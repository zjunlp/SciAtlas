---
name: scinet-quick-paper-search
description: Run SciNet's quick-paper-search workflow for lightweight low-level paper lookup and fast title/vector search. Use when the user wants a quick paper check, rapid candidate retrieval, smoke testing SciNet search, or a small list of papers before running heavier review workflows.
---

# SciNet Quick Paper Search

Use this skill for fast paper candidate retrieval when a full review, trend report, or idea evaluation would be more than the user needs.

## Workflow

1. Capture the search phrase as `--text` for quick lookup, or as `--query` if the user gives structured parameters.
2. Check SciNet access with `scinet health` if needed. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
3. Run the preset:

```bash
scinet skill run quick-paper-search --text "open world agent"
```

The built-in alias also works:

```bash
scinet skill run quick --text "open world agent"
```

4. Inspect `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json` when generated.
5. If the results look promising and the user needs depth, suggest the matching heavier workflow: `scinet-literature-review`, `scinet-trend-report`, or `scinet-idea-grounding`.

## Preset

- SciNet skill: `quick-paper-search`
- Alias: `quick`
- Underlying command: `paper-search`
- Defaults: `mode=vector`, `field=title`, `top_k=3`, `report_max_items=3`

## Parameter Hints

- Keep `top_k=3` for fast checks.
- Increase `--top-k` only when the user explicitly asks for more candidates.
- Prefer title-like search terms; this preset is intentionally lightweight.

## Output Style

Return the few candidate papers with brief relevance notes. Keep the answer short and hand off to a deeper skill when needed.
