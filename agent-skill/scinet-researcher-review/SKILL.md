---
name: scinet-researcher-review
description: Run SciNet's researcher-review workflow to profile a researcher from an author name or OpenAlex author ID. Use when the user asks for a researcher's background, publication trajectory, representative papers, related directions, collaboration context, or literature-grounded author profile.
---

# SciNet Researcher Review

Use this skill to create a SciNet-backed profile of a researcher instead of manually assembling an author biography.

## Workflow

1. Capture the person as `--author` unless the user provides an OpenAlex author ID supported by the CLI.
2. Use `--limit` to control the number of papers. Keep `--no-abstract` for fast summaries unless the user needs abstract-level detail.
3. Check SciNet access with `scinet health` if needed. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
4. Run the preset:

```bash
scinet skill run researcher-review --author "Yoshua Bengio"
```

5. Inspect `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
6. Report trajectory, major themes, representative works, and uncertainty about author disambiguation when present.

## Preset

- SciNet skill: `researcher-review`
- Aliases: `researcher`, `profile`
- Underlying command: `researcher-review`
- Defaults: `limit=10`, `no_abstract=true`, `report_max_items=10`

## Parameter Hints

- Use exact author names when possible.
- Increase `--limit` when the user asks for a fuller profile.
- Omit `--no-abstract` only if the user asks for richer paper summaries and runtime is acceptable.
- If several researchers may share a name, state the ambiguity and rely on returned identifiers or affiliations.

## Output Style

Return a compact profile: research trajectory, main topics, representative papers, collaboration or venue patterns if present, and caveats about disambiguation.
