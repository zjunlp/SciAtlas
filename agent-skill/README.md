# SciScholar Agent Skill Pack

This folder packages SciScholar workflows as a portable agent-skill pack. Each skill migrates SciScholar's base `search-papers` retrieval capability into an end-to-end downstream task: plan the search, run SciScholar, read generated artifacts, and complete the user's research goal.

These are project assets for coding and research agents. Each skill has a `SKILL.md` instruction file, with optional tool-specific UI metadata under `agents/`, so tools such as Codex, Claude Code, and other SKILL.md-aware agents can load or adapt the same workflow guidance.

They complement SciScholar's CLI retrieval layer:

- CLI commands make retrieval runnable from the terminal.
- The Agent Skill layer teaches an agent how to migrate retrieval into a downstream task deliverable.

## Included Skills

| Skill | Retrieval base | Downstream goal |
|---|---|---|
| `scischolar-quick-paper-search` | `search-papers` | Small evidence seed and downstream routing |
| `scischolar-literature-review` | `search-papers` | Evidence-backed reading lists and related-work reports |
| `scischolar-idea-grounding` | `search-papers` | Comparing a research idea with prior work |
| `scischolar-idea-evaluate` | `search-papers` | Checking novelty, feasibility, soundness, and differentiation |
| `scischolar-idea-generate` | `search-papers` | Generating literature-grounded research idea seeds |
| `scischolar-trend-report` | `search-papers` | Tracing topic evolution and representative papers over time |
| `scischolar-researcher-review` | `search-papers` plus author seed lookup | Profiling a researcher and representative works |

## Use

Copy any skill directory into the skill directory supported by your agent tool, then restart or refresh that tool's skill index.

Codex example on Windows PowerShell:

```powershell
Copy-Item -Recurse .\agent-skill\scischolar-literature-review "$env:USERPROFILE\.codex\skills\"
```

Codex example on macOS/Linux:

```bash
cp -R ./agent-skill/scischolar-literature-review ~/.codex/skills/
```

Claude Code and other agent tools can use the same `SKILL.md` folders when their skill/plugin loader supports filesystem skills. If a tool uses a different metadata filename, keep `SKILL.md` as the source of truth and adapt the `agents/` metadata as needed.

The skills expect the SciScholar CLI to be installed or available from this repository. Use the `scischolar` executable and configure `SCISCHOLAR_API_BASE_URL` and `SCISCHOLAR_API_KEY` before running hosted SciScholar tasks.

## Design Notes

- Keep each skill small enough to load quickly.
- Keep operational defaults aligned with `search-papers` and the downstream channel commands.
- Do not put API tokens or run artifacts in this folder.
- Use `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json` as the evidence trail after each workflow run.
