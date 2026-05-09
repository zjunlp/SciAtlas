# SciNet Agent Skill Pack

This folder packages SciNet workflows as a portable agent-skill pack. These are project assets for coding and research agents: each skill has a `SKILL.md` instruction file, with optional tool-specific UI metadata under `agents/`, so tools such as Codex, Claude Code, and other SKILL.md-aware agents can load or adapt the same workflow guidance.

They complement the SciNet CLI presets in `scinet/src/scinet/builtin_skills.json`:

- CLI presets make workflows runnable with `scinet skill run ...`.
- The Agent Skill layer teaches an agent when and how to use those workflows, which parameters to prefer, and where to read generated artifacts.

## Included Skills

| Skill | SciNet workflow | Best for |
|---|---|---|
| `scinet-literature-review` | `literature-review` | Evidence-backed reading lists and related-work reports |
| `scinet-idea-grounding` | `idea-grounding` | Comparing a research idea with prior work |
| `scinet-idea-evaluate` | `idea-evaluate` | Checking novelty, feasibility, soundness, and differentiation |
| `scinet-idea-generate` | `idea-generate` | Generating literature-grounded research idea seeds |
| `scinet-trend-report` | `trend-report` | Tracing topic evolution and representative papers over time |
| `scinet-researcher-review` | `researcher-review` | Profiling a researcher and representative works |
| `scinet-quick-paper-search` | `paper-search` | Fast paper candidate lookup before deeper workflows |

## Use

Copy any skill directory into the skill directory supported by your agent tool, then restart or refresh that tool's skill index.

Codex example on Windows PowerShell:

```powershell
Copy-Item -Recurse .\agent-skill\scinet-literature-review "$env:USERPROFILE\.codex\skills\"
```

Codex example on macOS/Linux:

```bash
cp -R ./agent-skill/scinet-literature-review ~/.codex/skills/
```

Claude Code and other agent tools can use the same `SKILL.md` folders when their skill/plugin loader supports filesystem skills. If a tool uses a different metadata filename, keep `SKILL.md` as the source of truth and adapt the `agents/` metadata as needed.

The skills expect the SciNet CLI to be installed or available from this repository. Configure `SCINET_API_BASE_URL` and `SCINET_API_KEY` before running hosted SciNet tasks.

## Design Notes

- Keep each skill small enough to load quickly.
- Keep operational defaults aligned with `builtin_skills.json`.
- Do not put API tokens or run artifacts in this folder.
- Use `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json` as the evidence trail after each workflow run.
