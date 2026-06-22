# SciAtlas Agent Skill Pack

This folder packages SciAtlas workflows as a portable agent-skill pack. Each skill migrates SciAtlas's base `search-papers` retrieval capability into an end-to-end downstream task: bootstrap a novice user's environment, obtain and configure the required API token with user feedback when needed, run only `search-papers`, read generated artifacts, and complete the user's research goal.

These are project assets for coding and research agents. Each skill has a `SKILL.md` instruction file, with optional tool-specific UI metadata under `agents/`, so tools such as Codex, Claude Code, and other SKILL.md-aware agents can load or adapt the same workflow guidance.

<p align="center">
  <img src="../imgs/agent-skill-demo.gif" alt="SciAtlas Agent Skill workflow demo" width="92%">
</p>

They complement SciAtlas's CLI retrieval layer:

- The CLI makes the base `search-papers` retrieval runnable from the terminal.
- The Agent Skill layer teaches an agent how to turn that one retrieval primitive into downstream task deliverables.

## Included Skills

| Skill | Retrieval base | Downstream goal |
|---|---|---|
| `sciatlas-quick-paper-search` | `search-papers` | Small evidence seed and downstream routing |
| `sciatlas-literature-review` | `search-papers` | Evidence-backed reading lists and related-work reports |
| `sciatlas-idea-grounding` | `search-papers` | Comparing a research idea with prior work |
| `sciatlas-idea-evaluate` | `search-papers` | Checking novelty, feasibility, soundness, and differentiation |
| `sciatlas-idea-generate` | `search-papers` | Generating literature-grounded research idea seeds |
| `sciatlas-trend-report` | `search-papers` | Tracing topic evolution and representative papers over time |
| `sciatlas-researcher-review` | `search-papers` only | Profiling a researcher from retrieved paper evidence |

## Use

### Git

Git is the source checkout for the Agent Skill pack. Clone it once, then pull the repository when you want updated skill instructions:

```bash
git clone https://github.com/zjunlp/SciAtlas.git
cd SciAtlas
git pull
```

Run the copy commands below from the repository root.

### Claude Code

Claude Code loads filesystem skills from `~/.claude/skills` on macOS/Linux or `%USERPROFILE%\.claude\skills` on Windows. Copy the packaged SciAtlas skill directories there, then restart Claude Code or refresh the workspace.

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills" | Out-Null
Copy-Item -Recurse .\agent-skill\sciatlas-* "$env:USERPROFILE\.claude\skills\"
```

```bash
# macOS / Linux
mkdir -p ~/.claude/skills
cp -R ./agent-skill/sciatlas-* ~/.claude/skills/
```

You can also keep the same skill directories in a project-local `.claude/skills/` folder when the workflows should stay scoped to one repository. To install only one workflow, replace `sciatlas-*` with a specific folder such as `sciatlas-literature-review`.

### Codex

Codex uses the installed Codex environment and loads skills from `~/.codex/skills` on macOS/Linux or `%USERPROFILE%\.codex\skills` on Windows. Copy the packaged SciAtlas skill directories there, then start a new Codex session.

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.codex\skills" | Out-Null
Copy-Item -Recurse .\agent-skill\sciatlas-* "$env:USERPROFILE\.codex\skills\"
```

```bash
# macOS / Linux
mkdir -p ~/.codex/skills
cp -R ./agent-skill/sciatlas-* ~/.codex/skills/
```

The same skill directories can be installed in both tools. Each helper skill is self-contained: keep `SKILL.md` as the source of truth, keep API tokens and run artifacts out of `agent-skill/`, and let the agent use `SCIATLAS_API_KEY` plus `runs/<run_id>/` artifacts during the task. Other agent tools can adapt the same folders when their skill/plugin loader supports filesystem skills.

The skills are written for zero-start users. If the CLI or token is missing, the agent should install/configure what it can, guide the user through browser registration at `http://sciatlas.openkg.cn/register`, ask for email/code/token only when human feedback is required, and continue until a `search-papers` run produces artifacts.

## Design Notes

- Keep each skill small enough to load quickly.
- Keep operational defaults aligned with `search-papers` only. Do not make these skills depend on SciAtlas downstream CLI commands.
- Do not put API tokens or run artifacts in this folder.
- Use `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json` as the evidence trail after each workflow run.
