# SciAtlas Skills

SciAtlas skills are editable JSON presets for downstream research workflows.

```bash
sciatlas skill list
sciatlas skill show literature-review
sciatlas skill run literature-review --query "open world agent" --keyword "high:open world agent"
sciatlas skill run literature-review-full --query "open world agent" --domain "artificial intelligence"
sciatlas skill run idea-evaluate --idea "LLM-based idea evaluation"
sciatlas skill run idea-evaluate-full --idea "LLM-based idea evaluation"
sciatlas skill init my-review --from literature-review
```

User-defined skills are loaded from:

1. `./skills/*.json`
2. `~/.sciatlas/skills/*.json`
3. paths in `SCIATLAS_SKILLS_DIR`

User skills override builtin skills with the same name.

## Flash vs Full

`literature-review`, `idea-evaluate`, and `idea-generate` expose two preset families:

| Preset | Mode | Use when |
|---|---|---|
| `literature-review`, `idea-evaluate`, `idea-generate` | `flash` | You want a fast interactive run, smoke test, or first-pass evidence. |
| `literature-review-full`, `idea-evaluate-full`, `idea-generate-full` | `full` | You need broader evidence, reviewer/rubric branches, formal review drafting, or a more complete idea-generation pass. |

The portable Agent Skill pack is packaged separately in `../agent-skill/`. Those folders are repository assets for tools such as Codex, Claude Code, and other coding agents. The intended Agent Skill contract is zero-start and novice-friendly: the agent installs or locates SciAtlas, guides browser registration, asks only for human-only values such as email, verification code, SciAtlas token, LLM/S2/KG credentials, or one task clarification, configures the environment, runs retrieval or the selected current workflow, reads `runs/<run_id>/`, and returns the final downstream result. Quick search, grounding, trend, and researcher-profile Skills may retrieve only through `search-papers`; literature review, idea evaluation, and idea generation use only their dedicated workflows. Agent Skills start dedicated workflows with `flash` and use `--workflow full` only for a deeper request or thin flash artifacts. The `*-full` names above are CLI JSON presets, not Agent Skill directory names. See [`../agent-skill/README.md`](../agent-skill/README.md) for setup commands. This CLI loader only reads JSON presets from the locations above.

Dedicated workflow presets require a full SciAtlas checkout plus
`python -m pip install -r requirements-workflows.txt`; a package-only CLI
installation supports only the core retrieval commands.
