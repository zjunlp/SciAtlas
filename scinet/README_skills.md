# SciScholar Skills

SciScholar skills are editable JSON presets for downstream research workflows.

```bash
scischolar skill list
scischolar skill show literature-review
scischolar skill run literature-review --query "open world agent" --keyword "high:open world agent"
scischolar skill init my-review --from literature-review
```

User-defined skills are loaded from:

1. `./skills/*.json`
2. `~/.scischolar/skills/*.json`
3. paths in `SCISCHOLAR_SKILLS_DIR`

User skills override builtin skills with the same name.

The portable Agent Skill pack is packaged separately in `../agent-skill/`. Those folders are repository assets for tools such as Codex, Claude Code, and other coding agents; they turn `search-papers` retrieval into end-to-end downstream task playbooks. This CLI loader only reads JSON presets from the locations above.
