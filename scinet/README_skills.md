# SciNet Skills

SciNet skills are editable JSON presets for downstream research workflows.

```bash
scinet skill list
scinet skill show literature-review
scinet skill run literature-review --query "open world agent" --keyword "high:open world agent"
scinet skill init my-review --from literature-review
```

User-defined skills are loaded from:

1. `./skills/*.json`
2. `~/.scinet/skills/*.json`
3. paths in `SCINET_SKILLS_DIR`

User skills override builtin skills with the same name.

The portable Agent Skill pack is packaged separately in `../agent-skill/`. Those folders are repository assets for tools such as Codex, Claude Code, and other coding agents; this CLI loader only reads JSON presets from the locations above.
