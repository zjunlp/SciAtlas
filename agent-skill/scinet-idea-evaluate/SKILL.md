---
name: scinet-idea-evaluate
description: Run SciNet's idea-evaluate workflow to collect evidence about a research idea's novelty, feasibility, soundness, and differentiation. Use when the user asks whether an idea is worth pursuing, wants prior-art checks, feasibility review, research-risk assessment, or literature-grounded idea evaluation.
---

# SciNet Idea Evaluate

Use this skill to evaluate a proposed research idea with SciNet's knowledge-graph retrieval rather than relying on unsupported judgment.

## Workflow

1. Capture the research proposal as `--idea`. Ask for a clearer idea only if the request lacks a concrete method, problem, or hypothesis.
2. Add `--domain`, `--time-range`, and `--keyword` anchors from the user's context.
3. Check SciNet access with `scinet health` if needed. If `scinet` is not on PATH inside this repository, try `./scinet/.venv/Scripts/scinet.exe` from the repository root or `./.venv/Scripts/scinet.exe` from `scinet/`.
4. Run the preset:

```bash
scinet skill run idea-evaluate --idea "Federated knowledge editing for large language models" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:knowledge editing" --keyword "middle:federated learning"
```

5. Inspect `runs/<run_id>/summary.txt`, `report.md`, `request.json`, and `response.json`.
6. Base the evaluation on returned evidence. Separate supported claims from open questions.

## Preset

- SciNet skill: `idea-evaluate`
- Aliases: `idea-eval`, `eval-idea`
- Underlying command: `idea-evaluate`
- Defaults: `retrieval_mode=hybrid`, `top_k=5`, `top_keywords=0`, `max_titles=0`, `max_refs=0`, `bias_keyword=high`, `bias_related=high`, `bias_citation=low`, `bias_exploration=low`, `ranking_profile=precision`, `report_max_items=5`

## Evaluation Frame

Cover these points when the artifacts contain enough evidence:

- Novelty: which prior work is closest and where the proposed idea differs.
- Feasibility: whether related methods, datasets, or systems suggest the idea can be built.
- Soundness: likely assumptions, missing evidence, and methodological risks.
- Positioning: how to phrase the contribution conservatively.

## Output Style

Return a decision-oriented assessment with evidence-backed strengths, risks, and next experiments or searches. Avoid a generic "promising idea" conclusion unless SciNet evidence supports it.
