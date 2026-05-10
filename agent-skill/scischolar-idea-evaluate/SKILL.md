---
name: scischolar-idea-evaluate
description: Use SciScholar search-papers as the retrieval base to evaluate a research idea's novelty, feasibility, soundness, and differentiation. Trigger when the user asks whether an idea is worth pursuing, wants literature-backed critique, or needs an evidence-based go/no-go assessment.
---

# SciScholar Idea Evaluate

This skill migrates `search-papers` into a research-idea evaluation workflow. The agent must retrieve evidence first, then judge the idea against that evidence.

## End-to-End Workflow

1. Extract the idea, claimed contribution, target problem, and likely baselines.
2. Build retrieval anchors:
   - Full proposal -> `--idea`
   - Main technical claim -> `--keyword "high:<term>"`
   - Baseline family or domain -> `--keyword "middle:<term>"`
   - Known papers -> `--title` or `--reference`
3. Run the idea-evaluate channel:

```bash
scischolar idea-evaluate --idea "Federated and privacy-preserving knowledge editing for large language models" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:knowledge editing" --keyword "middle:federated learning" --top-k 5
```

4. If the downstream command is unavailable, use base `search-papers`:

```bash
scischolar search-papers --retrieval-mode hybrid --query "Federated and privacy-preserving knowledge editing for large language models" --keyword "high:knowledge editing" --keyword "middle:federated learning" --top-k 5 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-keyword high --bias-related high --bias-citation low --bias-exploration low --ranking-profile precision --report-max-items 5
```

5. Read the generated artifacts in `runs/<run_id>/`.
6. Complete the evaluation using the retrieved evidence.

## Evaluation Rubric

Assess:

- Novelty: closest work and remaining differentiators.
- Feasibility: whether methods, datasets, or evaluations in related work make the idea buildable.
- Soundness: assumptions, confounds, and evaluation risks.
- Impact: what problem the idea would actually solve if successful.
- Next validation: one small experiment or follow-up retrieval.

## Deliverable

Return a decision-oriented review with a verdict such as `pursue`, `revise`, or `weak as stated`, plus evidence-backed reasons. Keep the tone conservative when retrieval evidence is limited.
