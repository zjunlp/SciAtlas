---
name: sciatlas-idea-evaluate
description: Use only the current SciAtlas automated review workflow (`review_pipeline`) to take a novice user from zero setup to a final automated review of a research idea or paper, including setup, registration guidance, workflow configuration, retrieval, artifact reading, and synthesis. Trigger when the user asks whether an idea is worth pursuing, wants automatic review, meta-review, rubric-based critique, novelty/feasibility/soundness assessment, or literature-backed evaluation.
---

# SciAtlas Idea Evaluate

Use this skill to run the repository automated review workflow. The workflow builds idea context, searches KG/S2 evidence, creates a manifest, grounds claims to paper paragraphs, builds rubric evidence, samples reviewer backgrounds, generates reviewer reports, and synthesizes a final report.

## Operating Contract

- Run only `sciatlas idea-evaluate` or `python run_sciatlas.py idea-evaluate` for this skill.
- Own the end-to-end novice flow: install or locate the CLI, guide registration, configure `.env` or shell variables, run the workflow, inspect artifacts, and synthesize the final review result.
- Ask the user only for human-only values: missing idea/PDF path, email, verification code, SciAtlas token, LLM/S2/KG credentials that are not already configured, or one necessary scope clarification.
- Do not ask the user to run shell commands when tool access is available.
- Use `--workflow flash` by default.
- Use `--workflow full` for a broader reviewer/rubric/evidence pass or when the user wants a more comprehensive review.
- Never disclose full API keys or tokens.
- Read saved artifacts before answering.

## Zero-Start Bootstrap

1. Check whether the repository command works:

```bash
python run_sciatlas.py idea-evaluate -h
```

If needed, fall back to `sciatlas idea-evaluate -h` after installing the full checkout.

2. This dedicated workflow requires a full SciAtlas checkout. If it is missing, clone the repository, change into it, then run `python -m pip install -e ./sciatlas` and `python -m pip install -r requirements-workflows.txt`. Do not use the GitHub `#subdirectory=sciatlas` package-only installation for this workflow.
3. Check current environment and `.env` for `SCIATLAS_API_KEY`, LLM settings, S2 settings, and KG settings before asking the user.
4. If no SciAtlas token is configured, guide the user to `http://sciatlas.openkg.cn/register`; ask for email, verification code, and returned `sciatlas_xxx` token only when needed.
5. If LLM/S2/KG credentials are required and missing, ask only for the missing values. Use the user's provider values without printing them back.
6. Configure the current shell or `.env` yourself, then run the workflow.

Configure workflow credentials in `.env` or the shell:

```bash
SCIATLAS_API_BASE_URL=http://sciatlas.openkg.cn
SCIATLAS_API_KEY=<sciatlas-token>
NEO4J_URI=<local-or-hosted-neo4j-uri>
NEO4J_USER=<neo4j-user>
NEO4J_PASSWORD=<neo4j-password>
S2_API_KEY=<semantic-scholar-key>
OPENAI_API_KEY=<llm-key>
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

The workflow also accepts `DMX-API-KEY`, `DMX_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_API_URL`, `SEARCH_LLM_API_KEY`, `SEARCH_LLM_API_URL`, and `SEARCH_LLM_MODEL`.

## Run Plan

Flash path for an idea:

```bash
python run_sciatlas.py idea-evaluate \
  --idea "<research idea>" \
  --workflow flash
```

Full path:

```bash
python run_sciatlas.py idea-evaluate \
  --idea "<research idea>" \
  --workflow full
```

For a paper PDF:

```bash
python run_sciatlas.py idea-evaluate \
  --pdf path/to/paper.pdf \
  --workflow full
```

Useful overrides:

- `--top-k N`, `--kg-top-k N`, `--s2-top-k N`, `--manifest-top-k N` tune evidence breadth.
- `--max-reviewers N` controls reviewer fan-out.
- `--grounding-final-top-k N` controls paragraph evidence depth.
- `--short-report` requests compact meta-review output in full mode.
- `--disable-review-llm` uses deterministic fallback when LLM review is unavailable.
- `--smoke` runs a stubbed pipeline for structural validation.

## Workflow Modes

`flash` compresses nonessential stages:

- smaller KG/S2/manifest budgets;
- one reviewer by default;
- compact evidence-card budgets;
- grounding refinement disabled;
- short meta-review enabled.

`full` keeps the comprehensive path:

- broader KG/S2/manifest budgets;
- reviewer fan-out and reviewer-background branch;
- rubric-source and rubric-LLM branches;
- paragraph grounding, reviewer reports, and full report synthesis.

## Artifacts To Read

Read the run directory:

- `summary.json`: wrapper status, workflow mode, subprocess command, and artifact pointers.
- `result.json`: review pipeline final payload and stage statuses.
- `report.md`: user-facing final report or compact meta-review.
- `idea_context/idea_context.json`: normalized idea/PDF context and structured idea extraction.
- `search/result.json`: KG/S2 search evidence.
- `manifest/manifest.json`: papers selected for paragraph extraction.
- `grounding/result.json`: paragraph grounding and experiment grounding.
- `rubric_sources/result.json`, `rubric_llm/result.json`: rubric evidence when available.
- `review/reviewer_reviews.index.json`: reviewer-level evaluations.
- `report/final_report.md`, `report/final_report.json`: native final outputs.
- `logs/*.log` and `logs/review_pipeline.*.txt`: stage logs when anything is partial or failed.

Treat `partial_error` as usable when the final report exists, but disclose which branch failed or was skipped.

## Deliverable

Return:

- exact command used, with credentials omitted;
- workflow mode and artifact paths;
- overall go/revise/no-go recommendation;
- novelty, feasibility, soundness, and differentiation assessment;
- closest-prior-art risk;
- strongest supporting evidence and biggest missing evidence;
- concrete revision advice and next SciAtlas query.

Keep judgments tied to retrieved papers, grounding snippets, reviewer reports, or rubric artifacts.
