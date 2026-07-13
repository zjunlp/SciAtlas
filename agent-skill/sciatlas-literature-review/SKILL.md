---
name: sciatlas-literature-review
description: Use only the current SciAtlas literature-review workflow (`literature_review_pipeline`) to take a novice user from zero setup to a final evidence-grounded survey outline, paper map, or literature review, including setup, registration guidance, workflow configuration, SciAtlas backend retrieval, artifact reading, and synthesis. Trigger when the user asks for a literature review, related work section, paper map, survey outline, reading path, or topic overview grounded in the provided SciAtlas paper backend.
---

# SciAtlas Literature Review

Use this skill to run the repository literature-review workflow. The workflow performs topic profiling, SciAtlas backend paper search, evidence organization, method clustering, time slicing, outline planning, evidence-pack construction, and optionally full section drafting/integration.

## Operating Contract

- Run only `sciatlas literature-review` or `python run_sciatlas.py literature-review` for this skill.
- Own the end-to-end novice flow: install or locate the CLI, guide registration, configure `.env` or shell variables, run the workflow, inspect artifacts, and synthesize the final review result.
- Ask the user only for human-only values: missing topic/domain, email, verification code, SciAtlas token, LLM credentials that are not already configured, or one necessary scope clarification.
- Do not ask the user to run shell commands when tool access is available.
- Use `--workflow flash` by default for interactive work.
- Use `--workflow full` when the user requests a comprehensive formal review or when flash artifacts are too thin.
- Never disclose full API keys or tokens.
- Read saved artifacts before answering.

## Zero-Start Bootstrap

1. Check whether the repository command works:

```bash
python run_sciatlas.py literature-review -h
```

If needed, fall back to `sciatlas literature-review -h` after installing the full checkout.

2. This dedicated workflow requires a full SciAtlas checkout. If it is missing, clone the repository, change into it, then run `python -m pip install -e ./sciatlas` and `python -m pip install -r requirements-workflows.txt`. Do not use the GitHub `#subdirectory=sciatlas` package-only installation for this workflow.
3. Check current environment and `.env` for `SCIATLAS_API_KEY` and LLM settings before asking the user.
4. If no SciAtlas token is configured, guide the user to `http://sciatlas.openkg.cn/register`; ask for email, verification code, and returned `sciatlas_xxx` token only when needed.
5. If LLM credentials are required and missing, ask only for the missing values. Use the user's provider values without printing them back.
6. Configure the current shell or `.env` yourself, then run the workflow.

Paper retrieval for this workflow must use only the provided SciAtlas backend through `run_sciatlas.py search-papers`. Do not add public-paper fallback retrieval (Semantic Scholar, OpenAlex, Crossref, arXiv scraping, or browser search) when the SciAtlas backend returns no papers or errors. The local literature-review code may still use LLM calls for planning, clustering, and synthesis.

Configure workflow credentials in `.env` or the shell:

```bash
SCIATLAS_API_BASE_URL=http://sciatlas.openkg.cn
SCIATLAS_API_KEY=<sciatlas-token>
OPENAI_API_KEY=<llm-key>
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

The workflow also accepts `DMX-API-KEY`, `DMX_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_API_URL`, `SEARCH_LLM_API_KEY`, `SEARCH_LLM_API_URL`, and `SEARCH_LLM_MODEL`.

## Run Plan

Flash path:

```bash
python run_sciatlas.py literature-review \
  --query "<topic>" \
  --domain "<optional field>" \
  --workflow flash
```

Full path:

```bash
python run_sciatlas.py literature-review \
  --query "<topic>" \
  --domain "<optional field>" \
  --workflow full
```

Useful overrides:

- `--top-k N` changes the default SciAtlas search budget.
- `--probe-top-k N` and `--round-top-k N` tune search breadth.
- `--round1-action-limit N` and `--round2-action-limit N` tune query planning.
- `--report-stop-after outline|packs|full` controls report generation depth.
- `--subject-domain general|chemistry|biology` selects prompt constraints.
- `--smoke` runs the mock search path for structure validation.

Smooth flash defaults when the backend is slow or unstable:

```bash
python run_sciatlas.py literature-review \
  --query "<topic>" \
  --domain "<optional field>" \
  --workflow flash \
  --probe-top-k 3 \
  --round-top-k 3 \
  --round1-action-limit 1 \
  --llm-paper-limit 12 \
  --report-stop-after packs \
  --workflow-timeout 600 \
  --llm-timeout 120 \
  --outline-timeout 180
```

Set `SCIATLAS_SEARCH_TIMEOUT` for the per-action hosted `search-papers` timeout. If SciAtlas returns `5xx`, keep the error in artifacts and do not switch to public retrieval.

## Workflow Modes

`flash` compresses nonessential stages:

- smaller probe and round search budgets;
- fewer Round 1 actions and no Round 2 by default;
- query cleaning and relevance guard enabled;
- report generation stops after evidence packs, producing a fast outline/evidence report.

`full` runs the broader path:

- larger probe and round search budgets;
- Round 2 refinement enabled;
- KG policy, query cleaning, and relevance guard enabled;
- full formal review drafting and integration.

## Artifacts To Read

Read the run directory:

- `summary.json`: status, workflow mode, subprocess logs, and artifact pointers.
- `report.md`: user-facing review, outline/evidence-pack summary, or full formal review.
- `lr_search/search_result.json`: topic profile, time windows, paper cards, clusters, coverage report, and search actions.
- `lr_search/organized_search_result.json`: deterministic evidence map when generated.
- `lr_review/formal_outline.json`: planned review structure.
- `lr_review/citation_plan.json`, `section_packs/`, `subsection_packs/`: flash evidence packs.
- `lr_review/formal_review.md` and `diagnostic_report.md`: full-mode final outputs.
- `logs/*.txt`: subprocess stdout/stderr when a stage fails.

In `flash`, a report that stops after `packs` is normal success. Do not rerun full unless the user asks or the evidence is insufficient for their requested depth.

## Deliverable

Return:

- exact command used, with credentials omitted;
- workflow mode and artifact paths;
- concise topic scope;
- 5-12 representative papers or paper groups;
- method/theme clusters and timeline notes;
- review outline or full review summary;
- gaps, caveats, and next SciAtlas queries.

Keep paper claims tied to artifact titles, clusters, and evidence fields.
