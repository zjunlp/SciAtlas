---
name: sciatlas-idea-generate
description: Use only the current SciAtlas multi-step idea-generation workflow (`sciatlas_idea_gen`) to take a novice user from zero setup to final literature-grounded research idea seeds, including setup, registration guidance, workflow configuration, KG retrieval, artifact reading, novelty checking, and synthesis. Trigger when the user wants new research directions, hypotheses, cross-topic combinations, project ideas, or brainstorming grounded in SciAtlas KG retrieval.
---

# SciAtlas Idea Generate

Use this skill to generate research ideas from SciAtlas KG evidence. Run only the repository workflow `python -m sciatlas_idea_gen.main`, which retrieves anchor papers, builds a research graph, searches same-field and cross-domain inspirations, generates ideas, and runs novelty checks.

## Operating Contract

- Own the end-to-end novice flow: install or locate the workflow, guide registration, configure `.env` or shell variables, run retrieval/generation, inspect artifacts, and synthesize final idea seeds.
- Ask the user only for human-only values: missing topic/domain, email, verification code, SciAtlas token, LLM key/base/model, S2 key, local KG credentials when explicitly using local KG, or one necessary scope clarification.
- Do not ask the user to run shell commands when tool access is available.
- Never disclose the full API token.
- Use `sciatlas_idea_gen` as the only idea-generation workflow.
- If `sciatlas_idea_gen` is unavailable or required LLM/KG setup is missing, report the blocker and fix setup if possible; do not substitute another route.
- Save and inspect artifacts before writing the final answer.

## KG Linkage

The workflow reaches the KG through `sciatlas_idea_gen.clients.SciAtlasClient`:

- Hosted mode: `SCIATLAS_USE_LOCAL_KG=0`; the adapter shells into this repository's `run_sciatlas.py` for KG-backed paper retrieval, then normalizes `data.result.ranking.papers` or `data.result.sources.kg.papers` to `papers`.
- Local KG mode: `SCIATLAS_USE_LOCAL_KG=1`; the adapter runs `references/search/run_search.py` with `--disable-s2`, `--disable-llm-ranking`, `--kg-top-k`, and `--final-top-k`, then normalizes the local KG response to `papers`.
- Keep hosted mode as the default unless the user says they have Neo4j and local KG models configured.

## Zero-Start Bootstrap

1. Check for the repository workflow:

```bash
python -m sciatlas_idea_gen.main -h
```

2. If missing, clone the full SciAtlas repository, change into it, then run `python -m pip install -e ./sciatlas` and `python -m pip install -r requirements-workflows.txt`. Do not use the GitHub `#subdirectory=sciatlas` package-only installation: it does not include `sciatlas_idea_gen`.
3. Check current environment and `.env` for hosted KG and LLM settings before asking the user.
4. If `SCIATLAS_API_KEY` is missing for hosted mode, guide the user to `http://sciatlas.openkg.cn/register`; ask for email, verification code, and returned `sciatlas_xxx` token only when needed.
5. Configure hosted KG yourself:

```powershell
$env:SCIATLAS_API_BASE_URL = "http://sciatlas.openkg.cn"
$env:SCIATLAS_API_KEY = "<token>"
$env:SCIATLAS_USE_LOCAL_KG = "0"
```

```bash
export SCIATLAS_API_BASE_URL="http://sciatlas.openkg.cn"
export SCIATLAS_API_KEY="<token>"
export SCIATLAS_USE_LOCAL_KG=0
```

6. Configure the LLM endpoint required by the multi-step workflow:

```bash
export LLM_API_KEY="<provider-key>"
export LLM_BASE_URL="https://your-provider.example/v1"
export LLM_MODEL="<model-name>"
```

Ask only for missing LLM provider values, then write them to the current shell or `.env` without echoing secrets.

7. Configure local KG only when requested:

```bash
export SCIATLAS_USE_LOCAL_KG=1
export SCIATLAS_LOCAL_SEARCH_ROOT="./references/search"
```

The local KG engine also needs its own `references/search/.env` with Neo4j/model settings.

## Run Plan

Use the flash path first for quick, interactive idea generation:

```bash
python -m sciatlas_idea_gen.main "<research topic>" --workflow flash --domain "<optional field>"
```

Run the full path when the user wants a broader research graph, more seed diversity, and a more comprehensive inspiration search:

```bash
python -m sciatlas_idea_gen.main "<research topic>" --workflow full --domain "<optional field>"
```

For a seed PDF:

```bash
python -m sciatlas_idea_gen.main "<research topic>" --workflow full --pdf path/to/paper.pdf
```

Useful overrides:

- `--workflow flash` for a faster path: 1 seed query, a compact graph, one same-field inspiration, one cross-domain inspiration, compressed Step 5/Step 8 gate stages, and no novelty feedback retry.
- `--workflow full` for the broader path: multi-query seed retrieval, larger graph construction, broader inspiration retrieval, and novelty feedback retry.
- `--k-step1 N` for more seed papers.
- `--anchor-top-k N` for broader Step 1 retrieval.
- `--graph-budget-min N` and `--graph-budget-max N` for graph size.
- `--seed-recent-years N` to include older foundational papers.
- `--resume-from-run-dir runs/<id>` to continue from saved artifacts.

## Artifacts To Read

Read the run directory before answering:

- `summary.json`: status, effective config, counts, and final ideas.
- `retrieval_trace.json`: KG/S2/OpenAlex retrieval events and returned papers.
- `step1_seed_papers.json`: anchor papers and refined query.
- `step2_research_graph.json`: graph nodes, edges, evidence scores, and phases.
- `step3_trend.txt`: trend summary from the graph.
- `step5_radius_plan.json`: inspiration radius plan. In `flash`, this is a deterministic compact plan and `summary.json` may mark Step 5 as `compressed`.
- `step6_inspiration_candidates.json`, `step7_inspirations.json`, `step8_selected_inspirations.json`: inspiration path.
- `step9_ideas.json` and `step9_ideas.md`: final ideas and citations.

In `flash`, treat `status: "compressed"` for Step 5 or Step 8 as normal success. Step 5 skips the LLM radius gate and uses the compact same-field/cross-domain plan. Step 8 skips the LLM selector and keeps the compact plausible inspiration set. Do not rerun full merely because these stages are compressed.

If the run is in smoke mode and fails partway, use the completed artifacts and clearly state the failed step. Smoke mode is now diagnostic; prefer `--workflow flash` for normal quick runs.

## Deliverable

Return 3-8 idea seeds when available. For each idea:

- Name.
- One-sentence hypothesis.
- KG evidence trail from seed papers, graph papers, and inspirations.
- Why it is nontrivial.
- Closest-prior-art or novelty risk.
- Minimal validation experiment.
- Next SciAtlas query or pipeline setting to test novelty.

Keep speculative claims clearly labeled and cite artifacts/paper titles from the run.
