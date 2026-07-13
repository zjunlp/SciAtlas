# SciAtlas Client

A lightweight pip-installable client and CLI for the hosted SciAtlas API service.

SciAtlas provides scientific knowledge-graph retrieval for paper search, related-author discovery, author-paper lookup, literature review, idea grounding/evaluation, idea generation, trend analysis, and researcher review.

The repository also includes a portable Agent Skill pack in `../agent-skill/`, which turns SciAtlas retrieval and the current literature-review, automated-review, and idea-generation workflows into zero-start downstream task playbooks for tools such as Codex, Claude Code, and other coding agents. The agent installs or locates the CLI, guides registration, configures environment variables, runs retrieval or the selected workflow, reads saved artifacts, and writes the final result. The user only supplies human-only values such as email, verification code, API token, LLM/S2/KG keys, or one necessary task clarification. Quick search, grounding, trend, and researcher-profile Skills are restricted to `search-papers`; the three dedicated Skills use their named workflows and start with `flash`, switching to `full` only when needed. The researcher profile is evidence-grounded rather than an authoritative CV. See [`../agent-skill/README.md`](../agent-skill/README.md) for the Git, Claude Code, and Codex setup commands.

<p align="center">
  <img src="../imgs/agent-skill-demo.gif" alt="SciAtlas Agent Skill workflow demo" width="92%">
</p>

Documentation: http://sciatlas.openkg.cn/api/docs/

## Installation

Recommended one-command download and install with `uv`:

Linux / macOS:

```bash
curl -LsSf https://raw.githubusercontent.com/zjunlp/SciAtlas/main/scripts/install-sciatlas-uv.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/zjunlp/SciAtlas/main/scripts/install-sciatlas-uv.ps1 | iex"
```

The installer downloads the full repository to `~/SciAtlas`, creates a `uv`
virtual environment, installs the SciAtlas CLI and workflow dependencies, and
also exposes an editable `sciatlas` command through `uv tool install`.

Package-only alternatives support the core CLI only. They do not include the
repository-level `literature-review`, `idea-evaluate`, or `idea-generate`
workflows.

Install directly from GitHub:

```bash
pip install "git+https://github.com/zjunlp/SciAtlas.git#subdirectory=sciatlas"
```

For isolated CLI usage:

```bash
pipx install "git+https://github.com/zjunlp/SciAtlas.git#subdirectory=sciatlas"
```

After installation:

```bash
sciatlas -h
```

## Get an API Token

Open:

```text
http://sciatlas.openkg.cn/register
```

Complete email verification and copy your personal token.

Then configure:

```bash
export SCIATLAS_API_BASE_URL="http://sciatlas.openkg.cn"
export SCIATLAS_API_KEY="your-personal-sciatlas-token"
```

You can also create a local `.env` from `.env.example`, although the CLI reads environment variables directly.

## Quick Start

```bash
sciatlas health
sciatlas config
```

Search papers:

```bash
sciatlas --timeout 900 search-papers \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --top-k 3 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0 \
  --report-max-items 3
```

For repository workflows, use a full checkout and install their dependencies:

```bash
git clone https://github.com/zjunlp/SciAtlas.git
cd SciAtlas
python -m pip install -e ./sciatlas
python -m pip install -r requirements-workflows.txt
```

Literature review:

```bash
sciatlas --timeout 900 literature-review \
  --workflow flash \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 5
```

Idea evaluation:

```bash
sciatlas --timeout 900 idea-evaluate \
  --workflow flash \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --keyword "high:idea evaluation" \
  --top-k 3
```

Researcher review:

```bash
sciatlas --timeout 900 researcher-review \
  --author "Yoshua Bengio" \
  --limit 10 \
  --no-abstract
```

## Python SDK

```python
from sciatlas import SciAtlasClient

client = SciAtlasClient()
print(client.health())

result = client.search_papers(
    query="open world agent",
    keywords=[{"text": "open world agent", "score": 10}],
    top_k=3,
)

print(result)
```

## Commands

| Command | Purpose |
|---|---|
| `health` | Check backend health |
| `config` | Show configuration |
| `build-plan` | Build a structured plan without calling backend |
| `search-papers` | Search related papers |
| `related-authors` | Retrieve related authors |
| `author-papers` | Query papers by author |
| `support-papers` | Retrieve support papers |
| `paper-search` | Lightweight low-level paper search |
| `literature-review` | Run the current literature-review workflow with compact `--workflow flash` or comprehensive `--workflow full` |
| `idea-grounding` | Ground a research idea against literature |
| `idea-evaluate` | Run the current automated review workflow with compact `--workflow flash` or comprehensive `--workflow full` |
| `idea-generate` | Run the current multi-step idea-generation workflow with compressed `--workflow flash` or comprehensive `--workflow full` |
| `trend-report` | Research trend analysis |
| `researcher-review` | Researcher background review |
| `make-report` | Regenerate Markdown report from saved artifacts |

## Workflow Modes

`literature-review`, `idea-evaluate`, and `idea-generate` support `--workflow flash|full`. `flash` is the default for interactive use; `full` keeps the comprehensive path.

| Command | `flash` | `full` |
|---|---|---|
| `literature-review` | Compact search and evidence-pack path. | Broader multi-round retrieval plus formal review drafting/integration. |
| `idea-evaluate` | Smaller reviewer, rubric, manifest, and evidence budgets. | Fuller reviewer/rubric/grounding/review/report path. |
| `idea-generate` | Compact graph and compressed gate/selection stages. | Broader graph, inspiration search, and novelty feedback. |

## Outputs

Each run saves artifacts under:

```text
runs/<run_id>/
  plan.json
  request.json
  response.json
  summary.txt
  report.md
  metadata.json
```

## Development

Install editable mode:

```bash
pip install -e .
sciatlas -h
```

Build package:

```bash
python -m pip install build twine
python -m build
twine check dist/*
```

## Security

Do not commit `.env`, API tokens, SMTP credentials, `.cache/`, or `runs/`.

## License

MIT.

<!-- SCIATLAS_FRONTEND_OPTIONAL_LLM_OPENALEX_START -->
## Frontend LLM and OpenAlex Configuration

Optional LLM settings are only for better keyword extraction before retrieval. They are not required for normal KG search.

| Variable | Required | Purpose |
|---|---|---|
| `SCIATLAS_API_BASE_URL` | yes | Hosted SciAtlas API base URL. |
| `SCIATLAS_API_KEY` | yes | SciAtlas token. |
| `LLM_PROVIDER` | optional | Keep as `chat_completions`. |
| `LLM_API_KEY` | optional | Your provider key; leave empty for local or no-auth services. |
| `LLM_BASE_URL` | optional | Provider base URL, usually ending in `/v1`. |
| `LLM_CHAT_COMPLETIONS_URL` | optional | Use only when your provider gives a full endpoint. |
| `LLM_MODEL` | optional | Model name from your provider. |
| `LLM_AUTH_HEADER` | optional | Use only for custom auth, such as `x-api-key: your-provider-api-key`. |
| `LLM_HTTP_HEADERS` | optional | Optional extra headers as JSON. |
| `GROBID_BASE_URL` | PDF tasks only | Required for `--pdf-path` workflows. |
| `OA_API_KEY` | optional | OpenAlex fallback/enrichment support. |
| `OPENALEX_MAILTO` | optional | OpenAlex contact email. |

If LLM variables are empty or the LLM call fails, SciAtlas falls back to built-in keyword extraction. If OpenAlex variables are empty, OpenAlex enrichment is skipped and normal KG retrieval still works.

User-editable config template: [.env.example](.env.example#L1-L52).
<!-- SCIATLAS_FRONTEND_OPTIONAL_LLM_OPENALEX_END -->
