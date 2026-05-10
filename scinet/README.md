# SciNet Client

A lightweight pip-installable client and CLI for the hosted SciNet / KG2API service.

SciNet provides scientific knowledge-graph retrieval for paper search, related-author discovery, author-paper lookup, literature review, idea grounding/evaluation, idea generation, trend analysis, and researcher review.

The repository also includes a portable Agent Skill pack in `../agent-skill/`, which wraps these workflows for tools such as Codex, Claude Code, and other coding agents.

Documentation: http://scinet.openkg.cn/api/docs/

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/zjunlp/SciNet.git
```

For isolated CLI usage:

```bash
pipx install git+https://github.com/zjunlp/SciNet.git
```

After installation:

```bash
scinet -h
```

## Get an API Token

Open:

```text
http://scinet.openkg.cn/register
```

Complete email verification and copy your personal token.

Then configure:

```bash
export SCINET_API_BASE_URL="http://scinet.openkg.cn"
export SCINET_API_KEY="your-personal-scinet-token"
```

You can also create a local `.env` from `.env.example`, although the CLI reads environment variables directly.

## Quick Start

```bash
scinet health
scinet config
```

Search papers:

```bash
scinet --timeout 900 search-papers \
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

Literature review:

```bash
scinet --timeout 900 literature-review \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 5
```

Idea evaluation:

```bash
scinet --timeout 900 idea-evaluate \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --keyword "high:idea evaluation" \
  --top-k 3
```

Researcher review:

```bash
scinet --timeout 900 researcher-review \
  --author "Yoshua Bengio" \
  --limit 10 \
  --no-abstract
```

## Python SDK

```python
from scinet import SciNetClient

client = SciNetClient()
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
| `literature-review` | Review-oriented paper retrieval |
| `idea-grounding` | Ground a research idea against literature |
| `idea-evaluate` | Collect evidence for idea evaluation |
| `idea-generate` | Discover idea seeds |
| `trend-report` | Research trend analysis |
| `researcher-review` | Researcher background review |
| `make-report` | Regenerate Markdown report from saved artifacts |

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
scinet -h
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

<!-- SCINET_FRONTEND_OPTIONAL_LLM_OPENALEX_START -->
## Frontend LLM and OpenAlex Configuration

Optional LLM settings are only for better keyword extraction before retrieval. They are not required for normal KG search.

| Variable | Required | Purpose |
|---|---|---|
| `SCINET_API_BASE_URL` | yes | Hosted SciNet API base URL. |
| `SCINET_API_KEY` | yes | SciNet token. |
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

If LLM variables are empty or the LLM call fails, SciNet falls back to built-in keyword extraction. If OpenAlex variables are empty, OpenAlex enrichment is skipped and normal KG retrieval still works.

User-editable config template: [.env.example](.env.example#L7-L26).
<!-- SCINET_FRONTEND_OPTIONAL_LLM_OPENALEX_END -->
