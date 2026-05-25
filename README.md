<div align="center">
  <h1>SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research</h1>
</div>

<p align="center">
  🌐 <strong>English</strong> · <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="http://scinet.openkg.cn/api/docs/">📚 SciAtlas Documentation</a>
</p>

<p align="center">
  A pip-installable client and CLI for literature-grounded scientific research workflows on top of the hosted SciAtlas API.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22878">📄 arXiv</a>
  ·
  <a href="http://scinet.openkg.cn/register">🔑 Get API Token</a>
  ·
  <a href="http://scinet.openkg.cn/healthz">🩺 API Health</a>
</p>

<p align="center">
  <a href="https://github.com/zjunlp/SciAtlas">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://github.com/zjunlp/SciAtlas/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/github/last-commit/zjunlp/SciAtlas?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
</p>

---

## ✨ Overview

SciAtlas is a research map you can use from the command line. Give it a topic, an idea, an author, or a paper trail, and it helps you look up literature, gather graph-backed evidence, and turn the result into readable reports and reusable JSON artifacts.

Behind that simple workflow is a large scientific knowledge graph. SciAtlas connects papers, authors, institutions, venues, keywords, citations, and a four-level research taxonomy from domains down to topics. That means a search is not limited to matching words: it can follow how research areas, people, concepts, and papers relate to one another.

This repository packages that capability as a lightweight **SciAtlas client**. New users can install it with `pip`, register an API token, and start running literature-grounded research tasks without setting up Neo4j, maintaining graph data, or touching backend infrastructure.

<p align="center">
  <img src="imgs/field_distribution_pie.png" alt="SciAtlas field distribution across research areas" width="92%">
</p>

<p align="center">
  <em>SciAtlas spans a broad research landscape, from medicine and social sciences to engineering, computer science, materials science, mathematics, and more.</em>
</p>

<p align="center">
  <img src="imgs/schema.png" alt="SciAtlas knowledge graph schema" width="92%">
</p>

<p align="center">
  <em>The graph links papers with authors, institutions, sources, keywords, citations, related work, and the domain-field-subfield-topic hierarchy.</em>
</p>

With the client, SciAtlas becomes a practical research assistant for:

- **graph-aware paper search**: combine keywords, semantic matching, title anchors, references, and graph propagation instead of stopping at plain keyword matching;
- **research workflow automation**: run literature review, idea grounding, idea evaluation, idea generation, trend analysis, related-author retrieval, and researcher profiling;
- **agent-friendly outputs**: keep reproducible machine-readable artifacts such as `request.json` and `response.json`, plus user-facing `summary.txt` and `report.md`;
- **editable CLI skills**: inspect, copy, modify, and rerun common downstream workflows as reusable JSON skills;
- **portable Agent Skill pack**: use [`agent-skill/`](agent-skill/) to migrate the base `search-papers` capability into end-to-end downstream tasks for tools such as Codex, Claude Code, and other coding agents.

## 📑 Table of Contents

- [✨ Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🔑 API Token](#-api-token)
- [🧩 Supported Tasks](#-supported-tasks)
- [🛠️ CLI-First Workflow](#️-cli-first-workflow)
- [🧰 Editable Skills](#-editable-skills)
- [🖊Agent Skill](#agent-skill)
- [🐍 Python SDK](#-python-sdk)
- [📦 Outputs and Artifacts](#-outputs-and-artifacts)
- [📂 Repository Layout](#-repository-layout)
- [🧯 Troubleshooting](#-troubleshooting)
- [📝 TODO](#-todo)
- [✍️ Citation](#️-citation)
- [📄 License](#-license)

## 🚀 Quick Start

### 1. Install

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

### 2. Register an API Token

Open:

```text
http://scinet.openkg.cn/register
```

Complete email verification and copy your personal token.

Quick link: [🔑 API Token](#-api-token).

### 3. Configure

At minimum, configure the hosted SciAtlas API endpoint and your personal token.

Linux / macOS:

```bash
export SCIATLAS_API_BASE_URL="http://scinet.openkg.cn"
export SCIATLAS_API_KEY="your-personal-sciatlas-token"
export SCIATLAS_TIMEOUT=900
export SCIATLAS_RUNS_DIR="./runs"
```

Windows CMD:

```bat
set SCIATLAS_API_BASE_URL=http://scinet.openkg.cn
set SCIATLAS_API_KEY=your-personal-sciatlas-token
set SCIATLAS_TIMEOUT=900
set SCIATLAS_RUNS_DIR=.\runs
```

📕 Optional: use your own LLM for keyword extraction

```bash
export LLM_PROVIDER="chat_completions"
export LLM_API_KEY="your-provider-api-key"
export LLM_BASE_URL="https://your-provider-or-gateway.example/v1"
export LLM_MODEL="your-model-name"
# Optional when your provider uses a custom endpoint or auth header:
# export LLM_CHAT_COMPLETIONS_URL="https://your-provider-or-gateway.example/v1/chat/completions"
# export LLM_AUTH_HEADER="x-api-key: your-provider-api-key"
export SCIATLAS_LLM_TIMEOUT=30
export SCIATLAS_LLM_TEMPERATURE=0
export SCIATLAS_LLM_MAX_TOKENS=512
```

This step is optional. Configure it only when you want SciAtlas to use your LLM API to turn a free-form query into better search keywords.

Keep `LLM_PROVIDER=chat_completions`, then replace `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` with your provider values. If your provider gives a full chat-completions endpoint, set `LLM_CHAT_COMPLETIONS_URL`; if it requires a custom auth header, set `LLM_AUTH_HEADER`.

Leave the LLM values empty if you do not need this. SciAtlas will use built-in keyword extraction, and normal search, review, idea, trend, and researcher workflows still run.

User-editable template: [.env.example](.env.example#L7-L19). Set these variables only if you want LLM-assisted keyword extraction.

🖊 Optional: OpenAlex metadata support

```bash
export OA_API_KEY=""
export OPENALEX_MAILTO=""
```

OpenAlex is useful when you want extra metadata or PDF-related support. It is not required for the main CLI examples in this README. If you leave these variables empty, normal SciAtlas retrieval still works.

User-editable template: [.env.example](.env.example#L24-L26). Set these only if you want OpenAlex-assisted metadata support.

🖌 Optional: GROBID for local PDF workflows

GROBID is only needed when you process local PDF files. It reads scientific PDFs and extracts titles, authors, abstracts, and references. If you are only running the text-based CLI commands above, you can skip this section.

Start GROBID locally:

```bash
docker pull lfoppiano/grobid:latest
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:latest
curl http://127.0.0.1:8070/api/isalive
```

Then set:

```bash
export GROBID_BASE_URL="http://127.0.0.1:8070"
```

Windows CMD:

```bat
set GROBID_BASE_URL=http://127.0.0.1:8070
```

User-editable template: [.env.example](.env.example#L21-L22). Leave `GROBID_BASE_URL` empty unless you process local PDFs.

Runtime variables:

| Variable | Required For | Notes |
|---|---|---|
| `SCIATLAS_API_BASE_URL` | all hosted SciAtlas tasks | Hosted SciAtlas API base URL. |
| `SCIATLAS_API_KEY` | all hosted SciAtlas tasks | Sent as `X-API-Key` and `Authorization: Bearer`. |
| `LLM_PROVIDER` | optional frontend enhancement | Keep as `chat_completions`. |
| `LLM_API_KEY` | optional frontend enhancement | Your provider key; leave empty for local or no-auth services. |
| `LLM_BASE_URL` | optional frontend enhancement | Provider base URL, usually ending in `/v1`. |
| `LLM_CHAT_COMPLETIONS_URL` | optional frontend enhancement | Use only when your provider gives a full endpoint. |
| `LLM_MODEL` | optional frontend enhancement | Model name from your provider. |
| `LLM_AUTH_HEADER` | optional frontend enhancement | Use only for custom auth, for example `x-api-key: your-provider-api-key`. |
| `LLM_HTTP_HEADERS` | optional frontend enhancement | Optional extra headers as JSON. |
| `GROBID_BASE_URL` | PDF tasks | Needed for `--pdf-path` workflows. |
| `OA_API_KEY` | optional | OpenAlex metadata/PDF support. |
| `OPENALEX_MAILTO` | optional | OpenAlex contact email. |

### 4. Test

```bash
sciatlas health
sciatlas config
```

### 5. Run a Paper Search

```bash
sciatlas search-papers \
  --query "open world agent" \
  --keyword "high:open world agent" \
  --top-k 10
```

---

## 🔑 API Token

SciAtlas uses personal API tokens for public access.

### Browser Registration

Visit:

```text
http://scinet.openkg.cn/register
```

Steps:

1. enter your name, email, organization, and use case;
2. click **Send code**;
3. check your inbox for the verification code;
4. enter the code and create a token;
5. copy the returned `sciatlas_xxx` token.

The token is shown only once.

### Check Token Status

```bash
curl -H "Authorization: Bearer $SCIATLAS_API_KEY" \
  http://scinet.openkg.cn/v1/auth/token/status
```

### Check Usage

```bash
curl -H "Authorization: Bearer $SCIATLAS_API_KEY" \
  "http://scinet.openkg.cn/v1/auth/usage?days=7"
```

---

## 🧩 Supported Tasks

| Command | Scenario | Main Output |
|---|---|---|
| `sciatlas search-papers` | Paper search | Related papers and Markdown report |
| `sciatlas related-authors` | Related-author discovery | Candidate authors and scores |
| `sciatlas author-papers` | Author paper lookup | Papers by a specified author |
| `sciatlas support-papers` | Support-paper retrieval | Evidence papers for candidate authors |
| `sciatlas paper-search` | Lightweight low-level paper search | Fast paper candidates |
| `sciatlas literature-review` | Literature review | Core paper pool, timeline, writing hints |
| `sciatlas idea-grounding` | Idea grounding | Similar works and differentiation evidence |
| `sciatlas idea-evaluate` | Idea evaluation | Evidence for novelty, feasibility, and soundness |
| `sciatlas idea-generate` | Idea generation | Topic combinations and idea seeds |
| `sciatlas trend-report` | Trend analysis | Evolution evidence and representative works |
| `sciatlas researcher-review` | Researcher background review | Research trajectory and representative works |
| `sciatlas skill` | Editable skill registry | Reusable workflow presets |

---

## 🛠️ CLI-First Workflow

SciAtlas is CLI-first: you can start with one command, inspect the saved artifacts, and then move into larger research workflows. If you are new, run help once, try a basic retrieval, then choose one of the downstream workflows below.

Documentation: [📚 SciAtlas Documentation](http://scinet.openkg.cn/api/docs/). Use it to check API setup, CLI commands, parameter meanings, and runnable examples.

### Help

```bash
sciatlas -h
sciatlas search-papers -h
sciatlas literature-review -h
sciatlas skill -h
```

### Input Styles

SciAtlas supports two input styles. For formal runs, prefer expert parameters because every field is explicit and easier to reproduce. Natural-language input is useful for quick trials or exploratory use.

#### Recommended: expert parameters

```bash
sciatlas --timeout 900 search-papers \
  --retrieval-mode hybrid \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --keyword "middle:embodied agent" \
  --title "middle:Voyager: An Open-Ended Embodied Agent with Large Language Models" \
  --reference "low:JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models" \
  --top-k 5 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0 \
  --bias-keyword high \
  --bias-related high \
  --bias-exploration low \
  --ranking-profile precision \
  --report-max-items 5
```

#### Compatible: natural-language input

Use `--text` when you want SciAtlas to parse the request from a short instruction. You can still add structured hints such as `keyword[high]: ...` in the text.

```bash
sciatlas --timeout 900 search-papers \
  --retrieval-mode hybrid \
  --text "Find papers related to open world agent in artificial intelligence since 2020. Return 3 papers.

keyword[high]: open world agent" \
  --top-k 3 \
  --top-keywords 1 \
  --max-titles 0 \
  --max-refs 0
```

### Basic Retrieval

Use this when you want a quick, evidence-backed paper list for one topic.

```bash
sciatlas search-papers \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --top-k 5 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0
```

### Downstream Workflows

Each workflow prints a concise terminal summary and saves full artifacts under `runs/<run_id>/`.

#### Literature Review

Build an initial reading list and get evidence for writing a literature review.

```bash
sciatlas literature-review \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 10
```

#### Idea Evaluation

Check whether a proposed research idea is novel, feasible, and well supported by existing work.

```bash
sciatlas idea-evaluate \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:idea evaluation" \
  --keyword "middle:LLM as a judge" \
  --top-k 10
```

#### Idea Generation

Explore promising topic combinations and generate candidate research directions.

```bash
sciatlas idea-generate \
  --query "knowledge editing for large language models" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:knowledge editing" \
  --keyword "middle:large language models" \
  --keyword "low:continual learning" \
  --top-k 10
```

#### Trend Report

Trace how a topic has developed and identify representative works along the way.

```bash
sciatlas trend-report \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --keyword "middle:knowledge graph" \
  --top-k 10
```

#### Researcher Review

Summarize a researcher's publication trajectory and representative papers.

```bash
sciatlas researcher-review \
  --author "Yoshua Bengio" \
  --limit 10 \
  --no-abstract
```

### Retrieval Modes

| Mode | Meaning | Best For |
|---|---|---|
| `keyword` | Keyword-driven KG retrieval | Clear terminology |
| `semantic` | Semantic retrieval | Broad semantic matching |
| `title` | Title-anchor retrieval | Known paper titles |
| `hybrid` | Keyword + semantic + title + graph walk | Default and recommended |

If `--retrieval-mode` is omitted, SciAtlas uses `hybrid`.

### Expert Anchors

Use anchors when you already know a strong keyword, title, or reference and want the graph search to start from it.

```bash
--keyword "high:open world agent"
--title "middle:Voyager: An Open-Ended Embodied Agent with Large Language Models"
--reference "low:JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models"
```

### Graph Bias Parameters

| Parameter | Meaning |
|---|---|
| `--bias-keyword` | Keyword association strength |
| `--bias-non-seed-keyword` | Non-seed keyword expansion |
| `--bias-citation` | Citation edge strength |
| `--bias-related` | Paper relatedness strength |
| `--bias-authorship` | Author-paper relation strength |
| `--bias-coauthorship` | Coauthor network strength |
| `--bias-cooccurrence` | Keyword co-occurrence strength |
| `--bias-exploration` | Graph exploration level |
| `--ranking-profile` | Ranking preference: `precision`, `balanced`, `discovery`, `impact` |

Recommended safe defaults:

```bash
--top-k 10
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

---

## 🧰 Editable Skills

SciAtlas skills are JSON presets for downstream research workflows. They make complex workflows easier to inspect, reuse, and customize.

```bash
sciatlas skill list
sciatlas skill show literature-review
sciatlas skill run literature-review --query "open world agent" --keyword "high:open world agent"
sciatlas skill run --dry-run literature-review --query "open world agent" --keyword "high:open world agent"
```

Create a custom skill:

```bash
sciatlas skill init my-review --from literature-review
```

This creates:

```text
./skills/my-review.json
```

Edit it, then run:

```bash
sciatlas skill run my-review --query "your topic"
```

User-defined skills are loaded from:

1. `./skills/*.json`
2. `~/.sciatlas/skills/*.json`
3. directories specified by `SCIATLAS_SKILLS_DIR`

User-defined skills can override built-in skills with the same name.

---

## 🖊Agent Skill

SciAtlas also ships a portable Agent Skill pack under [`agent-skill/`](agent-skill/). These are not runtime outputs or simple command aliases. They are downstream task playbooks that teach tools such as Codex, Claude Code, and other coding agents how to bootstrap a new user's environment, obtain/configure the API token with user feedback when needed, run only SciAtlas's base `search-papers` command, read saved artifacts, and complete a research goal.

Included skills:

| Skill | Retrieval base | Downstream use case |
|---|---|---|
| `sciatlas-quick-paper-search` | `search-papers` | Small evidence seed and downstream routing |
| `sciatlas-literature-review` | `search-papers` | Reading lists and related-work reports |
| `sciatlas-idea-grounding` | `search-papers` | Prior-art grounding for research ideas |
| `sciatlas-idea-evaluate` | `search-papers` | Novelty, feasibility, and soundness checks |
| `sciatlas-idea-generate` | `search-papers` | Literature-grounded idea seeds |
| `sciatlas-trend-report` | `search-papers` | Timeline and trend analysis |
| `sciatlas-researcher-review` | `search-papers` only | Researcher profiles from retrieved paper evidence |

To use one locally, copy its directory into the skill directory supported by your agent tool, then restart or refresh that tool. For Codex, that is usually `~/.codex/skills` or `%USERPROFILE%\.codex\skills`. Inside these Agent Skills, `search-papers` is the only SciAtlas retrieval primitive; the downstream review, idea, trend, or researcher result is produced by the agent reading `runs/<run_id>/` artifacts and synthesizing evidence.

---

## 🐍 Python SDK

SciAtlas also provides a lightweight Python client.

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

You can also pass credentials directly:

```python
from sciatlas import SciAtlasClient

client = SciAtlasClient(
    base_url="http://scinet.openkg.cn",
    api_key="your-personal-sciatlas-token",
)

print(client.token_status())
```

## 📦 Outputs and Artifacts

Terminal output is concise and table-based. Full outputs are saved under:

```text
runs/<run_id>/
```

Common artifacts:

| File | Description |
|---|---|
| `plan.json` | Structured search plan |
| `request.json` | Full request sent to SciAtlas API |
| `response.json` | Raw backend response |
| `summary.txt` | Short summary |
| `report.md` | User-facing Markdown report |
| `metadata.json` | Runtime metadata |

---

## 📂 Repository Layout

The tree below highlights the main user-facing areas of the repository. Generated outputs and local cache folders are omitted.

```text
SciAtlas/
  README.md / README_zh.md       # project documentation
  .env.example                   # root runtime configuration template
  requirements.txt
  run_sciatlas.py                  # lightweight local runner
  agent-skill/                   # portable Agent Skill pack
  docs/api/                      # unified static API and CLI documentation site
  imgs/                          # README figures
  sciatlas/                        # pip-installable SciAtlas client package
    pyproject.toml
    src/sciatlas/                  # packaged CLI, client, config, and skills
    core/ search/ tasks/         # retrieval planning and workflow logic
    evidence/ llm/ renderers/    # PDF evidence, optional LLM, report rendering
    examples/ tests/
  references/search/             # reference KG search implementation
  runs/                          # generated CLI outputs
```

---

## 🧯 Troubleshooting

### `sciatlas health` works but `search-papers` returns 401

Your token is missing or invalid.

```bash
echo $SCIATLAS_API_KEY
export SCIATLAS_API_KEY="your-personal-sciatlas-token"
```

Windows CMD:

```bat
set SCIATLAS_API_KEY=your-personal-sciatlas-token
```

### No email verification code

Check the email address, spam folder, and resend interval.

### Retrieval is slow or times out

Use lightweight settings:

```bash
--top-k 3
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

### `sciatlas` command is not found on Windows

Use the virtual environment executable directly:

```bat
.venv\Scripts\sciatlas.exe -h
```

or reinstall:

```bat
.venv\Scripts\python.exe -m pip install -e .
```

---

## 📝 TODO

- [x] **CLI Tools.** Add more user-facing CLI capabilities so downstream users and AI agents can invoke retrieval workflows without touching database internals.
- [x] **Portable Agent Skill pack.** Package reusable agent skills for common scientific discovery workflows and expose best practices as easier-to-load components.
- [ ] **More Knowledge.** Integrate more knowledge forms beyond paper-centric entities, such as datasets, code, standards, theorems, and experimental experience.
- [ ] **Benchmark and Evaluation.** Build dedicated benchmarks and evaluation protocols for downstream scientific research tasks supported by SciAtlas.
- [ ] **Dynamic Update**Improve dynamic knowledge updates toward a more systematic and frequent refresh mechanism.


---

## ✍️ Citation

If you find SciAtlas helpful, please cite:

```
@misc{qiao2026sciatlaslargescaleknowledgegraph,
      title={SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research}, 
      author={Shuofei Qiao and Yunxiang Wei and Jiazheng Fan and Bin Wu and Busheng Zhang and Mengru Wang and Yuqi Zhu and Ningyu Zhang and Keyan Ding and Qiang Zhang and Huajun Chen},
      year={2026},
      eprint={2605.22878},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.22878}, 
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
