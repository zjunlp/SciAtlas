# Search Reference

`references/search` keeps a standalone copy of the search logic used for InnoEval.

It is not the main runnable entrypoint of this repository. The root workflow is still `run_sciatlas.py`. This directory exists so the KG, S2, merge, and reranking logic stays easy to read and can still be run on its own when needed.

## What This Directory Is For

- reading the current search implementation in isolation
- demonstrating how KG search, S2 retrieval, result merging, and reranking fit together
- running S2-only search locally without depending on the full demo workflow

## Structure

```text
.
├── run_search.py
├── src/innoeval_search/
│   ├── combined/
│   ├── kg/
│   ├── s2/
│   └── shared/
└── tests/
```

## Quick Start

```bash
cd references/search
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install neo4j sentence-transformers pytest
cp .env.example .env
```

Copy `.env.example` to `.env` and fill in only the values required by the flow you want to inspect.

Shared LLM settings:

```env
OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your-model-name
```

Semantic Scholar settings:

```env
S2-API-KEY=replace-me
```

KG settings, only when KG retrieval is enabled:

```env
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=replace-me
NEO4J_DATABASE=neo4j
```

The KG flow also assumes the expected graph schema and indexes already exist.

## Typical Commands

S2-only idea-text search:

```bash
python3 run_search.py \
  --idea-text "research idea evaluation with large language models" \
  --disable-kg \
  --disable-llm-ranking \
  --pretty
```

S2-only PDF search:

```bash
python3 run_search.py \
  --pdf-path /absolute/path/to/paper.pdf \
  --disable-kg \
  --s2-mode hybrid \
  --grobid-base-url http://127.0.0.1:8070 \
  --disable-llm-ranking \
  --pretty
```

Combined KG + S2 search:

```bash
python3 run_search.py \
  --idea-text "research idea evaluation with large language models" \
  --pretty
```

## Notes

- `--disable-kg` is the default way to inspect the public search flow without the original Neo4j graph.
- `--disable-s2` is available when you want to inspect KG behavior in isolation.
- `--disable-llm-ranking` disables only the final reranking stage. Other stages may still use the configured LLM.
- GROBID is needed only for PDF-based flows.
- Result artifacts are written under `result/`, and cache artifacts under `target/`.

## Testing

```bash
pytest -q
```
