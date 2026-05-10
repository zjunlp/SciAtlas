---
name: scischolar-idea-generate
description: Use SciScholar search-papers as the retrieval base to generate literature-grounded research idea seeds. Trigger when the user wants new research directions, project ideas, hypotheses, cross-topic combinations, or brainstorming that should be grounded in retrieved papers.
---

# SciScholar Idea Generate

This skill turns `search-papers` into a downstream idea-generation workflow. The agent retrieves a paper/concept pool, identifies gaps and combinations, then proposes candidate ideas with evidence trails.

## End-to-End Workflow

1. Turn the user's broad interest into a searchable topic.
   - Topic or problem -> `--query`
   - Core concept -> `--keyword "high:<term>"`
   - Neighbor concepts -> `--keyword "middle:<term>"`
   - Desired field -> `--domain`
2. Run the idea-generate channel with exploratory graph settings:

```bash
scischolar idea-generate --query "knowledge editing for large language models" --domain "artificial intelligence" --time-range "2020-2025" --keyword "high:knowledge editing" --keyword "middle:large language models" --top-k 8
```

3. If the downstream command is unavailable, use base `search-papers`:

```bash
scischolar search-papers --retrieval-mode hybrid --query "knowledge editing for large language models" --keyword "high:knowledge editing" --keyword "middle:large language models" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-related high --bias-cooccurrence high --bias-exploration high --ranking-profile discovery --report-max-items 8
```

4. Read the generated artifacts in `runs/<run_id>/`.
5. Synthesize ideas from retrieved clusters, co-occurring topics, gaps, and mismatched assumptions.

## Generation Rules

- Every idea must cite the retrieval pattern that inspired it.
- Prefer combinations of two or more retrieved themes over unsupported imagination.
- Mark speculative leaps clearly.
- Avoid producing only generic "apply X to Y" ideas; include a concrete hypothesis and validation path.

## Deliverable

Return 3-8 idea seeds. For each seed include:

- Name.
- Core hypothesis.
- Evidence trail from retrieved papers or clusters.
- Why the idea might be nontrivial.
- First validation experiment or next `search-papers` query.
