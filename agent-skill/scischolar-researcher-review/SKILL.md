---
name: scischolar-researcher-review
description: Use SciScholar search-papers and author-aware retrieval to build an end-to-end researcher profile. Trigger when the user asks for a researcher's background, publication trajectory, representative works, topic evolution, collaborators, or author-centered literature review.
---

# SciScholar Researcher Review

This skill turns SciScholar's paper retrieval into a researcher-review workflow. The agent should resolve the author, retrieve author-related papers, then synthesize a trajectory and representative-work profile.

## End-to-End Workflow

1. Extract the author name or OpenAlex author ID.
2. If the author name is ambiguous, preserve ambiguity and use affiliation or topic hints from the user when available.
3. Run the researcher-review channel:

```bash
scischolar researcher-review --author "Yoshua Bengio" --limit 10 --no-abstract
```

4. If the downstream command is unavailable, run a two-step fallback:

```bash
scischolar author-papers --query "Yoshua Bengio" --limit 10 --no-abstract
```

Then use the strongest returned titles as anchors for base retrieval:

```bash
scischolar search-papers --retrieval-mode hybrid --query "Yoshua Bengio representative papers" --title "high:<seed paper title>" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-authorship high --bias-citation high --ranking-profile impact --report-max-items 10
```

5. Read the generated `runs/<run_id>/` artifacts.
6. Synthesize the profile from evidence rather than writing a biography from memory.

## Deliverable

Return:

- Identity and disambiguation notes.
- Main research themes.
- Timeline or stage-wise trajectory.
- Representative papers and why they matter.
- Collaboration, venue, or citation patterns when visible in the artifacts.
- Suggested follow-up search if the user needs a deeper profile.

## Evidence Rules

- Be explicit if the backend may have merged same-name authors.
- Do not overstate affiliation or career facts unless present in evidence.
- Prefer paper-backed research trajectory over generic biographical prose.
