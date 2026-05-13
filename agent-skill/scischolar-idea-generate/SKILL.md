---
name: scischolar-idea-generate
description: Use only SciScholar search-papers to take a novice user from zero setup to literature-grounded research idea seeds. Trigger when the user wants new research directions, hypotheses, cross-topic combinations, project ideas, or brainstorming grounded in retrieved papers.
---

# SciScholar Idea Generate

Use this skill to generate research ideas from retrieved literature. The only SciScholar retrieval command allowed is `scischolar search-papers`.

## Operating Contract

- Do not call any SciScholar downstream command. The only SciScholar retrieval command allowed is `search-papers`.
- Ask the user only for email, verification code, token, or one clarification about the desired topic/field.
- Complete setup, run `search-papers`, read artifacts, and generate evidence-grounded ideas.
- Never disclose the full API token.

## Zero-Start Bootstrap

1. Check whether the `scischolar` executable exists without invoking a SciScholar command: use `Get-Command scischolar -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v scischolar` on macOS/Linux. Install if missing using the local repo `python -m pip install -e ./scischolar` or GitHub `python -m pip install "git+https://github.com/zjunlp/SciScholar.git#subdirectory=scischolar"`.
2. If `SCISCHOLAR_API_KEY` is missing, guide the user to `http://scinet.openkg.cn/register`.
3. Ask for email/code/token only when needed by the registration flow.
4. Configure:

```powershell
$env:SCISCHOLAR_API_BASE_URL = "http://scinet.openkg.cn"
$env:SCISCHOLAR_API_KEY = "<token>"
setx SCISCHOLAR_API_BASE_URL "http://scinet.openkg.cn"
setx SCISCHOLAR_API_KEY "<token>"
```

```bash
export SCISCHOLAR_API_BASE_URL="http://scinet.openkg.cn"
export SCISCHOLAR_API_KEY="<token>"
```

5. If verification is needed, run a one-result `search-papers` smoke test.

## Search Plan

Generate ideas by retrieving a broad but relevant concept pool. Run only `search-papers`:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<broad topic>" --keyword "high:<core concept>" --keyword "middle:<neighbor concept>" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-related high --bias-cooccurrence high --bias-exploration high --ranking-profile discovery --report-max-items 10
```

If the first pool is too homogeneous, run one contrastive search:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<core concept> combined with <distant but plausible field>" --keyword "high:<core concept>" --keyword "middle:<distant field>" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-exploration high --ranking-profile discovery --report-max-items 10
```

## Reading Artifacts

Read `report.md`, `response.json`, and `request.json`. Build an idea raw-material board:

- Theme clusters.
- Common assumptions repeated across papers.
- Underexplored combinations.
- Methods that appear transferable.
- Evaluation gaps.
- Newer papers that change what is feasible.

## Idea Generation Method

Create ideas by combining evidence, not by free association:

1. Cluster retrieved papers into 3-5 themes.
2. For each theme, identify what the papers optimize, assume, or leave untested.
3. Find bridges between clusters: method from cluster A applied to unsolved problem in cluster B.
4. Form a hypothesis: "If we change X, then Y should improve because Z."
5. Define a minimal validation experiment: dataset, metric, baseline, and expected signal.
6. Check risk: which retrieved paper could make the idea less novel?
7. Keep speculative parts clearly labeled.

Reject ideas that are only "apply method X to domain Y" unless they include a mechanism and evaluation plan.

## Deliverable

Return 3-8 idea seeds. For each:

- Name.
- One-sentence hypothesis.
- Evidence trail from retrieved papers/themes.
- Why it is nontrivial.
- Closest-prior-art risk.
- Minimal validation experiment.
- Next `search-papers` query to test novelty.
