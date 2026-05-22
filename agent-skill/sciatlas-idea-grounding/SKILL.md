---
name: sciatlas-idea-grounding
description: Use only SciAtlas search-papers to take a novice user from zero setup to prior-art grounding for a research idea. Trigger when the user gives an idea and asks for similar work, differentiation evidence, related work, motivation support, or literature-grounded refinement.
---

# SciAtlas Idea Grounding

Use this skill to position a research idea against prior work using only `sciatlas search-papers`.

## Operating Contract

- Do not call any SciAtlas downstream command. The only SciAtlas retrieval command allowed is `search-papers`.
- Ask the user only for human-only setup values: email, verification code, token, or one concise clarification about the idea.
- Complete setup, run `search-papers`, inspect artifacts, and produce a grounded positioning memo.
- Do not reveal the full API token in final output.

## Zero-Start Bootstrap

1. Check whether the `sciatlas` executable exists without invoking a SciAtlas command: use `Get-Command sciatlas -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v sciatlas` on macOS/Linux. If missing, install with `python -m pip install -e ./sciatlas` in this repo or `python -m pip install "git+https://github.com/zjunlp/SciAtlas.git#subdirectory=sciatlas"`.
2. If `SCIATLAS_API_KEY` is absent, guide the user to `http://scinet.openkg.cn/register`.
3. Ask for the email verification code only when needed. Ask for the returned `sciatlas_xxx` token after registration.
4. Configure:

```powershell
$env:SCIATLAS_API_BASE_URL = "http://scinet.openkg.cn"
$env:SCIATLAS_API_KEY = "<token>"
setx SCIATLAS_API_BASE_URL "http://scinet.openkg.cn"
setx SCIATLAS_API_KEY "<token>"
```

```bash
export SCIATLAS_API_BASE_URL="http://scinet.openkg.cn"
export SCIATLAS_API_KEY="<token>"
```

5. If token validity is uncertain, test with a one-result `search-papers` query, not `health` or `config`.

## Idea Parsing

Before searching, decompose the idea into:

- Problem: what pain or research question it attacks.
- Proposed mechanism: the technical novelty the user claims.
- Target domain: where it applies.
- Evaluation target: what would prove it works.
- Neighboring literatures: 2-4 adjacent concepts.

If any of these are missing, infer reasonable defaults. Ask one short question only when the idea is too vague to search.

## Search Plan

Run only `search-papers`. Use at least one close-prior-art pass:

```bash
sciatlas search-papers --retrieval-mode hybrid --query "<full idea in one sentence>" --keyword "high:<core mechanism>" --keyword "middle:<target domain>" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-keyword high --bias-related high --bias-citation low --bias-exploration low --ranking-profile precision --report-max-items 8
```

If results are sparse, run a second pass around the problem rather than the mechanism:

```bash
sciatlas search-papers --retrieval-mode hybrid --query "<problem statement>" --keyword "high:<problem>" --keyword "middle:<neighbor concept>" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-related high --ranking-profile balanced --report-max-items 8
```

## Reading Artifacts

Read `summary.txt`, `report.md`, `request.json`, and `response.json`. Build a prior-art matrix:

- Paper.
- Overlap with user idea: problem, mechanism, domain, evaluation, or data.
- Difference from user idea.
- Threat level: direct prior, partial prior, background, or weakly related.
- Evidence phrase from abstract/report.

## Grounding Method

1. Restate the idea as a testable claim.
2. Identify the closest prior work first, even if it weakens novelty.
3. Separate overlap types. A paper can share the problem but not the mechanism, or share the method but not the domain.
4. Explain the differentiator only after showing the evidence.
5. Mark unsupported novelty claims. Use "not established by this search" rather than "novel" when evidence is insufficient.
6. Refine the idea into a sharper version that avoids the closest prior art.
7. Suggest the next `search-papers` query to test the weakest point.

## Deliverable

Return:

- Search command(s), with token omitted.
- Prior-art matrix.
- Closest prior work summary.
- Differentiation statement.
- Risks to novelty and how to refine the idea.
- Next search query.
