---
name: scischolar-idea-evaluate
description: Use only SciScholar search-papers to take a novice user from zero setup to an evidence-grounded evaluation of a research idea's novelty, feasibility, soundness, and differentiation. Trigger when the user asks whether an idea is worth pursuing or needs literature-backed critique.
---

# SciScholar Idea Evaluate

Use this skill to evaluate a research idea with evidence from `search-papers` only.

## Operating Contract

- Do not call any SciScholar downstream command. The only SciScholar retrieval command allowed is `search-papers`.
- Ask the user only for email, verification code, token, or one necessary clarification.
- Run setup and retrieval yourself where possible.
- Do not expose the full token.

## Zero-Start Bootstrap

1. Check whether the `scischolar` executable exists without invoking a SciScholar command: use `Get-Command scischolar -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v scischolar` on macOS/Linux. Install if missing with `python -m pip install -e ./scischolar` or `python -m pip install "git+https://github.com/zjunlp/SciScholar.git#subdirectory=scischolar"`.
2. If no `SCISCHOLAR_API_KEY` is available, open or provide `http://scinet.openkg.cn/register`.
3. Walk the novice user through registration. Ask for the email verification code when required and the returned `scischolar_xxx` token when registration finishes.
4. Configure the shell:

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

5. Use only `search-papers` for any token smoke test.

## Evaluation Frame

Convert the idea into four claims:

- Novelty claim: what is supposed to be new.
- Feasibility claim: why it can be built with available methods/data.
- Soundness claim: why the mechanism should work.
- Differentiation claim: why it is not just a minor variant.

## Search Plan

Run only `search-papers`. Use one direct search and, if needed, one stress-test search:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<research idea>" --keyword "high:<core mechanism>" --keyword "middle:<application domain>" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-keyword high --bias-related high --bias-citation low --bias-exploration low --ranking-profile precision --report-max-items 8
```

Stress-test novelty by searching for the closest alternative wording:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<closest competing formulation>" --keyword "high:<problem>" --keyword "middle:<neighbor method>" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-related high --ranking-profile precision --report-max-items 8
```

## Reading Artifacts

Read `report.md` and `response.json`. Build an evaluation ledger:

- Evidence for novelty.
- Evidence against novelty.
- Evidence for feasibility: data, implementation pattern, benchmark, or system precedent.
- Evidence for soundness: theory, empirical result, mechanism, or ablation.
- Missing evidence.

## Evaluation Method

Score each dimension as High, Medium, Low, or Unknown:

1. Novelty: High only if close papers solve a different problem or use a meaningfully different mechanism.
2. Feasibility: High only if retrieved work shows required components, data, or evaluation are realistic.
3. Soundness: High only if evidence supports the core causal mechanism, not merely the topic.
4. Differentiation: High only if the idea can be expressed as a clear contrast with closest prior work.

For each score:

- cite the supporting papers from the search result;
- explain the reasoning in plain language;
- state the biggest uncertainty;
- suggest one concrete improvement.

## Deliverable

Return:

- Search command(s), with token omitted.
- Evaluation table: novelty, feasibility, soundness, differentiation.
- Closest-prior-art risk.
- Go / revise / no-go recommendation.
- Revised idea statement and next `search-papers` query.
