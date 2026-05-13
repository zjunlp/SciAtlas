---
name: scischolar-trend-report
description: Use only SciScholar search-papers to take a novice user from zero setup to a timeline-oriented research trend report. Trigger when the user asks for topic history, field evolution, recent trends, representative papers over time, or emerging directions.
---

# SciScholar Trend Report

Use this skill to build a trend report from `search-papers` evidence only.

## Operating Contract

- Do not call any SciScholar downstream command. The only SciScholar retrieval command allowed is `search-papers`.
- Ask the user only for email, verification code, token, or the desired time range if it is essential.
- Run setup and retrieval yourself when possible.
- Do not reveal the full token.

## Zero-Start Bootstrap

1. Check whether the `scischolar` executable exists without invoking a SciScholar command: use `Get-Command scischolar -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v scischolar` on macOS/Linux. Install if needed with `python -m pip install -e ./scischolar` or `python -m pip install "git+https://github.com/zjunlp/SciScholar.git#subdirectory=scischolar"`.
2. If there is no `SCISCHOLAR_API_KEY`, guide registration at `http://scinet.openkg.cn/register`, asking for email/code/token only when needed.
3. Configure:

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

4. Use `search-papers --top-k 1` as the only token smoke test if needed.

## Search Plan

Run only `search-papers`. Use a time range whenever possible:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<topic>" --time-range "<start-end>" --keyword "high:<core topic>" --top-k 12 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-citation high --bias-exploration middle --ranking-profile impact --report-max-items 12
```

If the user asks for "recent trends", default to the last five complete years unless the current year matters.

If the field has older foundations, run one foundation search without the recent time range:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<topic> foundational work" --keyword "high:<core topic>" --top-k 6 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-citation high --ranking-profile impact --report-max-items 6
```

## Reading Artifacts

Read `report.md` and `response.json`. Build a timeline ledger:

- Paper title.
- Year.
- Main contribution.
- Theme.
- Whether it is foundation, transition, consolidation, or emerging direction.
- Evidence phrase.

## Trend Analysis Method

1. Sort papers by year.
2. Bucket years into phases. Use natural breaks in the result set, not arbitrary equal chunks.
3. For each phase, identify the dominant problem, method, and evaluation pattern.
4. Track what changed: data scale, model architecture, retrieval strategy, benchmark, deployment setting, or theoretical framing.
5. Identify representative papers for each phase and explain why they represent that phase.
6. Separate "well-supported trend" from "possible emerging signal." Emerging signals need at least 2 papers or a clear recent change in framing.
7. End with open questions and next searches.

## Deliverable

Return:

- Search command(s), with token omitted.
- Timeline table.
- 3-5 trend phases.
- Representative papers per phase.
- Emerging directions and weak signals.
- Next `search-papers` queries.
