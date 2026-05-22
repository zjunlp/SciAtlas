---
name: sciatlas-researcher-review
description: Use only SciAtlas search-papers to take a novice user from zero setup to an evidence-grounded researcher profile from retrieved papers. Trigger when the user asks for a researcher's background, representative works, topic trajectory, or author-centered literature overview.
---

# SciAtlas Researcher Review

Use this skill to create a researcher profile from `search-papers` evidence only. Because this skill does not call author-specific SciAtlas commands, treat the result as a literature-grounded profile, not an authoritative CV.

## Operating Contract

- Do not call any SciAtlas downstream, author, support, or report command. The only SciAtlas retrieval command allowed is `search-papers`.
- Use only `sciatlas search-papers` for retrieval.
- Ask the user only for email, verification code, token, or a clarification when the researcher name is ambiguous.
- Do not expose the full token.

## Zero-Start Bootstrap

1. Check whether the `sciatlas` executable exists without invoking a SciAtlas command: use `Get-Command sciatlas -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v sciatlas` on macOS/Linux. Install if missing with `python -m pip install -e ./sciatlas` or `python -m pip install "git+https://github.com/zjunlp/SciAtlas.git#subdirectory=sciatlas"`.
2. If `SCIATLAS_API_KEY` is missing, guide the user through `http://scinet.openkg.cn/register`; ask for email/code/token only when needed.
3. Configure:

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

4. Verify with `search-papers --top-k 1` only if needed.

## Search Plan

Run only `search-papers`. Use the researcher name as part of the query and, when the user provides a field, include it as a keyword:

```bash
sciatlas search-papers --retrieval-mode hybrid --query "<researcher name> representative papers <field>" --keyword "high:<researcher name>" --keyword "middle:<field or known topic>" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-authorship high --bias-citation high --ranking-profile impact --report-max-items 10
```

If the name is common, ask for one disambiguator: affiliation, field, or one known paper title. Then search:

```bash
sciatlas search-papers --retrieval-mode hybrid --query "<researcher name> <affiliation or known title>" --title "middle:<known paper title>" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-authorship high --ranking-profile precision --report-max-items 10
```

## Reading Artifacts

Read `report.md` and `response.json`. Build a profile ledger:

- Paper title.
- Year.
- Whether the researcher appears in authors if available.
- Topic.
- Contribution.
- Evidence confidence: confirmed author match, likely related, or weak match.

Discard or clearly mark papers where the author match is uncertain.

## Profile Construction Method

1. Start with a confidence note: this profile is based on `search-papers` results, not a complete bibliography.
2. Identify representative works with confirmed or likely author evidence.
3. Group papers by topic or period.
4. Describe trajectory: early focus -> later focus -> current direction, only when years support it.
5. Extract recurring methods, datasets, application domains, or collaborators only when visible in metadata.
6. Avoid claiming awards, positions, total citation counts, or full publication counts unless present in retrieved evidence.
7. Suggest follow-up searches for missing periods or ambiguous identity.

## Deliverable

Return:

- Search command(s), with token omitted.
- Confidence note.
- Representative works table.
- Topic trajectory.
- Research style / recurring themes.
- Missing evidence and next `search-papers` queries.
