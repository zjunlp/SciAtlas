---
name: scischolar-literature-review
description: Use only SciScholar search-papers to take a novice user from zero setup to an evidence-grounded literature review, related-work summary, survey outline, or reading list. Trigger when the user asks for a literature review, related work section, paper map, or topic overview.
---

# SciScholar Literature Review

Use this skill to turn `search-papers` output into a real literature review. The only SciScholar retrieval command allowed in this skill is `scischolar search-papers`.

## Operating Contract

- Do not call any SciScholar downstream command. The only SciScholar retrieval command allowed is `search-papers`.
- Run setup yourself when possible; ask the user only for email, verification code, token, or a necessary topic clarification.
- Keep going until `search-papers` has produced artifacts or the user must supply a human-only value.
- Never include the full API token in the final answer.

## Zero-Start Bootstrap

1. Check whether the `scischolar` executable exists without invoking a SciScholar command: use `Get-Command scischolar -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v scischolar` on macOS/Linux.
2. If missing, install from the local repo with `python -m pip install -e ./scischolar`; otherwise use `python -m pip install "git+https://github.com/zjunlp/SciScholar.git#subdirectory=scischolar"`.
3. If `SCISCHOLAR_API_KEY` is missing, send the user to `http://scinet.openkg.cn/register`.
4. Walk the user through the browser registration: enter email, wait for the code, ask for the code if you are controlling the form, then ask them to paste the returned `scischolar_xxx` token.
5. Configure the shell:

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

6. Use a tiny `search-papers --top-k 1` only if you must verify the token before the real run.

## Search Plan

Run only `search-papers`. Prefer two passes when the topic is broad:

Precision pass:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<topic>" --domain "<field>" --time-range "<years>" --keyword "high:<core concept>" --top-k 8 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-exploration low --ranking-profile precision --report-max-items 8
```

Coverage pass, only if the first result is too narrow:

```bash
scischolar search-papers --retrieval-mode hybrid --query "<topic plus neighboring concepts>" --keyword "high:<core concept>" --keyword "middle:<neighbor concept>" --top-k 10 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-related high --bias-exploration middle --ranking-profile balanced --report-max-items 10
```

If the user provides seed papers, put exact titles in `--title "high:<title>"` or `--reference "middle:<title>"`, still using `search-papers`.

## Reading Artifacts

For each run, read:

1. `summary.txt` for success/failure and result count.
2. `report.md` for ranked papers and generated descriptions.
3. `request.json` to record the exact query and anchors.
4. `response.json` to recover missing metadata and abstracts.

Create a working evidence ledger:

- Paper: title, year, venue/source if present.
- Problem: what question it addresses.
- Method: model, algorithm, system, dataset, or study design.
- Evidence: abstract phrase or report sentence that supports inclusion.
- Role in review: foundation, method, evaluation, application, critique, or recent extension.

## Review Construction Method

Do not summarize papers one by one only. Build a literature review:

1. Define the scope in 2-3 sentences: topic, period, and what counts as relevant.
2. Cluster papers by research theme. Use paper content, not only keywords.
3. Order clusters logically: foundations -> methods -> applications/evaluations -> limitations -> recent directions.
4. For each cluster, explain the shared research question and cite representative papers from the ledger.
5. Compare papers inside a cluster: what changed in method, data, assumptions, or evaluation.
6. Identify consensus, disagreement, and gaps. A gap must be tied to missing evaluation, weak assumptions, limited dataset, deployment challenge, or unconnected neighboring literature.
7. End with a reading path: first 3 papers to read and why.

## Deliverable

Return:

- The exact `search-papers` command(s), with token omitted.
- Evidence table of 6-10 papers.
- Thematic literature review with headings.
- Timeline or development notes when years are available.
- Gaps and next search queries.
