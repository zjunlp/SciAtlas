---
name: scischolar-quick-paper-search
description: Use only SciScholar search-papers to take a novice user from zero setup to a small evidence-backed paper search result. Trigger when the user wants a quick paper check, first-pass literature evidence, a retrieval smoke test, or help deciding which deeper SciScholar agent skill to use next.
---

# SciScholar Quick Paper Search

Use this skill to deliver a small, evidence-backed paper search from a blank machine. The only SciScholar retrieval command allowed in this skill is `scischolar search-papers`.

## Operating Contract

- Do not call any SciScholar downstream, author, support, report, config, health, or skill-management command for retrieval. The only SciScholar retrieval command allowed is `search-papers`.
- Run setup commands yourself when tool access is available. Do not ask a novice user to run shell commands unless you are blocked by missing permissions or missing human-only information.
- Ask the user only for values a program cannot know: email address, verification code, returned API token, or a clarification when the topic is genuinely ambiguous.
- Never print the full API token in the final answer. Mask it if you must mention it.
- If any step fails, explain the exact blocker in plain language, fix what you can, and continue until a `search-papers` run succeeds or the user must provide a missing human value.

## Zero-Start Bootstrap

1. Check whether the CLI executable exists without invoking a SciScholar command: use `Get-Command scischolar -ErrorAction SilentlyContinue` on Windows PowerShell or `command -v scischolar` on macOS/Linux.
2. If `scischolar` is unavailable:
   - If this repository is present, install locally with `python -m pip install -e ./scischolar`.
   - Otherwise install from GitHub with `python -m pip install "git+https://github.com/zjunlp/SciScholar.git#subdirectory=scischolar"`.
3. Check whether `SCISCHOLAR_API_KEY` is present in the current shell environment.
4. If no token is configured, guide the user through registration:
   - Open or provide `http://scinet.openkg.cn/register`.
   - Ask the user for the email only if the registration page or flow needs it and you cannot fill it yourself.
   - Ask for the email verification code when it arrives.
   - Ask the user to paste the returned `scischolar_xxx` token. The token is shown only once.
5. Configure the current shell before running retrieval:

Windows PowerShell:

```powershell
$env:SCISCHOLAR_API_BASE_URL = "http://scinet.openkg.cn"
$env:SCISCHOLAR_API_KEY = "<token>"
setx SCISCHOLAR_API_BASE_URL "http://scinet.openkg.cn"
setx SCISCHOLAR_API_KEY "<token>"
```

macOS/Linux:

```bash
export SCISCHOLAR_API_BASE_URL="http://scinet.openkg.cn"
export SCISCHOLAR_API_KEY="<token>"
```

6. If a token check is needed, do not call `health` or `config`; run a tiny `search-papers` request with `--top-k 1`.

## Search Recipe

Turn the user's request into one precise query:

- `--query`: the research topic in natural English.
- `--keyword "high:<core term>"`: the main concept that must stay central.
- `--keyword "middle:<supporting term>"`: optional nearby concept.
- `--time-range`: include only when the user asked for recent work or a period.
- `--top-k 3` for a quick answer; use `--report-max-items 3`.

Example:

```bash
scischolar search-papers --retrieval-mode hybrid --query "open world agent" --keyword "high:open world agent" --top-k 3 --top-keywords 0 --max-titles 0 --max-refs 0 --bias-exploration low --ranking-profile precision --report-max-items 3
```

## Reading Artifacts

After the command finishes:

1. Locate the run directory from stdout, or use the newest folder under `runs/`.
2. Read `summary.txt` first for a quick status.
3. Read `report.md` for user-facing paper descriptions.
4. Read `response.json` only when `report.md` lacks title, year, venue, abstract, score, or URL details.
5. Build an evidence ledger with columns: rank, title, year, authors if available, why it matched, strongest clue, URL/ID.

## Synthesis Method

Write the result as a decision aid, not a dump of titles:

1. State the search intent in one sentence.
2. List the top papers with 1-2 evidence sentences each.
3. Explain the pattern across the results: shared problem, shared method, datasets or evaluation style if visible.
4. Flag weak evidence clearly, for example "the title matches but the abstract is missing."
5. Recommend the next action:
   - use `scischolar-literature-review` for a broader related-work section;
   - use `scischolar-idea-grounding` if the user has a concrete idea;
   - use `scischolar-idea-evaluate` if the user wants a go/no-go critique;
   - use `scischolar-idea-generate` if the user wants new research directions;
   - use `scischolar-trend-report` for timeline analysis;
   - use `scischolar-researcher-review` for author-centered profiling.

## Deliverable

Return:

- Search command used, with token omitted.
- 3-paper evidence table.
- 3-5 sentence synthesis.
- Suggested next skill or next `search-papers` query.
