# Reviewer Selection Evaluation

This directory contains a lightweight reviewer-selection experiment. It stops before reviewer background generation and LLM review generation.

It produces two reviewer lists for one idea:

- `pipeline_reviewers`: existing KG author ranking, name merge, works-count filter, evenly spaced sampling.
- `baseline_reviewers`: top ranked KG papers, mapped to each paper's first `Author` node through the `AUTHORED` relationship.

Run from the repository root:

```bash
python reviewer-evaluate/select_reviewers.py \
  --idea-text "Your idea text here" \
  --reviewer-count 10 \
  --kg-top-k 50 \
  --baseline-scan-limit 50 \
  --env /home/weiyunxiang/yunx/.env \
  --pretty
```

For a PDF input, matching the top-level pipeline input style:

```bash
python reviewer-evaluate/select_reviewers.py \
  --pdf-path /path/to/paper.pdf \
  --reviewer-count 10 \
  --kg-top-k 50 \
  --baseline-scan-limit 50 \
  --env /home/weiyunxiang/yunx/.env \
  --pretty
```

To reuse an existing pipeline search artifact:

```bash
python reviewer-evaluate/select_reviewers.py \
  --search-result result/.../search/result.json \
  --reviewer-count 10 \
  --env /home/weiyunxiang/yunx/.env \
  --pretty
```

Outputs are written under `reviewer-evaluate/runs/<timestamp>/` by default:

- `search_result.json`
- `pipeline_selection.json`
- `baseline_first_author_selection.json`
- `reviewer_lists.json`

To run the whole `pairs_v2_final.json` dataset:

```bash
python reviewer-evaluate/run_pairs_dataset.py \
  --pair-json dataset/pairs_v2_final.json \
  --paper-key all \
  --reviewer-count 10 \
  --kg-top-k 50 \
  --baseline-scan-limit 50 \
  --env /home/weiyunxiang/yunx/.env
```

Useful batch options:

```bash
python reviewer-evaluate/run_pairs_dataset.py \
  --pair-json dataset/pairs_v2_final.json \
  --paper-key paper_nc \
  --start-index 0 \
  --max-papers 5 \
  --dry-run
```
