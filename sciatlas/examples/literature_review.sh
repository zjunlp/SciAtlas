#!/usr/bin/env bash
set -e

sciatlas --timeout 900 literature-review \
  --workflow flash \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 5

# For a broader formal review run:
# sciatlas --timeout 900 literature-review --workflow full --query "retrieval augmented generation" --domain "artificial intelligence"
