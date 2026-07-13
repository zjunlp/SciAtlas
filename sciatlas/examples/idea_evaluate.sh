#!/usr/bin/env bash
set -e

sciatlas --timeout 900 idea-evaluate \
  --workflow flash \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --keyword "high:idea evaluation" \
  --top-k 3

# For a broader reviewer/rubric/evidence pass:
# sciatlas --timeout 900 idea-evaluate --workflow full --idea "LLM-based multi-perspective evaluation for scientific research ideas"
