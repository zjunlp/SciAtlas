#!/usr/bin/env bash
set -e

sciatlas --timeout 900 idea-evaluate \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --keyword "high:idea evaluation" \
  --top-k 3
