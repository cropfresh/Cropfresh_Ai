#!/bin/bash
# Run complete evaluation suite
set -e
uv run python ai/evals/run_evals.py
