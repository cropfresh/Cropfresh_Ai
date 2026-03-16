#!/bin/bash
# Run complete evaluation suite
set -e
uv run python -m src.evaluation.eval_runner
