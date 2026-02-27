#!/bin/bash
# One-command dev environment setup
set -e
echo '🌱 Setting up CropFresh AI dev environment...'
uv sync --all-extras
cp -n .env.example .env 2>/dev/null || true
echo '✅ Done! Edit .env with your API keys.'
