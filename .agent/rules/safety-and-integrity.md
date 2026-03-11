---
description: Safety and integrity rules to prevent accidental data loss or destructive actions
---

# Safety & Integrity Rules

## Protected Files (Never Auto-Modify)

These files require **explicit user confirmation** before any modification:

1. `PLAN.md` — Master product vision
2. `AGENTS.md` — AI agent instruction set
3. `ai/rag/*` — RAG pipeline (check eval results first)
4. `src/agents/base_agent.py` — Agent base class
5. `docker-compose.yml` — Infrastructure config
6. `.env` — Secrets and API keys
7. `pyproject.toml` — Dependencies

## Data Integrity Rules

1. **Never delete tracking files** — Sprint outcomes, daily logs, and retros are permanent records
2. **Never truncate conversation history** — Append new entries, never remove old ones
3. **Never hardcode secrets** — Use environment variables and `.env` files
4. **Always backup before schema changes** — Database migrations must be reversible

## Code Safety

1. **No `print()` statements** — Use `loguru.logger` from `src/shared/logger.py`
2. **No raw SQL without parameterization** — Always use parameterized queries
3. **No `*` CORS origins in production** — Set explicit `ALLOWED_ORIGINS`
4. **No synchronous DB calls in async handlers** — Use `async` database clients

## Pre-Commit Checklist

Before committing any change:
- [ ] Tests pass: `uv run pytest -v`
- [ ] Type check clean: `uv run mypy src/`
- [ ] Lint clean: `uv run ruff check src/`
- [ ] WORKFLOW_STATUS.md updated with file changes
- [ ] No secrets in committed files
