# AGENTS.md — AI Agent Instructions for CropFresh

## Project Context
CropFresh AI is an intelligent agricultural marketplace platform connecting farmers with buyers.
Built with FastAPI + LangGraph + Qdrant + Neo4j + Supabase.

## Key Rules
1. Always refer to `PLAN.md` for the current development priorities
2. Follow coding standards in `docs/architecture/coding-standards.md`
3. Check `tracking/` for current sprint goals before starting work
4. All agents live in `src/agents/` — use `base_agent.py` as base class
5. RAG pipeline lives in `ai/rag/` — do not modify without checking eval results
6. Use structured logging via `src/shared/logger.py`

## Tech Stack
- **Backend**: FastAPI + Python 3.11+
- **AI Framework**: LangGraph + LangChain
- **Vector DB**: Qdrant Cloud
- **Graph DB**: Neo4j
- **Primary DB**: Supabase (PostgreSQL)
- **LLM**: Groq (Llama/Mixtral)
- **Voice**: Edge-TTS + Whisper
- **Package Manager**: uv

## File Structure Summary
- `src/` — Application source code (api, agents, scrapers, pipelines, shared)
- `ai/` — ML models, training data, evaluations, RAG pipeline
- `docs/` — All documentation (planning, architecture, decisions, features)
- `tracking/` — Development progress (goals, sprints, daily logs)
- `tests/` — Test infrastructure (e2e, integration, load)
- `infra/` — Deployment & monitoring configs
- `config/` — Database & service configurations
- `scripts/` — Automation utilities

## Cursor Cloud specific instructions

### Running the application
- Dev server: `uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000` (or `make dev`)
- The server starts on port 8000 and exposes Swagger docs at `/docs`
- `.env` must exist (copy from `.env.example`). The server starts even with placeholder API keys; LLM-dependent features (chat, RAG query) will return graceful error responses without a valid `GROQ_API_KEY`

### Lint / Test / Build
- Lint: `make lint` (uses `ruff`). Pre-existing lint issues exist in `ai/evals/run_evals.py` (malformed single-line file) and notebooks/scripts
- Type check: `make typecheck` (uses `mypy`). Known pre-existing syntax error in `src/agents/buyer_matching/agent.py` prevents full checking
- Tests: `make test` — all 18 current tests are **integration tests** requiring a running server (WebSocket on port 8000). Start the dev server before running tests
- Format: `make format` (uses `ruff format`)

### External services
- **Qdrant** (vector DB): Required for RAG endpoints. Not needed for basic server startup, health checks, or chat routing. If needed locally, use `docker compose up qdrant`
- **Neo4j**: Required for graph-based features. Use `docker compose up neo4j` if needed
- **Supabase / Groq / Redis**: Cloud-hosted or optional; configured via `.env` variables
- The server gracefully handles missing external services — endpoints return structured error responses rather than crashing

### Gotchas
- `.python-version` says 3.12; system Python 3.12.3 works. uv will use whatever Python matches `requires-python = ">=3.11"` from `pyproject.toml`
- The `sentence-transformers` model (`BAAI/bge-m3`, ~2GB) downloads on first RAG/embedding use — not at server startup
- Config uses `pydantic-settings` with `.env` file loading (`src/config/settings.py`). Default port in config is 8080, but `make dev` overrides to 8000
