# CLAUDE.md — Claude-Specific Project Rules

## Project: CropFresh AI
An AI-powered agricultural marketplace for Karnataka farmers.

## Architecture
- **Monorepo** with `src/`, `ai/`, `docs/`, `tracking/`, `tests/`, `infra/`, `config/`
- **Backend**: FastAPI (Python 3.11+)
- **AI**: LangGraph multi-agent system
- **Databases**: Supabase (PostgreSQL) + Qdrant (vectors) + Neo4j (graph)

## Coding Conventions
- Use `loguru` for all logging (never `print()` in production code)
- Type hints on all function signatures
- Pydantic models for all API request/response schemas
- Async-first: prefer `async def` for all I/O operations
- Use `tenacity` for retries on external API calls

## Agent Development Rules
- All agents extend `src/agents/base_agent.py`
- Agent prompts versioned in `src/agents/{name}/prompts/`
- Every agent must have corresponding eval in `ai/evals/`
- Log all agent interactions for observability

## Important Paths
- Entry point: `src/api/main.py`
- Settings: `src/api/config.py`
- Agent base: `src/agents/base_agent.py`
- RAG pipeline: `ai/rag/`
- Current plan: `PLAN.md`
