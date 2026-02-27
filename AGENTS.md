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
