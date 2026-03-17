# AGENTS.md - AI Agent Instructions for CropFresh

> **Read this before starting any work on CropFresh AI.**
> This is the authoritative context file for Claude Code, Antigravity, and any AI coding agent.
> Last Updated: 2026-03-17

---

## Project Context

**CropFresh AI** is India's intelligent agricultural marketplace AI service.

- Connects Karnataka farmers with buyers using multi-agent AI
- Supports voice-first interaction with a duplex websocket path plus hybrid STT/TTS
- Uses real-time mandi and marketplace data with RAG and shared rate services
- Built with FastAPI, LangGraph, PostgreSQL/Aurora, Qdrant, Neo4j, Redis, and AWS infrastructure

---

## Before Starting Any Work - Read These First

1. **`PLAN.md`** - Vision, architecture, core flows, NFRs, tech stack
2. **`tracking/PROJECT_STATUS.md`** - What is true right now, top priorities, blockers
3. **`tracking/sprints/sprint-07-voice-duplex-productionization.md`** - Next voice sprint handoff and source of truth for duplex voice work
4. **`TESTING/STRATEGY.md`** - What "done" means for each feature type

If the task is not voice-related, replace item 3 with the relevant active sprint file before coding.

---

## Development Rules

1. **Specs before code** - Always read the relevant sprint file and `PLAN.md` before generating code.
2. **Small, focused changes** - One endpoint, one agent, one function per change.
3. **Tests are mandatory** - Every new function or endpoint ships with unit tests (see `TESTING/STRATEGY.md`).
4. **Docs in the same commit** - Update `WORKFLOW_STATUS.md` and the relevant docs alongside code.
5. **No overwriting history** - Append to sprint outcomes and daily logs; use Git for history.
6. **ADRs for major decisions** - Create `docs/decisions/ADR-XXX.md` for significant architectural choices.
7. **Structured logging** - Use `src/shared/logger.py` for logging, never bare `print()`.
8. **Base class compliance** - All eligible agents extend `src/agents/base_agent.py`.
9. **RAG pipeline caution** - Do not modify `ai/rag/` without checking evaluation impact first.

---

## File Structure Summary

```text
CropFresh AI Root
|-- PLAN.md
|-- ROADMAP.md
|-- AGENTS.md
|-- WORKFLOW_STATUS.md
|-- CHANGELOG.md
|
|-- tracking/
|   |-- PROJECT_STATUS.md
|   |-- sprints/sprint-0X-*.md
|   `-- daily/YYYY-MM-DD.md
|
|-- docs/
|   |-- decisions/ADR-*.md
|   |-- agents/REGISTRY.md
|   |-- api/
|   `-- architecture/
|
|-- TESTING/
|   |-- STRATEGY.md
|   `-- CHECKLISTS.md
|
|-- src/
|   |-- agents/
|   |-- api/
|   |-- voice/
|   |-- scrapers/
|   `-- shared/
|
|-- ai/
|-- tests/
`-- scripts/
```

---

## Tech Stack Quick Reference

| Layer | Technology | Key Files |
|-------|------------|-----------|
| Backend | FastAPI + Python 3.11+ | `src/api/main.py` |
| AI Orchestration | LangGraph + LangChain | `src/agents/supervisor_agent.py` |
| Vector DB | Qdrant Cloud / pgvector transition | `ai/rag/knowledge_base.py` |
| Graph DB | Neo4j | `ai/rag/graph_retriever.py` |
| Primary DB | PostgreSQL / Aurora | `src/db/postgres/` |
| LLM Provider Layer | Groq, vLLM, Together (Bedrock legacy removal planned) | `src/orchestrator/llm_provider.py` |
| Scraping | Scrapling + Playwright | `src/scrapers/` |
| Voice | Duplex WebSocket + hybrid STT/TTS; Pipecat experimental | `src/api/websocket/voice_pkg/`, `src/voice/` |
| Caching | Redis | `src/shared/cache.py` |
| Package Manager | uv | `pyproject.toml` |

---

## Standard AI Prompts for Common Tasks

### Implementing a new endpoint

```text
"Read PLAN.md, tracking/PROJECT_STATUS.md, and the relevant sprint file.
Implement [endpoint name] in [file_path].
Follow our coding standards in docs/architecture/coding-standards.md.
Generate unit tests in tests/unit/ as part of this change.
Update WORKFLOW_STATUS.md with the new file in the changes log."
```

### Generating tests for existing code

```text
"Read TESTING/STRATEGY.md for our test philosophy.
Generate unit tests for [FunctionName] in [file_path].
Cover: happy path, edge cases, error cases.
Use pytest and mock external dependencies.
Add descriptive docstrings to each test."
```

### End-of-sprint review

```text
"Read the active sprint file and the last 5 daily logs.
For voice work, start with tracking/sprints/sprint-07-voice-duplex-productionization.md.
Summarize: what shipped, what slipped, 3 key learnings.
Format as the Sprint Outcome section and update PROJECT_STATUS.md."
```

### Planning next sprint

```text
"Based on PLAN.md, ROADMAP.md, and the latest sprint outcome,
propose the next sprint tasks for [theme].
Format into the sprint template at tracking/sprints/_template.md.
Keep scope realistic for a 2-week solo sprint."
```

---

## Things AI Agents Should NOT Do

- Overwrite or truncate existing sprint outcomes or daily logs
- Skip writing tests when implementing a new feature
- Modify `ai/rag/` pipeline files without reading eval results first
- Use `print()` instead of structured logging
- Make large refactors outside of a designated refactor sprint
- Create new files without updating `WORKFLOW_STATUS.md`
