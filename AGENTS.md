# AGENTS.md — AI Agent Instructions for CropFresh

> **Read this before starting any work on CropFresh AI.**
> This is the authoritative context file for Claude Code, Antigravity, and any AI coding agent.
> Last Updated: 2026-02-27

---

## 🧠 Project Context

**CropFresh AI** is India's intelligent agricultural marketplace AI service.
- Connects Karnataka farmers with buyers using multi-agent AI
- Voice-first Kannada interaction (STT + TTS via Pipecat)
- Real-time mandi price data (APMC scraping + Qdrant RAG)
- Built with: FastAPI + LangGraph + Qdrant + Neo4j + Supabase

---

## 📋 Before Starting Any Work — Read These First

1. **`PLAN.md`** — Vision, architecture, core flows, NFRs, tech stack
2. **`tracking/PROJECT_STATUS.md`** — What is true right now, top priorities, blockers
3. **`tracking/sprints/sprint-04-voice-pipeline.md`** — Current sprint tasks and scope
4. **`TESTING/STRATEGY.md`** — What "done" means for each feature type

---

## 🚦 Development Rules

1. **Specs before code** — Always read the relevant sprint file and PLAN.md before generating code
2. **Small, focused changes** — One endpoint, one agent, one function per change
3. **Tests are mandatory** — Every new function/endpoint ships with unit tests (see `TESTING/STRATEGY.md`)
4. **Docs in same commit** — Update `WORKFLOW_STATUS.md` and API docs alongside code
5. **No overwriting history** — Append to docs; use Git for history (never delete sprint outcomes)
6. **ADRs for major decisions** — Create `docs/decisions/ADR-XXX.md` for any significant tech choices
7. **Structured logging** — Use `src/shared/logger.py` for all logging, never bare `print()`
8. **Base class compliance** — All agents extend `src/agents/base_agent.py`
9. **RAG pipeline immutable** — Do not modify `ai/rag/` without checking eval results first

---

## 📂 File Structure Summary

```
📁 CropFresh AI Root
├── PLAN.md                        ← Master product + tech plan (start here)
├── ROADMAP.md                     ← Phase milestones and timeline
├── AGENTS.md                      ← This file — AI agent instructions
├── WORKFLOW_STATUS.md             ← Development workflow guide + file changes log
├── CHANGELOG.md                   ← Version-by-version changes
│
├── tracking/
│   ├── PROJECT_STATUS.md          ← Current state (always up-to-date)
│   ├── sprints/sprint-0X-*.md     ← Active sprint tasks
│   └── daily/YYYY-MM-DD.md        ← Per-session work logs
│
├── docs/
│   ├── decisions/ADR-*.md         ← Architecture Decision Records
│   ├── agents/REGISTRY.md         ← Agent specs and prompt versions
│   ├── api/                       ← API reference
│   └── architecture/              ← Architecture docs
│
├── TESTING/
│   ├── STRATEGY.md                ← Test pyramid and philosophy
│   └── CHECKLISTS.md              ← Per-feature-type done checklists
│
├── src/                           ← Application source code
│   ├── agents/                    ← All AI agents
│   ├── api/                       ← FastAPI routes and WebSocket
│   ├── voice/                     ← Pipecat STT/TTS pipeline
│   ├── scrapers/                  ← APMC and web scrapers
│   └── shared/                    ← Logger, config, utilities
│
├── ai/                            ← ML: RAG, evaluations, training
├── tests/                         ← Test infrastructure
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── scripts/                       ← Automation utilities
```

---

## 🤖 Tech Stack Quick Reference

| Layer | Technology | Key Files |
|-------|-----------|-----------|
| Backend | FastAPI + Python 3.11+ | `src/api/main.py` |
| AI Orchestration | LangGraph + LangChain | `src/agents/supervisor_agent.py` |
| Vector DB | Qdrant Cloud | `ai/rag/knowledge_base.py` |
| Graph DB | Neo4j | `ai/rag/graph_retriever.py` |
| Primary DB | Supabase (PostgreSQL) | `config/database/` |
| LLM | Groq (Llama-3.3-70B) | `src/agents/base_agent.py` |
| Scraping | Scrapling + Playwright | `src/scrapers/` |
| Voice | Pipecat + Edge-TTS + Whisper | `src/voice/` |
| Caching | Redis | `src/shared/cache.py` |
| Package Manager | uv | `pyproject.toml` |

---

## 💬 Standard AI Prompts for Common Tasks

### Implementing a new endpoint
```
"Read PLAN.md, tracking/PROJECT_STATUS.md, and the current sprint file.
Implement [endpoint name] in [file_path].
Follow our coding standards in docs/architecture/coding-standards.md.
Generate unit tests in tests/unit/ as part of this change.
Update WORKFLOW_STATUS.md with the new file in the changes log."
```

### Generating tests for existing code
```
"Read TESTING/STRATEGY.md for our test philosophy.
Generate unit tests for [FunctionName] in [file_path].
Cover: happy path, edge cases, error cases.
Use pytest, mock external dependencies (Qdrant, Groq API, Redis).
Add descriptive docstrings to each test."
```

### End-of-sprint review
```
"Read tracking/sprints/sprint-04-voice-pipeline.md and the last 5 daily logs.
Summarize: what shipped, what slipped, 3 key learnings.
Format as the Sprint Outcome section and update PROJECT_STATUS.md."
```

### Planning next sprint
```
"Based on PLAN.md, ROADMAP.md, and the sprint-04 outcome,
propose sprint-05 tasks for [theme].
Format into the sprint template at tracking/sprints/_template.md.
Keep scope realistic for a 2-week solo sprint."
```

---

## 🚫 Things AI Agents Should NOT Do

- Overwrite or truncate existing sprint outcomes or daily logs
- Skip writing tests when implementing a new feature
- Modify `ai/rag/` pipeline files without reading eval results first
- Use `print()` instead of structured logger
- Make large refactors outside of a designated refactor sprint
- Create new files without updating `WORKFLOW_STATUS.md`
