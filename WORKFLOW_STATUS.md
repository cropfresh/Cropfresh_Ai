# CropFresh AI — Development Workflow & Status Guide

> **Last Updated:** 2026-02-27 (15:01 IST)
> **Package Manager:** uv | **Python:** 3.11+ | **Stack:** FastAPI + LangGraph + Qdrant

This document is the **single entry point** for understanding how CropFresh AI is developed. It covers the development philosophy, workflow loop, documentation structure, and a running file changes log. AI agents should read this alongside `AGENTS.md` before starting any work.

---

## 📂 Documentation Map

```
/ (repo root)
  PLAN.md                      ← Master product + architecture plan (start here)
  ROADMAP.md                   ← Phase milestones (Feb–Aug 2026) 
  AGENTS.md                    ← AI agent rules, prompts, file structure map
  WORKFLOW_STATUS.md           ← This file: dev workflow + file changes log
  CHANGELOG.md                 ← Version history
  
  tracking/
    PROJECT_STATUS.md          ← Current state (update every sprint)
    sprints/sprint-0X-*.md     ← Each sprint's goals + outcomes
    daily/YYYY-MM-DD.md        ← Per-session work logs
    milestones/                ← Phase completion records  
    retros/                    ← Sprint retrospective notes
  
  docs/
    decisions/ADR-*.md         ← Architecture Decision Records
    agents/REGISTRY.md         ← Agent specs and prompt versions
    api/                       ← API reference docs
    architecture/              ← System architecture docs
  
  TESTING/
    STRATEGY.md                ← Test philosophy, pyramid, AI prompts
    CHECKLISTS.md              ← Per-feature done checklists
```

---

## 🧭 Core Development Principles

**1. Specs Before Code**  
Always read `PLAN.md`, `tracking/PROJECT_STATUS.md`, and the active sprint file before writing any code. Context-first development produces better results with AI agents.

**2. Documentation as Code**  
All docs live in the repo (`docs/`, `tracking/`, `TESTING/`), versioned in Git alongside source. Update docs in the same commit as the code changes.

**3. Small, Reviewable Changes**  
One endpoint, one agent, one refactor per change. Use branches for risky experiments. Even as a solo founder, review your own diffs before committing.

**4. Tests in the Loop**  
Ask AI to generate tests alongside every function. Run the test checklist from `TESTING/CHECKLISTS.md` before marking any task done.

**5. Continuous Refinement**  
After each sprint, update goals, outcomes, and documents so AI has the latest context. Stale docs produce hallucinations.

---

## 🔄 Sprint Workflow Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    CropFresh Dev Loop                           │
│                                                                 │
│  Step 1: Refine PLAN.md + ROADMAP.md (once per phase)          │
│    └─→ AI prompt: "Act as senior architect, review my plan…"   │
│                                                                 │
│  Step 2: Create sprint-XXX.md (start of each sprint)           │
│    └─→ AI prompt: "Based on PLAN.md + last sprint, plan 2-wk…" │
│                                                                 │
│  Step 3: Daily Execution                                        │
│    ├─→ Update tracking/daily/YYYY-MM-DD.md each session        │
│    ├─→ Implement with AI: spec → code → tests → docs           │
│    └─→ Commit with message: "Sprint-04: [task name]"           │
│                                                                 │
│  Step 4: Testing in Loop                                        │
│    ├─→ AI generates tests for every function/endpoint           │
│    └─→ AI reviews diff for bugs before merging                 │
│                                                                 │
│  Step 5: End-of-Sprint Review                                   │
│    ├─→ Fill sprint-XXX.md "Outcome" section                    │
│    ├─→ Update PROJECT_STATUS.md                                │
│    ├─→ Create ADRs for any architecture decisions              │
│    └─→ Git tag milestone releases                              │
│                                                                 │
│  Step 6: Refine Next Sprint                                     │
│    ├─→ Move unfinished but important tasks forward             │
│    ├─→ Adjust ROADMAP.md if macro milestones slip              │
│    └─→ Start back at Step 2                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💬 Standard AI Prompts (Copy-Paste Ready)

### 🏗️ Start of Sprint
```
"Here is my PLAN.md and ROADMAP.md. Based on last sprint's outcome in
tracking/sprints/sprint-03-*.md, propose a 2-week sprint plan with
5–7 concrete, testable tasks focused on [theme].
Format it into the sprint template at tracking/sprints/_template.md."
```

### ⚙️ Implement a Feature
```
"Read PLAN.md, tracking/PROJECT_STATUS.md, and tracking/sprints/sprint-04-voice-pipeline.md.
Implement [feature/endpoint name] in [file path].
Follow coding standards in docs/architecture/coding-standards.md.
Generate unit tests in tests/unit/ as part of this change.
Update WORKFLOW_STATUS.md with the new file in the changes log."
```

### 🧪 Generate Tests
```
"Read TESTING/STRATEGY.md for our test philosophy.
Generate pytest unit tests for [FunctionName] in [file path].
Cover: happy path, edge cases, and error cases.
Mock all external dependencies (Qdrant, Groq, Redis).
Add descriptive docstrings to each test."
```

### 🔍 Pre-merge Review
```
"Analyze this diff and identify:
- Functions or endpoints missing tests
- Unhandled error cases or missing validation
- Potential regressions
Reference TESTING/STRATEGY.md checklist and our coding standards."
```

### 📊 End-of-Sprint Summary
```
"Read tracking/sprints/sprint-04-voice-pipeline.md and the last 5 daily logs.
Summarize: what shipped, what slipped, 3 key learnings.
Format as the Sprint Outcome section and also update tracking/PROJECT_STATUS.md."
```

### 📐 Architecture Decision
```
"I need to decide [decision topic]. Context: [brief context].
Options: [option A], [option B].
Analyze the tradeoffs for a solo founder building CropFresh AI.
Format your recommendation as an ADR using docs/decisions/_template.md."
```

---

## 🗂️ Keeping History (Git-Centric Approach)

The user expressed concern about AI overwriting and losing history. Here is how this is prevented:

### Rule 1: Append, Never Overwrite Logs
- Sprint files: fill in the **Outcome section at sprint end**, never delete the goals section
- Daily logs: create a **new file** per day (`tracking/daily/YYYY-MM-DD.md`)
- PROJECT_STATUS.md: **update in place** but Git preserves all prior versions

### Rule 2: Meaningful Commits
```bash
git commit -m "Sprint-04: pipecat_bot.py WebSocket integration + unit tests"
git commit -m "Sprint-04: APMC scraper with Redis cache + APScheduler"
git commit -m "Docs: ADR-007 - chose Pipecat over raw WebSocket for voice"
```

### Rule 3: Tag Milestones
```bash
git tag -a v0.3-foundation -m "Phase 1 foundation complete"
git tag -a v0.4-mvp -m "MVP: farmer listing + voice + price query"
```

### Rule 4: Branch for Risky Changes
```bash
git checkout -b feature/sprint-04-apmc-scraper
# work, commit, test
git checkout main && git merge feature/sprint-04-apmc-scraper
```

---

## 📊 Current Component Status

| Component | Status | Progress | Sprint |
|-----------|--------|----------|--------|
| Project Structure | ✅ Complete | 100% | Sprint 01 |
| RAG Pipeline (RAPTOR + Hybrid) | ✅ Complete | 100% | Sprint 01 |
| Multi-Agent System | ✅ Complete | 100% | Sprint 01 |
| Memory System (Redis) | ✅ Complete | 100% | Sprint 01 |
| Voice Agent v1 (Edge-TTS + Whisper) | ✅ Complete | 90% | Sprint 01 |
| Pipecat Voice Pipeline | 🟡 In Progress | 40% | Sprint 04 |
| APMC Mandi Scraper | ❌ Not Started | 0% | Sprint 04 |
| Supabase Schema | ❌ Not Started | 0% | Sprint 05 |
| Vision Agent (YOLOv12 + DINOv2) | ❌ Not Started | 0% | Phase 3 |
| Evaluation Framework (LangSmith) | ❌ Not Started | 0% | Sprint 05 |
| Flutter Mobile App | ❌ Not Started | 0% | Phase 4 |

---

## 🚀 Quick Start

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 2. Populate Knowledge Base  
uv run python scripts/populate_qdrant.py

# 3. Run the service
uv run uvicorn src.api.main:app --reload --port 8000

# 4. Open Swagger UI
# http://localhost:8000/docs

# 5. Run tests
uv run pytest -v

# 6. Type checking + linting
uv run mypy src/
uv run ruff check src/
```

---

## 🌐 API Endpoints Reference

### Chat API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Multi-turn conversation with agent routing |
| `/api/v1/chat/stream` | POST | SSE streaming responses |
| `/api/v1/chat/session` | POST | Create new session |
| `/api/v1/chat/agents` | GET | List available agents |

### Voice API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/voice/process` | POST | Full voice-in → voice-out |
| `/api/v1/voice/transcribe` | POST | Audio → Text (STT) |
| `/api/v1/voice/synthesize` | POST | Text → Audio (TTS) |
| `/ws/voice/{user_id}` | WebSocket | Real-time streaming (Pipecat) |

### RAG API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Query knowledge base |
| `/api/v1/search` | POST | Semantic search |
| `/api/v1/ingest` | POST | Ingest documents |

### Health
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |

---

## 📝 File Changes Log

### 2026-02-27 — Workflow Documentation System

| Action | File | Description |
|--------|------|-------------|
| CREATE | `ROADMAP.md` | 6-phase milestone roadmap (Feb–Aug 2026) |
| UPDATE | `PLAN.md` | Full architecture diagram, user flows, NFRs, risk register, tech stack |
| UPDATE | `AGENTS.md` | Complete AI agent instructions: dev rules, file map, prompts, do-not-do list |
| UPDATE | `WORKFLOW_STATUS.md` | This file — comprehensive dev workflow + methodology |
| CREATE | `tracking/PROJECT_STATUS.md` | Always-current project state dashboard |
| CREATE | `tracking/sprints/sprint-04-voice-pipeline.md` | Current sprint: Pipecat + APMC scraping |
| CREATE | `tracking/daily/_template.md` | Daily log template |
| CREATE | `tracking/daily/2026-02-27.md` | Today's session log |
| CREATE | `TESTING/STRATEGY.md` | Test pyramid, philosophy, AI prompts for testing |
| CREATE | `TESTING/CHECKLISTS.md` | Per-feature-type done checklists |

### 2026-02-27 — Production Scraping Upgrade (Earlier Session)

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/scrapers/base_scraper.py` | Scrapling-based browser scraper with Camoufox stealth |
| UPDATE | `src/scrapers/apmc/` | Production-grade APMC scraper upgrade |
| UPDATE | `src/pipelines/` | Data pipeline with caching + APScheduler |

### 2026-02-26 — Voice Domain Separation

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/voice/pipecat/` | Pipecat submodule (STT, TTS services) |
| CREATE | `src/voice/pipecat/stt_service.py` | IndicWhisper STT + Groq fallback |
| CREATE | `src/voice/pipecat/tts_service.py` | Edge-TTS in Kannada/Hindi/English |
| CREATE | `src/voice/pipecat/__init__.py` | Module init with exports |
| CREATE | `src/voice/pipecat_bot.py` | Main Pipecat bot entry point |
| CREATE | `src/api/websocket/voice_ws.py` | WebSocket handler for Pipecat stream |

### 2026-02-26 — Domain Separation (RAG, Vision, Voice)

| Action | File | Description |
|--------|------|-------------|
| RESTRUCTURE | `ai/rag/` | Advanced RAG pipeline moved to dedicated AI folder |
| RESTRUCTURE | `ai/vision/` | Vision domain separated (YOLOv12 + DINOv2 placeholder) |
| RESTRUCTURE | `src/voice/` | Voice domain with dedicated agent |
| UPDATE | `ai/__init__.py` | Unified exports for all AI domains |

### 2026-02-27 — Advanced Folder Structure (Earliest Session)

| Action | File | Description |
|--------|------|-------------|
| CREATE | `ai/` | New top-level AI module (rag, vision, evaluations) |
| CREATE | `tracking/` | Development tracking (goals, sprints, daily, milestones) |
| CREATE | `infra/` | Deployment & monitoring configs |
| CREATE | `config/` | Database & service configurations |
| MOVE | `src/rag/` → `ai/rag/` | RAG pipeline moved to ai/ |

### January 10, 2026 — Advanced RAG Phase 1–4

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/tools/enam_client.py` | eNAM API client for live mandi prices |
| CREATE | `src/tools/imd_weather.py` | IMD Weather client with agro-advisories |
| CREATE | `src/rag/raptor.py` | RAPTOR hierarchical tree indexing with GMM clustering |
| CREATE | `src/rag/contextual_chunker.py` | Contextual chunking with entity extraction |
| CREATE | `src/rag/query_processor.py` | HyDE, multi-query, step-back, decomposition |
| CREATE | `src/rag/enhanced_retriever.py` | Parent Document, Sentence Window, MMR |
| CREATE | `src/rag/hybrid_search.py` | BM25 sparse + RRF fusion hybrid search |
| CREATE | `src/rag/reranker.py` | Cross-encoder reranking with MiniLM fallback |
| CREATE | `src/rag/graph_retriever.py` | Neo4j Graph RAG with entity extraction |
| CREATE | `src/rag/observability.py` | LangSmith tracing + RAG eval metrics |

### January 9, 2026 — Multi-Agent System + Voice v1

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/memory/state_manager.py` | Conversation memory + Redis session manager |
| CREATE | `src/agents/supervisor_agent.py` | LLM query routing with 0.9 confidence threshold |
| CREATE | `src/agents/agronomy_agent.py` | Crop cultivation, pest management, farming advice |
| CREATE | `src/agents/commerce_agent.py` | Market prices, AISP calculations, sell/hold |
| CREATE | `src/agents/platform_agent.py` | CropFresh app features, registration, support |
| CREATE | `src/agents/general_agent.py` | Greetings, fallback for unclear queries |
| CREATE | `src/voice/stt.py` | IndicWhisper STT + Groq fallback |
| CREATE | `src/voice/tts.py` | IndicTTS + Edge-TTS + gTTS |
| CREATE | `src/api/routes/chat.py` | Multi-turn chat + SSE streaming |

---

## ⚠️ Known Issues & Workarounds

### BGE-M3 Embedding Model Memory
Requires ~1GB RAM. Use `MiniLM-L6-v2` (90MB) for low-memory deployments.
```
Workaround: Set EMBEDDING_MODEL=minilm in .env
```

### Qdrant Client Compatibility
Use `query_points` not deprecated `search` for Qdrant 1.7+.
Fixed in: `ai/rag/knowledge_base.py`

### Pipecat WebSocket on Windows
Pipecat requires event loop policy adjustment on Windows:
```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

---

## 🛠️ Common Commands

```bash
# Start all required services
docker start qdrant

# Run full test suite
uv run pytest -v --cov=src --cov-report=term-missing

# Run multi-agent test
uv run python scripts/test_multi_agent.py

# Run RAG enhancements test
uv run python scripts/test_rag_enhancements.py

# Populate Qdrant knowledge base
uv run python scripts/populate_qdrant.py

# Type check
uv run mypy src/

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

---

_This file is the companion to `AGENTS.md`. Together they are the complete onboarding for any AI agent or developer joining CropFresh AI._
