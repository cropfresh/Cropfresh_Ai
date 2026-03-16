# CropFresh AI — Development Workflow & Status Guide

> **Last Updated:** 2026-03-16 (13:14 IST)
> **Package Manager:** uv | **Python:** 3.11+ | **Stack:** FastAPI + LangGraph + Qdrant Cloud + Neo4j AuraDB + Redis Labs

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

| Component                           | Status         | Progress | Sprint    |
| ----------------------------------- | -------------- | -------- | --------- |
| Project Structure                   | ✅ Complete    | 100%     | Sprint 01 |
| RAG Pipeline (RAPTOR + Hybrid)      | ✅ Complete    | 100%     | Sprint 01 |
| Multi-Agent System                  | ✅ Complete    | 100%     | Sprint 01 |
| Memory System (Redis)               | ✅ Complete    | 100%     | Sprint 01 |
| Voice Agent v1 (Edge-TTS + Whisper) | ✅ Complete    | 90%      | Sprint 01 |
| Pipecat Voice Pipeline              | 🟡 In Progress | 40%      | Sprint 04 |
| APMC Mandi Scraper                  | ❌ Not Started | 0%       | Sprint 04 |
| Supabase Schema                     | ❌ Not Started | 0%       | Sprint 05 |
| Vision Agent (YOLOv12 + DINOv2)     | ❌ Not Started | 0%       | Phase 3   |
| Evaluation Framework (LangSmith)    | ❌ Not Started | 0%       | Sprint 05 |
| Flutter Mobile App                  | ❌ Not Started | 0%       | Phase 4   |

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

| Endpoint               | Method | Description                                |
| ---------------------- | ------ | ------------------------------------------ |
| `/api/v1/chat`         | POST   | Multi-turn conversation with agent routing |
| `/api/v1/chat/stream`  | POST   | SSE streaming responses                    |
| `/api/v1/chat/session` | POST   | Create new session                         |
| `/api/v1/chat/agents`  | GET    | List available agents                      |

### Voice API

| Endpoint                   | Method    | Description                   |
| -------------------------- | --------- | ----------------------------- |
| `/api/v1/voice/process`    | POST      | Full voice-in → voice-out     |
| `/api/v1/voice/transcribe` | POST      | Audio → Text (STT)            |
| `/api/v1/voice/synthesize` | POST      | Text → Audio (TTS)            |
| `/ws/voice/{user_id}`      | WebSocket | Real-time streaming (Pipecat) |

### RAG API

| Endpoint         | Method | Description          |
| ---------------- | ------ | -------------------- |
| `/api/v1/query`  | POST   | Query knowledge base |
| `/api/v1/search` | POST   | Semantic search      |
| `/api/v1/ingest` | POST   | Ingest documents     |

### Health

| Endpoint        | Method | Description     |
| --------------- | ------ | --------------- |
| `/health`       | GET    | Health check    |
| `/health/ready` | GET    | Readiness check |

---

## 📝 File Changes Log

### 2026-03-16 - Kannada-Aware RAG Prompting

| Action | File                                      | Description                                                                  |
| ------ | ----------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/rag/language_support.py`             | Added Kannada/Kanglish language detection plus localized RAG fallback text  |
| UPDATE | `src/rag/agentic/speculative.py`          | Injected Kannada-aware response instructions into speculative RAG drafting   |
| UPDATE | `src/rag/graph_runtime/services.py`       | Localized no-doc and extractive fallback answers for Kannada RAG queries     |
| UPDATE | `src/rag/graph_runtime/nodes.py`          | Localized generation error fallback and threaded route into answer builder   |
| UPDATE | `src/rag/confidence_gate.py`              | Returned Kannada decline messages for safety/platform abstentions            |
| UPDATE | `src/rag/query_rewriter_prompts.py`       | Added Kannada and Kanglish retrieval guidance to rewrite prompts             |
| CREATE | `tests/unit/test_rag_language_support.py` | Added unit coverage for Kannada prompt injection and fallback behavior       |
| UPDATE | `tests/unit/test_confidence_gate.py`      | Added Kannada-localized decline response regression test                     |
| UPDATE | `WORKFLOW_STATUS.md`                      | Added this entry and refreshed the last-updated timestamp                    |

### 2026-03-16 - Live RAG Benchmark Lift


| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/routing/models.py`                     | Extracted routing enums and decision models from legacy `query_analyzer.py`  |
| CREATE | `ai/rag/routing/prefilter.py`                  | Rule-based router prefilters and Kannada-aware fallback routing              |
| CREATE | `ai/rag/routing/classifier.py`                 | LLM routing and query-analysis helpers                                       |
| CREATE | `ai/rag/routing/router.py`                     | Thin router/analyzer orchestration layer                                     |
| CREATE | `ai/rag/routing/__init__.py`                   | Routing package exports                                                      |
| UPDATE | `ai/rag/query_analyzer.py`                     | Converted oversized module into compatibility re-export layer                |
| CREATE | `src/agents/knowledge_models.py`               | Typed response models for user-facing and benchmark/debug knowledge output    |
| CREATE | `src/agents/knowledge_mapping.py`              | Source-detail and citation extraction helpers for benchmark runs             |
| CREATE | `src/agents/knowledge_runtime.py`              | Benchmark embedding toggle helper for `KnowledgeAgent`                       |
| UPDATE | `src/agents/knowledge_agent.py`                | Added `answer_with_debug()` while keeping `answer()` contract stable         |
| CREATE | `src/rag/export_map.py`                        | Added lazy export registry so `src.rag` submodules can load without cycles   |
| UPDATE | `src/rag/__init__.py`                          | Replaced eager package imports with lazy export resolution                   |
| UPDATE | `src/rag/query_analyzer.py`                    | Made `src.rag` the app-facing routing facade over the split adaptive router  |
| UPDATE | `src/rag/grader.py`                            | Made `src.rag` the app-facing grading facade over the new `src/rag/grading/` package |
| UPDATE | `src/rag/export_map.py`                        | Expanded lazy exports to include adaptive router and enhanced grader symbols |
| UPDATE | `src/rag/graph.py`                             | Replaced monolith with compatibility facade over `ai.rag.graph`              |
| MOVE   | `ai/rag/routing/` -> `src/rag/routing/`        | Consolidated adaptive router implementation into the canonical `src/rag` tree |
| MOVE   | `ai/rag/retrieval/` -> `src/rag/retrieval/`    | Consolidated advanced retrieval helpers into `src/rag`                       |
| MOVE   | `ai/rag/agentic/` -> `src/rag/agentic/`        | Consolidated agentic RAG internals into `src/rag`                            |
| MOVE   | `ai/rag/browser_rag_pkg/` -> `src/rag/browser_rag_pkg/` | Consolidated browser-augmented RAG internals into `src/rag`          |
| MOVE   | `ai/rag/graph/` -> `src/rag/graph_runtime/`    | Moved LangGraph runtime internals under `src/rag` without colliding with `src/rag/graph.py` |
| MOVE   | `ai/rag/evaluation/` -> `src/rag/benchmark/`   | Moved live benchmark pipeline under `src/rag` without colliding with legacy `src/rag/evaluation.py` |
| MOVE   | `ai/rag/agri_embeddings.py` -> `src/rag/agri_embeddings.py` | Moved AgriEmbeddingWrapper into canonical `src/rag` surface      |
| MOVE   | `ai/rag/citation_engine.py` -> `src/rag/citation_engine.py` | Moved citation engine into canonical `src/rag` surface           |
| MOVE   | `ai/rag/confidence_gate.py` -> `src/rag/confidence_gate.py` | Moved confidence gate into canonical `src/rag` surface           |
| MOVE   | `ai/rag/query_rewriter.py` -> `src/rag/query_rewriter.py` | Moved query rewriting into canonical `src/rag` surface            |
| MOVE   | `ai/rag/query_rewriter_prompts.py` -> `src/rag/query_rewriter_prompts.py` | Moved query rewriting prompts into `src/rag`        |
| MOVE   | `ai/rag/browser_rag.py` -> `src/rag/browser_rag.py` | Moved browser RAG redirect into canonical `src/rag` surface     |
| MOVE   | `ai/rag/agentic_orchestrator.py` -> `src/rag/agentic_orchestrator.py` | Moved agentic orchestrator redirect into `src/rag` |
| MOVE   | `ai/rag/knowledge_base/` -> `src/rag/knowledge_base_data/` | Moved RAG knowledge assets under the canonical `src/rag` tree   |
| CREATE | `src/rag/grading/models.py`                    | Extracted grading models and constants from the oversized enhanced grader     |
| CREATE | `src/rag/grading/relevance.py`                 | Extracted relevance scoring and freshness-aware grading logic                  |
| CREATE | `src/rag/grading/hallucination.py`             | Extracted hallucination checking logic                                         |
| CREATE | `src/rag/grading/__init__.py`                  | Public package exports for the new canonical grading package                  |
| CREATE | `src/rag/agri_terms.py`                        | Extracted the large bilingual agri term map from `agri_embeddings.py`         |
| CREATE | `ai/rag/export_map.py`                         | Added lazy export registry for combined `ai.rag` package exports             |
| UPDATE | `ai/rag/__init__.py`                           | Reduced `ai.rag` to a compatibility-only namespace over the canonical `src.rag` modules |
| UPDATE | `ai/rag/export_map.py`                         | Redirected compatibility exports to `src.rag.*` instead of local `ai.rag` implementations |
| DELETE | `ai/rag/advanced_reranker.py`                  | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/contextual_chunker.py`                 | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/embeddings.py`                         | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/evaluation.py`                         | Removed shadowing module in favor of the real moved benchmark package       |
| DELETE | `ai/rag/graph.py`                              | Removed shadowing module in favor of the moved graph runtime package        |
| DELETE | `ai/rag/graph_constructor.py`                  | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/graph_retriever.py`                    | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/graph_store.py`                        | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/hybrid_search.py`                      | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/knowledge_base.py`                     | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/knowledge_injection.py`                | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/observability.py`                      | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/production.py`                         | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/raptor.py`                             | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/reranker.py`                           | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/retriever.py`                          | Removed duplicate top-level proxy to reduce `ai/rag` confusion              |
| DELETE | `ai/rag/grader.py`                             | Removed the old enhanced grader module after moving logic into `src/rag/grading/` |
| DELETE | `ai/rag/query_analyzer.py`                     | Removed the old query analyzer shim after moving routing ownership into `src/rag` |
| DELETE | `ai/rag/query_processor.py`                    | Removed redundant query processor proxy after consolidating on `src/rag/query_processor/` |
| DELETE | `ai/rag/enhanced/`                             | Removed unused duplicate enhanced RAG package after consolidating on `src/rag/enhanced/` |
| DELETE | `ai/rag/enhanced_retriever/`                   | Removed unused duplicate enhanced retriever package after consolidating on `src/rag/enhanced_retriever/` |
| CREATE | `ai/rag/graph/services.py`                     | Shared retrieval/generation helpers for graph runtime                        |
| UPDATE | `ai/rag/graph/state.py`                        | Added route, tool-call, and document fields for benchmark observability      |
| UPDATE | `ai/rag/graph/nodes.py`                        | Routed retrieval through adaptive router and shared services                 |
| UPDATE | `ai/rag/graph/nodes_safety.py`                 | Enabled injected web search and retained gate/evaluator flow                 |
| UPDATE | `ai/rag/graph/builder.py`                      | Fixed final answer mapping so declined answers surface correctly             |
| CREATE | `ai/rag/evaluation/models.py`                  | JSON dataset entry, resolved reference, and live-run extra models            |
| CREATE | `ai/rag/evaluation/dataset_loader.py`          | Loader for `core_live` and `full` JSON benchmark datasets                    |
| CREATE | `ai/rag/evaluation/reference_resolver.py`      | Static/live reference resolver with Agmarknet freshness checks               |
| CREATE | `ai/rag/evaluation/pipeline_adapter.py`        | Benchmark adapter for the canonical `KnowledgeAgent` runtime path            |
| CREATE | `ai/rag/evaluation/live_runner.py`             | Semantic benchmark runner using `src.evaluation.ragas_evaluator`             |
| CREATE | `ai/rag/evaluation/runtime.py`                 | Benchmark runtime factory for `KnowledgeAgent` + configured LLM              |
| UPDATE | `ai/rag/evaluation/golden_dataset.py`          | Converted dataset entrypoint to JSON-backed loader helpers                   |
| UPDATE | `ai/rag/evaluation/guardrail.py`               | Added explicit `live` and `heuristic` guardrail modes over real pipeline     |
| UPDATE | `ai/rag/evaluation/reporting.py`               | Writes semantic reports plus guardrail and extras artifacts                  |
| UPDATE | `ai/rag/evaluation/__init__.py`                | Refreshed package exports for new benchmark modules                          |
| CREATE | `ai/rag/evaluation/datasets/core_live.json`    | Phase-1 benchmark subset for market, agronomy, pest, scheme, and Kannada     |
| CREATE | `ai/rag/evaluation/datasets/full.json`         | Full regression dataset including deferred weather and multi-hop items       |
| UPDATE | `scripts/eval_guardrail.py`                    | Added ASCII-safe CLI with `--live`, `--heuristic`, `--subset`, and `--runs`  |
| UPDATE | `docs/architecture/rag-benchmark-baseline.md`  | Reframed baseline doc around live vs heuristic benchmark modes               |
| UPDATE | `tests/unit/test_evaluation.py`                | Replaced canned-answer tests with live guardrail and JSON dataset coverage   |
| CREATE | `tests/unit/test_routing.py`                   | Routing regression tests for Kannada and compatibility exports               |
| CREATE | `tests/unit/test_benchmark_dataset.py`         | Dataset loader and reference resolver unit tests                             |
| CREATE | `tests/unit/test_knowledge_agent_debug.py`     | `KnowledgeAgent.answer_with_debug()` unit coverage                           |
| CREATE | `tests/unit/test_rag_module_boundaries.py`     | Verifies `src.rag` is the public surface and duplicate `ai/rag` files stay removed |
| CREATE | `tests/unit/test_rag_graph_edges_extra.py`     | Regression test for single-pass web-search fallback routing                  |
| CREATE | `tests/unit/test_rag_graph_facade.py`          | Compatibility facade mapping tests for `src/rag/graph.py`                    |
| CREATE | `tests/unit/test_guardrail_cli.py`             | CLI summary and ASCII-safety tests                                           |
| UPDATE | `tests/integration/test_rag_integration.py`    | Canonical runtime integration tests with mocked KB and live source adapters  |
| DELETE | `ai/evals/`                                    | Removed unused legacy evaluation scaffold so `src/evaluation` is the single maintained evaluation surface |
| UPDATE | `scripts/run-all-evals.sh`                     | Pointed the evaluation entrypoint at `python -m src.evaluation.eval_runner`  |
| UPDATE | `tests/unit/test_ragas_evaluator.py`           | Cleaned imports while keeping coverage on the canonical `src.evaluation` stack |
| UPDATE | `TESTING/STRATEGY.md`                          | Replaced stale `ai/evaluations` references with `src/evaluation` datasets and commands |

### 2026-03-14 — Phase 1: Anti-Hallucination Pipeline (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/query_rewriter.py`                     | HyDE, step-back, and multi-query expansion strategies (178 LOC)              |
| CREATE | `ai/rag/query_rewriter_prompts.py`             | Prompt templates for query rewriting (74 LOC)                                |
| CREATE | `ai/rag/citation_engine.py`                    | Inline [1], [2] citation markers with source attribution (172 LOC)           |
| CREATE | `ai/rag/confidence_gate.py`                    | "I don't know" safety gate with Kannada keywords (195 LOC)                   |
| UPDATE | `ai/rag/grader.py`                             | Continuous 0–1 scoring, time-decay for stale market docs, Kannada keywords   |
| CREATE | `tests/unit/test_query_rewriter.py`            | 12 tests: strategies, classification, edge cases, LLM fallback              |
| CREATE | `tests/unit/test_citation_engine.py`           | 8 tests: heuristic/LLM citations, source extraction, formatting             |
| CREATE | `tests/unit/test_confidence_gate.py`           | 9 tests: safety classification, grounding, gating, Kannada                   |
| CREATE | `tests/unit/test_grader_enhanced.py`           | 11 tests: continuous scoring, time-decay, batch grading                      |
| CREATE | `docs/planning/advanced-agentic-rag-plan.md`   | Comprehensive 4-phase implementation plan                                    |
| CREATE | `docs/decisions/ADR-010.md`                    | Architecture Decision Record for Advanced Agentic RAG upgrade                |

### 2026-03-14 — Phase 3: LangGraph State Machine (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/graph/state.py`                        | RAG state TypedDict + GraphRunResult Pydantic model (77 LOC)                 |
| CREATE | `ai/rag/graph/nodes.py`                        | 5 pipeline nodes: rewrite, retrieve, grade, generate, cite (199 LOC)         |
| CREATE | `ai/rag/graph/nodes_safety.py`                 | 3 safety nodes: gate, evaluate, web_search (175 LOC)                         |
| CREATE | `ai/rag/graph/edges.py`                        | Conditional edge routing: after_grade, after_evaluate, after_gate (83 LOC)   |
| CREATE | `ai/rag/graph/builder.py`                      | Graph assembly + `run_rag_graph()` public API (143 LOC)                      |
| CREATE | `ai/rag/graph/__init__.py`                     | Package exports (15 LOC)                                                     |
| CREATE | `tests/unit/test_rag_graph.py`                 | 17 tests: edge routing, node functions, graph compilation                    |

### 2026-03-14 — Phase 4: Advanced Retrieval (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/retrieval/contextual_enricher.py`      | Anthropic-style chunk enrichment with section inference (155 LOC)             |
| CREATE | `ai/rag/retrieval/query_decomposer.py`         | Multi-part query splitting with Kannada conjunctions (164 LOC)               |
| CREATE | `ai/rag/retrieval/time_aware.py`               | Freshness-boosted scoring: market/weather/scheme/evergreen (175 LOC)         |
| CREATE | `ai/rag/retrieval/advanced_retriever.py`       | Retrieval coordinator: decompose → retrieve → rerank (130 LOC)               |
| CREATE | `ai/rag/retrieval/__init__.py`                 | Package exports (27 LOC)                                                     |
| CREATE | `tests/unit/test_advanced_retrieval.py`        | 18 tests: enricher, decomposer, time-aware, edge cases                       |

### 2026-03-14 — Phase 5: Evaluation & Guardrails (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/evaluation/metrics.py`                 | RAGAS multi-metric evaluator: faithfulness, relevancy, precision (196 LOC)   |
| CREATE | `ai/rag/evaluation/golden_dataset.py`          | 30 golden queries: market, agronomy, pest, scheme, weather, Kannada          |
| CREATE | `ai/rag/evaluation/guardrail.py`               | CI quality gate with configurable thresholds (154 LOC)                       |
| CREATE | `ai/rag/evaluation/__init__.py`                | Package exports (31 LOC)                                                     |
| CREATE | `scripts/eval_guardrail.py`                    | CLI entry point for CI/CD pipeline (85 LOC)                                  |
| CREATE | `tests/unit/test_evaluation.py`                | 17 tests: evaluator, dataset integrity, guardrail logic                      |

### 2026-03-14 — Phase 6: Documentation, Integration Tests & Benchmark (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| MODIFY | `docs/architecture/data-flow.md`               | Replaced Section 5 with Advanced Agentic RAG architecture (6 subsections)    |
| CREATE | `docs/architecture/rag-benchmark-baseline.md`  | Evaluation benchmark baseline report with heuristic scores                   |
| CREATE | `tests/integration/test_rag_integration.py`    | 10 integration tests: pipeline, LangGraph, evaluation, cross-module          |

### 2026-03-11 — LLM Provider Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/orchestrator/llm_provider/models.py`          | Extracted `LLMMessage` and `LLMResponse` models                              |
| CREATE | `src/orchestrator/llm_provider/base.py`            | Extracted `BaseLLMProvider` interface                                        |
| CREATE | `src/orchestrator/llm_provider/bedrock.py`         | Extracted Amazon Bedrock provider                                            |
| CREATE | `src/orchestrator/llm_provider/groq.py`            | Extracted Groq provider                                                      |
| CREATE | `src/orchestrator/llm_provider/together.py`        | Extracted Together AI provider                                               |
| CREATE | `src/orchestrator/llm_provider/vllm.py`            | Extracted vLLM provider                                                      |
| CREATE | `src/orchestrator/llm_provider/factory.py`         | Extracted `create_llm_provider` factory                                      |
| CREATE | `src/orchestrator/llm_provider/__init__.py`        | Initialized `llm_provider` package                                           |
| UPDATE | `src/orchestrator/llm_provider.py`                 | Reduced (477 -> 24 lines) by converting into an import proxy file            |

### 2026-03-11 — Real-Time Data Manager Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/realtime_data/models.py`             | Extracted health and data models                                             |
| CREATE | `src/scrapers/realtime_data/health.py`             | Extracted `HealthMixin` for status tracking                                  |
| CREATE | `src/scrapers/realtime_data/fetchers.py`           | Extracted `FetchersMixin` for unified API fallback logic                     |
| CREATE | `src/scrapers/realtime_data/manager.py`            | Extracted core `RealTimeDataManager` class and singleton initialization      |
| CREATE | `src/scrapers/realtime_data/__init__.py`           | Initialized the `realtime_data` package                                      |
| UPDATE | `src/scrapers/realtime_data.py`                    | Reduced (480 -> 21 lines) by converting into an import proxy                 |
| UPDATE | `src/tools/realtime_data.py`                       | Reduced (480 -> 21 lines) by converging to the same import proxy             |

### 2026-03-11 — Deep Research Tool Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/tools/deep_research/models.py`                | Extracted tool data models (`PageContent`, `DeepResearchResult`)             |
| CREATE | `src/tools/deep_research/constants.py`             | Extracted timeouts, token limits, system models                              |
| CREATE | `src/tools/deep_research/fetching.py`              | Extracted Jina parallel page fetcher logic                                   |
| CREATE | `src/tools/deep_research/llm.py`                   | Extracted helper for Groq LLM                                                |
| CREATE | `src/tools/deep_research/map_reduce.py`            | Extracted Map-Reduce parallel fact extraction and reduction                  |
| CREATE | `src/tools/deep_research/tool.py`                  | Extracted core `DeepResearchTool` orchestration class                        |
| CREATE | `src/tools/deep_research/__init__.py`              | Initialized tool package                                                     |
| UPDATE | `src/tools/deep_research.py`                       | Reduced (484 -> 54 lines) by converting into an import proxy and registry    |

### 2026-03-11 — Duplex Pipeline Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/duplex/models.py`                       | Extracted pipeline enums and models                                          |
| CREATE | `src/voice/duplex/initializers.py`                 | Extracted component initializers (LLM, STT, TTS)                             |
| CREATE | `src/voice/duplex/processing.py`                   | Extracted core audio rendering & STT logic                                   |
| CREATE | `src/voice/duplex/pipeline.py`                     | Rebuilt `DuplexPipeline` class combining mixins                              |
| CREATE | `src/voice/duplex/__init__.py`                     | Created package public interface                                             |
| UPDATE | `src/voice/duplex_pipeline.py`                     | Reduced (488 -> 17 lines) by converting into an import proxy file            |

### 2026-03-11 — Price Prediction Agent Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/price_prediction/models.py`            | Extracted `PricePrediction` model                                            |
| CREATE | `src/agents/price_prediction/constants.py`         | Extracted seasonal calender and rule thresholds                              |
| CREATE | `src/agents/price_prediction/analysis.py`          | Extracted `AnalysisMixin` with features, rules, trend logic                  |
| CREATE | `src/agents/price_prediction/__init__.py`          | Provided a clean public interface for the agent                              |
| UPDATE | `src/agents/price_prediction/agent.py`             | Reduced (514 -> 241 lines) by importing mixins and models                    |

### 2026-03-11 — Base Agent Modular Refactoring (Protected File Compatible)

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/base/models.py`                        | Extracted `AgentResponse` and `AgentConfig` models                           |
| CREATE | `src/agents/base/retrieval.py`                     | Extracted `RetrievalMixin` for KB search and context formatting              |
| CREATE | `src/agents/base/tools.py`                         | Extracted `ToolMixin` for safe tool execution and tracking                   |
| CREATE | `src/agents/base/llm.py`                           | Extracted `LLMMixin` for memory injection, LLM generation, and retries       |
| CREATE | `src/agents/base/agent.py`                         | Created `BaseAgent` core combining all mixins                                |
| CREATE | `src/agents/base/__init__.py`                      | Exposed base agent components                                                |
| UPDATE | `src/agents/base_agent.py`                         | Reduced (516 -> 15 lines) by converting into an import proxy file            |

### 2026-03-11 — VAD Module Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/vad/models.py`                          | Extracted data models: `VADState`, `VADEvent`, `SpeechSegment`               |
| CREATE | `src/voice/vad/utils.py`                           | Extracted utility functions for silence and audio bytes conversion           |
| CREATE | `src/voice/vad/silero.py`                          | Extracted `SileroVAD` core implementation and logic                          |
| CREATE | `src/voice/vad/bargein.py`                         | Extracted `BargeinDetector` class                                            |
| CREATE | `src/voice/vad/__init__.py`                        | Exposed `SileroVAD`, `BargeinDetector`, models, and utils                    |
| UPDATE | `src/voice/vad.py`                                 | Reduced (527 -> 20 lines) by converting into an import proxy file            |

### 2026-03-11 — AI Kosha Client Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/aikosha/models.py`                   | Extracted data models: `AIKoshaCategory`, `AIKoshaDataset`, search results   |
| CREATE | `src/scrapers/aikosha/catalog.py`                  | Extracted the static hardcoded datasets catalog                              |
| CREATE | `src/scrapers/aikosha/client.py`                   | Extracted `AIKoshaClient` core logic for APIs and catalog search            |
| CREATE | `src/scrapers/aikosha/__init__.py`                 | Exposed `AIKoshaClient`, models, and catalog methods                        |
| UPDATE | `src/scrapers/aikosha_client.py`                   | Reduced (530 -> 19 lines) by converting into an import proxy file            |

### 2026-03-11 — WebRTC Transport Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/webrtc/models.py`                       | Extracted `ConnectionState`, `WebRTCConfig`, `AudioChunk`                    |
| CREATE | `src/voice/webrtc/tracks.py`                       | Extracted `AudioReceiveTrack` and `AudioSendTrack`                           |
| CREATE | `src/voice/webrtc/transport.py`                    | Extracted `WebRTCTransport` core logic                                       |
| CREATE | `src/voice/webrtc/signaling.py`                    | Extracted `WebRTCSignaling` logic                                            |
| CREATE | `src/voice/webrtc/__init__.py`                     | Exposed WebRTC transport models and classes                                  |
| UPDATE | `src/voice/webrtc_transport.py`                    | Reduced (538 -> 24 lines) by converting into an import proxy file            |

### 2026-03-11 — Agri Scrapers Tool Proxy

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| UPDATE | `src/tools/agri_scrapers.py`                       | Reduced (539 -> 26 lines) by converting into an import proxy file            |

### 2026-03-11 — Contextual Chunker Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/rag/contextual_chunker/models.py`             | Extracted data schemas `EnrichedChunk` and `ChunkingConfig`                  |
| CREATE | `src/rag/contextual_chunker/constants.py`          | Extracted `CONTEXT_PROMPT` and `ENTITY_PATTERNS`                            |
| CREATE | `src/rag/contextual_chunker/extractor.py`          | Extracted `ExtractorMixin` for headers, entities, and keywords              |
| CREATE | `src/rag/contextual_chunker/splitter.py`           | Extracted `SplitterMixin` for semantic and simple chunking bounds           |
| CREATE | `src/rag/contextual_chunker/llm_context.py`        | Extracted `LLMContextMixin` to handle LLM completions                        |
| CREATE | `src/rag/contextual_chunker/chunker.py`            | Modularized `ContextualChunker` core using extraction and splitting mixins   |
| CREATE | `src/rag/contextual_chunker/__init__.py`           | Exposed factory methods and schemas                                          |
| UPDATE | `src/rag/contextual_chunker.py`                    | Reduced (553 -> 20 lines) by converting into an import proxy file            |
| UPDATE | `ai/rag/contextual_chunker.py`                     | Reduced (553 -> 20 lines) by converting into an import proxy file            |

### 2026-03-11 — TTS (Voice) Module Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/tts/models.py`                          | Extracted schemas `TTSVoice`, `TTSEmotion`, `SynthesisResult`                |
| CREATE | `src/voice/tts/utils.py`                           | Extracted helper utilities                                                   |
| CREATE | `src/voice/tts/indic.py`                           | Extracted `IndicTTS` implementation for AI4Bharat                            |
| CREATE | `src/voice/tts/edge.py`                            | Extracted `EdgeTTSProvider` implementation for Edge TTS                      |
| CREATE | `src/voice/tts/__init__.py`                        | Exposed public interfaces for seamless imports                               |
| DELETE | `src/voice/tts.py`                                 | Cleaned up monolithic (566 lines) file for <200 bounds compliance            |

### 2026-03-11 — Buyer Matching Agent Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/buyer_matching/constants.py`           | Extracted matcher tuning constants `MAX_MATCH_DISTANCE_KM`, `GRADE_ORDER`    |
| CREATE | `src/agents/buyer_matching/models.py`              | Extracted data schemas `BuyerProfile`, `ListingProfile`, `MatchResult`       |
| CREATE | `src/agents/buyer_matching/engine.py`              | Isolated the multi-factor scoring logic `MatchingEngine`                     |
| CREATE | `src/agents/buyer_matching/cache.py`               | Extracted `BuyerMatchingCacheMixin` for memory-safety                        |
| CREATE | `src/agents/buyer_matching/mock_data.py`           | Extracted `BuyerMatchingMockDataMixin` for isolated unit testing             |
| UPDATE | `src/agents/buyer_matching/agent.py`               | Refactored `BuyerMatchingAgent` to consume core mixins (569 -> 167 lines)    |
| CREATE | `src/agents/buyer_matching/__init__.py`            | Exposed public interfaces for seamless imports                               |

### 2026-03-11 — Listing Service Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/api/services/listing_service/constants.py`    | Extracted standard `SHELF_LIFE_DAYS` and `GRADE_ORDER`                       |
| CREATE | `src/api/services/listing_service/models.py`       | Extracted `CreateListingRequest` and `ListingResponse` structures            |
| CREATE | `src/api/services/listing_service/enrichment.py`   | Extracted external ML `ListingEnrichmentMixin` dependencies                  |
| CREATE | `src/api/services/listing_service/storage.py`      | Extracted AuroraPostgresClient wrappers `ListingStorageMixin`                |
| CREATE | `src/api/services/listing_service/service.py`      | Re-implemented `ListingService` incorporating separation of concerns         |
| CREATE | `src/api/services/listing_service/__init__.py`     | Final backward-compatible API export point                                   |
| DELETE | `src/api/services/listing_service.py`              | Cleaned up monolithic (571 lines) file for <200 bounds compliance            |

### 2026-03-11 — Google AMED Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/tools/google_amed/models.py`                  | Extracted AMED structures `CropType`, `HealthStatus`, `CropMonitoringData`   |
| CREATE | `src/tools/google_amed/mock_data.py`               | Extracted `AMEDMockDataMixin` for robust synthetic response generation       |
| CREATE | `src/tools/google_amed/client.py`                  | Extracted main `GoogleAMEDClient` integrating the data mixes                 |
| CREATE | `src/tools/google_amed/__init__.py`                | Re-exported `get_amed_client` factory ensuring backwards compatibility       |
| DELETE | `src/tools/google_amed.py`                         | Dismantled monolithic file (579 lines) to respect 200-line modular rule      |

### 2026-03-11 — Digital Twin Engine Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/digital_twin/engine/utils.py`          | Pure analysis, confidence math, and grading estimation functions             |
| CREATE | `src/agents/digital_twin/engine/storage.py`        | `StorageMixin` handling Postgres persistence and in-memory caches            |
| CREATE | `src/agents/digital_twin/engine/report.py`         | `DiffReportMixin` handling the orchestration of ML similarity diffing        |
| CREATE | `src/agents/digital_twin/engine/core.py`           | `DigitalTwinEngine` orchestrator integrating via inheritance mixins          |
| CREATE | `src/agents/digital_twin/engine/__init__.py`       | Exposed identically to original module structure API exports                 |
| DELETE | `src/agents/digital_twin/engine.py`                | Swept monolithic file (586 lines) under the 200-line compliance rule         |

### 2026-03-11 — Agri Scrapers Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/agri_scrapers/models.py`             | Extracted `MandiPrice`, `WeatherData`, `NewsArticle` schemas                 |
| CREATE | `src/scrapers/agri_scrapers/constants.py`          | Centralized data URLs and source constants                                   |
| CREATE | `src/scrapers/agri_scrapers/enam.py`               | Extracted `ENAMScraper` for StealthyFetcher-powered dashboard parsing        |
| CREATE | `src/scrapers/agri_scrapers/imd.py`                | Extracted `IMDWeatherScraper` for agricultural advisories logic              |
| CREATE | `src/scrapers/agri_scrapers/rss.py`                | Extracted `RSSNewsScraper` for feedparser logic                              |
| CREATE | `src/scrapers/agri_scrapers/api.py`                | Extracted `AgriculturalDataAPI` handling fallback logic                      |
| CREATE | `src/scrapers/agri_scrapers/__init__.py`           | Initialized the package identical to old file exports                        |
| DELETE | `src/scrapers/agri_scrapers.py`                    | Deleted monolithic file (611 lines) replacing it with the new package        |

### 2026-03-11 — Web Scraping Agent Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/web_scraping_agent/models.py`          | Extracted `ScrapingResult` and `ScrapingConfig`                              |
| CREATE | `src/agents/web_scraping_agent/base.py`            | Extracted `BaseWebScraper` for Playwright initialization                     |
| CREATE | `src/agents/web_scraping_agent/cache.py`           | Extracted `ScraperCacheMixin` for caching HTML results locally               |
| CREATE | `src/agents/web_scraping_agent/parser.py`          | Extracted `HTMLParserMixin` for markdown translation                         |
| CREATE | `src/agents/web_scraping_agent/browser.py`         | Extracted `BrowserScraperMixin` for basic orchestration                      |
| CREATE | `src/agents/web_scraping_agent/extractor.py`       | Extracted `LLMExtractorMixin` for structured LLM data extraction             |
| CREATE | `src/agents/web_scraping_agent/agent.py`           | Reassembled `WebScrapingAgent` orchestrator via Mixin composition            |
| CREATE | `src/agents/web_scraping_agent/__init__.py`        | Exposed unified clean API representing the old monolithic file               |
| DELETE | `src/agents/web_scraping_agent.py`                 | Removed monolithic file (624 lines) fulfilling the 200-line rule             |

### 2026-03-11 — State Manager Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/shared/memory/state_manager/models.py`        | Extracted memory models (`ConversationContext`, etc.)                        |
| CREATE | `src/shared/memory/state_manager/base.py`          | Base state holding Redis context and cache structures                        |
| CREATE | `src/shared/memory/state_manager/session.py`       | `SessionManagerMixin` for conversation messages and history window           |
| CREATE | `src/shared/memory/state_manager/execution.py`     | `ExecutionTrackerMixin` for tracking execution state steps                   |
| CREATE | `src/shared/memory/state_manager/voice.py`         | `VoiceSessionMixin` for NFR6 WebRTC SLA rehydration checks                   |
| CREATE | `src/shared/memory/state_manager/manager.py`       | Extracted `AgentStateManager` combining all mixins                           |
| CREATE | `src/shared/memory/state_manager/__init__.py`      | Exposed identical API replacing the old module                               |
| DELETE | `src/shared/memory/state_manager.py`               | Removed monolithic file (630 lines) to comply with 200-line limit            |

### 2026-03-11 — Query Processor Modular Refactoring

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/rag/query_processor/models.py`            | Extracted data models (ExpandedQuery, QueryProcessorConfig)                  |
| CREATE | `src/rag/query_processor/prompts.py`           | Extracted processing prompts                                                 |
| CREATE | `src/rag/query_processor/hyde.py`              | Extracted HyDE expansion logic                                               |
| CREATE | `src/rag/query_processor/multi_query.py`       | Extracted Multi-Query expansion logic                                        |
| CREATE | `src/rag/query_processor/step_back.py`         | Extracted Step-Back generation logic                                         |
| CREATE | `src/rag/query_processor/decompose.py`         | Extracted query decomposition logic                                          |
| CREATE | `src/rag/query_processor/rewrite.py`           | Extracted query rewriting logic                                              |
| CREATE | `src/rag/query_processor/processor.py`         | Extracted `AdvancedQueryProcessor` orchestrator                              |
| CREATE | `src/rag/query_processor/__init__.py`          | Initialized package                                                          |
| DELETE | `src/rag/query_processor.py`                   | Deleted monolithic file to comply with 200-line rule                         |
| UPDATE | `ai/rag/query_processor.py`                    | Converted duplicate file to an import proxy for the new package              |

### 2026-03-11 — IMD Weather Client Modular Refactoring

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/imd_weather/models.py`           | Extracted data models (CurrentWeather, DailyForecast, WeatherAlert, etc.)    |
| CREATE | `src/scrapers/imd_weather/constants.py`        | Extracted API constants and district coordinates                             |
| CREATE | `src/scrapers/imd_weather/mock_data.py`        | Extracted mock data generation functions                                     |
| CREATE | `src/scrapers/imd_weather/cache.py`            | Extracted `IMDCacheManager` class                                            |
| CREATE | `src/scrapers/imd_weather/advisory.py`         | Extracted logic to generate agricultural advisories from weather             |
| CREATE | `src/scrapers/imd_weather/client.py`           | Extracted orchestrator `IMDWeatherClient` class                              |
| CREATE | `src/scrapers/imd_weather/__init__.py`         | Exposed models and client instance                                           |
| DELETE | `src/scrapers/imd_weather.py`                  | Removed monolithic file (>600 lines) to adhere to 200-line rule              |
| UPDATE | `src/tools/imd_weather.py`                     | Replaced duplicate monolithic code with an import proxy to the new package   |

### 2026-03-11 — eNAM Client Modular Refactoring

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/enam_client/models.py`           | Extracted data models (MandiPrice, PriceTrend, etc.)                         |
| CREATE | `src/scrapers/enam_client/constants.py`        | Extracted API and state/commodity constants                                  |
| CREATE | `src/scrapers/enam_client/mock_data.py`        | Extracted mock data generation functions                                     |
| CREATE | `src/scrapers/enam_client/cache.py`            | Extracted `ENAMCacheManager` class                                           |
| CREATE | `src/scrapers/enam_client/api_fetch.py`        | Extracted API call and parsing logic (`fetch_live_prices`, etc.)             |
| CREATE | `src/scrapers/enam_client/trends.py`           | Extracted trend calculation and market summary generation                    |
| CREATE | `src/scrapers/enam_client/client.py`           | Extracted orchestrator `ENAMClient` class                                    |
| CREATE | `src/scrapers/enam_client/__init__.py`         | Exposed models and client instance                                           |
| DELETE | `src/scrapers/enam_client.py`                  | Removed monolithic file (>600 lines) to adhere to 200-line rule              |
| UPDATE | `src/tools/enam_client.py`                     | Replaced duplicate monolithic code with an import proxy to the new package   |

### 2026-03-11 — Supervisor Agent Modular Refactoring

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/supervisor/models.py`              | Extracted RoutingDecision model                                              |
| CREATE | `src/agents/supervisor/prompts.py`             | Extracted ROUTING_PROMPT and prompt generation                               |
| CREATE | `src/agents/supervisor/rules.py`               | Extracted rule-based fallback routing logic                                  |
| CREATE | `src/agents/supervisor/router.py`              | Extracted LLM routing coordination logic                                     |
| CREATE | `src/agents/supervisor/session.py`             | Extracted session and context management logic                               |
| CREATE | `src/agents/supervisor/utils.py`               | Extracted response merging utilities                                         |
| CREATE | `src/agents/supervisor/agent.py`               | Extracted core SupervisorAgent class orchestrating the new modules           |
| CREATE | `src/agents/supervisor/__init__.py`            | Created package index exporting SupervisorAgent and RoutingDecision          |
| DELETE | `src/agents/supervisor_agent.py`               | Removed monolithic file (>600 lines) to adhere to 200-line rule              |
| UPDATE | `src/agents/__init__.py`                       | Updated imports to use the new `src.agents.supervisor` package               |

### 2026-03-11 — Agronomy Agent Multilingual Accuracy Improvements

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| UPDATE | `src/agents/agronomy_prompt.py`                | Added language tagging, CoT translation, and code-mixed few-shot examples    |
| UPDATE | `src/agents/agronomy_helpers.py`               | Made _FOLLOWUP_SECTION_RE regex robust for translated section headers        |
| UPDATE | `src/agents/agronomy_agent.py`                 | Stripped [LANG: xx] tag from final LLM response before returning             |
| UPDATE | `tests/unit/agents/test_agronomy_agent.py`     | Updated tests to verify parsing of natively translated follow-up headers     |
| UPDATE | `src/agents/agronomy_prompt.py`                | Expanded Kannada-specific vocabulary mapping, tone, and grammar enforcement  |
| CREATE | `scripts/test_llm_routing.py`                  | Standalone LLM prompt testing script to view agent routing visualizations    |
| CREATE | `tests/unit/agents/test_supervisor_agent.py`   | Robust LLM parsing and keyword fallback unit tests for Supervisor Agent      |
| UPDATE | `.env`                                         | Switched the active LLM provider from Bedrock to Groq (`llama-3.3-70b-versatile`) |
| UPDATE | `src/orchestrator/llm_provider.py`             | Updated `GroqProvider` default arguments for `llama-3.3-70b-versatile`       |
| UPDATE | `scripts/test_llm_routing.py`                  | Adjusted expected outputs to match exact routing prompt instructions; fixed Windows console Unicode display issues |

### 2026-02-27 — Workflow Documentation System

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ROADMAP.md`                                   | 6-phase milestone roadmap (Feb–Aug 2026)                                     |
| UPDATE | `PLAN.md`                                      | Full architecture diagram, user flows, NFRs, risk register, tech stack       |
| UPDATE | `AGENTS.md`                                    | Complete AI agent instructions: dev rules, file map, prompts, do-not-do list |
| UPDATE | `WORKFLOW_STATUS.md`                           | This file — comprehensive dev workflow + methodology                         |
| CREATE | `tracking/PROJECT_STATUS.md`                   | Always-current project state dashboard                                       |
| CREATE | `tracking/sprints/sprint-04-voice-pipeline.md` | Current sprint: Pipecat + APMC scraping                                      |
| CREATE | `tracking/daily/_template.md`                  | Daily log template                                                           |
| CREATE | `tracking/daily/2026-02-27.md`                 | Today's session log                                                          |
| CREATE | `TESTING/STRATEGY.md`                          | Test pyramid, philosophy, AI prompts for testing                             |
| CREATE | `TESTING/CHECKLISTS.md`                        | Per-feature-type done checklists                                             |

### 2026-02-27 — Production Scraping Upgrade (Earlier Session)

| Action | File                           | Description                                           |
| ------ | ------------------------------ | ----------------------------------------------------- |
| CREATE | `src/scrapers/base_scraper.py` | Scrapling-based browser scraper with Camoufox stealth |
| UPDATE | `src/scrapers/apmc/`           | Production-grade APMC scraper upgrade                 |
| UPDATE | `src/pipelines/`               | Data pipeline with caching + APScheduler              |

### 2026-02-26 — Voice Domain Separation

| Action | File                               | Description                           |
| ------ | ---------------------------------- | ------------------------------------- |
| CREATE | `src/voice/pipecat/`               | Pipecat submodule (STT, TTS services) |
| CREATE | `src/voice/pipecat/stt_service.py` | IndicWhisper STT + Groq fallback      |
| CREATE | `src/voice/pipecat/tts_service.py` | Edge-TTS in Kannada/Hindi/English     |
| CREATE | `src/voice/pipecat/__init__.py`    | Module init with exports              |
| CREATE | `src/voice/pipecat_bot.py`         | Main Pipecat bot entry point          |
| CREATE | `src/api/websocket/voice_ws.py`    | WebSocket handler for Pipecat stream  |

### 2026-02-26 — Domain Separation (RAG, Vision, Voice)

| Action      | File             | Description                                            |
| ----------- | ---------------- | ------------------------------------------------------ |
| RESTRUCTURE | `ai/rag/`        | Advanced RAG pipeline moved to dedicated AI folder     |
| RESTRUCTURE | `ai/vision/`     | Vision domain separated (YOLOv12 + DINOv2 placeholder) |
| RESTRUCTURE | `src/voice/`     | Voice domain with dedicated agent                      |
| UPDATE      | `ai/__init__.py` | Unified exports for all AI domains                     |

### 2026-02-27 — Advanced Folder Structure (Earliest Session)

| Action | File                   | Description                                              |
| ------ | ---------------------- | -------------------------------------------------------- |
| CREATE | `ai/`                  | New top-level AI module (rag, vision, evaluations)       |
| CREATE | `tracking/`            | Development tracking (goals, sprints, daily, milestones) |
| CREATE | `infra/`               | Deployment & monitoring configs                          |
| CREATE | `config/`              | Database & service configurations                        |
| MOVE   | `src/rag/` → `ai/rag/` | RAG pipeline moved to ai/                                |

### January 10, 2026 — Advanced RAG Phase 1–4

| Action | File                            | Description                                           |
| ------ | ------------------------------- | ----------------------------------------------------- |
| CREATE | `src/tools/enam_client.py`      | eNAM API client for live mandi prices                 |
| CREATE | `src/tools/imd_weather.py`      | IMD Weather client with agro-advisories               |
| CREATE | `src/rag/raptor.py`             | RAPTOR hierarchical tree indexing with GMM clustering |
| CREATE | `src/rag/contextual_chunker.py` | Contextual chunking with entity extraction            |
| CREATE | `src/rag/query_processor.py`    | HyDE, multi-query, step-back, decomposition           |
| CREATE | `src/rag/enhanced_retriever.py` | Parent Document, Sentence Window, MMR                 |
| CREATE | `src/rag/hybrid_search.py`      | BM25 sparse + RRF fusion hybrid search                |
| CREATE | `src/rag/reranker.py`           | Cross-encoder reranking with MiniLM fallback          |
| CREATE | `src/rag/graph_retriever.py`    | Neo4j Graph RAG with entity extraction                |
| CREATE | `src/rag/observability.py`      | LangSmith tracing + RAG eval metrics                  |

### January 9, 2026 — Multi-Agent System + Voice v1

| Action | File                             | Description                                       |
| ------ | -------------------------------- | ------------------------------------------------- |
| CREATE | `src/memory/state_manager.py`    | Conversation memory + Redis session manager       |
| CREATE | `src/agents/supervisor_agent.py` | LLM query routing with 0.9 confidence threshold   |
| CREATE | `src/agents/agronomy_agent.py`   | Crop cultivation, pest management, farming advice |
| CREATE | `src/agents/commerce_agent.py`   | Market prices, AISP calculations, sell/hold       |
| CREATE | `src/agents/platform_agent.py`   | CropFresh app features, registration, support     |
| CREATE | `src/agents/general_agent.py`    | Greetings, fallback for unclear queries           |
| CREATE | `src/voice/stt.py`               | IndicWhisper STT + Groq fallback                  |
| CREATE | `src/voice/tts.py`               | IndicTTS + Edge-TTS + gTTS                        |
| CREATE | `src/api/routes/chat.py`         | Multi-turn chat + SSE streaming                   |

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

---

### 2026-02-27 — RAG 2027: Advanced Agentic RAG Research & Documentation (Sprint 05 Prep)

**Research conducted**: Comprehensive RAG paradigm shift analysis for 2027 competitiveness. Identified 10 major innovation areas. Created sprint-integrated implementation roadmap.

#### New ADR Files

| Action | File                                                 | Description                                                                                          |
| ------ | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| CREATE | `docs/decisions/ADR-007-agentic-rag-orchestrator.md` | Decision: Replace fixed 4-node pipeline with autonomous retrieval planner + speculative draft engine |
| CREATE | `docs/decisions/ADR-008-adaptive-query-router.md`    | Decision: 8-strategy adaptive router with explicit cost signals (₹0.03–₹0.55/query)                  |
| CREATE | `docs/decisions/ADR-009-agri-embeddings.md`          | Decision: Two-layer agri embedding strategy (wrapper L1 Sprint 05 + fine-tuned L2 Phase 4)           |
| CREATE | `docs/decisions/ADR-010-browser-scraping-rag.md`     | Decision: Browser-augmented RAG using Scrapling for live gov/news data                               |

#### New Architecture Docs

| Action | File                                         | Description                                                                           |
| ------ | -------------------------------------------- | ------------------------------------------------------------------------------------- |
| CREATE | `docs/architecture/agentic_rag_system.md`    | Full architecture: Retrieval Planner → Speculative Engine → Verifier → Self-Evaluator |
| CREATE | `docs/architecture/adaptive_query_router.md` | 8-strategy router: decision tree, cost table, A/B rollout plan                        |
| CREATE | `docs/architecture/agri_embeddings.md`       | Layer 1 (AgriEmbeddingWrapper) + Layer 2 (fine-tuned model) architecture              |
| CREATE | `docs/architecture/browser_scraping_rag.md`  | Source registry, SourceSelector, ContentExtractor, TTL lifecycle, fallbacks           |

#### Updated Docs

| Action | File                       | Description                                                                                                                              |
| ------ | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| UPDATE | `docs/rag_architecture.md` | Full rewrite: 9-component architecture diagram, performance targets table, mermaid data flow, linked to all new ADRs + architecture docs |
| UPDATE | `WORKFLOW_STATUS.md`       | Added this entry; updated Last Updated timestamp                                                                                         |

#### Key Design Decisions (Sprint 05-06)

- **Adaptive Router** reduces avg query cost ₹0.44 → ₹0.21 (–52%) by routing simple queries to `DIRECT_LLM`
- **Speculative RAG** reduces voice latency –51% via 3 parallel Groq 8B drafts + Gemini Flash verifier
- **AgriEmbeddingWrapper** wraps BGE-M3 with domain instruction prefix + 50-term Hindi/Kannada normalization map
- **Browser RAG** extends Scrapling infrastructure to 14+ ag-specific gov/news sources with TTL-gated Qdrant live collection
- All new components are **feature-flagged** to allow A/B testing and gradual rollout

#### Next Actions (Sprint 05 Implementation)

```
[ ] Create ai/rag/agri_embeddings.py (AgriEmbeddingWrapper, Layer 1)
[ ] Create ai/rag/agentic_orchestrator.py (Retrieval Planner, basic version)
[ ] Upgrade ai/rag/query_analyzer.py (AdaptiveQueryRouter, 8 strategies)
[ ] Create ai/rag/browser_rag.py (BrowserRAGIntegration)
[ ] Create scripts/test_adaptive_router.py
[ ] Register for eNAM API access (enam.gov.in/register)
[ ] Establish RAGAS evaluation baseline (20 golden queries)
```

### 2026-03-16 — Vision Training Contracts Slice

| Action | File | Description |
| ------ | ---- | ----------- |
| CREATE | `src/agents/quality_assessment/training/commodity_registry.py` | Stable commodity slug/id registry with launch cohort defaults |
| CREATE | `src/agents/quality_assessment/training/manifest_schema.py` | Canonical manifest schema for lot/image/commodity/grade metadata |
| CREATE | `src/agents/quality_assessment/training/splitter.py` | Deterministic grouped split assignment by farm/lot |
| CREATE | `src/agents/quality_assessment/training/dataset_exporter.py` | Manifest-to-classification/YOLO export pipeline and dataset YAML writer |
| CREATE | `src/agents/quality_assessment/training/model_contracts.py` | ONNX contract validation for YOLO, DINO, and ResNet runtime artifacts |
| CREATE | `src/agents/quality_assessment/training/dinov2_model.py` | Commodity-conditioned DINO classifier model definition |
| CREATE | `src/agents/quality_assessment/training/dinov2_data.py` | Manifest-backed DINO dataloaders |
| CREATE | `src/agents/quality_assessment/training/dinov2_training.py` | DINO training/eval/export orchestration |
| CREATE | `src/agents/quality_assessment/training/similarity_model.py` | ResNet similarity model definition |
| CREATE | `src/agents/quality_assessment/training/similarity_dataset.py` | Triplet CSV dataset loader for similarity training |
| CREATE | `src/agents/quality_assessment/training/similarity_training.py` | ResNet similarity training/export orchestration |
| CREATE | `src/agents/quality_assessment/dinov2_runtime.py` | DINO preprocessing, softmax, and commodity tensor helpers |
| CREATE | `src/agents/digital_twin/similarity_runtime.py` | ResNet runtime helpers and contract-aware loader |
| CREATE | `scripts/export_quality_dataset.py` | Thin CLI for exporting canonical manifests into training layouts |
| CREATE | `scripts/train_yolo_defects.py` | Thin CLI for training/exporting validated CropFresh YOLO defect models |
| UPDATE | `scripts/train_dinov2_classifier.py` | Refactored to thin CLI over manifest-backed DINO training |
| UPDATE | `scripts/train_resnet_similarity.py` | Refactored to thin CLI over ResNet similarity training |
| UPDATE | `src/agents/quality_assessment/dinov2_classifier.py` | Validates ONNX contract, passes commodity_id, and keeps fallback behavior |
| UPDATE | `src/agents/quality_assessment/yolo_detector.py` | Rejects invalid YOLO contracts and stays within small-file boundaries |
| UPDATE | `src/agents/quality_assessment/vision_models.py` | Passes commodity through to the DINO runtime |
| UPDATE | `src/agents/digital_twin/similarity.py` | Slim wrapper over extracted runtime helpers; preserves existing API |
| CREATE | `tests/unit/test_quality_training_manifest.py` | Manifest schema validation coverage |
| CREATE | `tests/unit/test_quality_training_splits.py` | Deterministic grouped split coverage |
| CREATE | `tests/unit/test_quality_training_exporter.py` | Dataset export and YOLO layout coverage |
| CREATE | `tests/unit/test_vision_model_contracts.py` | Rejects placeholder ONNX artifacts, accepts valid mocked contracts |
| CREATE | `tests/unit/test_resnet_similarity_contract.py` | Runtime contract enforcement coverage for ResNet similarity |
| UPDATE | `tests/unit/test_vision_dinov2.py` | Verifies commodity_id is passed into DINO inference |
