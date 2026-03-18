# CropFresh AI â€” Development Workflow & Status Guide

> **Last Updated:** 2026-03-18 (vision lab API and static wiring)
> **Package Manager:** uv | **Python:** 3.11+ | **Stack:** FastAPI + LangGraph + Qdrant Cloud + Neo4j AuraDB + Redis Labs

This document is the **single entry point** for understanding how CropFresh AI is developed. It covers the development philosophy, workflow loop, documentation structure, and a running file changes log. AI agents should read this alongside `AGENTS.md` before starting any work.

---

## Latest Session Snapshot

**2026-03-18 - Vision Lab API and Static Wiring**

- Added a small shared Vision API surface under `src/api/routers/vision.py` so the static suite can call the real quality agent instead of routing quality checks through unrelated chat-only flows.
- `GET /api/v1/vision/health` now exposes whether the shared quality agent is available and whether the runtime is in full model mode or rule-based fallback, while `POST /api/v1/vision/assess` returns the grading/HITL contract plus a ready-to-reuse listing-grade payload.
- Added `static/vision_lab.html` with focused JS/CSS helpers so image or description-based quality checks, model-mode visibility, and `/api/v1/listings/{id}/grade` attachment all live in one connected testing page.
- Added a small session-storage bridge in `static/assets/js/lab-state.js` and surfaced `workflow_context` on `/api/v1/voice/process` so Voice Hub can carry the last created listing id into Vision Lab for a truthful voice -> vision -> listing demo path.
- Updated the dashboard quick quality check to use the new vision route, expanded shared suite navigation to include Vision Lab, and added focused API/static contract tests for the new route and page.
- Verification snapshot: run `uv run pytest tests/api/test_vision_routes.py tests/unit/test_static_testing_lab.py -v` plus targeted `ruff check` on the new/updated Python tests.

**Fastest way to resume from this implementation**

- Review `src/api/routers/vision.py` and `src/api/rest/voice.py`
- Review `static/vision_lab.html`, `static/assets/js/vision-lab.js`, and `static/assets/js/lab-state.js`
- Review `static/assets/js/dashboard.js`, `static/assets/js/voice-agent-rest.js`, and `static/assets/js/agent-workflows.js`
- Run `uv run pytest tests/api/test_vision_routes.py tests/unit/test_static_testing_lab.py -v`

**2026-03-18 - Static Agent Journey Wiring**

- Upgraded the static testing lab from a premium skin into a more accurate operator dashboard by adding explicit workflow boards for route validation, downstream service handoff, and realtime contract visibility.
- The dashboard now shows connected scenario cards plus a routed-run board that explains which agent answered a quick check, what tools were used, and which surface should be opened next for deeper validation.
- The voice hub now includes a REST route board that spells out detected language, intent, missing listing fields, and expected service handoff, plus a duplex contract board that stays honest about the current websocket limitation: no intent or entity payloads yet.
- Fixed the voice inspector health panel so it now reads the real `/api/v1/voice/health` fields (`stt_providers`, `tts_provider`) instead of stale placeholder names.
- Added a focused static-contract test slice to keep dashboard/voice DOM hooks, script ordering, shared workflow assets, and the small-file rule from drifting.
- Hardened the static shell against stale browser cache by adding direct critical rail CSS for suite pages and the premium voice demo, plus versioned CSS asset links so the dashboard rail cannot fall back into raw document flow above the main content.
- Verification snapshot: run `uv run pytest tests/unit/test_static_testing_lab.py -v` and targeted `ruff check` for the new Python test before closing this slice.

**Fastest way to resume from this implementation**

- Review `static/index.html` and `static/voice_agent.html`
- Review `static/assets/css/suite/critical-shell.css` and `static/assets/css/premium-voice/critical-rail.css`
- Review `static/assets/js/agent-workflow-data.js`, `static/assets/js/agent-workflows.js`, and `static/assets/js/voice-hub-lab.js`
- Review `static/assets/js/voice-agent-rest.js`, `static/assets/js/voice-agent-ws.js`, and `static/assets/js/voice-agent-tools.js`
- Run `uv run pytest tests/unit/test_static_testing_lab.py -v`

**2026-03-18 - Premium Static Suite Refresh**

- Refreshed the entire `static/` experience so the dashboard, RAG lab, voice hub, realtime stream, quick voice UI, and premium voice demo now feel like one product family instead of separate utility pages.
- Added a shared suite chrome layer for premium header, navigation, shell glow, footer, and card polish under `static/assets/css/suite/`, then upgraded the form controls so inputs, buttons, and result panes feel more intentional.
- Expanded that into a real freemium testing-lab layout with persistent suite rails, page-specific theming, and connected navigation so the static pages no longer feel like the same layout repeated with different titles.
- Updated each static page with product-style header tags, richer section framing, shared footer navigation, and page-specific side context while preserving all existing IDs and JS wiring.
- Tightened the premium voice page so it now sits inside the same connected suite concept instead of feeling visually isolated from the other static surfaces.
- Verification note: no automated frontend test or screenshot pipeline was run for this static-only pass; changes were validated by code-level sanity checks only on 2026-03-18.

**Fastest way to resume from this implementation**

- Review `static/assets/css/suite/chrome.css`, `static/assets/css/suite/forms.css`, and `static/assets/css/premium-voice/layout.css`
- Review `static/index.html`, `static/voice_agent.html`, `static/rag_test.html`, `static/voice_realtime.html`, `static/voice_test_ui.html`, and `static/premium_voice.html`
- Open the pages locally and compare mobile vs desktop layout before making further visual changes

**2026-03-18 - Centralized Language Resolution Wiring**

- Split the new shared language layer into small reusable modules under `src/shared/` so language normalization, script detection, transliterated Kannada detection, and context shaping now live behind one stable import path: `src.shared.language`.
- Wired that shared resolver into chat session preparation, supervisor session context, prompt building, LLM generation guidance, and supervisor fallback routing so Kannada support is determined centrally instead of by scattered per-agent prompt tweaks.
- Replaced duplicate chat route implementations with one shared `src/api/chat_pkg/` router and kept both legacy import paths as thin compatibility shims.
- Updated the voice compatibility layer so `VoiceEntityExtractor.detect_language_from_text(...)` now uses the shared detector and the legacy `VoiceAgent()` import path still works without manual dependency setup.
- Expanded the shared Kannada prompt library with dedicated domain sections for crop listing, buyer matching, quality assessment, logistics, crop recommendation/ADCL, and price prediction instead of relying on only the older agronomy/commerce/platform buckets.
- Added central Kannada domain-context injection in the LLM mixin so custom-prompt agents also inherit the right Kannada vocabulary and guidance, then patched the main agent call sites to pass normalized `context` through to the LLM layer.
- Upgraded the shared Kannada prompt into an advanced multi-dialect system layer with stronger slang handling, code-mixing rules, safety/output guidance, and a district-aware dialect hint builder.
- Added reusable runtime prompt rendering for `[DIALECT_LEXICON: ...]` and `[CONTEXT_KANNADA_INFO]` blocks so local Kannada vocabulary and crop context can be injected centrally from shared session context.
- Expanded the overall shared Kannada library with dialect bucket guidance, reusable response patterns, few-shot interaction examples, and deeper domain content for listing, buyer matching, quality, logistics, price prediction, and crop recommendation.
- Added a scalable structured Kannada retrieval layer with JSONL seed corpora, cached loaders, runtime enrichment, and query-aware retrieval so we can grow Kannada coverage into thousands of entries without bloating the base system prompt.
- Updated the LLM mixin to inject retrieval-only Kannada blocks even when a static shared Kannada prompt already exists, keeping the static layer compact while still adding query-specific dialect and domain hints.
- Verification snapshot: targeted Ruff checks passed and the focused multilingual test slice passed (`54 passed`) on 2026-03-18.

**Fastest way to resume from this implementation**

- Review `src/shared/language.py`, `src/shared/language_context.py`, and `src/shared/language_detection.py`
- Review `src/api/chat_pkg/session.py` and `src/api/chat_pkg/router.py`
- Review `src/agents/base/llm.py`, `src/agents/kannada/builder.py`, `src/agents/kannada/retriever.py`, and `src/agents/kannada/data/`
- Run `uv run pytest tests/unit/test_kannada_retriever.py tests/unit/test_kannada_builder.py tests/unit/test_llm_language_guidance.py tests/unit/test_shared_language.py tests/unit/test_chat_session_flow.py tests/unit/test_supervisor_routing.py tests/unit/test_voice_agent_task16.py tests/unit/test_rag_language_support.py -v`

**2026-03-18 - Retro Template Cleanup**

- Normalized `tracking/retros/sprint-04-retro.md` to the repo retro template without changing the underlying sprint learnings.
- Cleaned up `tracking/retros/sprint-05-retro.md` so it now follows the exact template structure instead of a custom metadata-heavy variant.
- Left `tracking/retros/_template.md` unchanged and treated it as the source of truth for retrospective formatting.

**2026-03-17 - Docs-First Voice Duplex Sprint Handoff**

- Added `tracking/sprints/sprint-07-voice-duplex-productionization.md` as the next voice sprint source of truth for duplex productionization, realistic latency targets, Bedrock removal, local-language quality, and live testing cleanup.
- Updated the core voice docs so they now point to `/api/v1/voice/ws` and `/api/v1/voice/ws/duplex` instead of the removed legacy websocket path.
- Reframed the model-provider docs around `groq`, `vllm`, and `together` while keeping Bedrock references explicit as legacy code paths still present on 2026-03-17.
- Aligned the README, environment/setup guides, and AWS markdown with the actual current voice stack and planned Sprint 07 direction.

**Fastest way to resume from this handoff**

- Read `tracking/sprints/sprint-07-voice-duplex-productionization.md`
- Read `tracking/PROJECT_STATUS.md`
- Read `docs/api/websocket-voice.md` and `docs/features/voice-pipeline.md`
- Review `src/api/websocket/voice_pkg/router.py` and `src/api/websocket/voice_pkg/duplex.py`
- Open `static/premium_voice.html` before changing the live voice flow

**2026-03-17 - Sprint 05 Retrospective Update**

- Added `tracking/retros/sprint-05-retro.md` using the repo retro template headings so Sprint 05 now has a formal retrospective artifact.
- Captured what shipped well, what slipped, and the carry-forward actions created by the Sprint 05 to Sprint 06 pivot.
- Kept the retro honest about the mixed sprint outcome: strong docs/rate-hub/ADCL momentum, but original RAG sprint goals still need explicit carryover handling.

**2026-03-17 - ADCL Productionization Implementation**

- Implemented the canonical `ADCLService` and split it into smaller helper modules to keep the source files reviewable and below the 200-line target.
- Added Aurora ADCL persistence helpers, the `(week_start, district)` migration, listing persistence updates, and shared app-state wiring for ADCL, listings, and voice.
- Shipped `GET /api/v1/adcl/weekly`, voice/listing compatibility updates, APScheduler refresh jobs, and focused tests for the ADCL slice.
- Added ADCL API docs, a 20-query golden set, and a live backtest runbook so the next session can move from fixture validation to Aurora validation.

**Fastest way to resume from this implementation**

- Read `tracking/sprints/sprint-06-adcl-productionization.md`
- Read `tracking/daily/2026-03-17.md`
- Review `src/agents/adcl/service.py`, `src/api/routes/adcl.py`, and `src/api/runtime/services.py`
- Use `src/evaluation/reports/adcl_backtest_2026-03-17.md` for the live validation checklist

**2026-03-17 - Sprint 06 ADCL Planning Handoff**

- Created `tracking/sprints/sprint-06-adcl-productionization.md` as the next sprint source of truth for ADCL productionization.
- Added `docs/decisions/ADR-012-adcl-district-first-service-contract.md` and `docs/decisions/ADR-013-adcl-source-precedence-and-evidence.md` to lock the service contract and live-data policy before coding starts.
- Updated `ROADMAP.md`, `tracking/PROJECT_STATUS.md`, `tracking/tasks/backlog.md`, and the Sprint 05 file so Sprint 06 consistently points to ADCL and the Supabase/auth follow-up sits behind it.
- Kept Sprint 05 as the active sprint; this was a docs-first handoff so the next session can start implementation without re-planning scope.

**Fastest way to resume from this handoff**

- Read `tracking/sprints/sprint-06-adcl-productionization.md`
- Read `docs/decisions/ADR-012-adcl-district-first-service-contract.md`
- Read `docs/decisions/ADR-013-adcl-source-precedence-and-evidence.md`
- Read the ADCL handoff addendum in `tracking/daily/2026-03-17.md`

**2026-03-17 â€” Multi-Source Karnataka Rate Hub**

- Added a shared `src/rates/` domain for official-first aggregation of mandi, support/reference, fuel, gold, and validator/retail rate sources.
- Refactored agentic orchestration and tool registration so `multi_source_rates` can be reused by agents, API routes, planner fallback, graph-runtime retrieval, and scheduler jobs.
- Added `POST /api/v1/prices/query` and `GET /api/v1/prices/source-health` plus scheduled refresh jobs for the main Karnataka data categories.
- Added focused tests, sprint notes, daily log entries, and ADR-011 so the slice is reviewable without reopening every code diff.
- Verification snapshot: targeted rate-hub tests passed, targeted Ruff passed, and full-repo Ruff/mypy still have older unrelated backlog that should be cleaned up separately.

**Fastest way to review this session**

- Read `docs/decisions/ADR-011-multi-source-rate-hub.md` for the architecture decision and precedence model.
- Read `tracking/daily/2026-03-17.md` for the execution log and `tracking/sprints/sprint-05-advanced-rag.md` for sprint-level impact.
- Use the `2026-03-17 - Multi-Source Karnataka Rate Hub` section in the file log below when you want the full touched-file inventory.
- Open `tests/unit/rates/` if you want concrete examples of connector normalization, comparison behavior, and service fan-out expectations.

---

## ðŸ“‚ Documentation Map

```
/ (repo root)
  PLAN.md                      â† Master product + architecture plan (start here)
  ROADMAP.md                   â† Phase milestones (Febâ€“Aug 2026)
  AGENTS.md                    â† AI agent rules, prompts, file structure map
  WORKFLOW_STATUS.md           â† This file: dev workflow + file changes log
  CHANGELOG.md                 â† Version history

  tracking/
    PROJECT_STATUS.md          â† Current state (update every sprint)
    sprints/sprint-0X-*.md     â† Each sprint's goals + outcomes
    daily/YYYY-MM-DD.md        â† Per-session work logs
    milestones/                â† Phase completion records
    retros/                    â† Sprint retrospective notes

  docs/
    decisions/ADR-*.md         â† Architecture Decision Records
    agents/REGISTRY.md         â† Agent specs and prompt versions
    api/                       â† API reference docs
    architecture/              â† System architecture docs

  TESTING/
    STRATEGY.md                â† Test philosophy, pyramid, AI prompts
    CHECKLISTS.md              â† Per-feature done checklists
```

---

## ðŸ§­ Core Development Principles

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

## ðŸ”„ Sprint Workflow Loop

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    CropFresh Dev Loop                           â”‚
â”‚                                                                 â”‚
â”‚  Step 1: Refine PLAN.md + ROADMAP.md (once per phase)          â”‚
â”‚    â””â”€â†’ AI prompt: "Act as senior architect, review my planâ€¦"   â”‚
â”‚                                                                 â”‚
â”‚  Step 2: Create sprint-XXX.md (start of each sprint)           â”‚
â”‚    â””â”€â†’ AI prompt: "Based on PLAN.md + last sprint, plan 2-wkâ€¦" â”‚
â”‚                                                                 â”‚
â”‚  Step 3: Daily Execution                                        â”‚
â”‚    â”œâ”€â†’ Update tracking/daily/YYYY-MM-DD.md each session        â”‚
â”‚    â”œâ”€â†’ Implement with AI: spec â†’ code â†’ tests â†’ docs           â”‚
â”‚    â””â”€â†’ Commit with message: "Sprint-04: [task name]"           â”‚
â”‚                                                                 â”‚
â”‚  Step 4: Testing in Loop                                        â”‚
â”‚    â”œâ”€â†’ AI generates tests for every function/endpoint           â”‚
â”‚    â””â”€â†’ AI reviews diff for bugs before merging                 â”‚
â”‚                                                                 â”‚
â”‚  Step 5: End-of-Sprint Review                                   â”‚
â”‚    â”œâ”€â†’ Fill sprint-XXX.md "Outcome" section                    â”‚
â”‚    â”œâ”€â†’ Update PROJECT_STATUS.md                                â”‚
â”‚    â”œâ”€â†’ Create ADRs for any architecture decisions              â”‚
â”‚    â””â”€â†’ Git tag milestone releases                              â”‚
â”‚                                                                 â”‚
â”‚  Step 6: Refine Next Sprint                                     â”‚
â”‚    â”œâ”€â†’ Move unfinished but important tasks forward             â”‚
â”‚    â”œâ”€â†’ Adjust ROADMAP.md if macro milestones slip              â”‚
â”‚    â””â”€â†’ Start back at Step 2                                    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ðŸ’¬ Standard AI Prompts (Copy-Paste Ready)

### ðŸ—ï¸ Start of Sprint

```
"Here is my PLAN.md and ROADMAP.md. Based on last sprint's outcome in
tracking/sprints/sprint-03-*.md, propose a 2-week sprint plan with
5â€“7 concrete, testable tasks focused on [theme].
Format it into the sprint template at tracking/sprints/_template.md."
```

### âš™ï¸ Implement a Feature

```
"Read PLAN.md, tracking/PROJECT_STATUS.md, and the relevant sprint file.
For voice work, start with tracking/sprints/sprint-07-voice-duplex-productionization.md.
Implement [feature/endpoint name] in [file path].
Follow coding standards in docs/architecture/coding-standards.md.
Generate unit tests in tests/unit/ as part of this change.
Update WORKFLOW_STATUS.md with the new file in the changes log."
```

### ðŸ§ª Generate Tests

```
"Read TESTING/STRATEGY.md for our test philosophy.
Generate pytest unit tests for [FunctionName] in [file path].
Cover: happy path, edge cases, and error cases.
Mock all external dependencies (Qdrant, Groq, Redis).
Add descriptive docstrings to each test."
```

### ðŸ” Pre-merge Review

```
"Analyze this diff and identify:
- Functions or endpoints missing tests
- Unhandled error cases or missing validation
- Potential regressions
Reference TESTING/STRATEGY.md checklist and our coding standards."
```

### ðŸ“Š End-of-Sprint Summary

```
"Read the active sprint file and the last 5 daily logs.
For voice work, start with tracking/sprints/sprint-07-voice-duplex-productionization.md.
Summarize: what shipped, what slipped, 3 key learnings.
Format as the Sprint Outcome section and also update tracking/PROJECT_STATUS.md."
```

### ðŸ“ Architecture Decision

```
"I need to decide [decision topic]. Context: [brief context].
Options: [option A], [option B].
Analyze the tradeoffs for a solo founder building CropFresh AI.
Format your recommendation as an ADR using docs/decisions/_template.md."
```

---

## ðŸ—‚ï¸ Keeping History (Git-Centric Approach)

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

## ðŸ“Š Current Component Status

| Component                           | Status         | Progress | Sprint    |
| ----------------------------------- | -------------- | -------- | --------- |
| Project Structure                   | âœ… Complete    | 100%     | Sprint 01 |
| RAG Pipeline (RAPTOR + Hybrid)      | âœ… Complete    | 100%     | Sprint 01 |
| Multi-Agent System                  | âœ… Complete    | 100%     | Sprint 01 |
| Memory System (Redis)               | âœ… Complete    | 100%     | Sprint 01 |
| Voice Agent v1 (Edge-TTS + Whisper) | âœ… Complete    | 90%      | Sprint 01 |
| Duplex WebSocket Voice Path         | In Progress    | 80%      | Sprint 07 |
| Pipecat Voice Pipeline              | ðŸŸ¡ In Progress | 40%      | Sprint 07 (experimental) |
| APMC Mandi Scraper                  | âŒ Not Started | 0%       | Sprint 04 |
| ADCL Service (district-first)      | In Progress    | 80%      | Sprint 06 |
| Supabase/Auth Hardening            | âŒ Not Started | 0%       | Sprint 08 |
| Vision Agent (YOLOv12 + DINOv2)     | âŒ Not Started | 0%       | Phase 3   |
| Evaluation Framework (LangSmith)    | âŒ Not Started | 0%       | Sprint 05 |
| Flutter Mobile App                  | âŒ Not Started | 0%       | Phase 4   |

---

## ðŸš€ Quick Start

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

## ðŸŒ API Endpoints Reference

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
| `/api/v1/voice/process`    | POST      | Full voice-in â†’ voice-out     |
| `/api/v1/voice/transcribe` | POST      | Audio â†’ Text (STT)            |
| `/api/v1/voice/synthesize` | POST      | Text â†’ Audio (TTS)            |
| `/api/v1/voice/ws`         | WebSocket | Compatibility voice streaming |
| `/api/v1/voice/ws/duplex`  | WebSocket | Canonical realtime duplex path |

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

## ðŸ“ File Changes Log

### 2026-03-18 - Retro Template Cleanup

| Action | File | Description |
|--------|------|-------------|
| UPDATE | `tracking/retros/sprint-04-retro.md` | Rewrote the Sprint 04 retrospective to match the retro template exactly while preserving the sprint learnings |
| UPDATE | `tracking/retros/sprint-05-retro.md` | Normalized the Sprint 05 retrospective to the same template structure and retained the carry-forward actions |
| UPDATE | `WORKFLOW_STATUS.md` | Added this retro cleanup entry so the markdown-only change is recorded for the next session |

### 2026-03-17 - Docs-First Voice Duplex Handoff

| Action | File | Description |
|--------|------|-------------|
| CREATE | `tracking/sprints/sprint-07-voice-duplex-productionization.md` | Added the next voice sprint source of truth for duplex productionization, latency targets, Bedrock removal, and live test cleanup |
| UPDATE | `tracking/PROJECT_STATUS.md` | Kept Sprint 06 current, added Sprint 07 next, and aligned the next-session entry with the voice handoff |
| UPDATE | `tracking/tasks/backlog.md` | Promoted voice latency, Bedrock removal, Pipecat cleanup, and live-test work into explicit Sprint 07 backlog items |
| UPDATE | `AGENTS.md` | Pointed startup guidance and prompts to Sprint 07 instead of the removed legacy voice sprint file |
| UPDATE | `docs/api/websocket-voice.md` | Rewrote the websocket contract around `/api/v1/voice/ws` and `/api/v1/voice/ws/duplex` |
| UPDATE | `docs/api/endpoints-reference.md` | Corrected the voice REST and websocket contract to match the current runtime |
| UPDATE | `docs/features/voice-pipeline.md` | Rewrote the voice stack doc around duplex websocket, hybrid STT/TTS, and current latency reality |
| UPDATE | `docs/architecture/system-architecture.md` | Aligned the architecture summary with the Bedrock-free provider direction and duplex-first voice path |
| UPDATE | `docs/architecture/data-flow.md` | Corrected the voice and provider data flows to the current duplex contract |
| UPDATE | `docs/guides/environment-variables.md` | Reframed provider setup around `groq`, `vllm`, and `together` |
| UPDATE | `docs/guides/getting-started.md` | Replaced Bedrock-first onboarding with Groq and vLLM quick-start guidance |
| UPDATE | `README.md` | Removed Bedrock-first positioning and described the current duplex voice stack accurately |
| UPDATE | `infra/aws/iam-roles.md` | Removed Bedrock IAM guidance from the recommended production path |
| UPDATE | `infra/aws/vpc-network.md` | Removed Bedrock-specific networking assumptions from the recommended path |
| UPDATE | `WORKFLOW_STATUS.md` | Added this handoff entry, refreshed prompts, and corrected the voice API summary |

### 2026-03-17 - Sprint 05 Retrospective Update

| Action | File | Description |
|--------|------|-------------|
| CREATE | `tracking/retros/sprint-05-retro.md` | Added the Sprint 05 retrospective using the repo template and current sprint-pivot context |

### 2026-03-17 - ADCL Productionization Implementation

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/agents/adcl/service.py` | Canonical district-first ADCL service contract for REST, voice, listings, and wrappers |
| CREATE | `src/agents/adcl/report_utils.py` | Shared ADCL evidence, metadata, and empty-report helpers |
| CREATE | `src/agents/adcl/price_runtime.py` | Shared rate-hub-backed price signal builder for ADCL |
| CREATE | `src/agents/adcl/scheduler.py` | APScheduler wrapper for weekly report generation and source refresh |
| CREATE | `src/api/routes/adcl.py` | `GET /api/v1/adcl/weekly` endpoint |
| CREATE | `src/api/runtime/` | Shared startup wiring for DB, ADCL, listings, voice, and scheduler state |
| CREATE | `src/db/postgres/adcl.py` | Aurora ADCL read/write helpers |
| CREATE | `src/db/migrations/004_adcl_reports_district_persistence.sql` | Composite-key ADCL report migration with freshness and source health |
| UPDATE | `src/api/services/listing_service/enrichment.py` | Listing ADCL checks now prefer `is_recommended_crop(...)` |
| UPDATE | `src/agents/voice/handlers_ext.py` | Voice weekly-demand flow now uses the canonical ADCL contract |
| UPDATE | `src/api/rest/voice.py` | Voice REST routes now resolve shared runtime services |
| UPDATE | `src/api/main.py` | Thin FastAPI entrypoint with shared runtime lifespan and router setup |
| CREATE | `tests/api/test_adcl_routes.py` | API coverage for `/api/v1/adcl/weekly` |
| UPDATE | `tests/unit/test_adcl_agent.py` | Focused tests for the canonical ADCL service behavior |
| CREATE | `tests/unit/test_api_config.py` | Regression coverage for tolerant debug env parsing |
| CREATE | `src/evaluation/datasets/adcl_golden_queries.json` | 20-query ADCL golden set |
| CREATE | `src/evaluation/reports/adcl_backtest_2026-03-17.md` | Execution-ready ADCL backtest runbook and verification snapshot |
| UPDATE | `docs/api/overview.md` | Added the ADCL weekly-report surface to the API overview |
| UPDATE | `docs/api/endpoints-reference.md` | Added the ADCL route contract and caller-facing response shape |

### 2026-03-17 - Sprint 06 ADCL Planning Handoff

| Action | File | Description |
| ------ | ---- | ----------- |
| CREATE | `tracking/sprints/sprint-06-adcl-productionization.md` | Added the next sprint plan for ADCL productionization, live data wiring, and hardening |
| CREATE | `docs/decisions/ADR-012-adcl-district-first-service-contract.md` | Locked the canonical district-first ADCL service contract for Sprint 06 |
| CREATE | `docs/decisions/ADR-013-adcl-source-precedence-and-evidence.md` | Locked the marketplace-first source precedence and evidence policy for ADCL |
| UPDATE | `tracking/sprints/sprint-05-advanced-rag.md` | Added a forward-planning note pointing the next session at Sprint 06 ADCL work |
| UPDATE | `tracking/PROJECT_STATUS.md` | Added the Sprint 06 ADCL handoff, next-session reading order, and ADR references |
| UPDATE | `ROADMAP.md` | Re-sequenced Sprint 06 around ADCL and moved Supabase/auth follow-up behind it |
| UPDATE | `tracking/tasks/backlog.md` | Re-prioritized Sprint 06 backlog items around ADCL productionization |
| UPDATE | `tracking/daily/2026-03-17.md` | Added an ADCL implementation handoff note for the next working session |
| UPDATE | `WORKFLOW_STATUS.md` | Added this entry and refreshed the last-updated timestamp |

### 2026-03-16 - Vision Training Accuracy Hardening

| Action | File | Description |
| ------ | ---- | ----------- |
| UPDATE | `src/shared/logger.py` | Made shared logger setup idempotent so training scripts can use structured logging safely |
| CREATE | `src/agents/quality_assessment/training/dinov2_artifacts.py` | Split DINO seed/export helpers into a small companion module to keep training orchestration under the 200-line rule |
| UPDATE | `src/agents/quality_assessment/training/dinov2_model.py` | Added partial backbone unfreezing support for higher-accuracy DINO fine-tuning |
| UPDATE | `src/agents/quality_assessment/training/dinov2_data.py` | Added deterministic loaders, inverse-frequency class weights, and balanced train sampling metadata |
| CREATE | `src/agents/quality_assessment/training/dinov2_metrics.py` | Added exact per-grade/per-commodity metrics and JSON training report generation |
| UPDATE | `src/agents/quality_assessment/training/dinov2_training.py` | Reworked DINO training to use balanced loss, deterministic seeds, early stopping, ONNX validation, and structured metrics reports |
| CREATE | `src/agents/quality_assessment/training/yolo_reporting.py` | Added YOLO validation metric extraction, release gates, and JSON report writing |
| UPDATE | `scripts/train_dinov2_classifier.py` | Added CLI controls for seed, patience, label smoothing, backbone fine-tuning, balancing, and report output |
| UPDATE | `scripts/train_yolo_defects.py` | Added validation gates, report generation, and structured logging before exporting the detector |
| CREATE | `tests/unit/test_dinov2_metrics.py` | Added unit coverage for exact DINO evaluation/report payloads |
| CREATE | `tests/unit/test_yolo_reporting.py` | Added unit coverage for YOLO metric extraction, threshold failures, and report writing |
| UPDATE | `WORKFLOW_STATUS.md` | Added this entry and refreshed the last-updated timestamp |

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

### 2026-03-14 â€” Phase 1: Anti-Hallucination Pipeline (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/query_rewriter.py`                     | HyDE, step-back, and multi-query expansion strategies (178 LOC)              |
| CREATE | `ai/rag/query_rewriter_prompts.py`             | Prompt templates for query rewriting (74 LOC)                                |
| CREATE | `ai/rag/citation_engine.py`                    | Inline [1], [2] citation markers with source attribution (172 LOC)           |
| CREATE | `ai/rag/confidence_gate.py`                    | "I don't know" safety gate with Kannada keywords (195 LOC)                   |
| UPDATE | `ai/rag/grader.py`                             | Continuous 0â€“1 scoring, time-decay for stale market docs, Kannada keywords   |
| CREATE | `tests/unit/test_query_rewriter.py`            | 12 tests: strategies, classification, edge cases, LLM fallback              |
| CREATE | `tests/unit/test_citation_engine.py`           | 8 tests: heuristic/LLM citations, source extraction, formatting             |
| CREATE | `tests/unit/test_confidence_gate.py`           | 9 tests: safety classification, grounding, gating, Kannada                   |
| CREATE | `tests/unit/test_grader_enhanced.py`           | 11 tests: continuous scoring, time-decay, batch grading                      |
| CREATE | `docs/planning/advanced-agentic-rag-plan.md`   | Comprehensive 4-phase implementation plan                                    |
| CREATE | `docs/decisions/ADR-010.md`                    | Architecture Decision Record for Advanced Agentic RAG upgrade                |

### 2026-03-14 â€” Phase 3: LangGraph State Machine (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/graph/state.py`                        | RAG state TypedDict + GraphRunResult Pydantic model (77 LOC)                 |
| CREATE | `ai/rag/graph/nodes.py`                        | 5 pipeline nodes: rewrite, retrieve, grade, generate, cite (199 LOC)         |
| CREATE | `ai/rag/graph/nodes_safety.py`                 | 3 safety nodes: gate, evaluate, web_search (175 LOC)                         |
| CREATE | `ai/rag/graph/edges.py`                        | Conditional edge routing: after_grade, after_evaluate, after_gate (83 LOC)   |
| CREATE | `ai/rag/graph/builder.py`                      | Graph assembly + `run_rag_graph()` public API (143 LOC)                      |
| CREATE | `ai/rag/graph/__init__.py`                     | Package exports (15 LOC)                                                     |
| CREATE | `tests/unit/test_rag_graph.py`                 | 17 tests: edge routing, node functions, graph compilation                    |

### 2026-03-14 â€” Phase 4: Advanced Retrieval (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/retrieval/contextual_enricher.py`      | Anthropic-style chunk enrichment with section inference (155 LOC)             |
| CREATE | `ai/rag/retrieval/query_decomposer.py`         | Multi-part query splitting with Kannada conjunctions (164 LOC)               |
| CREATE | `ai/rag/retrieval/time_aware.py`               | Freshness-boosted scoring: market/weather/scheme/evergreen (175 LOC)         |
| CREATE | `ai/rag/retrieval/advanced_retriever.py`       | Retrieval coordinator: decompose â†’ retrieve â†’ rerank (130 LOC)               |
| CREATE | `ai/rag/retrieval/__init__.py`                 | Package exports (27 LOC)                                                     |
| CREATE | `tests/unit/test_advanced_retrieval.py`        | 18 tests: enricher, decomposer, time-aware, edge cases                       |

### 2026-03-14 â€” Phase 5: Evaluation & Guardrails (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ai/rag/evaluation/metrics.py`                 | RAGAS multi-metric evaluator: faithfulness, relevancy, precision (196 LOC)   |
| CREATE | `ai/rag/evaluation/golden_dataset.py`          | 30 golden queries: market, agronomy, pest, scheme, weather, Kannada          |
| CREATE | `ai/rag/evaluation/guardrail.py`               | CI quality gate with configurable thresholds (154 LOC)                       |
| CREATE | `ai/rag/evaluation/__init__.py`                | Package exports (31 LOC)                                                     |
| CREATE | `scripts/eval_guardrail.py`                    | CLI entry point for CI/CD pipeline (85 LOC)                                  |
| CREATE | `tests/unit/test_evaluation.py`                | 17 tests: evaluator, dataset integrity, guardrail logic                      |

### 2026-03-14 â€” Phase 6: Documentation, Integration Tests & Benchmark (ADR-010)

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| MODIFY | `docs/architecture/data-flow.md`               | Replaced Section 5 with Advanced Agentic RAG architecture (6 subsections)    |
| CREATE | `docs/architecture/rag-benchmark-baseline.md`  | Evaluation benchmark baseline report with heuristic scores                   |
| CREATE | `tests/integration/test_rag_integration.py`    | 10 integration tests: pipeline, LangGraph, evaluation, cross-module          |

### 2026-03-11 â€” LLM Provider Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/orchestrator/llm_provider/models.py`          | Extracted `LLMMessage` and `LLMResponse` models                              |
| CREATE | `src/orchestrator/llm_provider/base.py`            | Extracted `BaseLLMProvider` interface                                        |
| CREATE | `src/orchestrator/llm_provider/bedrock.py`         | Extracted legacy Amazon Bedrock provider                                     |
| CREATE | `src/orchestrator/llm_provider/groq.py`            | Extracted Groq provider                                                      |
| CREATE | `src/orchestrator/llm_provider/together.py`        | Extracted Together AI provider                                               |
| CREATE | `src/orchestrator/llm_provider/vllm.py`            | Extracted vLLM provider                                                      |
| CREATE | `src/orchestrator/llm_provider/factory.py`         | Extracted `create_llm_provider` factory                                      |
| CREATE | `src/orchestrator/llm_provider/__init__.py`        | Initialized `llm_provider` package                                           |
| UPDATE | `src/orchestrator/llm_provider.py`                 | Reduced (477 -> 24 lines) by converting into an import proxy file            |

### 2026-03-11 â€” Real-Time Data Manager Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/realtime_data/models.py`             | Extracted health and data models                                             |
| CREATE | `src/scrapers/realtime_data/health.py`             | Extracted `HealthMixin` for status tracking                                  |
| CREATE | `src/scrapers/realtime_data/fetchers.py`           | Extracted `FetchersMixin` for unified API fallback logic                     |
| CREATE | `src/scrapers/realtime_data/manager.py`            | Extracted core `RealTimeDataManager` class and singleton initialization      |
| CREATE | `src/scrapers/realtime_data/__init__.py`           | Initialized the `realtime_data` package                                      |
| UPDATE | `src/scrapers/realtime_data.py`                    | Reduced (480 -> 21 lines) by converting into an import proxy                 |
| UPDATE | `src/tools/realtime_data.py`                       | Reduced (480 -> 21 lines) by converging to the same import proxy             |

### 2026-03-11 â€” Deep Research Tool Modular Refactoring

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

### 2026-03-11 â€” Duplex Pipeline Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/duplex/models.py`                       | Extracted pipeline enums and models                                          |
| CREATE | `src/voice/duplex/initializers.py`                 | Extracted component initializers (LLM, STT, TTS)                             |
| CREATE | `src/voice/duplex/processing.py`                   | Extracted core audio rendering & STT logic                                   |
| CREATE | `src/voice/duplex/pipeline.py`                     | Rebuilt `DuplexPipeline` class combining mixins                              |
| CREATE | `src/voice/duplex/__init__.py`                     | Created package public interface                                             |
| UPDATE | `src/voice/duplex_pipeline.py`                     | Reduced (488 -> 17 lines) by converting into an import proxy file            |

### 2026-03-11 â€” Price Prediction Agent Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/price_prediction/models.py`            | Extracted `PricePrediction` model                                            |
| CREATE | `src/agents/price_prediction/constants.py`         | Extracted seasonal calender and rule thresholds                              |
| CREATE | `src/agents/price_prediction/analysis.py`          | Extracted `AnalysisMixin` with features, rules, trend logic                  |
| CREATE | `src/agents/price_prediction/__init__.py`          | Provided a clean public interface for the agent                              |
| UPDATE | `src/agents/price_prediction/agent.py`             | Reduced (514 -> 241 lines) by importing mixins and models                    |

### 2026-03-11 â€” Base Agent Modular Refactoring (Protected File Compatible)

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/base/models.py`                        | Extracted `AgentResponse` and `AgentConfig` models                           |
| CREATE | `src/agents/base/retrieval.py`                     | Extracted `RetrievalMixin` for KB search and context formatting              |
| CREATE | `src/agents/base/tools.py`                         | Extracted `ToolMixin` for safe tool execution and tracking                   |
| CREATE | `src/agents/base/llm.py`                           | Extracted `LLMMixin` for memory injection, LLM generation, and retries       |
| CREATE | `src/agents/base/agent.py`                         | Created `BaseAgent` core combining all mixins                                |
| CREATE | `src/agents/base/__init__.py`                      | Exposed base agent components                                                |
| UPDATE | `src/agents/base_agent.py`                         | Reduced (516 -> 15 lines) by converting into an import proxy file            |

### 2026-03-11 â€” VAD Module Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/vad/models.py`                          | Extracted data models: `VADState`, `VADEvent`, `SpeechSegment`               |
| CREATE | `src/voice/vad/utils.py`                           | Extracted utility functions for silence and audio bytes conversion           |
| CREATE | `src/voice/vad/silero.py`                          | Extracted `SileroVAD` core implementation and logic                          |
| CREATE | `src/voice/vad/bargein.py`                         | Extracted `BargeinDetector` class                                            |
| CREATE | `src/voice/vad/__init__.py`                        | Exposed `SileroVAD`, `BargeinDetector`, models, and utils                    |
| UPDATE | `src/voice/vad.py`                                 | Reduced (527 -> 20 lines) by converting into an import proxy file            |

### 2026-03-11 â€” AI Kosha Client Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/scrapers/aikosha/models.py`                   | Extracted data models: `AIKoshaCategory`, `AIKoshaDataset`, search results   |
| CREATE | `src/scrapers/aikosha/catalog.py`                  | Extracted the static hardcoded datasets catalog                              |
| CREATE | `src/scrapers/aikosha/client.py`                   | Extracted `AIKoshaClient` core logic for APIs and catalog search            |
| CREATE | `src/scrapers/aikosha/__init__.py`                 | Exposed `AIKoshaClient`, models, and catalog methods                        |
| UPDATE | `src/scrapers/aikosha_client.py`                   | Reduced (530 -> 19 lines) by converting into an import proxy file            |

### 2026-03-11 â€” WebRTC Transport Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/webrtc/models.py`                       | Extracted `ConnectionState`, `WebRTCConfig`, `AudioChunk`                    |
| CREATE | `src/voice/webrtc/tracks.py`                       | Extracted `AudioReceiveTrack` and `AudioSendTrack`                           |
| CREATE | `src/voice/webrtc/transport.py`                    | Extracted `WebRTCTransport` core logic                                       |
| CREATE | `src/voice/webrtc/signaling.py`                    | Extracted `WebRTCSignaling` logic                                            |
| CREATE | `src/voice/webrtc/__init__.py`                     | Exposed WebRTC transport models and classes                                  |
| UPDATE | `src/voice/webrtc_transport.py`                    | Reduced (538 -> 24 lines) by converting into an import proxy file            |

### 2026-03-11 â€” Agri Scrapers Tool Proxy

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| UPDATE | `src/tools/agri_scrapers.py`                       | Reduced (539 -> 26 lines) by converting into an import proxy file            |

### 2026-03-11 â€” Contextual Chunker Modular Refactoring

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

### 2026-03-11 â€” TTS (Voice) Module Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/voice/tts/models.py`                          | Extracted schemas `TTSVoice`, `TTSEmotion`, `SynthesisResult`                |
| CREATE | `src/voice/tts/utils.py`                           | Extracted helper utilities                                                   |
| CREATE | `src/voice/tts/indic.py`                           | Extracted `IndicTTS` implementation for AI4Bharat                            |
| CREATE | `src/voice/tts/edge.py`                            | Extracted `EdgeTTSProvider` implementation for Edge TTS                      |
| CREATE | `src/voice/tts/__init__.py`                        | Exposed public interfaces for seamless imports                               |
| DELETE | `src/voice/tts.py`                                 | Cleaned up monolithic (566 lines) file for <200 bounds compliance            |

### 2026-03-11 â€” Buyer Matching Agent Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/buyer_matching/constants.py`           | Extracted matcher tuning constants `MAX_MATCH_DISTANCE_KM`, `GRADE_ORDER`    |
| CREATE | `src/agents/buyer_matching/models.py`              | Extracted data schemas `BuyerProfile`, `ListingProfile`, `MatchResult`       |
| CREATE | `src/agents/buyer_matching/engine.py`              | Isolated the multi-factor scoring logic `MatchingEngine`                     |
| CREATE | `src/agents/buyer_matching/cache.py`               | Extracted `BuyerMatchingCacheMixin` for memory-safety                        |
| CREATE | `src/agents/buyer_matching/mock_data.py`           | Extracted `BuyerMatchingMockDataMixin` for isolated unit testing             |
| UPDATE | `src/agents/buyer_matching/agent.py`               | Refactored `BuyerMatchingAgent` to consume core mixins (569 -> 167 lines)    |
| CREATE | `src/agents/buyer_matching/__init__.py`            | Exposed public interfaces for seamless imports                               |

### 2026-03-11 â€” Listing Service Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/api/services/listing_service/constants.py`    | Extracted standard `SHELF_LIFE_DAYS` and `GRADE_ORDER`                       |
| CREATE | `src/api/services/listing_service/models.py`       | Extracted `CreateListingRequest` and `ListingResponse` structures            |
| CREATE | `src/api/services/listing_service/enrichment.py`   | Extracted external ML `ListingEnrichmentMixin` dependencies                  |
| CREATE | `src/api/services/listing_service/storage.py`      | Extracted AuroraPostgresClient wrappers `ListingStorageMixin`                |
| CREATE | `src/api/services/listing_service/service.py`      | Re-implemented `ListingService` incorporating separation of concerns         |
| CREATE | `src/api/services/listing_service/__init__.py`     | Final backward-compatible API export point                                   |
| DELETE | `src/api/services/listing_service.py`              | Cleaned up monolithic (571 lines) file for <200 bounds compliance            |

### 2026-03-11 â€” Google AMED Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/tools/google_amed/models.py`                  | Extracted AMED structures `CropType`, `HealthStatus`, `CropMonitoringData`   |
| CREATE | `src/tools/google_amed/mock_data.py`               | Extracted `AMEDMockDataMixin` for robust synthetic response generation       |
| CREATE | `src/tools/google_amed/client.py`                  | Extracted main `GoogleAMEDClient` integrating the data mixes                 |
| CREATE | `src/tools/google_amed/__init__.py`                | Re-exported `get_amed_client` factory ensuring backwards compatibility       |
| DELETE | `src/tools/google_amed.py`                         | Dismantled monolithic file (579 lines) to respect 200-line modular rule      |

### 2026-03-11 â€” Digital Twin Engine Modular Refactoring

| Action | File                                               | Description                                                                  |
| ------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `src/agents/digital_twin/engine/utils.py`          | Pure analysis, confidence math, and grading estimation functions             |
| CREATE | `src/agents/digital_twin/engine/storage.py`        | `StorageMixin` handling Postgres persistence and in-memory caches            |
| CREATE | `src/agents/digital_twin/engine/report.py`         | `DiffReportMixin` handling the orchestration of ML similarity diffing        |
| CREATE | `src/agents/digital_twin/engine/core.py`           | `DigitalTwinEngine` orchestrator integrating via inheritance mixins          |
| CREATE | `src/agents/digital_twin/engine/__init__.py`       | Exposed identically to original module structure API exports                 |
| DELETE | `src/agents/digital_twin/engine.py`                | Swept monolithic file (586 lines) under the 200-line compliance rule         |

### 2026-03-11 â€” Agri Scrapers Modular Refactoring

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

### 2026-03-11 â€” Web Scraping Agent Modular Refactoring

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

### 2026-03-11 â€” State Manager Modular Refactoring

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

### 2026-03-11 â€” Query Processor Modular Refactoring

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

### 2026-03-11 â€” IMD Weather Client Modular Refactoring

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

### 2026-03-11 â€” eNAM Client Modular Refactoring

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

### 2026-03-11 â€” Supervisor Agent Modular Refactoring

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

### 2026-03-11 â€” Agronomy Agent Multilingual Accuracy Improvements

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| UPDATE | `src/agents/agronomy_prompt.py`                | Added language tagging, CoT translation, and code-mixed few-shot examples    |
| UPDATE | `src/agents/agronomy_helpers.py`               | Made _FOLLOWUP_SECTION_RE regex robust for translated section headers        |
| UPDATE | `src/agents/agronomy_agent.py`                 | Stripped [LANG: xx] tag from final LLM response before returning             |
| UPDATE | `tests/unit/agents/test_agronomy_agent.py`     | Updated tests to verify parsing of natively translated follow-up headers     |
| UPDATE | `src/agents/agronomy_prompt.py`                | Expanded Kannada-specific vocabulary mapping, tone, and grammar enforcement  |
| CREATE | `scripts/test_llm_routing.py`                  | Standalone LLM prompt testing script to view agent routing visualizations    |
| CREATE | `tests/unit/agents/test_supervisor_agent.py`   | Robust LLM parsing and keyword fallback unit tests for Supervisor Agent      |
| UPDATE | `.env`                                         | Switched the active LLM provider from legacy Bedrock to Groq (`llama-3.3-70b-versatile`) |
| UPDATE | `src/orchestrator/llm_provider.py`             | Updated `GroqProvider` default arguments for `llama-3.3-70b-versatile`       |
| UPDATE | `scripts/test_llm_routing.py`                  | Adjusted expected outputs to match exact routing prompt instructions; fixed Windows console Unicode display issues |

### 2026-02-27 â€” Workflow Documentation System

| Action | File                                           | Description                                                                  |
| ------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| CREATE | `ROADMAP.md`                                   | 6-phase milestone roadmap (Febâ€“Aug 2026)                                     |
| UPDATE | `PLAN.md`                                      | Full architecture diagram, user flows, NFRs, risk register, tech stack       |
| UPDATE | `AGENTS.md`                                    | Complete AI agent instructions: dev rules, file map, prompts, do-not-do list |
| UPDATE | `WORKFLOW_STATUS.md`                           | This file â€” comprehensive dev workflow + methodology                         |
| CREATE | `tracking/PROJECT_STATUS.md`                   | Always-current project state dashboard                                       |
| CREATE | `tracking/sprints/(historical voice plan)`     | Initial voice sprint plan created during early repo setup                    |
| CREATE | `tracking/daily/_template.md`                  | Daily log template                                                           |
| CREATE | `tracking/daily/2026-02-27.md`                 | Today's session log                                                          |
| CREATE | `TESTING/STRATEGY.md`                          | Test pyramid, philosophy, AI prompts for testing                             |
| CREATE | `TESTING/CHECKLISTS.md`                        | Per-feature-type done checklists                                             |

### 2026-02-27 â€” Production Scraping Upgrade (Earlier Session)

| Action | File                           | Description                                           |
| ------ | ------------------------------ | ----------------------------------------------------- |
| CREATE | `src/scrapers/base_scraper.py` | Scrapling-based browser scraper with Camoufox stealth |
| UPDATE | `src/scrapers/apmc/`           | Production-grade APMC scraper upgrade                 |
| UPDATE | `src/pipelines/`               | Data pipeline with caching + APScheduler              |

### 2026-02-26 â€” Voice Domain Separation

| Action | File                               | Description                           |
| ------ | ---------------------------------- | ------------------------------------- |
| CREATE | `src/voice/pipecat/`               | Pipecat submodule (STT, TTS services) |
| CREATE | `src/voice/pipecat/stt_service.py` | IndicWhisper STT + Groq fallback      |
| CREATE | `src/voice/pipecat/tts_service.py` | Edge-TTS in Kannada/Hindi/English     |
| CREATE | `src/voice/pipecat/__init__.py`    | Module init with exports              |
| CREATE | `src/voice/pipecat_bot.py`         | Main Pipecat bot entry point          |
| CREATE | `src/api/websocket/voice_ws.py`    | WebSocket handler for Pipecat stream  |

### 2026-02-26 â€” Domain Separation (RAG, Vision, Voice)

| Action      | File             | Description                                            |
| ----------- | ---------------- | ------------------------------------------------------ |
| RESTRUCTURE | `ai/rag/`        | Advanced RAG pipeline moved to dedicated AI folder     |
| RESTRUCTURE | `ai/vision/`     | Vision domain separated (YOLOv12 + DINOv2 placeholder) |
| RESTRUCTURE | `src/voice/`     | Voice domain with dedicated agent                      |
| UPDATE      | `ai/__init__.py` | Unified exports for all AI domains                     |

### 2026-02-27 â€” Advanced Folder Structure (Earliest Session)

| Action | File                   | Description                                              |
| ------ | ---------------------- | -------------------------------------------------------- |
| CREATE | `ai/`                  | New top-level AI module (rag, vision, evaluations)       |
| CREATE | `tracking/`            | Development tracking (goals, sprints, daily, milestones) |
| CREATE | `infra/`               | Deployment & monitoring configs                          |
| CREATE | `config/`              | Database & service configurations                        |
| MOVE   | `src/rag/` â†’ `ai/rag/` | RAG pipeline moved to ai/                                |

### January 10, 2026 â€” Advanced RAG Phase 1â€“4

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

### January 9, 2026 â€” Multi-Agent System + Voice v1

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

## âš ï¸ Known Issues & Workarounds

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

## ðŸ› ï¸ Common Commands

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

### 2026-02-27 â€” RAG 2027: Advanced Agentic RAG Research & Documentation (Sprint 05 Prep)

**Research conducted**: Comprehensive RAG paradigm shift analysis for 2027 competitiveness. Identified 10 major innovation areas. Created sprint-integrated implementation roadmap.

#### New ADR Files

| Action | File                                                 | Description                                                                                          |
| ------ | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| CREATE | `docs/decisions/ADR-007-agentic-rag-orchestrator.md` | Decision: Replace fixed 4-node pipeline with autonomous retrieval planner + speculative draft engine |
| CREATE | `docs/decisions/ADR-008-adaptive-query-router.md`    | Decision: 8-strategy adaptive router with explicit cost signals (â‚¹0.03â€“â‚¹0.55/query)                  |
| CREATE | `docs/decisions/ADR-009-agri-embeddings.md`          | Decision: Two-layer agri embedding strategy (wrapper L1 Sprint 05 + fine-tuned L2 Phase 4)           |
| CREATE | `docs/decisions/ADR-010-browser-scraping-rag.md`     | Decision: Browser-augmented RAG using Scrapling for live gov/news data                               |

#### New Architecture Docs

| Action | File                                         | Description                                                                           |
| ------ | -------------------------------------------- | ------------------------------------------------------------------------------------- |
| CREATE | `docs/architecture/agentic_rag_system.md`    | Full architecture: Retrieval Planner â†’ Speculative Engine â†’ Verifier â†’ Self-Evaluator |
| CREATE | `docs/architecture/adaptive_query_router.md` | 8-strategy router: decision tree, cost table, A/B rollout plan                        |
| CREATE | `docs/architecture/agri_embeddings.md`       | Layer 1 (AgriEmbeddingWrapper) + Layer 2 (fine-tuned model) architecture              |
| CREATE | `docs/architecture/browser_scraping_rag.md`  | Source registry, SourceSelector, ContentExtractor, TTL lifecycle, fallbacks           |

#### Updated Docs

| Action | File                       | Description                                                                                                                              |
| ------ | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| UPDATE | `docs/rag_architecture.md` | Full rewrite: 9-component architecture diagram, performance targets table, mermaid data flow, linked to all new ADRs + architecture docs |
| UPDATE | `WORKFLOW_STATUS.md`       | Added this entry; updated Last Updated timestamp                                                                                         |

#### Key Design Decisions (Sprint 05-06)

- **Adaptive Router** reduces avg query cost â‚¹0.44 â†’ â‚¹0.21 (â€“52%) by routing simple queries to `DIRECT_LLM`
- **Speculative RAG** reduces voice latency â€“51% via 3 parallel Groq 8B drafts + Gemini Flash verifier
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

### 2026-03-16 â€” Vision Training Contracts Slice

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

### 2026-03-17 - Multi-Source Karnataka Rate Hub

| Action | File | Description |
| ------ | ---- | ----------- |
| CREATE | `src/rates/__init__.py` | Public exports for the shared multi-source rate hub |
| CREATE | `src/rates/enums.py` | RateKind, authority tier, fetch mode, and comparison depth enums |
| CREATE | `src/rates/models.py` | Shared query, record, health, and response models |
| CREATE | `src/rates/precedence.py` | Official-first source precedence, TTLs, and discrepancy thresholds |
| CREATE | `src/rates/query_builder.py` | Request normalization and deterministic cache keys |
| CREATE | `src/rates/cache.py` | Redis or in-memory TTL cache for rate responses |
| CREATE | `src/rates/repository.py` | Raw and normalized rate persistence with mandi dual-write support |
| CREATE | `src/rates/service.py` | Parallel fan-out, retries, caching, circuit breaker, and aggregation |
| CREATE | `src/rates/factory.py` | Shared service factory for API, tools, and scheduler consumers |
| CREATE | `src/rates/settings.py` | Safe Agmarknet API-key helper for partially configured environments |
| CREATE | `src/db/schema_rates.sql` | `normalized_rates` schema for the generic rate hub |
| CREATE | `src/rates/connectors/base.py` | Base connector contract and HTTP/browser fallback helpers |
| CREATE | `src/rates/connectors/html_utils.py` | Shared HTML parsing helpers for table and text sources |
| CREATE | `src/rates/connectors/pending_sources.py` | Metadata-only pending-access sources |
| CREATE | `src/rates/connectors/registry.py` | Enabled connector registry for all public sources |
| CREATE | `src/rates/connectors/agmarknet_ogd.py` | Official AGMARKNET OGD connector |
| CREATE | `src/rates/connectors/agmarknet_scrape.py` | Official AGMARKNET scrape connector |
| CREATE | `src/rates/connectors/enam_dashboard.py` | Public eNAM dashboard connector |
| CREATE | `src/rates/connectors/krama_daily.py` | KRAMA daily mandi connector |
| CREATE | `src/rates/connectors/krama_floor_price.py` | KRAMA support-price connector |
| CREATE | `src/rates/connectors/kapricom_reference.py` | KAPRICOM reference-price connector |
| CREATE | `src/rates/connectors/napanta.py` | Validator connector for NaPanta |
| CREATE | `src/rates/connectors/agriplus.py` | Validator connector for AgriPlus |
| CREATE | `src/rates/connectors/commoditymarketlive.py` | Validator connector for CommodityMarketLive |
| CREATE | `src/rates/connectors/shyali.py` | Validator connector for Shyali |
| CREATE | `src/rates/connectors/vegetablemarketprice.py` | Retail produce connector |
| CREATE | `src/rates/connectors/todaypricerates.py` | Retail produce connector for TodayPriceRates |
| CREATE | `src/rates/connectors/petroldieselprice.py` | Fuel connector |
| CREATE | `src/rates/connectors/parkplus_fuel.py` | Fuel connector fallback |
| CREATE | `src/rates/connectors/businessline_gold.py` | Gold connector |
| CREATE | `src/rates/connectors/iifl_gold.py` | Gold connector fallback |
| CREATE | `src/agents/agent_groups.py` | Extracted grouped agent builders to keep registry slim |
| CREATE | `src/agents/tool_registry_setup.py` | Shared agent tool-registry assembly |
| CREATE | `src/agents/tool_setup/__init__.py` | Lazy exports for agent tool-setup helpers |
| CREATE | `src/agents/tool_setup/commerce.py` | Commerce tool registration wrappers |
| CREATE | `src/agents/tool_setup/agronomy.py` | Agronomy tool registration wrappers |
| CREATE | `src/agents/tool_setup/research.py` | Research tool registration wrappers |
| CREATE | `src/agents/tool_setup/rates.py` | Rate-hub tool registration wrappers |
| CREATE | `src/tools/multi_source_rates.py` | Global tool entry points for `multi_source_rates` and `price_api` |
| CREATE | `src/tools/registry_types.py` | Annotation helpers for tool schema generation |
| CREATE | `src/scrapers/scheduler_runtime.py` | Refactored scheduler runtime with legacy and rate-hub jobs |
| UPDATE | `src/agents/agent_registry.py` | Slimmed registry to orchestration-only factory logic |
| UPDATE | `src/agents/__init__.py` | Lazy package exports to avoid circular imports |
| UPDATE | `src/tools/__init__.py` | Lazy-safe auto-registration imports |
| UPDATE | `src/tools/registry.py` | Better annotation handling for Optional and generic types |
| UPDATE | `src/api/routes/prices.py` | Added `/prices/query` and `/prices/source-health` on the shared service |
| UPDATE | `src/rag/agentic/orchestrator.py` | Routes `multi_source_rates` and mandi alias through extracted handlers |
| UPDATE | `src/rag/agentic/planner.py` | Plans price, support-price, fuel, and gold queries to the rate hub |
| CREATE | `src/rag/agentic/executor.py` | Extracted retrieval-plan execution helper |
| CREATE | `src/rag/agentic/tool_handlers.py` | Extracted tool call handlers for standalone agentic RAG |
| UPDATE | `src/rag/graph_runtime/services.py` | Uses the shared rate hub for live price documents |
| UPDATE | `src/scrapers/scraper_scheduler.py` | Thin compatibility export over refactored scheduler runtime |
| CREATE | `tests/unit/rates/test_query_builder.py` | Query normalization and cache-key tests |
| CREATE | `tests/unit/rates/test_comparison.py` | Official-first precedence and warning tests |
| CREATE | `tests/unit/rates/test_service.py` | Cache and circuit-breaker tests for RateService |
| CREATE | `tests/unit/rates/connectors/test_connectors.py` | Fixture-based parsing tests for all enabled connectors |
| CREATE | `tests/api/rates/test_prices_routes.py` | `/prices/query` and `/prices/source-health` endpoint tests |
| CREATE | `tests/unit/rag_agentic/test_rate_planning.py` | Planner fallback and tool-registry coverage for rate-hub tools |
| CREATE | `tests/unit/test_scheduler_rate_jobs.py` | Scheduler job registration and `force_live=True` coverage |
| UPDATE | `docs/api/endpoints-reference.md` | Documented multi-source rate endpoints |
| UPDATE | `docs/features/scraping-system.md` | Documented source tiers and shared rate-refresh scheduler |
| UPDATE | `tracking/sprints/sprint-05-advanced-rag.md` | Added multi-source rate-hub sprint progress update |
| CREATE | `tracking/daily/2026-03-17.md` | Daily implementation log for the rate-hub slice |
| CREATE | `docs/decisions/ADR-011-multi-source-rate-hub.md` | Architecture decision for the generic rate hub and conflict policy |
| UPDATE | `WORKFLOW_STATUS.md` | Added this log entry and refreshed the timestamp |

### 2026-03-18 - Centralized Language Resolution Wiring

| Action | File | Description |
| ------ | ---- | ----------- |
| CREATE | `src/shared/language_values.py` | Shared language constants, aliases, and Kannada transliteration hints |
| CREATE | `src/shared/script_language.py` | Raw unicode-script detector extracted into a neutral shared module |
| CREATE | `src/shared/language_detection.py` | Canonical language normalization and transliterated Kannada detection |
| CREATE | `src/shared/language_context.py` | Canonical context shaping for `user_profile`, `entities`, and response language |
| CREATE | `src/shared/language.py` | Small public facade for the shared language API |
| UPDATE | `src/rag/language_support.py` | Reused the shared detector for multilingual RAG language guidance |
| UPDATE | `src/memory/state_pkg/manager.py` | Added user-profile persistence support for language state |
| UPDATE | `src/agents/supervisor/session.py` | Persisted current language and built normalized agent context per turn |
| UPDATE | `src/agents/base/llm.py` | Injected per-turn language guidance into non-supervisor LLM calls |
| CREATE | `src/agents/kannada/listing_terms.py` | Kannada listing-flow guidance and vocabulary |
| CREATE | `src/agents/kannada/matching_terms.py` | Kannada buyer-matching guidance and vocabulary |
| CREATE | `src/agents/kannada/quality_terms.py` | Kannada quality-grading guidance and vocabulary |
| CREATE | `src/agents/kannada/logistics_terms.py` | Kannada logistics guidance and vocabulary |
| CREATE | `src/agents/kannada/adcl_terms.py` | Kannada crop-recommendation and weekly-demand guidance |
| CREATE | `src/agents/kannada/price_prediction_terms.py` | Kannada price-forecast guidance and vocabulary |
| CREATE | `src/agents/kannada/dialect_context.py` | District-aware Kannada dialect bucket hints for the shared prompt builder |
| CREATE | `src/agents/kannada/dialect_patterns.py` | Shared Kannada dialect bucket descriptions and style-matching guidance |
| CREATE | `src/agents/kannada/conversation_patterns.py` | Reusable Kannada clarification, confirmation, and recommendation patterns |
| CREATE | `src/agents/kannada/few_shot_examples.py` | Shared Kannada few-shot examples for rural, market, and platform interactions |
| CREATE | `src/agents/kannada/runtime_context.py` | Shared rendering of runtime dialect lexicon and local Kannada context blocks |
| CREATE | `src/agents/kannada/data_loader.py` | Cached JSONL loaders for structured Kannada lexicon and domain context corpora |
| CREATE | `src/agents/kannada/domain_resolution.py` | Shared Kannada domain alias resolution for prompt building and retrieval |
| CREATE | `src/agents/kannada/retriever.py` | Query-aware Kannada runtime enrichment from structured seed corpora |
| CREATE | `src/agents/kannada/data/dialect_lexicon.jsonl` | Seed Kannada dialect lexicon corpus for scalable runtime injection |
| CREATE | `src/agents/kannada/data/domain_context.jsonl` | Seed Kannada domain-context corpus for scalable runtime injection |
| UPDATE | `src/agents/kannada/guidelines.py` | Replaced the basic Kannada note with advanced multi-dialect behavior, safety, and output rules |
| UPDATE | `src/agents/kannada/dialect_context.py` | Exposed reusable district-signal and dialect-bucket helpers for shared retrieval |
| UPDATE | `src/agents/kannada/builder.py` | Composed advanced shared Kannada sections plus retrieval-aware runtime enrichment in one builder |
| UPDATE | `src/agents/kannada/adcl_terms.py` | Deepened Kannada crop-recommendation wording, risks, and output pattern guidance |
| UPDATE | `src/agents/kannada/listing_terms.py` | Deepened Kannada listing flow, selling guidance, and clearer marketplace vocabulary |
| UPDATE | `src/agents/kannada/matching_terms.py` | Deepened Kannada buyer-matching explanations, trust cues, and comparison wording |
| UPDATE | `src/agents/kannada/quality_terms.py` | Deepened Kannada quality-result wording, photo guidance, and defect vocabulary |
| UPDATE | `src/agents/kannada/logistics_terms.py` | Deepened Kannada logistics tradeoff language, timing, and spoilage guidance |
| UPDATE | `src/agents/kannada/price_prediction_terms.py` | Deepened Kannada forecast framing, confidence language, and hold/sell wording |
| UPDATE | `src/agents/prompt_context.py` | Passed normalized context into the shared Kannada builder for dialect-aware prompt injection |
| UPDATE | `src/agents/base/llm.py` | Passed full runtime context into shared Kannada domain injection for custom-prompt agents |
| CREATE | `src/agents/supervisor/multilingual_rules.py` | Reused multilingual voice intent keywords for supervisor fallback routing |
| UPDATE | `src/agents/supervisor/rules.py` | Routed non-English fallback queries through the shared multilingual helper |
| UPDATE | `src/agents/general_agent.py` | Passed normalized context into shared LLM generation |
| UPDATE | `src/agents/commerce_agent.py` | Passed normalized context into shared LLM generation |
| UPDATE | `src/agents/platform_agent.py` | Passed normalized context into shared LLM generation |
| UPDATE | `src/agents/buyer_matching/agent.py` | Passed normalized context into shared LLM generation |
| UPDATE | `src/agents/quality_assessment/agent.py` | Passed normalized context into shared LLM generation |
| UPDATE | `src/agents/adcl_wrapper_agent.py` | Passed normalized context and ADCL domain into shared prompting |
| UPDATE | `src/agents/logistics_wrapper_agent.py` | Passed normalized context and logistics domain into shared prompting |
| CREATE | `src/api/chat_pkg/models.py` | Shared chat request/response models |
| CREATE | `src/api/chat_pkg/supervisor.py` | Shared supervisor dependency builder for chat endpoints |
| CREATE | `src/api/chat_pkg/session.py` | Centralized session preparation and context persistence for chat |
| CREATE | `src/api/chat_pkg/router.py` | Canonical chat router implementation reused by both route entry points |
| UPDATE | `src/api/routes/chat.py` | Reduced to a thin compatibility export over the shared chat router |
| UPDATE | `src/api/routers/chat.py` | Reduced to a thin compatibility export over the shared chat router |
| UPDATE | `src/voice/entity_extractor/_language.py` | Pointed voice raw detection at the shared script detector |
| UPDATE | `src/voice/entity_extractor/__init__.py` | Exposed the shared response-language detector through the voice package |
| UPDATE | `src/agents/voice_agent.py` | Restored backward-compatible no-arg construction for the legacy wrapper |
| CREATE | `tests/unit/test_shared_language.py` | Coverage for normalization, transliterated Kannada, and context splitting |
| CREATE | `tests/unit/test_state_manager_profiles.py` | Coverage for persisted user-profile language updates |
| CREATE | `tests/unit/test_llm_language_guidance.py` | Coverage for centralized language instruction injection |
| CREATE | `tests/unit/test_kannada_retriever.py` | Coverage for structured Kannada retrieval and runtime-context merging |
| CREATE | `tests/unit/test_kannada_builder.py` | Coverage for advanced Kannada prompt sections, few-shot content, dialect hints, and runtime context blocks |
| CREATE | `tests/unit/test_chat_session_flow.py` | Coverage for stateful and stateless chat language handling |
| UPDATE | `tests/unit/test_llm_language_guidance.py` | Added query-specific retrieval coverage for shared Kannada injection with and without static prompts |
| UPDATE | `tests/unit/test_supervisor_routing.py` | Added Kannada routing coverage for fallback supervisor rules |
| UPDATE | `tests/unit/test_voice_agent_task16.py` | Added transliterated Kannada detection coverage |
| UPDATE | `WORKFLOW_STATUS.md` | Added this handoff entry and refreshed the timestamp |

### 2026-03-18 - Premium Static Suite Refresh

| Action | File | Description |
| ------ | ---- | ----------- |
| UPDATE | `static/assets/css/suite.css` | Added the shared premium suite chrome layer to the static design stack |
| CREATE | `static/assets/css/suite/chrome.css` | Shared premium shell styling for headers, nav, cards, glow, and footers |
| CREATE | `static/assets/css/suite/lab-shell.css` | Connected freemium-lab rail layout shared across the static suite |
| CREATE | `static/assets/css/suite/page-themes.css` | Page-specific premium themes for dashboard, voice, RAG, realtime, and quick voice UI |
| UPDATE | `static/assets/css/suite/forms.css` | Added richer focus, depth, and premium interaction styling for controls |
| UPDATE | `static/assets/css/premium-voice.css` | Added the premium rail layer to the premium voice style stack |
| CREATE | `static/assets/css/premium-voice/rail.css` | Connected-suite rail layout for the premium voice demo |
| UPDATE | `static/assets/css/premium-voice/layout.css` | Upgraded premium voice header treatment and added shared footer layout |
| UPDATE | `static/assets/css/premium-voice/components.css` | Refined premium voice nav and footer chip styling |
| UPDATE | `static/index.html` | Moved the dashboard into the connected suite rail layout and upgraded page framing |
| UPDATE | `static/voice_agent.html` | Moved the voice hub into the connected suite rail layout and upgraded operator framing |
| UPDATE | `static/rag_test.html` | Moved the RAG lab into the connected suite rail layout and upgraded knowledge-lab framing |
| UPDATE | `static/voice_realtime.html` | Moved the realtime stream page into the connected suite rail layout and upgraded streaming-lab framing |
| UPDATE | `static/voice_test_ui.html` | Moved the quick voice UI page into the connected suite rail layout and upgraded utility-lab framing |
| UPDATE | `static/premium_voice.html` | Connected the premium voice demo to the broader suite via a dedicated premium rail layout |
| UPDATE | `WORKFLOW_STATUS.md` | Added this static-suite refresh log entry |

### 2026-03-18 - Static Agent Journey Wiring

| Action | File | Description |
| ------ | ---- | ----------- |
| CREATE | `static/assets/css/suite/workflows.css` | Shared workflow-board and scenario-card styling for route-aware testing surfaces |
| CREATE | `static/assets/css/suite/critical-shell.css` | Direct critical rail layout CSS to prevent unstyled suite-rail content above the dashboard |
| CREATE | `static/assets/css/premium-voice/critical-rail.css` | Direct critical premium rail layout CSS to prevent premium demo shell regressions under stale cache |
| CREATE | `static/assets/js/agent-workflow-data.js` | Shared voice scenario catalog and route metadata for the static lab |
| CREATE | `static/assets/js/agent-workflows.js` | Shared renderers for dashboard route boards and voice intent workflow boards |
| CREATE | `static/assets/js/voice-hub-lab.js` | Voice Hub-specific duplex contract board and scenario bootstrap |
| UPDATE | `static/index.html` | Added connected scenario cards, routed-run board, and versioned critical shell asset links |
| UPDATE | `static/voice_agent.html` | Added REST route and duplex contract boards plus versioned critical shell asset links |
| UPDATE | `static/rag_test.html` | Added versioned critical shell asset links for the shared suite rail |
| UPDATE | `static/voice_realtime.html` | Added versioned critical shell asset links for the shared suite rail |
| UPDATE | `static/voice_test_ui.html` | Added versioned critical shell asset links for the shared suite rail |
| UPDATE | `static/premium_voice.html` | Added versioned critical premium rail asset links |
| UPDATE | `static/assets/js/dashboard.js` | Wired quick checks into the new dashboard route board |
| UPDATE | `static/assets/js/voice-agent-rest.js` | Wired REST voice results into the route board |
| UPDATE | `static/assets/js/voice-agent-ws.js` | Wired duplex websocket events into the contract board and improved error/full-text handling |
| UPDATE | `static/assets/js/voice-agent-tools.js` | Fixed voice health rendering to use the current endpoint payload fields |
| CREATE | `tests/unit/test_static_testing_lab.py` | Contract checks for DOM wiring, script ordering, workflow assets, and file-size guardrails |
| UPDATE | `WORKFLOW_STATUS.md` | Added this agent-journey wiring log entry |

