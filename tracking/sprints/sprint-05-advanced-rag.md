# Sprint 05 — Advanced RAG & Documentation

> **Period:** 2026-03-10 → 2026-03-24
> **Theme:** Upgrade RAG pipeline with adaptive routing and establish RAGAS evaluation baseline
> **Sprint Status:** 🟡 In Progress

---

## 🎯 Goals (Measurable)

1. AgriEmbeddingWrapper wraps BGE-M3 with domain instruction prefix + 50-term Hindi/Kannada map
2. Adaptive Query Router routes simple queries to DIRECT_LLM (target: −52% cost)
3. RAGAS evaluation baseline established with 20 golden queries
4. Full documentation system created for docs/ and tracking/
5. Browser RAG extends Scrapling to ≥5 ag-specific gov/news sources

---

## 📋 Scope (Stories / Tasks)

### RAG Pipeline Upgrades
- [ ] `ai/rag/agri_embeddings.py` — AgriEmbeddingWrapper (Layer 1)
- [ ] `ai/rag/agentic_orchestrator.py` — Basic Retrieval Planner
- [ ] `ai/rag/query_analyzer.py` — Adaptive Query Router (8 strategies)
- [ ] `ai/rag/browser_rag.py` — Browser RAG integration

### Evaluation
- [ ] `scripts/test_adaptive_router.py` — Router evaluation script
- [ ] Create 20 golden queries for RAGAS baseline
- [ ] Register for eNAM API access

### Rate Intelligence (added during Sprint 05)
- [x] `src/rates/` — Shared multi-source Karnataka rate hub with precedence, caching, persistence, and comparison logic
- [x] `src/tools/multi_source_rates.py` — Reusable tool wrapper plus mandi-only compatibility alias
- [x] `src/api/routes/prices.py` — Multi-source rate query and source-health endpoints
- [x] `src/rag/agentic/` — Planner and orchestration wiring for price, support-price, fuel, and gold queries
- [x] `src/scrapers/scheduler_runtime.py` — Scheduled refresh jobs for official mandi, support/reference, validator/retail, fuel, and gold categories
- [ ] Live smoke tests for enabled public sources once unrestricted validation is available
- [ ] Repo-wide Ruff and mypy cleanup needed to remove unrelated CI noise from this slice

### Documentation (Sprint 05 deliverable)
- [x] `docs/architecture/system-architecture.md` — Full architecture with Mermaid
- [x] `docs/architecture/data-flow.md` — End-to-end data flows
- [x] `docs/agents/REGISTRY.md` — All 15 agents documented
- [x] `docs/api/endpoints-reference.md` — Full API reference
- [x] `docs/features/` — Voice, RAG, scraping, pricing docs
- [x] `docs/guides/` — Getting started, workflow, env vars
- [x] `tracking/PROJECT_STATUS.md` — Status dashboard
- [x] `.agent/rules/` — Antigravity doc rules
- [x] `config/project-context.yaml` — Master context YAML

---

## 🚫 Out of Scope

- No Supabase schema (Sprint 06)
- No Flutter app (Phase 4)
- No Vision Agent (Phase 3)

---

## ⚠️ Risks / Open Questions

- eNAM API registration may take 1-2 weeks
- BGE-M3 fine-tuning (Layer 2) is Phase 4 — keep wrapper simple
- Repo-wide Ruff and mypy backlog can hide feature-level verification quality if not tracked separately

---

## 📊 Sprint Outcome (fill at end of sprint)

**What Shipped:**
-

**What Slipped to Next Sprint:**
-

**Key Learnings:**
-

---

## 🔗 Related Files

- `docs/decisions/ADR-007-agentic-rag-orchestrator.md`
- `docs/decisions/ADR-008-adaptive-query-router.md`
- `docs/decisions/ADR-009-agri-embeddings.md`
- `docs/decisions/ADR-010-browser-augmented-rag.md`
- `docs/decisions/ADR-011-multi-source-rate-hub.md`
- `docs/architecture/agentic_rag_system.md`

## 2026-03-17 Progress Update — Multi-Source Rate Hub

**What shipped in this slice:**
- Added a generic `src/rates/` domain with normalized records, precedence rules, caching, persistence, and multi-source comparison
- Added 16 public connectors plus pending-access metadata for gated sources
- Wired `multi_source_rates` into the API, agent tool registries, planner fallback, graph-runtime live price retrieval, and APScheduler refresh jobs
- Added fixture-driven tests for rate normalization, comparison, connectors, API endpoints, planner fallback, and scheduler jobs

**Tracking follow-up added this session:**
- Refreshed `tracking/PROJECT_STATUS.md` and `tracking/tasks/backlog.md` with rate-hub hardening as active follow-up work
- Rewrote `tracking/daily/2026-03-17.md` in the daily-log template shape
- Created `tracking/weekly/2026-W12.md` from the weekly template for the new slice

**Still open:**
- Live network smoke tests for the public sources once unrestricted verification is available
- Full strict-mypy cleanup across older adjacent modules (`src/tools/registry.py`, legacy agentic files, and connector wrappers)
- Deeper scheduler target coverage beyond the initial Karnataka refresh tuples

## 2026-03-17 Forward Planning Note

- Sprint 06 planning is now captured in `tracking/sprints/sprint-06-adcl-productionization.md`
- ADCL productionization has been pulled forward as the next sprint so the next build session can start from one service contract and one live-data plan
- Supabase and user-management follow-up work has moved behind that sprint in roadmap and backlog references
- Next implementation session should start with:
  - `tracking/sprints/sprint-06-adcl-productionization.md`
  - `docs/decisions/ADR-012-adcl-district-first-service-contract.md`
  - `docs/decisions/ADR-013-adcl-source-precedence-and-evidence.md`
