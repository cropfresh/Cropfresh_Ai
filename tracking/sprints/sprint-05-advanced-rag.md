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
- `docs/architecture/agentic_rag_system.md`
