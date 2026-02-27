# 📊 PROJECT_STATUS.md — CropFresh AI Current State

> **Last Updated:** 2026-02-27 (15:38 IST)
> **Version/Tag:** `v0.4-rag-agentic`
> **Current Sprint:** Sprint 05 — Agentic RAG & Adaptive Intelligence
> **Sprint Status:** 🔲 Planning (starts 2026-03-13)

---

## 🟢 What Is True Right Now

| Component | Status | Notes |
|-----------|--------|-------|
| Project Structure | ✅ Stable | Advanced folder layout, all modules organized |
| RAG Pipeline (Base) | ✅ Stable | RAPTOR + Hybrid Search + Qdrant Cloud + Neo4j |
| Query Processor | ✅ Stable | HyDE, multi-query, step-back, decompose implemented |
| Multi-Agent System | ✅ Stable | Supervisor routes with 0.9 confidence |
| Memory System | ✅ Stable | Redis-backed session manager |
| Voice Agent | 🟡 In Progress | Pipecat integration started, STT/TTS services built |
| APMC Scraper | 🟡 In Progress | Scrapling base scraper ready; eNAM + IMD clients built |
| RAG Documentation | ✅ Updated | 4 new ADRs, 4 new architecture docs, rag_architecture.md rewritten |
| **Adaptive Query Router** | 📋 Sprint 05 | ADR-008 written; implementation planned Sprint 05 |
| **Agentic Orchestrator** | 📋 Sprint 05 | ADR-007 written; implementation planned Sprint 05 |
| **AgriEmbedding Layer** | 📋 Sprint 05 | ADR-009 written; implementation planned Sprint 05 |
| **Browser-Augmented RAG** | 📋 Sprint 06 | ADR-010 written; implementation planned Sprint 06 |
| **Speculative Draft Engine** | 📋 Sprint 06 | Architecture doc written; implementation Sprint 06 |
| RAGAS Evaluation | 📋 Sprint 05 | Golden dataset creation planned Sprint 05 |
| Vision Agent | ❌ Not Started | YOLOv12 + DINOv2, planned for Phase 3 |
| Supabase Schema | ❌ Not Started | Schema migrations blocked on Sprint 06 |
| Fine-tuned Agri Embeddings | 📋 Phase 4 | Model training planned for 2027 |

---

## 🎯 Top 3 Priorities (Sprint 05 — starts March 13)

1. **Adaptive Query Router** — Upgrade `ai/rag/query_analyzer.py` with 8-strategy router (target: –52% avg query cost)
2. **AgriEmbeddingWrapper Layer 1** — Create `ai/rag/agri_embeddings.py` (target: +8–12% retrieval precision)
3. **RAGAS Baseline** — Create golden 20-query evaluation dataset, measure baseline scores before any optimizations

---

## 🚧 Active Blockers

- [ ] `pipecat_bot.py` — Pipecat WebSocket integration still needs live audio testing (Sprint 04 carry-over)
- [ ] `eNAM API` — Registration at enam.gov.in required before `LIVE_PRICE_API` strategy can use real data (submit March 13)
- [ ] `Supabase` — Schema not yet migrated; farmer/listing data flows still blocked

---

## ✅ Recently Completed Milestones

| Date | Milestone |
|------|-----------|
| Feb 27, 2026 (15:38) | **RAG 2027 Research Complete** — 10 paradigm shifts, 4 ADRs, 4 arch docs, sprint-05 + sprint-06 planned |
| Feb 27, 2026 (15:27) | RAG architecture docs updated — rag_architecture.md rewritten with 9-component diagram |
| Feb 27, 2026 (15:01) | Advanced development workflow documentation system established |
| Feb 27, 2026 | Production-grade scraping upgrade (Scrapling + eNAM + IMD clients) |
| Feb 26, 2026 | Voice domain separated; Pipecat submodule built |
| Feb 26, 2026 | Domain separation complete: RAG, Vision, Voice as distinct ai/ modules |
| Jan 10, 2026 | Advanced RAG Phase 1–4 complete (RAPTOR, Hybrid, HyDE, MMR, Contextual Chunking) |
| Jan 9, 2026 | Multi-agent system live (Supervisor + 4 domain agents) |

---

## 📦 Stack Reference

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11+ |
| AI Framework | LangGraph + LangChain |
| Vector DB | Qdrant Cloud |
| Graph DB | Neo4j |
| Primary DB | Supabase (PostgreSQL) |
| LLM (Generation) | Gemini Flash 2.0 + Groq Llama-3.3-70B |
| LLM (Router/Planner) | Groq Llama-3.1-8B-Instant (~80ms, ~₹0.001/call) |
| Scraping | Scrapling (Playwright + Camoufox StealthyFetcher) |
| Voice | Pipecat + Edge-TTS + IndicWhisper |
| Caching | Redis |
| Embeddings | BGE-M3 (1024-dim) + AgriEmbeddingWrapper (Sprint 05) |
| Evaluation | RAGAS + LangSmith |
| Package Manager | uv |

---

## 📁 Key File Locations

| Purpose | Path |
|---------|------|
| Product Vision | `PLAN.md` |
| Phase Milestones | `ROADMAP.md` |
| Previous Sprint | `tracking/sprints/sprint-04-voice-pipeline.md` |
| **Current Sprint** | `tracking/sprints/sprint-05-agentic-rag.md` |
| Next Sprint | `tracking/sprints/sprint-06-browser-rag.md` |
| Daily Logs | `tracking/daily/YYYY-MM-DD.md` |
| Architecture Decisions | `docs/decisions/ADR-*.md` (ADR-001 to ADR-010) |
| RAG Architecture | `docs/rag_architecture.md` |
| Agentic RAG System | `docs/architecture/agentic_rag_system.md` |
| Adaptive Router | `docs/architecture/adaptive_query_router.md` |
| Agri Embeddings | `docs/architecture/agri_embeddings.md` |
| Browser RAG | `docs/architecture/browser_scraping_rag.md` |
| RAG 2027 Research | `rag_2027_research_report.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |

---

## 📈 Metrics Dashboard

| Metric | Target (Sprint 05) | Target (2027) | Current |
|--------|-------------------|--------------|---------|
| Agent routing accuracy | > 88% | > 95% | ~87% (mock) |
| Voice P95 latency | < 2.5s | < 1.5s | ~4.5s (estimated) |
| API cost per query avg | < ₹0.25 | < ₹0.18 | ~₹0.44 (fixed pipeline) |
| RAGAS faithfulness | > 0.80 | > 0.92 | — (no baseline yet) |
| Context precision | + baseline | +25% vs baseline | — (no baseline yet) |
| KB documents indexed | > 100 | > 10,000 | 32 |
| Test coverage | ≥ 45% | ≥ 80% | ~35% |
| eNAM mandis integrated | 5+ | 1,000+ | 0 (registration pending) |

---

_Update this file at the end of every sprint and after major milestones. AI agents should read this first before starting any work._


---

## 🟢 What Is True Right Now

| Component | Status | Notes |
|-----------|--------|-------|
| Project Structure | ✅ Stable | Advanced folder layout, all modules organized |
| RAG Pipeline | ✅ Stable | RAPTOR + Hybrid Search + Qdrant Cloud |
| Multi-Agent System | ✅ Stable | Supervisor routes with 0.9 confidence |
| Memory System | ✅ Stable | Redis-backed session manager |
| Voice Agent | 🟡 In Progress | Pipecat integration underway, STT/TTS services built |
| APMC Scraper | ❌ Not Started | Scrapling-based scraper planned |
| Vision Agent | ❌ Not Started | YOLOv12 + DINOv2, planned for Phase 3 |
| Supabase Schema | ❌ Not Started | Schema migrations pending |
| Evaluation Framework | ❌ Not Started | LangSmith setup planned |

---

## 🎯 Top 3 Priorities (This Week)

1. **Complete Pipecat voice pipeline** — `src/voice/pipecat_bot.py` integration with real STT/TTS
2. **APMC mandi scraper** — Scrapling-based real-time price data pipeline
3. **Sprint 04 review & Sprint 05 plan** — Close current sprint, plan next with realistic scope

---

## 🚧 Active Blockers

- [ ] `pipecat_bot.py` — Pipecat WebSocket integration needs testing with live audio
- [ ] `APMC_SCRAPER` — No live APMC API; scraping strategy needs validation with Scrapling
- [ ] `Supabase` — Schema not yet migrated; farmer/listing data flows blocked

---

## ✅ Recently Completed Milestones

| Date | Milestone |
|------|-----------|
| Feb 27, 2026 | Advanced folder restructure complete |
| Feb 27, 2026 | Production-grade scraping upgrade (Scrapling + caching + scheduling) |
| Feb 26, 2026 | Voice domain separated into own module (pipecat/) |
| Feb 26, 2026 | Domain separation: RAG, Vision, Voice as distinct domains |
| Jan 10, 2026 | Advanced RAG Phase 1–4 complete (RAPTOR, Hybrid, HyDE, MMR) |
| Jan 9, 2026 | Multi-agent system live (Supervisor + 4 domain agents) |
| Jan 9, 2026 | Voice agent v1 (Edge-TTS + Whisper STT) |

---

## 📦 Stack Reference

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11+ |
| AI Framework | LangGraph + LangChain |
| Vector DB | Qdrant Cloud |
| Graph DB | Neo4j |
| Primary DB | Supabase (PostgreSQL) |
| LLM | Groq (Llama/Mixtral) + Gemini Flash |
| Scraping | Scrapling (Playwright + Camoufox) |
| Voice | Pipecat + Edge-TTS + IndicWhisper |
| Caching | Redis |
| Package Manager | uv |

---

## 📁 Key File Locations

| Purpose | Path |
|---------|------|
| Product Vision | `PLAN.md` |
| Phase Milestones | `ROADMAP.md` |
| Current Sprint | `tracking/sprints/sprint-04-voice-pipeline.md` |
| Daily Logs | `tracking/daily/YYYY-MM-DD.md` |
| Architecture Decisions | `docs/decisions/ADR-*.md` |
| Agent Registry | `docs/agents/REGISTRY.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |

---

## 📈 Lightweight Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Agent routing accuracy | > 90% | ~87% (mock eval) |
| Voice round-trip latency | < 3s | TBD (Pipecat testing) |
| API cost per query | < ₹0.50 | ~₹0.20 (Groq) |
| Knowledge base documents | > 50 | 32 indexed |
| Test coverage | > 60% | ~35% |

---

_Update this file at the end of every sprint and after major milestones. AI agents should read this first before starting any work._
