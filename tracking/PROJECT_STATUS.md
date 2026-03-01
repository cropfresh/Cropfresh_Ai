# 📊 PROJECT_STATUS.md — CropFresh AI Current State

> **Last Updated:** 2026-02-28 (21:14 IST)
> **Version/Tag:** `v0.5-business-aligned`
> **Current Sprint:** Sprint 05 — Core Agent Completion (Phase 1 of Upgrade)
> **Sprint Status:** 🟢 Active (starts 2026-03-01)

---

## 🎯 North Star (From Business Model PDF)

Build India's most trusted AI-powered agri-marketplace with:
- **5 AI Agents** working together as the intelligence layer
- Farmer income up **+40–60%** vs. mandi
- Disputes **<2%** via Digital Twin + HITL
- Logistics cost **<₹2.5/kg** via DPLE mesh routing
- Voice-first UI for **zero-literacy-barrier** farmer access

---

## 🟢 What Is True Right Now

| Component | Status | Accuracy vs Business Model |
|-----------|--------|---------------------------|
| RAG Pipeline (RAPTOR + Hybrid) | ✅ Stable | ✅ Fully aligned |
| Multi-Agent Supervisor | ✅ Stable | 🟡 Routes exist, some agents are stubs |
| Memory System (Redis) | ✅ Stable | ✅ OK |
| Agronomy Agent | ✅ Stable | ✅ Good, needs ADCL integration |
| Commerce Agent | ✅ Stable | 🟡 AISP math exists, not fully wired |
| **Pricing Agent (DPLE)** | 🟡 Partial | ❌ Missing: deadhead factor, risk buffer, mandi cap |
| **Voice Agent** | 🟡 Partial | ❌ All major intents have `# TODO` stubs |
| Pipecat Voice Pipeline | 🟡 In Progress | ❌ WebSocket not fully tested |
| **Matchmaking Engine** | ❌ Stub | ❌ `NotImplementedError` — highest priority |
| **CV-QG Vision Agent** | ❌ Not Started | ❌ YOLOv8 code not written |
| **Digital Twin Engine** | ❌ Not Started | ❌ Critical for <2% dispute target |
| **DPLE Logistics Routing** | ❌ Not Started | ❌ Multi-pickup clustering missing |
| **ADCL Agent** | ❌ Not Started | ❌ Weekly demand list generator |
| APMC Live Scraper | ❌ Not Started | ❌ eNAM registration still pending |
| Supabase DB Schema | ❌ Not Started | ❌ All transactional tables missing |
| Vision Agent (ai/vision/) | ❌ Not Started | ❌ Only placeholder exists |
| RAGAS Evaluation | ❌ Not Started | ❌ No golden dataset yet |

---

## 🎯 Current Sprint 05 — Core Agent Completion (Phase 1)

**Goal:** Bring all 5 core AI agents from stub → working implementation

### Sprint Tasks (2026-03-01 to 2026-03-14)

| # | Task | File | Priority |
|---|------|------|----------|
| 1 | Fix Pricing Agent (deadhead + risk buffer + mandi cap) | `src/agents/pricing_agent.py` | 🔴 P0 |
| 2 | Implement Matchmaking Engine | `src/agents/matchmaking_agent.py` | 🔴 P0 |
| 3 | Implement Quality Assessment Agent | `src/agents/quality_assessment/agent.py` | 🔴 P0 |
| 4 | Wire Voice Agent TODOs to real agents | `src/agents/voice_agent.py` | 🔴 P0 |
| 5 | Write unit tests for all Phase 1 agents | `tests/unit/` | 🟠 P1 |
| 6 | Create Supabase schema migrations | `config/supabase_schema.sql` | 🟠 P1 |
| 7 | Add 5 new voice intents | `src/agents/voice_agent.py` | 🟡 P2 |

### Sprint KPIs
- [ ] 0 `NotImplementedError` in core agents
- [ ] AISP calculation includes risk buffer
- [ ] Matching returns at least 1 valid farmer→buyer pair in unit test
- [ ] Voice `create_listing` intent creates a DB record (stub OK)
- [ ] Test coverage increases from 35% → 45%

---

## 🗺️ 6-Phase Upgrade Roadmap

| Phase | Focus | Target | Status |
|-------|-------|--------|--------|
| **Phase 1** (Wk 1–4) | Core 5 agents → working | 0 stubs, unit tests | 🟢 Active |
| **Phase 2** (Wk 3–6) | CV-QG Vision + Digital Twin | <500ms grading | 📋 Planned |
| **Phase 3** (Wk 5–8) | DPLE Logistics Routing | <₹2.5/kg, >70% util | 📋 Planned |
| **Phase 4** (Wk 7–10) | Supabase + Live Scraper | Real mandi data | 📋 Planned |
| **Phase 5** (Wk 9–12) | Voice in 10+ languages + ADCL | Zero literacy barrier | 📋 Planned |
| **Phase 6** (Wk 12–18) | Production hardening | 99.5% uptime | 📋 Planned |

---

## 🚧 Active Blockers

| Blocker | Impact | Owner |
|---------|--------|-------|
| eNAM API registration pending | Live mandi data blocked | external |
| Supabase schema not migrated | Farmer/listing DB flows blocked | Sprint 05 |
| Pipecat WebSocket untested on Windows | Voice streaming broken | Sprint 05 |
| YOLOv8 model weights not downloaded | CV-QG grading blocked | Phase 2 |

---

## ✅ Completed Milestones

| Date | Milestone |
|------|-----------|
| Feb 28, 2026 | **Business model analysis complete** — PDF mapped to exact code gaps |
| Feb 28, 2026 | **Architecture docs updated** — ARCHITECTURE.md v0.5 with 5-agent model |
| Feb 28, 2026 | **Upgrade implementation plan created** — 6 phases, 18 weeks |
| Feb 27, 2026 | RAG 2027 Research — 4 ADRs, 4 arch docs, sprint-05 + sprint-06 planned |
| Feb 27, 2026 | Advanced dev workflow documentation system established |
| Feb 27, 2026 | Production-grade scraping upgrade (Scrapling + eNAM + IMD clients) |
| Feb 26, 2026 | Voice domain separated; Pipecat submodule built |
| Jan 10, 2026 | Advanced RAG Phase 1–4 (RAPTOR, Hybrid, HyDE, MMR, Contextual Chunking) |
| Jan 9, 2026 | Multi-agent system live (Supervisor + 4 domain agents) |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11+ |
| AI Framework | LangGraph + LangChain |
| Vision | YOLOv8 + ViT-B/16 + ONNX Runtime |
| Vector DB | Qdrant Cloud |
| Graph DB | Neo4j |
| Primary DB | Supabase (PostgreSQL) |
| LLM (Generation) | Gemini Flash 2.0 + Groq Llama-3.3-70B |
| LLM (Router) | Groq Llama-3.1-8B-Instant (~80ms, ~₹0.001/call) |
| Scraping | Scrapling + Camoufox (stealth) |
| Voice | Pipecat + Edge-TTS + IndicWhisper |
| Caching | Redis |
| Embeddings | BGE-M3 + AgriEmbeddingWrapper |
| Evaluation | RAGAS + LangSmith |
| Monitoring | Prometheus + Grafana |
| Package Manager | uv |

---

## 📈 Business-Aligned Metrics Dashboard

| Metric | PDF Target | Sprint 05 Target | Current |
|--------|-----------|-----------------|---------|
| AI grading accuracy | >95% (Month 12) | N/A (Phase 2) | — |
| Dispute rate | <2% | — | — |
| Voice P95 latency | <2s | <3s | ~4.5s (est.) |
| AISP accuracy | ±5% of real landed cost | Correct formula | Risk buffer missing |
| Logistics cost/kg | <₹2.5/kg | — | — |
| Agent routing accuracy | >90% | >88% | ~87% (mock) |
| API cost per query avg | <₹0.25 | <₹0.30 | ~₹0.44 |
| RAGAS faithfulness | >0.80 | baseline run | — (no baseline) |
| KB documents indexed | >1,000 | >100 | 32 |
| Test coverage | >80% (Phase 6) | >45% | ~35% |
| eNAM mandis integrated | 1,000+ (Y2) | 5+ | 0 |

---

## 📁 Key File Locations

| Purpose | Path |
|---------|------|
| Product Vision | `PLAN.md` |
| Phase Milestones | `ROADMAP.md` |
| **Architecture (v0.5)** | `docs/architecture/ARCHITECTURE.md` |
| **Upgrade Plan** | `C:/Users/shrik/.gemini/antigravity/brain/.../implementation_plan.md` |
| Current Sprint | `tracking/sprints/sprint-05-core-agents.md` |
| Architecture Decisions | `docs/decisions/ADR-*.md` (ADR-001 to ADR-010) |
| RAG Architecture | `docs/rag_architecture.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |

---

_Update this file at the end of every sprint and after major milestones._
_AI agents: read this + ARCHITECTURE.md before starting any work._
