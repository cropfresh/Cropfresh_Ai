# 📊 PROJECT_STATUS.md — CropFresh AI Current State

> **Last Updated:** 2026-03-01 (14:15 IST)
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
| **Buyer Matching Agent** | ✅ Updated | ✅ Task 2 complete: 5-factor scoring + reverse matching + cache + routing |
| **Pricing Agent (DPLE)** | ✅ Updated | ✅ Task 1 complete: deadhead + 2% risk buffer + mandi cap + trend/seasonality |
| **Voice Agent** | 🟡 Partial | ❌ All major intents have `# TODO` stubs |
| Pipecat Voice Pipeline | 🟡 In Progress | ❌ WebSocket not fully tested |
| **Matchmaking Engine** | 🟡 In Progress | ✅ Buyer-side matching core implemented; full cluster optimization pending |
| **CV-QG Vision Agent** | ✅ Task 3 Complete | ✅ Quality pipeline + fallback + HITL + digital twin linkage + supervisor wiring |
| **Digital Twin Engine** | ❌ Not Started | ❌ Critical for <2% dispute target |
| **DPLE Logistics Routing** | ❌ Not Started | ❌ Multi-pickup clustering missing |
| **ADCL Agent** | ❌ Not Started | ❌ Weekly demand list generator |
| APMC Live Scraper | ❌ Not Started | ❌ eNAM registration still pending |
| RDS PostgreSQL Schema | ✅ Created | ✅ `schema.sql` + `postgres_client.py` ready |
| Vision Agent (ai/vision/) | ❌ Not Started | ❌ Only placeholder exists |
| RAGAS Evaluation | ❌ Not Started | ❌ No golden dataset yet |

---

## 🎯 Current Sprint 05 — Core Agent Completion (Phase 1)

**Goal:** Bring all 5 core AI agents from stub → working implementation

### Sprint Tasks (2026-03-01 to 2026-03-14)

| # | Task | File | Priority |
|---|------|------|----------|
| 1 | ✅ Fix Pricing Agent (deadhead + risk buffer + mandi cap + trend/seasonality) | `src/agents/pricing_agent.py` | 🔴 P0 |
| 2 | ✅ Implement Buyer Matching Engine core (5-factor score + reverse matching + cache) | `src/agents/buyer_matching/agent.py` | 🔴 P0 |
| 3 | ✅ Implement Quality Assessment Agent | `src/agents/quality_assessment/agent.py` | 🔴 P0 |
| 4 | Wire Voice Agent TODOs to real agents | `src/agents/voice_agent.py` | 🔴 P0 |
| 5 | ✅ Write unit tests for quality + routing increment | `tests/unit/` | 🟠 P1 |
| 6 | ~~Create Supabase schema~~ → RDS schema done | `src/db/schema.sql` | ✅ Done |
| 7 | Add 5 new voice intents | `src/agents/voice_agent.py` | 🟡 P2 |

### Sprint KPIs
- [ ] 0 `NotImplementedError` in core agents
- [x] AISP calculation includes risk buffer
- [x] Matching returns at least 1 valid farmer→buyer pair in unit test
- [ ] Voice `create_listing` intent creates a DB record (stub OK)
- [ ] Test coverage increases from 35% → 45%

---

## 🗺️ 6-Phase Upgrade Roadmap

| Phase | Focus | Target | Status |
|-------|-------|--------|--------|
| **Phase 1** (Wk 1–4) | Core 5 agents → working | 0 stubs, unit tests | 🟢 Active |
| **Phase 2** (Wk 3–6) | CV-QG Vision + Digital Twin | <500ms grading | 📋 Planned |
| **Phase 3** (Wk 5–8) | DPLE Logistics Routing | <₹2.5/kg, >70% util | 📋 Planned |
| **Phase 4** (Wk 7–10) | RDS PostgreSQL + Live Scraper | Real mandi data | 📋 Planned |
| **Phase 5** (Wk 9–12) | Voice in 10+ languages + ADCL | Zero literacy barrier | 📋 Planned |
| **Phase 6** (Wk 12–18) | Production hardening | 99.5% uptime | 📋 Planned |

---

## 🚧 Active Blockers

| Blocker | Impact | Owner |
|---------|--------|-------|
| eNAM API registration pending | Live mandi data blocked | external |
| AWS Free Tier limit blocks RDS creation | Production DB not provisioned yet | manual (upgrade account) |
| pgvector not installed locally (needs VS Build Tools) | Local vector search uses Qdrant fallback | dev workaround |
| Pipecat WebSocket untested on Windows | Voice streaming broken | Sprint 05 |
| YOLOv8 model weights not downloaded | CV-QG grading blocked | Phase 2 |

---

## ✅ Completed Milestones

| Date | Milestone |
|------|-----------|
| **Mar 01, 2026** | **Task 3 complete — Quality Assessment Agent**: A+/A/B/C grading pipeline, HITL threshold at `0.7`, digital twin assessment IDs, fallback mode without model weights, and supervisor/chat integration with passing tests |
| **Mar 01, 2026** | **Live streaming static UI validated**: ChatGPT-style token streaming UI shipped on `static/voice_realtime.html` using `POST /api/v1/chat/stream`, with runtime verification for health, static serving, and SSE events |
| **Mar 01, 2026** | **Task 2 complete — Buyer Matching Engine**: 5-factor weighted scoring, haversine proximity, reverse matching, 5-min cache, and supervisor routing integration with passing unit tests |
| **Mar 01, 2026** | **Task 1 complete — Pricing Agent (DPLE)**: implemented utilization-based deadhead, fixed 2% risk buffer, mandi cap at `modal × 1.05`, plus trend and seasonality methods with passing unit tests |
| **Mar 01, 2026** | **LLM provider migration** — Groq → AWS Bedrock (dual-provider strategy with `BedrockProvider`) |
| **Mar 01, 2026** | **Database migration started** — Qdrant+Supabase → RDS PostgreSQL+pgvector (`AuroraPostgresClient`, `schema.sql`) |
| **Mar 01, 2026** | **AWS infra provisioned** — subnet group + security group created, RDS blocked by free tier |
| **Mar 01, 2026** | **Local PostgreSQL setup** — `setup_local_db.sql` for dev, Qdrant fallback for vectors |
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
| **LLM (Production)** | **AWS Bedrock — Claude Sonnet 4** |
| **LLM (Router/Dev)** | **Groq — Llama-3.1-8B (~80ms)** |
| **Primary DB + Vectors** | **RDS PostgreSQL + pgvector** (replaces Supabase + Qdrant) |
| Graph DB | Neo4j |
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
| AISP accuracy | ±5% of real landed cost | Correct formula | Formula aligned (Task 1 complete) |
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
| Architecture Decisions | `docs/decisions/ADR-*.md` (ADR-001 to ADR-012) |
| RAG Architecture | `docs/rag_architecture.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |

---

_Update this file at the end of every sprint and after major milestones._
_AI agents: read this + ARCHITECTURE.md before starting any work._
