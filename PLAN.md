# ðŸ”¥ PLAN.md â€” CropFresh AI Master Living Plan

> **Last Updated:** 2026-03-17
> **Status:** Active Development
> **Current Phase:** Phase 2 â€” Business Services
> **Current Sprint:** Sprint 05 â€” Advanced RAG & Documentation
> **Owner:** CropFresh team (solo founder + AI agents)

---

## ðŸŽ¯ Vision

Build India's most intelligent agricultural marketplace, connecting Karnataka farmers directly with buyers using AI agents, voice-first interaction in Kannada, and real-time market intelligence.

**Target Users:**
- **Farmers** (primary) â€” smallholder farmers in Karnataka, primarily Kannada-speaking
- **Buyers** â€” mandis, FPOs, exporters, retail chains
- **Platform operators** â€” CropFresh team managing listings, logistics, payments

---

## ðŸ—ï¸ System Architecture Overview

```
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                    â”‚      Mobile App          â”‚
                    â”‚  (Flutter - Kannada UI)  â”‚
                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                               â”‚ REST / WebSocket
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                    â”‚     FastAPI Backend       â”‚
                    â”‚  Multi-Agent Supervisor   â”‚
                    â””â”€â”€â”¬â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                       â”‚    â”‚    â”‚    â”‚
             â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â”‚    â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
             â–¼              â–¼    â–¼               â–¼
     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
     â”‚  Agronomy  â”‚ â”‚Commerce  â”‚ â”‚  Voice   â”‚ â”‚  Vision  â”‚
     â”‚   Agent   â”‚ â”‚  Agent   â”‚ â”‚  Agent   â”‚ â”‚  Agent   â”‚
     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚              â”‚
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚     Knowledge Layer               â”‚
    â”‚  Qdrant (RAG) â”‚ Neo4j (Graph) â”‚  â”‚
    â”‚  Supabase (DB) â”‚ Redis (Cache) â”‚  â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Current Architecture Focus (March 2026)

- Sprint 05 is still centered on advanced RAG and documentation, but the platform now also includes a shared rate-intelligence layer for Karnataka market data.
- The new `src/rates/` domain provides one official-first aggregation path for mandi wholesale prices, support/reference prices, fuel, gold, and selected retail/validator feeds.
- One `multi_source_rates` capability is now reused by API routes, agent tools, the agentic planner/orchestrator, graph-runtime live retrieval, and APScheduler refresh jobs.
- This keeps price discovery logic in one place and makes it easier to compare sources, track freshness, and add new data connectors without duplicating business rules.
- The next planned sprint is `Sprint 06 - ADCL Productionization & Full Hardening`, with district-first live demand intelligence, persistence, and app wiring now pulled ahead of Supabase/auth follow-up work.

---

## âœ… Core User Flows

### 1. Farmer Onboarding (Priority 1)
`Farmer installs app â†’ registers via OTP/Aadhaar â†’ sets up profile â†’ creates first crop listing`

### 2. Price Discovery (Priority 1)
`Farmer asks (voice/text): "Tomato price in Mysore today?" â†’ planner calls multi_source_rates â†’ official-first rates + comparison evidence are retrieved â†’ response is returned in Kannada`

### 3. Buyer Order (Priority 2)
`Buyer browses listings â†’ matches with the farmer â†’ places order â†’ escrow â†’ delivery â†’ payment released`

### 4. AI Advisory (Priority 2)
`Farmer describes crop problem (voice/photo) â†’ Vision + Agronomy Agent diagnoses â†’ advice returned`

---

## ðŸ§  AI Agents in the System

### 5 Core Agents (Business-Aligned, from PDF Business Model)

| Agent | Domain | Key Capabilities | Status |
|-------|--------|-----------------|--------|
| **DPLE Pricing** | Pricing Engine | AISP = Farmer Ask + Logistics + Margin(4-8%) + Risk Buffer(2%) | Partial |
| **DPLE Logistics** | Route Optimizer | Multi-pickup clustering, vehicle auto-select, deadhead calc | TODO |
| **Matchmaking Engine** | Supply-Demand | GPS clustering, buyer preference matrix, margin optimization | TODO |
| **CV-QG Vision** | Quality Grading | YOLOv8 grading, HITL trigger, Digital Twin, Dispute Diff Engine | TODO |
| **Voice Accessibility** | Farmer Interface | 10+ Indian languages, all farmer flows voice-driven | Partial |
| **RAG Advisory + ADCL** | Crop Intelligence | ADCL weekly demand list, agronomy, pest alerts, forecasts | Partial |

### Supporting Agents

| Agent | Domain | Key Capabilities | Status |
|-------|--------|-----------------|--------|
| Supervisor | Routing | Query classification, 0.9 confidence threshold | Stable |
| Agronomy | Farming | Crop guides, pest/disease, irrigation advice | Stable |
| Commerce | Market | AISP, mandi prices, sell/hold decisions | Stable |
| Platform | App Support | Registration, FAQ, order tracking | Stable |
| General | Fallback | Greetings, unclear queries | Stable |
| ADCL | Crop Demand | Weekly assured demand crop list generator | Partial |

---

## ðŸ”‘ Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Voice response latency | < 3s (< 2s goal) |
| API response latency | < 500ms P95 |
| Agent routing accuracy | > 90% |
| Multi-language support | Kannada (primary), Hindi, English |
| API cost per query | < â‚¹0.50 |
| Uptime (Phase 6+) | > 99.5% |
| Data privacy | No farmer data used for LLM training |

---

## ðŸš© Current Risks & Open Questions

| Risk | Severity | Mitigation |
|------|----------|------------|
| APMC scraping rate limits | High | Scrapling + Camoufox stealth, respectful delays |
| BGE-M3 model memory on low-RAM | Medium | MiniLM fallback configured |
| Kannada ASR accuracy | Medium | IndicWhisper + Groq Whisper fallback |
| Pipecat production stability | Medium | Thorough WebSocket testing before launch |
| Supabase schema changes mid-sprint | Low | ADRs document all schema decisions |

---

## ðŸ“ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI 0.115+ + Python 3.11+ |
| AI Orchestration | LangGraph + LangChain |
| Vector DB | Qdrant Cloud |
| Graph DB | Neo4j |
| Primary DB | Supabase (PostgreSQL) |
| LLM | Groq (Llama-3.3-70B / Mixtral) + Gemini Flash |
| Scraping | Scrapling (Playwright + Camoufox) |
| Voice | Pipecat + Edge-TTS + IndicWhisper/Groq STT |
| Caching | Redis |
| Scheduling | APScheduler |
| Monitoring | LangSmith tracing + Prometheus + Grafana |
| Package Manager | uv |

---

## ðŸ“Š Key Metrics to Track

- Agent routing accuracy (target: > 90%)
- Voice round-trip latency (target: < 3s)
- API cost per query (target: < â‚¹0.50)
- Farmer adoption rate (target: 50 active in Phase 6)
- Successful transaction percent (target: > 80%)
- Test coverage (target: > 60%)

---

## ðŸ“Ž Key Links

| Purpose | Path |
|---------|------|
| Phase Milestones | `ROADMAP.md` |
| Current Status | `tracking/PROJECT_STATUS.md` |
| Sprint Tracking | `tracking/sprints/` |
| Daily Logs | `tracking/daily/` |
| Agent Registry | `docs/agents/REGISTRY.md` |
| Architecture Decisions | `docs/decisions/ADR-*.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |
| API Reference | `docs/api/` |

---

> **Update this file only when direction, architecture, or user personas change materially.**
> For daily progress, use `tracking/daily/`. For sprint goals, use `tracking/sprints/`.

