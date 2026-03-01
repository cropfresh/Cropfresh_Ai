# 🔥 PLAN.md — CropFresh AI Master Living Plan

> **Last Updated:** 2026-02-27
> **Status:** Active Development
> **Current Phase:** Phase 1 — Foundation & Data Pipeline
> **Current Sprint:** Sprint 04 — Voice Pipeline & Scraping
> **Owner:** CropFresh team (solo founder + AI agents)

---

## 🎯 Vision

Build India's most intelligent agricultural marketplace, connecting Karnataka farmers directly with buyers using AI agents, voice-first interaction in Kannada, and real-time market intelligence.

**Target Users:**
- **Farmers** (primary) — smallholder farmers in Karnataka, primarily Kannada-speaking
- **Buyers** — mandis, FPOs, exporters, retail chains
- **Platform operators** — CropFresh team managing listings, logistics, payments

---

## 🏗️ System Architecture Overview

```
                    ┌─────────────────────────┐
                    │      Mobile App          │
                    │  (Flutter - Kannada UI)  │
                    └──────────┬──────────────┘
                               │ REST / WebSocket
                    ┌──────────▼──────────────┐
                    │     FastAPI Backend       │
                    │  Multi-Agent Supervisor   │
                    └──┬────┬────┬────┬───────┘
                       │    │    │    │
             ┌─────────┘    │    │    └──────────┐
             ▼              ▼    ▼               ▼
     ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
     │  Agronomy  │ │Commerce  │ │  Voice   │ │  Vision  │
     │   Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
     └────────────┘ └──────────┘ └──────────┘ └──────────┘
             │              │
    ┌────────┴──────────────┴───────────┐
    │     Knowledge Layer               │
    │  Qdrant (RAG) │ Neo4j (Graph) │  │
    │  Supabase (DB) │ Redis (Cache) │  │
    └───────────────────────────────────┘
```

---

## ✅ Core User Flows

### 1. Farmer Onboarding (Priority 1)
`Farmer installs app → registers via OTP/Aadhaar → sets up profile → creates first crop listing`

### 2. Price Discovery (Priority 1)
`Farmer asks (voice/text): "Tomato price in Mysore today?" → AI retrieves APMC data → responds in Kannada`

### 3. Buyer Order (Priority 2)
`Buyer browses listings → matches with the farmer → places order → escrow → delivery → payment released`

### 4. AI Advisory (Priority 2)
`Farmer describes crop problem (voice/photo) → Vision + Agronomy Agent diagnoses → advice returned`

---

## 🧠 AI Agents in the System

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
| ADCL | Crop Demand | Weekly assured demand crop list generator | TODO |

---

## 🔑 Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Voice response latency | < 3s (< 2s goal) |
| API response latency | < 500ms P95 |
| Agent routing accuracy | > 90% |
| Multi-language support | Kannada (primary), Hindi, English |
| API cost per query | < ₹0.50 |
| Uptime (Phase 6+) | > 99.5% |
| Data privacy | No farmer data used for LLM training |

---

## 🚩 Current Risks & Open Questions

| Risk | Severity | Mitigation |
|------|----------|------------|
| APMC scraping rate limits | High | Scrapling + Camoufox stealth, respectful delays |
| BGE-M3 model memory on low-RAM | Medium | MiniLM fallback configured |
| Kannada ASR accuracy | Medium | IndicWhisper + Groq Whisper fallback |
| Pipecat production stability | Medium | Thorough WebSocket testing before launch |
| Supabase schema changes mid-sprint | Low | ADRs document all schema decisions |

---

## 📐 Tech Stack

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

## 📊 Key Metrics to Track

- Agent routing accuracy (target: > 90%)
- Voice round-trip latency (target: < 3s)
- API cost per query (target: < ₹0.50)
- Farmer adoption rate (target: 50 active in Phase 6)
- Successful transaction percent (target: > 80%)
- Test coverage (target: > 60%)

---

## 📎 Key Links

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
