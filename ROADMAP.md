# ROADMAP.md — CropFresh AI Milestones & Phases

> **Last Updated:** 2026-03-01
> **Current Phase:** Phase 2 — Business Services (active)
> **Horizon:** 6-month view (Feb → Aug 2026)

---

## Overview

CropFresh AI builds India's most intelligent agricultural marketplace, connecting Karnataka farmers with buyers via AI agents, voice-first Kannada interaction, and real-time market intelligence. This roadmap tracks major milestones, deliverables, and success criteria per phase.

---

## Phase 1 — Foundation & Core Agents
**Duration:** Feb 2026 → Mar 2026
**Status:** ✅ Complete

### Goals
- Stand up production-grade FastAPI backend with full project structure
- Complete RAG pipeline with PostgreSQL pgvector for agricultural knowledge
- Establish multi-agent architecture (Supervisor + domain agents)
- All 5 core AI agents fully implemented with no `NotImplementedError`
- Voice agent with Kannada/Hindi/English intent handling

### Key Deliverables
- [x] Advanced folder structure & project scaffold
- [x] RAG pipeline (pgvector + RAPTOR + Hybrid Search)
- [x] Multi-agent system (Supervisor, Agronomy, Commerce, Voice, Quality)
- [x] Voice agent — 10+ intents, multi-turn flows, 3-language templates (Task 4)
- [x] Pricing Agent (DPLE) — deadhead, risk buffer, mandi cap, trend/seasonality (Task 1)
- [x] Buyer Matching Engine — 5-factor scoring, reverse matching, Redis cache (Task 2)
- [x] Quality Assessment Agent — HITL threshold, A+/A/B/C grading, digital twin linkage (Task 3)
- [x] Price Prediction Agent — rule-based, numpy trend, Karnataka seasonal calendar, LLM fallback (Task 5)
- [x] LLM migration — Groq + AWS Bedrock dual-provider strategy

### Success Criteria
- [x] 0 `NotImplementedError` in core agent paths
- [x] Voice `create_listing` creates real DB record (AC6 verified)
- [x] AISP calculation includes risk buffer

---

## Phase 2 — Business Services (DB + Marketplace APIs)
**Duration:** Mar 2026 → Apr 2026
**Status:** 🟢 Active — Tasks 6–9 complete, Tasks 10–11 next

### Goals
- Production-grade PostgreSQL schema for all business entities
- Crop listing marketplace API fully wired end-to-end
- Order lifecycle with escrow state machine
- Farmer/buyer registration with OTP + JWT

### Key Deliverables
- [x] Database Schema — 10 business tables, PostGIS GEOGRAPHY, migration runner, SHA-256 checksums (Task 6)
- [x] Crop Listing Service — auto-price, shelf-life expiry, QR code, ADCL tag, 7 REST endpoints (Task 7)
- [x] Order Management Service — 11-status state machine, escrow flow, AISP breakdown, Digital Twin dispute diff, 8 REST endpoints (Task 8)
- [x] Farmer/Buyer Registration API — OTP phone auth, stdlib JWT, profile CRUD, 31-district language preference, voice `register_farmer()` compat, 6 REST endpoints (Task 9)
- [ ] APMC mandi data scraper — Scrapling-based, 8 Karnataka mandis, 11 commodities (Task 13)
- [ ] TESTING/STRATEGY.md and end-to-end test checklists (Task 19)

### Success Criteria
- End-to-end: farmer registers → creates listing → order created → escrow held → settled
- All 21 REST endpoints (7 listings + 8 orders + 6 auth) pass integration tests
- Test coverage ≥ 45% (**current: ~57%** — 340 tests / 15 files)

---

## Phase 3 — Intelligence & Digital Twin
**Duration:** Apr 2026 → May 2026
**Status:** 🔲 Planned

### Goals
- Digital Twin Engine for departure/arrival quality comparison
- DPLE Logistics Routing (HDBSCAN + OR-Tools CVRP)
- ADCL Agent for weekly crop demand list
- Real-time APMC price pipeline

### Key Deliverables
- [x] Digital Twin Engine — departure twin, arrival comparison, SSIM diff, liability matrix (Task 10)
- [ ] DPLE Logistics Routing — multi-pickup clustering, <₹2.5/kg target (Task 11)
- [ ] ADCL Agent — weekly demand list, 90-day order data aggregation (Task 12)
- [ ] APMC Live Scraper — Agmarknet.gov.in, 10 AM IST daily scrape (Task 13)
- [ ] Historical price analysis dashboard

### Success Criteria
- Dispute diff engine detects quality degradation in >80% of test cases
- Logistics cost per kg < ₹2.5 on multi-pickup routes
- Price prediction within ±10% of actual mandi price for major crops

---

## Phase 4 — Mobile & WhatsApp
**Duration:** Apr 2026 → Jun 2026
**Status:** 🔲 Planned

### Goals
- Flutter mobile app prototype (farmer-facing)
- WhatsApp bot for order updates, price alerts, listing creation
- Kannada voice fully production-grade (Pipecat WebSocket e2e)

### Key Deliverables
- [ ] Pipecat WebSocket voice streaming — Silero VAD + IndicWhisper + EdgeTTS (Task 14)
- [ ] WhatsApp Bot Agent — Meta Cloud API, text/voice/image/location handling (Task 15)
- [ ] Voice Agent in 10+ languages — kn, hi, en, ta, te, mr, bn, gu, pa, ml (Task 16)
- [ ] Flutter mobile app (iOS + Android prototype)
- [ ] Push notifications for price alerts and order status

### Success Criteria
- Voice round-trip latency < 2s P95 for Kannada
- WhatsApp bot answers price and weather queries accurately
- 10 internal testers using mobile app for crop queries

---

## Phase 5 — Testing, Evaluation & Hardening
**Duration:** Jun 2026 → Jul 2026
**Status:** 🔲 Planned

### Goals
- Comprehensive test coverage (unit + integration + E2E)
- RAGAS evaluation framework with golden dataset
- Production security hardening

### Key Deliverables
- [ ] Unit Tests — coverage 35% → 60% across all agents + services (Task 17)
- [ ] RAGAS Evaluation Framework — 50+ Q&A golden dataset, faithfulness/relevancy/precision metrics (Task 18)
- [ ] Integration Tests — 3 E2E flows: farmer listing, order lifecycle, voice round-trip (Task 19)
- [ ] Production Hardening — JWT auth, RBAC, rate limiting, structured logging, Prometheus metrics (Task 20)

### Success Criteria
- Test coverage ≥ 60%
- RAGAS faithfulness > 0.80
- 0 P0 security vulnerabilities

---

## Phase 6 — Beta Launch (50-Farmer Karnataka Pilot)
**Duration:** Jul 2026 → Aug 2026
**Status:** 🔲 Planned

### Goals
- 50-farmer pilot in Karnataka districts
- Performance optimization (< 2s P95 response time)
- Monitoring dashboards live

### Key Deliverables
- [ ] Security audit (auth, rate limiting, API key rotation)
- [ ] Performance profiling and optimization
- [ ] Farmer onboarding kit (Kannada guide + video)
- [ ] Monitoring dashboards (Grafana + Prometheus)
- [ ] Bug bash and regression test suite

### Success Criteria
- 50 farmers onboarded, 20+ active weekly users
- < 1% error rate in prod, < 2s P95 latency
- NPS score > 40 from pilot farmers

---

## High-Level Timeline

| Phase | Period | Status |
|-------|--------|--------|
| Phase 1 — Foundation & Core Agents | Feb–Mar 2026 | ✅ Complete |
| Phase 2 — Business Services | Mar–Apr 2026 | 🟢 Active (Tasks 6-9 done) |
| Phase 3 — Intelligence & Digital Twin | Apr–May 2026 | 🔲 Planned |
| Phase 4 — Mobile & WhatsApp | Apr–Jun 2026 | 🔲 Planned |
| Phase 5 — Testing & Evaluation | Jun–Jul 2026 | 🔲 Planned |
| Phase 6 — Beta Launch | Jul–Aug 2026 | 🔲 Planned |

---

## Links
- Product vision & tech plan: [`PLAN.md`](./PLAN.md)
- Current sprint: [`tracking/sprints/sprint-05-core-agents.md`](./tracking/sprints/sprint-05-core-agents.md)
- Architecture decisions: [`docs/decisions/`](./docs/decisions/)
- Project status: [`tracking/PROJECT_STATUS.md`](./tracking/PROJECT_STATUS.md)

