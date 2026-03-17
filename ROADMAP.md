# ROADMAP.md - CropFresh AI Milestones & Phases

> **Last Updated:** 2026-03-17
> **Current Phase:** Phase 2 - Business Services (active)
> **Horizon:** 6-month view (Feb -> Aug 2026)

---

## Overview

CropFresh AI builds India's intelligent agricultural marketplace, connecting Karnataka farmers with buyers via AI agents, voice-first Kannada interaction, and real-time market intelligence.

### Sprint Sequencing Note (2026-03-17)

Sprint 06 is now reserved for **ADCL Productionization & Full Hardening** as the first district-first Phase 3 delivery slice. Supabase and user-management follow-up work moves behind it into Sprint 07 so the next build session can focus on one production-grade ADCL path instead of splitting effort across unrelated tracks.

---

## Near-Term Sprint Sequencing

| Sprint | Period | Focus | Status |
|--------|--------|-------|--------|
| Sprint 05 | 2026-03-10 -> 2026-03-24 | Advanced RAG, docs, and multi-source rate-hub hardening | In Progress |
| Sprint 06 | 2026-03-25 -> 2026-04-07 | ADCL productionization, live data, persistence, API wiring, evaluation | Planned |
| Sprint 07 | 2026-04-08 -> 2026-04-21 | Supabase/auth hardening and deferred marketplace follow-up | Planned |

---

## Phase 1 - Foundation & Core Agents

**Duration:** Feb 2026 -> Mar 2026
**Status:** Complete

### Goals

- Stand up production-grade FastAPI backend with full project structure
- Complete RAG pipeline with PostgreSQL pgvector for agricultural knowledge
- Establish multi-agent architecture (Supervisor + domain agents)
- All 5 core AI agents fully implemented with no `NotImplementedError`
- Voice agent with Kannada/Hindi/English intent handling

### Key Deliverables

- [x] Advanced folder structure and project scaffold
- [x] RAG pipeline (pgvector + RAPTOR + Hybrid Search)
- [x] Multi-agent system (Supervisor, Agronomy, Commerce, Voice, Quality)
- [x] Voice agent - 10+ intents, multi-turn flows, 3-language templates (Task 4)
- [x] Pricing Agent (DPLE) - deadhead, risk buffer, mandi cap, trend/seasonality (Task 1)
- [x] Buyer Matching Engine - 5-factor scoring, reverse matching, Redis cache (Task 2)
- [x] Quality Assessment Agent - HITL threshold, A+/A/B/C grading, digital twin linkage (Task 3)
- [x] Price Prediction Agent - rule-based, numpy trend, Karnataka seasonal calendar, LLM fallback (Task 5)
- [x] LLM migration - Groq + AWS Bedrock dual-provider strategy

### Success Criteria

- [x] 0 `NotImplementedError` in core agent paths
- [x] Voice `create_listing` creates real DB record
- [x] AISP calculation includes risk buffer

---

## Phase 2 - Business Services (DB + Marketplace APIs)

**Duration:** Mar 2026 -> Apr 2026
**Status:** Active - core marketplace services exist; follow-up hardening continues behind Sprint 06

### Goals

- Stabilize production-grade PostgreSQL business entities and service wiring
- Keep crop listing marketplace APIs and order workflows reliable end-to-end
- Harden farmer and buyer onboarding flows

### Key Deliverables

- [x] Database schema and migration runner foundation
- [x] Crop listing marketplace APIs
- [x] Order lifecycle and escrow workflow
- [x] Farmer and buyer registration foundation
- [ ] Remaining hardening and cleanup tasks moved to Sprint 07

### Success Criteria

- End-to-end: farmer registers -> creates listing -> order created -> escrow held -> settled
- Core marketplace APIs remain stable while ADCL is integrated as a new production surface

---

## Phase 3 - Intelligence & Digital Twin

**Duration:** Apr 2026 -> May 2026
**Status:** Queued next via Sprint 06

### Goals

- Digital Twin Engine for departure and arrival quality comparison
- DPLE logistics routing
- ADCL as a production-grade weekly crop-demand intelligence service
- Real-time market and weather context feeding recommendation quality

### Key Deliverables

- [x] Digital Twin Engine - departure twin, arrival comparison, SSIM diff, liability matrix (Task 10)
- [ ] Sprint 06 - ADCL productionization and full hardening
- [ ] DPLE Logistics Routing - multi-pickup clustering, target <Rs2.5/kg (Task 11)
- [ ] Real-time APMC and external context hardening
- [ ] Historical price analysis dashboard

### Success Criteria

- Dispute diff engine detects quality degradation in >80% of test cases
- Logistics cost per kg < Rs2.5 on multi-pickup routes
- ADCL weekly report is live, district-aware, evidence-backed, and reused across REST, voice, and listings

---

## Phase 4 - Mobile & WhatsApp

**Duration:** Apr 2026 -> Jun 2026
**Status:** Planned

### Goals

- Flutter mobile app prototype (farmer-facing)
- WhatsApp bot for order updates, price alerts, and listing creation
- Kannada voice fully production-grade

### Key Deliverables

- [ ] Pipecat WebSocket voice streaming
- [ ] WhatsApp Bot Agent
- [ ] Voice agent in 10+ languages
- [ ] Flutter mobile app prototype
- [ ] Push notifications for price alerts and order status

### Success Criteria

- Voice round-trip latency < 2s P95 for Kannada
- WhatsApp bot answers price and weather queries accurately
- 10 internal testers using the mobile app for crop queries

---

## Phase 5 - Testing, Evaluation & Hardening

**Duration:** Jun 2026 -> Jul 2026
**Status:** Planned

### Goals

- Comprehensive test coverage
- Evaluation framework with golden datasets
- Production security hardening

### Key Deliverables

- [ ] Unit test coverage growth across agents and services
- [ ] RAGAS evaluation framework
- [ ] Integration tests for end-to-end flows
- [ ] Production hardening for auth, RBAC, rate limiting, logging, and monitoring

### Success Criteria

- Test coverage >= 60%
- RAGAS faithfulness > 0.80
- 0 P0 security vulnerabilities

---

## Phase 6 - Beta Launch (50-Farmer Karnataka Pilot)

**Duration:** Jul 2026 -> Aug 2026
**Status:** Planned

### Goals

- 50-farmer pilot in Karnataka districts
- Performance optimization (< 2s P95 response time)
- Monitoring dashboards live

### Key Deliverables

- [ ] Security audit
- [ ] Performance profiling and optimization
- [ ] Farmer onboarding kit
- [ ] Monitoring dashboards
- [ ] Bug bash and regression test suite

### Success Criteria

- 50 farmers onboarded, 20+ active weekly users
- <1% error rate in prod, <2s P95 latency
- NPS score > 40 from pilot farmers

---

## High-Level Timeline

| Phase | Period | Status |
|-------|--------|--------|
| Phase 1 - Foundation & Core Agents | Feb-Mar 2026 | Complete |
| Phase 2 - Business Services | Mar-Apr 2026 | Active |
| Phase 3 - Intelligence & Digital Twin | Apr-May 2026 | Queued next |
| Phase 4 - Mobile & WhatsApp | Apr-Jun 2026 | Planned |
| Phase 5 - Testing & Evaluation | Jun-Jul 2026 | Planned |
| Phase 6 - Beta Launch | Jul-Aug 2026 | Planned |

---

## Links

- Product vision and tech plan: [`PLAN.md`](./PLAN.md)
- Current sprint: [`tracking/sprints/sprint-05-advanced-rag.md`](./tracking/sprints/sprint-05-advanced-rag.md)
- Next sprint: [`tracking/sprints/sprint-06-adcl-productionization.md`](./tracking/sprints/sprint-06-adcl-productionization.md)
- Architecture decisions: [`docs/decisions/`](./docs/decisions/)
- Project status: [`tracking/PROJECT_STATUS.md`](./tracking/PROJECT_STATUS.md)
