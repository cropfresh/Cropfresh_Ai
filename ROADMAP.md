# 🗺️ ROADMAP.md — CropFresh AI Milestones & Phases

> **Last Updated:** 2026-02-27
> **Current Phase:** Phase 1 — Foundation & Data Pipeline
> **Horizon:** 6-month view (Feb → Aug 2026)

---

## Overview

CropFresh AI builds India's most intelligent agricultural marketplace, connecting Karnataka farmers with buyers via AI agents, voice-first Kannada interaction, and real-time market intelligence. This roadmap tracks major milestones, deliverables, and success criteria per phase.

---

## 🏁 Phase 1 — Foundation & Data Pipeline
**Duration:** Feb 2026 → Mar 2026  
**Status:** 🟡 In Progress

### Goals
- Stand up production-grade FastAPI backend with full project structure
- Complete RAG pipeline with Qdrant + Neo4j for agricultural knowledge
- Build APMC mandi data scraping pipeline (real-time prices)
- Establish multi-agent architecture (Supervisor + domain agents)
- Voice agent with Kannada STT/TTS

### Key Deliverables
- [x] Advanced folder structure & project scaffold
- [x] RAG pipeline (Qdrant vector DB + RAPTOR + Hybrid Search)
- [x] Multi-agent system (Supervisor, Agronomy, Commerce, Platform, Voice)
- [x] Voice agent (Pipecat + Edge-TTS + Whisper)
- [ ] APMC mandi data scraper (Scrapling-based)
- [ ] Supabase schema: farmers, listings, transactions
- [ ] TESTING/STRATEGY.md and first test checklists

### Success Criteria
- All core agents route correctly with > 90% accuracy on eval set
- Voice round-trip latency < 3s for Kannada
- At least 50 agricultural documents in Qdrant knowledge base

---

## 🚀 Phase 2 — First Agent MVP (Crop Listing)
**Duration:** Mar 2026 → Apr 2026  
**Status:** 🔲 Not Started

### Goals
- Farmer can register, create crop listing, and get AI pricing advice
- Evaluation framework for all agents (LangSmith + custom metrics)
- AI Kosha integration for government agricultural data
- Agent prompt versioning system

### Key Deliverables
- [ ] Crop Listing Agent (draft listing from farmer voice input)
- [ ] Farmer Registration API (Supabase)
- [ ] Evaluation framework with LangSmith baselines
- [ ] Prompt versioning & A/B comparison tooling
- [ ] API Reference docs for all MVP endpoints

### Success Criteria
- End-to-end: farmer registers → creates listing → gets price advice (all via voice)
- Agent evaluation framework capturing accuracy, latency, cost per query

---

## 📊 Phase 3 — Market Intelligence
**Duration:** Apr 2026 → May 2026  
**Status:** 🔲 Not Started

### Goals
- Real-time price prediction with historical APMC data
- Weather integration with agro-advisories
- Buyer matching prototype (ML-based demand signal)

### Key Deliverables
- [ ] Price Prediction Agent (LSTM/Prophet + APMC historical data)
- [ ] IMD Weather integration with farming advisories
- [ ] Historical price analysis dashboard (internal)
- [ ] Buyer Matching prototype (rule-based v1)

### Success Criteria
- Price prediction within ±10% of actual mandi price for major crops
- Weather advisory delivered alongside every crop listing

---

## 📱 Phase 4 — Mobile & WhatsApp (In Parallel)
**Duration:** Apr 2026 → Jun 2026  
**Status:** 🔲 Not Started

### Goals
- Flutter mobile app prototype (farmer-facing)
- WhatsApp bot for market price queries and alerts
- Kannada voice fully production-grade

### Key Deliverables
- [ ] Flutter mobile app (iOS + Android prototype)
- [ ] WhatsApp Business API bot integration
- [ ] Kannada voice agent (production-grade, < 2s latency)
- [ ] Push notifications for price alerts

### Success Criteria
- 10 internal testers using mobile app for crop queries
- WhatsApp bot answers price and weather queries accurately

---

## 🏪 Phase 5 — Marketplace Core
**Duration:** Jun 2026 → Jul 2026  
**Status:** 🔲 Not Started

### Goals
- Buyer Matching Agent (ML-driven)
- Order management and escrow
- UPI payment integration
- AISP (All-Inclusive Sourcing Price) live calculation

### Key Deliverables
- [ ] Buyer Matching Agent (personalized recommendations)
- [ ] Order Management API (create, track, complete)
- [ ] UPI payment gateway integration
- [ ] Escrow + payout pipeline (farmer gets paid on delivery)
- [ ] Transaction dispute resolution flow

### Success Criteria
- End-to-end: farmer lists → buyer matches → order placed → paid via UPI
- Escrow resolves within 2 hours of delivery confirmation

---

## 🎯 Phase 6 — Beta Launch (50-Farmer Karnataka Pilot)
**Duration:** Jul 2026 → Aug 2026  
**Status:** 🔲 Not Started

### Goals
- 50-farmer pilot in Karnataka districts
- Production security hardening and rate limiting
- Performance optimization (< 2s P95 response time)

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

## 📅 High-Level Timeline

| Phase | Period | Status |
|-------|--------|--------|
| Phase 1 — Foundation & Data Pipeline | Feb–Mar 2026 | 🟡 In Progress |
| Phase 2 — First Agent MVP | Mar–Apr 2026 | 🔲 Planned |
| Phase 3 — Market Intelligence | Apr–May 2026 | 🔲 Planned |
| Phase 4 — Mobile & WhatsApp | Apr–Jun 2026 | 🔲 Planned |
| Phase 5 — Marketplace Core | Jun–Jul 2026 | 🔲 Planned |
| Phase 6 — Beta Launch | Jul–Aug 2026 | 🔲 Planned |

---

## 🔗 Links
- Product vision & tech plan: [`PLAN.md`](./PLAN.md)
- Current sprint: [`tracking/sprints/`](./tracking/sprints/)
- Architecture decisions: [`docs/decisions/`](./docs/decisions/)
- Project status: [`tracking/PROJECT_STATUS.md`](./tracking/PROJECT_STATUS.md)
