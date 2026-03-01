# PROJECT_STATUS.md � CropFresh AI Current State

> **Last Updated:** 2026-03-01 (Task 10 complete)
> **Version/Tag:** `v0.8-digital-twin`
> **Current Sprint:** Sprint 05 — Core Agent Completion (Phase 1 + Phase 2)
> **Sprint Status:** 🟢 Active — Tasks 1–10 complete

---

## North Star (From Business Model PDF)

Build India's most trusted AI-powered agri-marketplace with:
- **5 AI Agents** working together as the intelligence layer
- Farmer income up **+40�60%** vs. mandi
- Disputes **<2%** via Digital Twin + HITL
- Logistics cost **<?2.5/kg** via DPLE mesh routing
- Voice-first UI for **zero-literacy-barrier** farmer access

---

## What Is True Right Now

| Component | Status | Notes |
|-----------|--------|-------|
| RAG Pipeline (RAPTOR + Hybrid) | ? Stable | Fully aligned |
| Multi-Agent Supervisor | ? Stable | Routes exist, some agents are stubs |
| Memory System (Redis) | ? Stable | OK |
| Agronomy Agent | ? Stable | Good, needs ADCL integration |
| Commerce Agent | ? Stable | AISP math exists, not fully wired |
| **Buyer Matching Agent** | ? Task 2 Complete | 5-factor scoring + reverse matching + cache + routing |
| **Pricing Agent (DPLE)** | ? Task 1 Complete | Deadhead + 2% risk buffer + mandi cap + trend/seasonality |
| **Price Prediction Agent** | ? Task 5 Complete | Class fixed, rule-based prediction, numpy trend, Karnataka seasonal calendar, sell/hold rec, LLM fallback, 25 tests pass |
| **Voice Agent** | ? Task 4 Complete | All 10+ intents wired, multi-turn flows (create_listing/find_buyer/register), 3-language templates (en/hi/kn), 20 tests pass |
| Pipecat Voice Pipeline | ?? In Progress | WebSocket not fully tested |
| **Matchmaking Engine** | ?? In Progress | Buyer-side matching core implemented; full cluster optimization pending |
| **CV-QG Vision Agent** | ? Task 3 Complete | Quality pipeline + fallback + HITL + digital twin linkage + supervisor wiring |
| **Digital Twin Engine** | ✅ Task 10 Complete | Departure twin, arrival diff, SSIM/pHash/rule-based similarity, 6-rule liability matrix, 42 tests |
| **DPLE Logistics Routing** | ? Not Started | Multi-pickup clustering missing |
| **ADCL Agent** | ? Not Started | Weekly demand list generator |
| APMC Live Scraper | ? Not Started | eNAM registration still pending |
| **RDS PostgreSQL Schema** | ? Task 6 Complete | 10 business tables (PostGIS GEOGRAPHY, JSONB, UUID PKs, check constraints), 4-file migration runner with SHA-256 checksum tracking, 17 CRUD methods, 32 tests pass |
| **Crop Listing Service** | ? Task 7 Complete | ListingService (auto-price, shelf-life expiry, QR code, ADCL tag, quality trigger), 7 REST endpoints, CropListingAgent NL interface, 50 tests pass |
| **Order Management Service** | ? Task 8 Complete | OrderService with 11-status state machine, escrow flow (held?released/refunded), AISP ratio breakdown, Digital Twin dispute diff trigger, 8 REST endpoints, 73 tests pass |
| Vision Agent (ai/vision/) | ? Not Started | Only placeholder exists |
| RAGAS Evaluation | ? Not Started | No golden dataset yet |

---

## Current Sprint 05 � Core Agent Completion

**Goal:** Bring all 5 core AI agents + marketplace services from stub ? working implementation

### Sprint Tasks (2026-03-01 to 2026-03-14)

| # | Task | File | Priority | Done? |
|---|------|------|----------|-------|
| 1 | Fix Pricing Agent (deadhead + risk buffer + mandi cap + trend/seasonality) | `src/agents/pricing_agent.py` | P0 | ? |
| 2 | Implement Buyer Matching Engine core (5-factor score + reverse matching + cache) | `src/agents/buyer_matching/agent.py` | P0 | ? |
| 3 | Implement Quality Assessment Agent | `src/agents/quality_assessment/agent.py` | P0 | ? |
| 4 | Wire Voice Agent TODOs to real agents (all 10+ intents, multi-turn, 3-language) | `src/agents/voice_agent.py` | P0 | ? |
| 5 | Write unit tests for quality + routing increment | `tests/unit/` | P1 | ? |
| 6 | Extend DB schema � 10 business tables, migrations, CRUD | `src/db/` | P1 | ? |
| 7 | Crop Listing Service � full REST API + NL agent | `src/api/services/listing_service.py` | P1 | ? |
| 8 | Order Management Service � state machine + escrow + disputes | `src/api/services/order_service.py` | P1 | ? |

### Sprint KPIs

- [x] 0 `NotImplementedError` in core agents (0 remaining)
- [x] AISP calculation includes risk buffer
- [x] Matching returns at least 1 valid farmer�buyer pair in unit test
- [x] Voice `create_listing` intent creates a real DB record (AC6 verified 2026-03-01)
- [x] Test coverage increases from 35% ? 45% (**now at ~56%** � 276 tests / 14 test files)
- [x] Order lifecycle: confirmed ? pickup_scheduled ? in_transit ? delivered ? settled
- [x] Escrow state machine: held ? released / refunded enforced in tests

---

## 6-Phase Upgrade Roadmap

| Phase | Focus | Target | Status |
|-------|-------|--------|--------|
| **Phase 1** (Wk 1�4) | Core 5 agents working | 0 stubs, unit tests | ? Complete |
| **Phase 2** (Wk 3�6) | DB schema + Listing + Order services | REST APIs, escrow | ? Active � Tasks 6�8 done |
| **Phase 3** (Wk 5�8) | DPLE Logistics Routing + Digital Twin | <?2.5/kg, >70% util | ?? Planned |
| **Phase 4** (Wk 7�10) | APMC Live Scraper + ADCL Agent | Real mandi data | ?? Planned |
| **Phase 5** (Wk 9�12) | Voice in 10+ languages + Registration | Zero literacy barrier | ?? Planned |
| **Phase 6** (Wk 12�18) | Production hardening | 99.5% uptime | ?? Planned |

---

## Active Blockers

| Blocker | Impact | Owner |
|---------|--------|-------|
| eNAM API registration pending | Live mandi data blocked | external |
| AWS Free Tier limit blocks RDS creation | Production DB not provisioned yet | manual (upgrade account) |
| pgvector not installed locally (needs VS Build Tools) | Local vector search uses Qdrant fallback | dev workaround |
| Pipecat WebSocket untested on Windows | Voice streaming blocked | Sprint 05 carry-over |
| YOLOv8 model weights not downloaded | CV-QG grading blocked | Phase 3 |

---

## Completed Milestones

| Date | Milestone |
|------|-----------|
| **Mar 01, 2026** | **Task 10 complete — Digital Twin Engine**: `DigitalTwinEngine` in `src/agents/digital_twin/` — `create_departure_twin()`, `compare_arrival()`, `generate_diff_report()`; SSIM → perceptual hash → rule-based similarity chain; 6-rule liability matrix (farmer/hauler/buyer/shared/none); `QualityAssessmentAgent.compare_twin()` + `create_departure_twin()`; `OrderService._trigger_twin_diff()` dual-path (engine + QA fallback); `get_digital_twin()` + `update_dispute_diff_report()` in postgres_client; 42 unit tests; full suite **382 passed** |
| **Mar 01, 2026** | **Task 8 complete — Order Management Service**: `OrderService` with 11-status state machine (`VALID_TRANSITIONS`), escrow flow (held?released/refunded), AISP ratio breakdown (80/10/6/4%), Digital Twin dispute diff trigger, 4 new DB CRUD methods, 8 REST endpoints, 73 tests pass; full suite **276 passed** |
| **Mar 01, 2026** | **Task 7 complete � Crop Listing Service**: `ListingService` with 5-step auto-enrichment (price suggestion, shelf-life expiry, QR code, ADCL tag, quality trigger); 7 REST endpoints; `CropListingAgent` NL interface; AC6 verified (voice ? DB.create_listing); 50 tests pass; 203 total suite passes |
| **Mar 01, 2026** | **Task 6 complete � Database Schema Extension**: 10 business tables (`field_agents`, `haulers`, `buyers`, `farmers`, `listings`, `digital_twins`, `orders`, `disputes`, `price_history`, `adcl_reports`); PostGIS GEOGRAPHY/JSONB/UUID PKs; 4-file migration runner with SHA-256 checksum tracking; 13 CRUD methods; 32 tests pass |
| **Mar 01, 2026** | **Task 5 complete � Price Prediction Agent**: fixed corrupted class name, implemented hybrid rule-based prediction with numpy trend analysis, Karnataka 6-crop seasonal calendar, sell/hold recommendation engine, LLM fallback, and 25 unit tests pass |
| **Mar 01, 2026** | **Task 4 complete � Voice Agent full wiring**: all 10+ intents route to real services; multi-turn flows for `create_listing`, `find_buyer`, `register`; graceful fallbacks; 3-language templates (en/hi/kn); 20 unit tests pass |
| **Mar 01, 2026** | **Task 3 complete � Quality Assessment Agent**: A+/A/B/C grading pipeline, HITL threshold at 0.7, digital twin assessment IDs, fallback mode without model weights, and supervisor/chat integration with passing tests |
| **Mar 01, 2026** | **Live streaming static UI validated**: ChatGPT-style token streaming UI shipped on `static/voice_realtime.html` using `POST /api/v1/chat/stream`, with runtime verification for health, static serving, and SSE events |
| **Mar 01, 2026** | **Task 2 complete � Buyer Matching Engine**: 5-factor weighted scoring, haversine proximity, reverse matching, 5-min cache, and supervisor routing integration with passing unit tests |
| **Mar 01, 2026** | **Task 1 complete � Pricing Agent (DPLE)**: implemented utilization-based deadhead, fixed 2% risk buffer, mandi cap at modal � 1.05, plus trend and seasonality methods with passing unit tests |
| **Mar 01, 2026** | **LLM provider migration** � Groq ? AWS Bedrock (dual-provider strategy with `BedrockProvider`) |
| **Mar 01, 2026** | **Database migration** � Qdrant+Supabase ? RDS PostgreSQL+pgvector (`AuroraPostgresClient`, `schema.sql`) |
| **Mar 01, 2026** | **AWS infra provisioned** � subnet group + security group created, RDS blocked by free tier |
| **Mar 01, 2026** | **Local PostgreSQL setup** � `setup_local_db.sql` for dev, Qdrant fallback for vectors |
| Feb 28, 2026 | **Business model analysis complete** � PDF mapped to exact code gaps |
| Feb 28, 2026 | **Architecture docs updated** � ARCHITECTURE.md v0.5 with 5-agent model |
| Feb 28, 2026 | **Upgrade implementation plan created** � 6 phases, 18 weeks |
| Feb 27, 2026 | RAG 2027 Research � 4 ADRs, 4 arch docs, sprint-05 + sprint-06 planned |
| Feb 27, 2026 | Advanced dev workflow documentation system established |
| Feb 27, 2026 | Production-grade scraping upgrade (Scrapling + eNAM + IMD clients) |
| Feb 26, 2026 | Voice domain separated; Pipecat submodule built |
| Jan 10, 2026 | Advanced RAG Phase 1�4 (RAPTOR, Hybrid, HyDE, MMR, Contextual Chunking) |
| Jan 9, 2026 | Multi-agent system live (Supervisor + 4 domain agents) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11+ |
| AI Framework | LangGraph + LangChain |
| Vision | YOLOv8 + ViT-B/16 + ONNX Runtime |
| **LLM (Production)** | **AWS Bedrock � Claude Sonnet 4** |
| **LLM (Router/Dev)** | **Groq � Llama-3.1-8B (~80ms)** |
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

## Business-Aligned Metrics Dashboard

| Metric | PDF Target | Sprint 05 Target | Current |
|--------|-----------|-----------------|---------|
| AI grading accuracy | >95% (Month 12) | N/A (Phase 3) | � |
| Dispute rate | <2% | � | � |
| Voice P95 latency | <2s | <3s | ~4.5s (est.) |
| AISP accuracy | �5% of real landed cost | Correct formula | Formula aligned (Task 1 + Task 8 complete) |
| Logistics cost/kg | <?2.5/kg | � | � |
| Agent routing accuracy | >90% | >88% | ~87% (mock) |
| API cost per query avg | <?0.25 | <?0.30 | ~?0.44 |
| RAGAS faithfulness | >0.80 | baseline run | � (no baseline) |
| KB documents indexed | >1,000 | >100 | 32 |
| Test coverage | >80% (Phase 6) | >45% | **~57%** (382 tests / 15 files) |
| REST endpoints live | � | � | **15** (7 listings + 8 orders) |
| eNAM mandis integrated | 1,000+ (Y2) | 5+ | 0 |

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Product Vision | `PLAN.md` |
| Phase Milestones | `ROADMAP.md` |
| **Architecture (v0.5)** | `docs/architecture/ARCHITECTURE.md` |
| Current Sprint | `tracking/sprints/sprint-05-core-agents.md` |
| Architecture Decisions | `docs/decisions/ADR-*.md` (ADR-001 to ADR-012) |
| RAG Architecture | `docs/rag_architecture.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |

---

_Update this file at the end of every sprint and after major milestones._
_AI agents: read this + ARCHITECTURE.md before starting any work._

