# System Architecture — CropFresh AI
> **Last Updated:** 2026-02-28
> **Version:** v0.5-business-aligned
> **Status:** Active Upgrade — aligning with PDF business model

---

## Vision

CropFresh AI is the **intelligence layer** of an agri-marketplace connecting Indian farmers directly to urban buyers. The AI operates as 5 specialized agents (from the business model PDF) supervised by a LangGraph orchestrator.

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            CLIENTS                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Farmer   │  │ Buyer    │  │ Agent (Mitra) │  │ Admin / Ops    │  │
│  │  App     │  │  App     │  │  App          │  │  Dashboard     │  │
│  │ (Voice)  │  │ (REST)   │  │  (Verify)     │  │  (Internal)    │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  └───────┬────────┘  │
└───────┼─────────────┼────────────────┼───────────────────┼───────────┘
        │ WebSocket   │ REST           │ REST               │ REST
        ▼             ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway  (src/api/)                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  /chat   │  │  /voice  │  │ /vision    │  │  /marketplace    │  │
│  │  /rag    │  │  /ws     │  │ /logistics │  │  /dispute        │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └────────┬─────────┘  │
│       └─────────────┴──────────────┴───────────────────┘            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │        Middleware: Auth (JWT) │ Rate Limit │ RBAC │ Logging    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                LangGraph Multi-Agent Supervisor                      │
│                     (src/agents/supervisor_agent.py)                 │
│                                                                      │
│   Routes queries at 0.9 confidence threshold to:                    │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   5 CORE AI AGENTS (PDF-aligned)              │  │
│  │                                                               │  │
│  │  ┌─────────────────┐   ┌──────────────────────────────────┐  │  │
│  │  │ 1. DPLE         │   │ 2. MATCHMAKING ENGINE            │  │  │
│  │  │ (Pricing +      │   │ (Supply ↔ Demand Optimizer)      │  │  │
│  │  │  Logistics)     │   │                                  │  │  │
│  │  │ pricing_agent.py│   │ matchmaking_agent.py             │  │  │
│  │  │ logistics_      │   │ buyer_matching/agent.py          │  │  │
│  │  │  routing_agent  │   │                                  │  │  │
│  │  └─────────────────┘   └──────────────────────────────────┘  │  │
│  │                                                               │  │
│  │  ┌─────────────────┐   ┌──────────────────────────────────┐  │  │
│  │  │ 3. CV-QG        │   │ 4. VOICE ACCESSIBILITY           │  │  │
│  │  │ (Computer       │   │ (Farmer-First UI)                │  │  │
│  │  │  Vision Quality │   │                                  │  │  │
│  │  │  Grader)        │   │ voice_agent.py                   │  │  │
│  │  │ ai/vision/      │   │ src/voice/ (STT + TTS)           │  │  │
│  │  │  quality_grader │   │ Pipecat WebRTC                   │  │  │
│  │  │  digital_twin   │   │ 10+ Indian languages             │  │  │
│  │  └─────────────────┘   └──────────────────────────────────┘  │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │ 5. RAG ADVISORY AGENT                                  │  │  │
│  │  │ (Crop Intelligence + Agronomy + ADCL)                  │  │  │
│  │  │                                                        │  │  │
│  │  │ agronomy_agent.py  +  ai/rag/ (full pipeline)          │  │  │
│  │  │ adcl_agent.py  +  knowledge_agent.py                   │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Supporting Agents                                 │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │ Agronomy │  │ Commerce │  │ Platform │  │ General     │  │  │
│  │  │  Agent   │  │  Agent   │  │  Agent   │  │ (Fallback)  │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────┬───────────────────────────┘
                        │                 │
           ┌────────────┘                 └──────────────┐
           ▼                                             ▼
┌────────────────────────────┐            ┌──────────────────────────┐
│    AI / ML Layer           │            │  Real-Time Data Layer    │
│                            │            │                          │
│  ai/rag/                   │            │  src/scrapers/           │
│  ├── RAPTOR (hierarchical) │            │  ├── agmarknet.py        │
│  ├── hybrid_search.py      │            │  ├── enam_client.py      │
│  ├── enhanced_retriever.py │            │  ├── imd_weather.py      │
│  ├── query_processor.py    │            │  └── realtime_data.py    │
│  ├── agentic_orchestrator  │            │                          │
│  └── adaptive_query_router │            │  APScheduler: 6h refresh │
│                            │            │  Redis TTL cache         │
│  ai/vision/                │            │                          │
│  ├── quality_grader.py     │            └──────────────────────────┘
│  ├── digital_twin.py       │
│  └── defect_library.py     │
└─────────────────┬──────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Data & Storage Layer                           │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Supabase    │  │    Qdrant    │  │  Neo4j   │  │   Redis   │  │
│  │  (Postgres)  │  │  (Vectors)   │  │  (Graph) │  │  (Cache)  │  │
│  │              │  │              │  │          │  │           │  │
│  │  farmers     │  │  kb_chunks   │  │ entities │  │ sessions  │  │
│  │  listings    │  │  embeddings  │  │ relations│  │ prices TTL│  │
│  │  orders      │  │  digital_    │  │ crop     │  │ routes    │  │
│  │  disputes    │  │  twin_index  │  │  graph   │  │           │  │
│  │  haulers     │  │              │  │          │  │           │  │
│  │  buyers      │  │              │  │          │  │           │  │
│  │  price_hist  │  │              │  │          │  │           │  │
│  └──────────────┘  └──────────────┘  └──────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Responsibilities (Business Model Alignment)

| PDF Business Agent | Code Location | Business Function |
|---|---|---|
| **DPLE** (Dynamic Pricing & Logistics Engine) | `src/agents/pricing_agent.py` + `src/agents/logistics_routing_agent.py` | Calculates AISP = Farmer Ask + Logistics + Margin(4–8%) + Risk Buffer(2%) |
| **Matchmaking Engine** | `src/agents/matchmaking_agent.py` | Clusters farmers by GPS, matches to buyers by grade/demand/route |
| **CV-QG** (Computer Vision Quality Grader) | `ai/vision/quality_grader.py` + `ai/vision/digital_twin.py` | YOLOv8 grading + HITL trigger + Digital Twin + Dispute Diff Engine |
| **Voice Accessibility Agent** | `src/agents/voice_agent.py` + `src/voice/` | Voice-first in 10+ Indian languages, all farmer flows voice-driven |
| **RAG Advisory Agent** | `src/agents/agronomy_agent.py` + `ai/rag/` + `src/agents/adcl_agent.py` | Crop advisory, ADCL (weekly demand list), pest alerts, price forecasts |

---

## Key Business Flows

### Farmer Listing Flow
```
Farmer speaks (voice) → STT → Entity Extraction → Voice Agent
  → Create Listing API → Quality Agent (AI grade + HITL trigger)
  → Digital Twin created → Matchmaking Engine
  → DPLE calculates AISP → Buyer sees verified listing
  → Order → Escrow → Hauler routed (DPLE) → Delivery
  → AI Diff Engine (dispute check) → UPI auto-split settlement
```

### AISP Calculation
```
AISP = Farmer Ask
     + Logistics Cost (DPLE: distance × vehicle type × deadhead factor)
     + Platform Margin (4–8% dynamic)
     + Risk Buffer (2%)
```

### Dispute Resolution Flow
```
Buyer claims damage → Uploads video + QR scan
  → AI Diff Engine: departure Digital Twin vs arrival photos
  → Liability determination: Farmer / Hauler / Buyer
  → Auto-deduction from responsible party wallet
  → Resolution pool covers small claims (<₹500)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI + Python 3.11 | REST + WebSocket API |
| AI Orchestration | LangGraph + LangChain | Multi-agent routing |
| LLM (Generation) | Gemini Flash 2.0 + Groq Llama-3.3-70B | Response generation |
| LLM (Router) | Groq Llama-3.1-8B (~80ms) | Intent routing |
| Vision | YOLOv8 + ViT-B/16 + ONNX Runtime | Quality grading <500ms |
| Vector DB | Qdrant Cloud | RAG semantic search |
| Graph DB | Neo4j | Entity relationships in RAG |
| Primary DB | Supabase (PostgreSQL) | Transactional data |
| Cache | Redis | Sessions, price TTL |
| Voice | Pipecat + IndicWhisper + Edge-TTS | Real-time voice pipeline |
| Scraping | Scrapling + Camoufox + APScheduler | Live mandi price data |
| Embeddings | BGE-M3 + AgriEmbeddingWrapper | Domain-tuned retrieval |
| Evaluation | RAGAS + LangSmith | Quality metrics |
| Monitoring | Prometheus + Grafana | Production observability |
| Package Manager | uv | Fast dependency management |

---

## Performance Targets (Aligned with Business SLA)

| Metric | Target | Business Source |
|---|---|---|
| Voice round-trip latency | <2s | PDF Section 11 |
| AI grading inference | <500ms | PDF Section 9 |
| AISP calculation | <1s | PDF Section 11 |
| Payment settlement | <5s | PDF Section 8 |
| Dispute resolution | <15 min | PDF Section 6 |
| Dispute rate | <2% | PDF Section 5 |
| Hauler utilization | >70% | PDF Section 7 |
| Avg logistics cost | <₹2.5/kg | PDF Section 7 |
| AI grading accuracy | >95% by Month 12 | PDF Section 9 |
| System uptime | >99.5% | PDF Section 11 |

---

## HITL (Human-in-the-Loop) Architecture

```
Farmer Photo Upload
        │
        ▼
AI Grading (CV-QG)
        │
        ├── Confidence ≥ 95% ──────────→ Auto Grade A/B/C → Digital Twin
        │
        └── Confidence < 95% ──────────→ HITL Flag
                                               │
                                               ▼
                                    Agent App Notification
                                               │
                                    Physical Farm Visit (if needed)
                                               │
                                    Agent Captures Standardized Photos
                                               │
                                    Agent Grading Decision
                                    (Confirm A / Downgrade B / Reject)
                                               │
                                    Ground Truth → ML Retraining Weekly
                                               │
                                    Digital Twin Created → QR Tag
```

---

## Module Map

```
Cropfresh_Ai/
├── src/
│   ├── agents/                  ← AI Agents (5 core + supporting)
│   │   ├── supervisor_agent.py  ← LangGraph router
│   │   ├── pricing_agent.py     ← DPLE pricing (partial ✅)
│   │   ├── matchmaking_agent.py ← Supply↔demand (❌ TODO)
│   │   ├── logistics_routing_agent.py  ← DPLE routing (❌ TODO)
│   │   ├── voice_agent.py       ← Voice flow (🟡 partial)
│   │   ├── agronomy_agent.py    ← RAG advisory ✅
│   │   ├── adcl_agent.py        ← ADCL crop list (❌ TODO)
│   │   └── quality_assessment/  ← CV-QG stub (❌ TODO)
│   ├── voice/                   ← STT, TTS, Pipecat (🟡 partial)
│   ├── scrapers/                ← Agmarknet, eNAM, IMD (🟡 partial)
│   ├── api/                     ← FastAPI routes + middleware
│   └── shared/                  ← Logging, resilience, memory
├── ai/
│   ├── rag/                     ← Full RAG pipeline ✅
│   └── vision/                  ← CV-QG (❌ TODO)
├── config/
│   └── supabase_schema.sql      ← DB schema (❌ TODO)
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

---

> **See Also:**
> - `docs/architecture/adaptive_query_router.md` — RAG query routing
> - `docs/architecture/agentic_rag_system.md` — RAG orchestration
> - `docs/architecture/agri_embeddings.md` — Domain embeddings
> - `docs/architecture/browser_scraping_rag.md` — Live data RAG
> - `tracking/PROJECT_STATUS.md` — Current build status
> - `implementation_plan.md` — Upgrade plan (in brain/)
