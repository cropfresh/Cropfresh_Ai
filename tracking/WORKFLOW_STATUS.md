# CropFresh AI Service - Workflow Status

**Last Updated:** March 01, 2026 (14:15 IST)  
**Package Manager:** uv  
**Python Version:** 3.11+

---

## 📊 Current Status

| Component | Status | Progress |
|-----------|--------|----------|
| Voice Agent | ✅ Complete | 100% |
| RAG System | ✅ Complete | 100% |
| Multi-Agent System | ✅ Complete | 100% |
| Memory System | ✅ Complete | 100% |
| Tool Integration | ✅ Complete | 90% |
| **LLM Provider (Bedrock+Groq)** | ✅ Complete | 100% |
| **DB Migration (pgvector)** | 🟡 In Progress | 70% |
| Vision Agent | ✅ Task 3 Complete | 100% (for Task 3 scope) |
| Pricing Agent | ✅ Task 1 Complete | 100% (for Task 1 scope) |
| Buyer Matching Agent | ✅ Task 2 Complete | 100% (for Task 2 scope) |
| LangGraph Orchestrator | ⚠️ Partial | 70% |

---

## 📁 File Changes Log

### March 01, 2026 — Task 4 Progress + Live Streaming UI Validation

| Action | File | Description |
|--------|------|-------------|
| UPDATE | `src/agents/voice_agent.py` | Added multi-turn `create_listing` flow and improved service-backed handlers for price/order intents |
| UPDATE | `src/voice/entity_extractor.py` | Added asking-price extraction for listing turn completion |
| CREATE | `tests/unit/test_voice_agent.py` | Added unit tests for multi-turn listing, pricing service usage, and order-status service usage |
| UPDATE | `static/voice_realtime.html` | Rebuilt as ChatGPT-style streaming chat surface |
| UPDATE | `static/assets/css/voice-realtime.css` | Added conversation bubbles, stream metadata, and typing caret animation |
| UPDATE | `static/assets/js/voice-realtime.js` | Implemented SSE-based token streaming UI (`/api/v1/chat/stream`) with stop/clear controls |
| VERIFY | Runtime check | `/health` responded `200` on `localhost:8000` |
| VERIFY | Static page check | `/static/voice_realtime.html` served with streaming controls (`STATIC_OK`) |
| VERIFY | SSE stream check | `/api/v1/chat/stream` emitted `token` + `done` events (`SSE_OK`) |

### March 01, 2026 — Task 3 Completion (Quality Assessment Agent + Static Dashboard)

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/agents/quality_assessment/vision_models.py` | Added `CropVisionPipeline` with optional ONNX loading, fallback grading, defects, and shelf-life estimation |
| UPDATE | `src/agents/quality_assessment/agent.py` | Integrated vision pipeline, `execute()` contract, HITL threshold policy (`<0.7`), and digital twin assessment ID linkage |
| UPDATE | `src/agents/supervisor_agent.py` | Added `quality_assessment_agent` routing prompt/keywords and rule-based routing path |
| UPDATE | `src/api/routes/chat.py` | Registered `quality_assessment_agent` in supervisor bootstrap |
| UPDATE | `src/api/routers/chat.py` | Registered `quality_assessment_agent` in supervisor bootstrap |
| UPDATE | `tests/unit/test_quality_assessment.py` | Added Task 3-aligned tests for fallback, HITL policy, execute contract, and digital twin linkage |
| UPDATE | `tests/unit/test_supervisor_routing.py` | Added quality-assessment routing coverage |
| UPDATE | `static/index.html` | Added quick buyer-matching and quick quality-check cards |
| UPDATE | `static/assets/js/dashboard.js` | Added handlers for quick buyer matching and quality check actions |
| UPDATE | `tracking/tasks/task3.md` | Marked Task 3 complete with acceptance and validation mapping |
| UPDATE | `tracking/sprints/sprint-05-core-agents.md` | Marked Task 3 and Task 8 complete |
| UPDATE | `tracking/PROJECT_STATUS.md` | Updated sprint task table and milestone for Task 3 completion |

### March 01, 2026 — Task 2 Completion (Buyer Matching Agent)

| Action | File | Description |
|--------|------|-------------|
| UPDATE | `src/agents/buyer_matching/agent.py` | Added 5-factor `MatchingEngine`, ranked matching, reverse matching, and 5-minute cache (redis + local fallback) |
| UPDATE | `src/agents/supervisor_agent.py` | Added buyer-matching intent routing support in rule-based fallback and prompt |
| UPDATE | `src/api/routes/chat.py` | Registered `buyer_matching_agent` in supervisor bootstrap |
| UPDATE | `src/api/routers/chat.py` | Registered `buyer_matching_agent` in supervisor bootstrap |
| UPDATE | `tests/unit/test_buyer_matching.py` | Added Task 2 alignment tests (ranking, reverse matching, cache, scoring) |
| UPDATE | `tests/unit/test_supervisor_routing.py` | Added buyer-matching intent route test |
| UPDATE | `tracking/tasks/task2.md` | Marked Task 2 complete with acceptance mapping and validation |
| UPDATE | `tracking/sprints/sprint-05-core-agents.md` | Marked Task 2 and associated tests complete |
| UPDATE | `tracking/PROJECT_STATUS.md` | Updated component status, sprint task list, KPI, and milestones |

### March 01, 2026 — Task 1 Completion (Pricing Agent DPLE)

| Action | File | Description |
|--------|------|-------------|
| UPDATE | `src/agents/pricing_agent.py` | Implemented business-aligned AISP formula with deadhead by utilization, 2% risk buffer, mandi cap (`×1.05`), plus trend and seasonal adjustment methods |
| UPDATE | `src/tools/agmarknet.py` | Added `get_historical_prices()` and mock-history fallback for trend analysis |
| UPDATE | `tests/unit/test_pricing_agent.py` | Added/updated unit tests for AISP formula, cap, deadhead utilization, trend, and seasonality |
| UPDATE | `tracking/tasks/task1.md` | Marked Task 1 complete with implementation and validation notes |
| UPDATE | `tracking/sprints/sprint-05-core-agents.md` | Marked pricing task and pricing unit tests as complete |
| UPDATE | `tracking/PROJECT_STATUS.md` | Updated sprint tracker and KPI line for completed pricing formula work |

### March 01, 2026 — AWS Migration Session

#### LLM Provider Migration (Groq → Bedrock)
| Action | File | Description |
|--------|------|-------------|
| UPDATE | `src/shared/orchestrator/llm_provider.py` | Added `BedrockProvider` class with boto3 runtime |
| UPDATE | `src/config/settings.py` | Added `has_llm_configured`, Bedrock fields, dual-provider support |
| UPDATE | `src/api/routers/rag.py` | Replaced Groq-only guard with `has_llm_configured` |
| UPDATE | `src/api/routes/chat.py` | Replaced Groq-only guard with `has_llm_configured` |
| UPDATE | `src/shared/production/config.py` | LLMConfig defaults to Bedrock, load_config updated |
| UPDATE | `src/production/config.py` | Same updates (duplicate file) |
| UPDATE | `scripts/test_llm.py` | Provider-agnostic tests (Groq + Bedrock via CLI flags) |
| UPDATE | `scripts/test_rag.py` | Provider-agnostic `has_llm_configured` check |
| UPDATE | `scripts/test_multi_agent.py` | Provider-agnostic `has_llm_configured` check |
| UPDATE | `scripts/verify_setup.py` | Added Bedrock connection check alongside Groq |
| CREATE | `.env.example` | Both Groq + Bedrock config sections documented |

#### Database Migration (Qdrant + Supabase → RDS PostgreSQL + pgvector)
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/db/schema.sql` | Full schema: pgvector extension, agri_knowledge (1024-dim), users, chat_history, produce |
| CREATE | `src/db/postgres_client.py` | `AuroraPostgresClient` — unified vector search + relational CRUD via asyncpg |
| CREATE | `scripts/setup_local_db.sql` | Local PostgreSQL setup script (graceful pgvector fallback) |
| UPDATE | `src/config/settings.py` | Added `vector_db_provider`, `pg_host/database/port/user/password/use_iam_auth` |
| UPDATE | `src/api/config.py` | Same Aurora PostgreSQL config additions |
| UPDATE | `src/db/__init__.py` | Added `get_postgres`, `AuroraPostgresClient` exports |
| UPDATE | `src/api/db/__init__.py` | Same exports |
| UPDATE | `src/shared/production/config.py` | `DatabaseConfig` + `load_config()` with pgvector fields |
| UPDATE | `src/production/config.py` | Same updates (duplicate file) |
| UPDATE | `pyproject.toml` | `asyncpg` + `pgvector` as core deps; `qdrant-client` → optional |
| UPDATE | `.env.example` | Aurora PostgreSQL section, VECTOR_DB_PROVIDER toggle |

#### Documentation Updates
| Action | File | Description |
|--------|------|-------------|
| CREATE | `docs/decisions/ADR-011-aws-bedrock-llm.md` | ADR for Bedrock migration |
| CREATE | `docs/decisions/ADR-012-aurora-pgvector-consolidation.md` | ADR for DB consolidation |
| UPDATE | `docs/decisions/ADR-001-qdrant-vector-db.md` | Marked superseded by ADR-012 |
| UPDATE | `docs/decisions/ADR-002-supabase-primary-db.md` | Marked superseded by ADR-012 |
| UPDATE | `docs/architecture/tech-stack.md` | Rewritten with Bedrock + pgvector stack |
| UPDATE | `tracking/PROJECT_STATUS.md` | Tech stack, milestones, blockers updated |
| UPDATE | `tracking/COST.md` | AWS cost estimates added |
| UPDATE | `tracking/WORKFLOW_STATUS.md` | This file — full session log |

---

### January 10, 2026 (Afternoon Session)

#### Advanced RAG Phase 1: Real-Time Data Integration
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/tools/enam_client.py` | eNAM API client for live mandi prices, trends, market summaries |
| CREATE | `src/tools/imd_weather.py` | IMD Weather client with forecasts and agro advisories |
| CREATE | `src/tools/google_amed.py` | Google AMED for satellite crop monitoring and season info |
| CREATE | `src/tools/realtime_data.py` | Unified RealTimeDataManager with fallbacks and health checks |
| UPDATE | `src/tools/__init__.py` | Added exports for all new real-time data modules |
| CREATE | `scripts/test_realtime_data.py` | Test suite for Phase 1 components |
| CREATE | `docs/diagrams/advanced_rag_architecture.png` | Architecture diagram for advanced RAG |
| CREATE | `docs/diagrams/raptor_tree_structure.png` | RAPTOR hierarchical retrieval diagram |
| UPDATE | `docs/rag_architecture.md` | Updated with diagrams and new components |
| CREATE | `docs/advanced_rag_implementation_plan.md` | Comprehensive implementation plan |

#### Advanced RAG Phase 2: Advanced Retrieval Techniques
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/rag/raptor.py` | RAPTOR hierarchical tree indexing with GMM clustering |
| CREATE | `src/rag/contextual_chunker.py` | Contextual chunking with entity extraction |
| UPDATE | `src/rag/__init__.py` | Added exports for RAPTOR and contextual chunking |
| CREATE | `scripts/test_advanced_retrieval.py` | Test suite for Phase 2 components |

#### Advanced RAG Phase 3-4: Query Processing & Enhanced Retrieval
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/rag/query_processor.py` | HyDE, multi-query, step-back, decomposition, rewriting |
| CREATE | `src/rag/enhanced_retriever.py` | Parent Document, Sentence Window, MMR retrievers |
| UPDATE | `src/rag/__init__.py` | Added exports for Phase 3-4 modules |
| CREATE | `scripts/test_query_retrieval.py` | Test suite for Phase 3-4 components |

---

### January 10, 2026 (Morning Session)

#### Next-Level RAG Enhancements
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/rag/hybrid_search.py` | BM25 sparse retrieval + RRF fusion for hybrid search |
| CREATE | `src/rag/reranker.py` | Cross-encoder reranking with MiniLM fallback |
| CREATE | `src/rag/graph_retriever.py` | Neo4j Graph RAG with entity extraction |
| CREATE | `src/rag/observability.py` | LangSmith tracing + RAG evaluation metrics |
| UPDATE | `src/rag/__init__.py` | Added exports for all new enhancement modules |
| CREATE | `scripts/test_rag_enhancements.py` | Comprehensive test suite for enhancements |
| UPDATE | `.env` | Added LangSmith configuration section |

---

### January 9, 2026 (Evening Session)

#### Advanced Agentic RAG System
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/memory/state_manager.py` | Conversation memory, session management, Redis support |
| CREATE | `src/tools/registry.py` | Dynamic tool registration with OpenAI/Anthropic schemas |
| CREATE | `src/agents/base_agent.py` | Abstract base with LLM, tools, memory integration |
| CREATE | `src/agents/supervisor_agent.py` | Query routing (0.9 confidence), multi-agent orchestration |
| CREATE | `src/agents/agronomy_agent.py` | Crop cultivation, pest management, farming advice |
| CREATE | `src/agents/commerce_agent.py` | Market prices, AISP calculations, sell/hold decisions |
| CREATE | `src/agents/platform_agent.py` | CropFresh app features, registration, support |
| CREATE | `src/agents/general_agent.py` | Greetings, fallback, unclear queries |
| CREATE | `src/tools/weather.py` | Agricultural weather with advisories (mock) |
| CREATE | `src/tools/calculator.py` | AISP, yield estimates, unit conversions |
| CREATE | `src/tools/web_search.py` | Real-time web search (mock) |
| CREATE | `src/api/routes/chat.py` | Multi-turn chat + SSE streaming + session management |
| UPDATE | `src/rag/knowledge_base.py` | Fixed Qdrant API compatibility (query_points/search) |
| UPDATE | `src/api/main.py` | Added chat API routes |
| CREATE | `scripts/populate_qdrant.py` | 12 agricultural documents for testing |
| CREATE | `scripts/test_multi_agent.py` | Comprehensive multi-agent test suite |

#### Voice Agent Implementation (Morning Session)
| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/voice/__init__.py` | Module exports |
| CREATE | `src/voice/audio_utils.py` | Audio format detection, FFmpeg |
| CREATE | `src/voice/stt.py` | IndicWhisper STT + Groq fallback |
| CREATE | `src/voice/tts.py` | IndicTTS + Edge TTS + gTTS |
| CREATE | `src/voice/entity_extractor.py` | Intent + entity extraction |
| CREATE | `src/agents/voice_agent.py` | Two-way voice orchestrator |
| CREATE | `src/api/rest/voice.py` | REST API endpoints |
| CREATE | `src/api/websocket.py` | WebSocket streaming |

---

## ✅ All Tests Passing

```
============================================================
  TEST SUMMARY
============================================================
   state_manager: ✅ PASS
   tool_registry: ✅ PASS
   agent_routing: ✅ PASS
   general_agent: ✅ PASS
   commerce_agent: ✅ PASS
   multi_agent_pipeline: ✅ PASS
   llm_pipeline: ✅ PASS

🎉 All tests passed!
```

---

## 🔧 Quick Start

### 1. Start Databases
```bash
# Option A: Local PostgreSQL (relational) + Qdrant (vectors)
psql -U postgres -f scripts/setup_local_db.sql   # creates cropfresh DB
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Option B: AWS RDS (when provisioned) — set PG_HOST in .env
```

### 2. Set Environment
```bash
# Copy and edit .env
copy .env.example .env
# Set VECTOR_DB_PROVIDER=qdrant for local dev
# Set PG_HOST=localhost, PG_PASSWORD=cropfresh_dev_2026
```

### 3. Populate Knowledge Base
```bash
.venv\Scripts\python scripts\populate_qdrant.py
```

### 4. Run the Service
```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### 5. Open Swagger UI
```
http://localhost:8000/docs
```

---

## � API Endpoints

### Chat API (NEW)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Multi-turn conversation with agent routing |
| `/api/v1/chat/stream` | POST | SSE streaming responses |
| `/api/v1/chat/session` | POST | Create new session |
| `/api/v1/chat/session/{id}` | GET | Get session info |
| `/api/v1/chat/agents` | GET | List available agents |
| `/api/v1/chat/tools` | GET | List available tools |

### Voice API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/voice/process` | POST | Full voice-in → voice-out |
| `/api/v1/voice/transcribe` | POST | Audio → Text |
| `/api/v1/voice/synthesize` | POST | Text → Audio |
| `/api/v1/voice/languages` | GET | Supported languages |
| `/ws/voice/{user_id}` | WebSocket | Real-time streaming |

### RAG API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Query knowledge base |
| `/api/v1/search` | POST | Semantic search |
| `/api/v1/ingest` | POST | Ingest documents |

### Health
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |

---

## 🤖 Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                              │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Supervisor Agent                            │
│         (LLM Routing with 0.9 confidence)                  │
└────────────────────────┬────────────────────────────────────┘
                         ▼
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Agronomy    │ │  Commerce    │ │   Platform   │
│    Agent     │ │    Agent     │ │    Agent     │
├──────────────┤ ├──────────────┤ ├──────────────┤
│ Crop guides  │ │ Market prices│ │ App features │
│ Pest mgmt    │ │ AISP calcs   │ │ Registration │
│ Irrigation   │ │ Sell/hold    │ │ FAQs         │
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Knowledge Base                            │
│  Production: RDS PostgreSQL + pgvector (vector search)      │
│  Dev:        Qdrant fallback (VECTOR_DB_PROVIDER=qdrant)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tools Available

| Tool | Category | Purpose |
|------|----------|---------|
| `get_current_weather` | weather | Current conditions + agricultural advisories |
| `get_weather_forecast` | weather | 5-day forecast with farm recommendations |
| `calculate_aisp` | calculator | All-Inclusive Sourcing Price breakdown |
| `estimate_yield` | calculator | Crop yield estimates |
| `convert_units` | calculator | Agricultural unit conversions |
| `web_search` | search | Real-time information retrieval |

---

## 📦 Knowledge Base

**32 documents indexed** across 4 categories:

| Category | Documents | Topics |
|----------|-----------|--------|
| Agronomy | 5 | Tomato, onion, organic farming, drip irrigation, pest management |
| Market | 2 | Mandi pricing, sell/hold decisions |
| Platform | 3 | Registration, quality grades, payments |
| General | 2 | About CropFresh, Prashna Krishi AI |

---

## 📝 Future Improvements

### High Priority
- [x] **Hybrid Search**: BM25 sparse retrieval + RRF fusion ✅ DONE
- [x] **Cross-Encoder Re-ranking**: MiniLM-based reranking ✅ DONE
- [ ] **Lighter Embedding Model**: Add MiniLM option for low-memory systems
- [x] **AWS Bedrock Integration**: Dual-provider strategy (Bedrock + Groq) ✅ DONE
- [x] **Database Consolidation**: Qdrant+Supabase → RDS PostgreSQL+pgvector ✅ DONE
- [ ] **True LLM Token Streaming**: Stream tokens directly from Bedrock/Groq API

### Medium Priority
- [ ] **Vision Agent**: YOLOv12 + DINOv2 for crop disease detection
- [ ] **Database Query Tool**: Access order/transaction data
- [x] **LangSmith/LangFuse Tracing**: Observability + evaluation ✅ DONE
- [x] **Graph RAG**: Neo4j integration for relationships ✅ DONE
- [ ] **Redis Session Storage**: Production-grade session persistence

### Lower Priority
- [ ] **Multi-hop Reasoning**: Complex queries requiring multiple KB lookups
- [ ] **Contextual Compression**: Extract only relevant chunks from documents
- [ ] **Agent Collaboration**: Multi-agent responses for complex queries
- [ ] **Voice + Chat Integration**: Unified interface for voice and text

---

## ⚠️ Known Issues

### Embedding Model Memory
The BGE-M3 embedding model requires ~1GB RAM. On low-memory systems, retrieval may fail but LLM fallback ensures responses still work.

```
Solution: Use MiniLM-L6-v2 (90MB) for lighter deployments
```

### Qdrant Client Compatibility
Fixed API compatibility for Qdrant client 1.7+ (use `query_points` instead of deprecated `search`).  
> **Note:** Qdrant is now a dev-only fallback. Production uses pgvector on RDS PostgreSQL.

### AWS Free Tier
RDS instance creation blocked by free tier spending limits. Use local PostgreSQL for dev, upgrade account for production.

---

## 🧪 Testing Commands

```bash
# Run all multi-agent tests
uv run python scripts/test_multi_agent.py

# Run knowledge base search test
uv run python scripts/test_kb_search.py

# Run RAG enhancements tests (NEW)
uv run python scripts/test_rag_enhancements.py

# Populate Qdrant with sample data
uv run python scripts/populate_qdrant.py

# Run pytest suite
uv run pytest -v

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

---

## 🛠️ Common Issues

### Qdrant not connecting
```bash
# Start container
docker start qdrant

# Or restart fresh
docker rm -f qdrant && docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### FFmpeg not found
```bash
# Windows (with chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg
```

### GROQ_API_KEY / Bedrock not configured
```bash
# For Groq (fast dev)
GROQ_API_KEY=your_api_key_here
LLM_PROVIDER=groq

# For Bedrock (production)
LLM_PROVIDER=bedrock
# Uses AWS CLI credentials (aws configure)
```

### PostgreSQL not connecting locally
```bash
# Run setup script (enter postgres password when prompted)
psql -U postgres -f scripts/setup_local_db.sql

# Verify connection
psql -U cropfresh_app -d cropfresh -c "SELECT 1"
```

### Virtual environment not activating
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```
