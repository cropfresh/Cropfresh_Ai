# CropFresh AI — System Architecture

> **Last Updated:** 2026-03-11
> **Version:** 2.1.0
> **Status:** Active Development (Phase 1 — Foundation & Data Pipeline)

---

## Overview

CropFresh AI is India's intelligent agricultural marketplace AI service. It connects Karnataka farmers with buyers using a **multi-agent AI system**, **voice-first Kannada interaction**, and **real-time market intelligence**.

The platform is built as a **FastAPI backend** with a **Supervisor-routed multi-agent architecture** where 14 specialized domain agents handle everything from price queries to crop listings, quality assessment, and voice interactions in 10+ Indian languages.

---

## High-Level Architecture

```mermaid
graph TD
    subgraph "Client Layer"
        A1["📱 Mobile App<br/>(Flutter — Kannada UI)"]
        A2["🌐 Web Dashboard"]
        A3["💬 WhatsApp Bot"]
        A4["🎙️ Voice (WebRTC)"]
    end

    subgraph "API Gateway"
        B["⚡ FastAPI Backend<br/>src/api/main.py<br/>Port 8080"]
    end

    subgraph "Middleware"
        M1["🔐 API Key Auth"]
        M2["🌍 CORS"]
    end

    subgraph "Agent Orchestration Layer"
        C["🧠 SupervisorAgent<br/>src/agents/supervisor_agent.py<br/>Query routing (LLM + rule-based)"]
    end

    subgraph "Domain Agents (14 Total)"
        D1["🌾 AgronomyAgent"]
        D2["💰 CommerceAgent"]
        D3["📋 PlatformAgent"]
        D4["🤖 GeneralAgent"]
        D5["🔬 PricingAgent<br/>(DPLE Engine)"]
        D6["📊 PricePredictionAgent"]
        D7["🤝 BuyerMatchingAgent"]
        D8["✅ QualityAssessmentAgent"]
        D9["📝 CropListingAgent"]
        D10["🚚 LogisticsAgent"]
        D11["📰 ADCLAgent"]
        D12["🌐 WebScrapingAgent"]
        D13["🔍 BrowserAgent"]
        D14["📚 ResearchAgent"]
        D15["🧠 KnowledgeAgent"]
    end

    subgraph "Voice Pipeline"
        V1["🎤 STT<br/>(IndicWhisper / Groq)"]
        V2["🗣️ TTS<br/>(Edge-TTS / IndicTTS)"]
        V3["🔇 VAD<br/>(Silero)"]
        V4["📝 Entity Extractor"]
        V5["🎙️ VoiceAgent<br/>(10 Indian languages)"]
    end

    subgraph "Tools & Services"
        T1["🔧 ToolRegistry<br/>src/tools/registry.py"]
        T2["🌡️ Weather Tool"]
        T3["📈 Agmarknet Scraper"]
        T4["🔎 Deep Research"]
        T5["📰 News Sentiment"]
        T6["🧮 ML Forecaster"]
        T7["🌐 Web Search"]
    end

    subgraph "Knowledge Layer"
        K1["🔮 RAG Pipeline<br/>src/rag/<br/>(21 modules)"]
        K2["📊 RAPTOR Indexing"]
        K3["🔀 Hybrid Search<br/>(BM25 + Dense + RRF)"]
        K4["🏆 Reranker<br/>(Cross-Encoder)"]
    end

    subgraph "Data Layer"
        DB1[("🐘 PostgreSQL<br/>(Aurora / Supabase)")]
        DB2[("🔷 Qdrant<br/>(Vector DB)")]
        DB3[("🕸️ Neo4j<br/>(Graph DB)")]
        DB4[("⚡ Redis<br/>(Cache + Sessions)")]
    end

    subgraph "Observability"
        O1["📊 Prometheus"]
        O2["📈 Grafana"]
        O3["🔍 LangSmith"]
        O4["📡 OpenTelemetry"]
    end

    A1 & A2 & A3 --> B
    A4 --> B
    B --> M1 --> M2 --> C
    B --> V1 & V2 & V3
    V1 --> V4 --> V5
    V5 --> C
    C --> D1 & D2 & D3 & D4
    C --> D5 & D6 & D7 & D8
    C --> D9 & D10 & D11
    C --> D12 & D13 & D14 & D15
    D1 & D2 & D5 --> T1
    T1 --> T2 & T3 & T4 & T5 & T6 & T7
    D14 & D15 --> K1
    K1 --> K2 & K3 --> K4
    K1 --> DB2
    K1 --> DB3
    D5 & D9 & D7 --> DB1
    C --> DB4
    B --> O1 --> O2
    B --> O3 & O4
```

---

## Component Inventory

### `src/` Directory Structure

| Directory | Purpose | Key Files | Lines |
|-----------|---------|-----------|-------|
| `src/agents/` | 14 domain agents + supervisor | `supervisor_agent.py`, `base_agent.py`, `agent_registry.py` | ~2,500+ |
| `src/api/` | FastAPI app, 9 routers, middleware | `main.py`, `config.py`, `routes/`, `routers/` | ~1,500+ |
| `src/voice/` | STT, TTS, VAD, WebRTC, entity extraction | `stt.py`, `tts.py`, `vad.py`, `voice_agent.py` | ~2,000+ |
| `src/rag/` | 21-module RAG pipeline | `raptor.py`, `hybrid_search.py`, `reranker.py` | ~3,000+ |
| `src/tools/` | 16 agent tools | `registry.py`, `agmarknet.py`, `weather.py` | ~2,000+ |
| `src/scrapers/` | 13 web scrapers | `base_scraper.py`, `agmarknet.py`, `enam_client.py` | ~1,800+ |
| `src/db/` | Database clients | `postgres_client.py`, `neo4j_client.py`, `supabase_client.py` | ~700+ |
| `src/memory/` | Session + state management | `state_manager.py` (729 lines) | 729 |
| `src/orchestrator/` | LLM provider abstraction | `llm_provider.py` | ~500+ |
| `src/evaluation/` | RAG evaluation (RAGAS) | `eval_runner.py`, `ragas_evaluator.py` | ~400+ |
| `src/resilience/` | Circuit breaker, health monitor | `circuit_breaker.py`, `health_monitor.py` | ~600+ |
| `src/autonomous/` | Goal-driven autonomous agent | `goal_agent.py`, `pear_loop.py` | ~400+ |
| `src/production/` | Production infra (rate limiter, cache) | `rate_limiter.py`, `cache.py`, `config.py` | ~350+ |
| `src/mcp/` | MCP browser server | `browser_server.py` | ~250+ |
| `src/workflows/` | JSON workflow definitions | 5 workflow JSON files | — |
| `src/config/` | App settings (Pydantic) | `settings.py` | ~250+ |
| `src/pipelines/` | Data pipeline stubs | Stub files | — |
| `src/models/` | Pydantic data models | `__init__.py` only | — |

### Agent Groups (from `agent_registry.py`)

| Group | Agents | Notes |
|-------|--------|-------|
| **Core** | AgronomyAgent, CommerceAgent, PlatformAgent, GeneralAgent | All inherit `BaseAgent` |
| **Pricing** | PricingAgent, PricePredictionAgent | PricingAgent does NOT inherit BaseAgent |
| **Marketplace** | BuyerMatchingAgent, QualityAssessmentAgent, CropListingAgent | Located in subdirectories |
| **Web** | WebScrapingAgent, BrowserAgent, ResearchAgent | WebScrapingAgent/BrowserAgent don't inherit BaseAgent |
| **Wrapper** | ADCLWrapperAgent, LogisticsWrapperAgent | Wrap standalone engines |
| **Knowledge** | KnowledgeAgent | Qdrant-backed RAG, doesn't inherit BaseAgent |

---

## Technology Stack

| Layer | Technology | Configuration |
|-------|-----------|---------------|
| **Backend** | FastAPI 0.115+ / Python 3.11+ | `src/api/main.py` |
| **LLM** | Amazon Bedrock (Claude) / Groq (Llama-3.3-70B) / Together / vLLM | `src/api/config.py` → `llm_provider` |
| **Vector DB** | Qdrant Cloud (or pgvector) | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY` |
| **Graph DB** | Neo4j 5 Community | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |
| **Primary DB** | PostgreSQL (Aurora / Supabase) | `PG_HOST`, `PG_DATABASE`, `PG_USER` |
| **Cache** | Redis 7 Alpine | `REDIS_URL` |
| **Embeddings** | BAAI/bge-m3 (MiniLM fallback) | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE` |
| **STT** | IndicWhisper / Groq Whisper | `WHISPER_MODEL_SIZE` |
| **TTS** | Edge-TTS / IndicTTS | Default: Edge-TTS |
| **VAD** | Silero VAD | Pre-downloaded at startup |
| **Vision** | YOLOv11m (planned) | `YOLO_MODEL_PATH` |
| **Scraping** | Scrapling + Playwright + Camoufox | `src/scrapers/base_scraper.py` |
| **Monitoring** | Prometheus + Grafana | `infra/monitoring/` |
| **Tracing** | LangSmith + OpenTelemetry | `LANGSMITH_API_KEY`, `OTEL_ENDPOINT` |
| **Workflows** | n8n | Port 5678 |
| **Package Mgr** | uv | `pyproject.toml`, `uv.lock` |

---

## Docker Compose Services

```mermaid
graph LR
    subgraph "docker-compose.yml"
        API["🚀 api<br/>Port 8000"]
        REDIS["⚡ redis<br/>Port 6379"]
        QDRANT["🔷 qdrant<br/>Port 6333-6334"]
        NEO4J["🕸️ neo4j<br/>Port 7474, 7687"]
        PROM["📊 prometheus<br/>Port 9090"]
        GRAF["📈 grafana<br/>Port 3000"]
        N8N["⚙️ n8n<br/>Port 5678"]
    end

    API -->|depends_on| QDRANT
    API -->|depends_on| REDIS
    API -->|depends_on| NEO4J
    PROM -->|scrapes| API
    GRAF -->|reads| PROM
    N8N -->|workflows| API
```

| Service | Image | Volume | Health Check |
|---------|-------|--------|-------------|
| `api` | Custom Dockerfile | `src/`, `ai/` | `wget /health` |
| `redis` | `redis:7-alpine` | `redis_data` | `redis-cli ping` |
| `qdrant` | `qdrant/qdrant:latest` | `qdrant_data` | `wget /healthz` |
| `neo4j` | `neo4j:5-community` | `neo4j_data`, `neo4j_logs` | `wget :7474` |
| `prometheus` | `prom/prometheus:latest` | `prometheus_data` | — |
| `grafana` | `grafana/grafana:latest` | `grafana_data` | — |
| `n8n` | `n8nio/n8n:latest` | `n8n_data` | — |

---

## Startup Sequence

The application initializes in this order (see `src/api/main.py` → `lifespan()`):

```mermaid
sequenceDiagram
    participant App as FastAPI App
    participant LM as LangSmith
    participant OT as OpenTelemetry
    participant RD as Redis
    participant LLM as LLM Provider
    participant KA as KnowledgeAgent
    participant AR as AgentRegistry
    participant VAD as Silero VAD

    App->>LM: 1. Configure LangSmith tracing
    App->>OT: 2. Setup OpenTelemetry
    App->>RD: 3. Connect Redis cache
    App->>LLM: 4. Create LLM provider (Bedrock/Groq/Together/vLLM)
    App->>KA: 5. Init KnowledgeAgent (Qdrant connection)
    App->>AR: 6. create_agent_system() → wire all 15 agents
    AR-->>App: Return SupervisorAgent + StateManager
    App->>VAD: 7. Pre-download Silero VAD model
    App-->>App: ✅ Ready to serve
```

---

## Non-Functional Requirements

| Requirement | Target | Current Status |
|-------------|--------|---------------|
| Voice response latency | < 3s (< 2s goal) | ~3-4s (optimizing) |
| API response latency | < 500ms P95 | Met for cached queries |
| Agent routing accuracy | > 90% | ~85% (improving) |
| Multi-language support | Kannada, Hindi, English + 7 more | 10 languages active |
| API cost per query | < ₹0.50 | ~₹0.44 (adaptive router reduces to ~₹0.21) |
| Uptime (Phase 6+) | > 99.5% | N/A (pre-production) |
| Data privacy | No farmer data used for LLM training | ✅ Enforced |

---

## Security Architecture

1. **API Key Authentication** — `X-API-Key` header via `APIKeyMiddleware` (skipped in development)
2. **Environment-driven CORS** — `ALLOWED_ORIGINS` env var (never `*` in production)
3. **Firebase Auth** — JWT Bearer tokens for user-facing endpoints
4. **Supabase RLS** — Row Level Security on database tables
5. **No Swagger in Production** — `/docs` and `/redoc` hidden when `ENVIRONMENT=production`
6. **Secret Management** — All credentials via env vars, never hardcoded

---

## Related Documentation

| Document | Path |
|----------|------|
| Data Flow Diagrams | [`docs/architecture/data-flow.md`](data-flow.md) |
| Module Dependency Map | [`docs/diagrams/module-map.md`](../diagrams/module-map.md) |
| Agent Registry | [`docs/agents/REGISTRY.md`](../agents/REGISTRY.md) |
| API Reference | [`docs/api/endpoints-reference.md`](../api/endpoints-reference.md) |
| Environment Variables | [`docs/guides/environment-variables.md`](../guides/environment-variables.md) |
