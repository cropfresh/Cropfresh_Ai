<div align="center">

<br/>

# 🌾 CropFresh AI

### _India's Intelligent Agricultural Marketplace — Powered by AI Agents_

<br/>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![AWS Bedrock](https://img.shields.io/badge/AWS_Bedrock-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **Connecting Karnataka's farmers directly with buyers — eliminating middlemen, one crop at a time.**
>
> Voice-first Kannada interaction · Real-time APMC market data · AI crop quality grading · Direct farmer-to-buyer marketplace

<br/>

[**Quick Start**](#-quick-start) &nbsp;·&nbsp; [**Architecture**](#-architecture) &nbsp;·&nbsp; [**AI Agents**](#-ai-agent-system) &nbsp;·&nbsp; [**API Reference**](#-api-reference) &nbsp;·&nbsp; [**Roadmap**](#-roadmap)

<br/>

</div>

---

## 💡 The Problem We Solve

> **Indian farmers lose 30–40% of their crop value to intermediaries.** Most lack access to real-time market data, fair quality assessments, or direct buyer connections — especially in their native language.

**CropFresh AI** closes this gap with a fully intelligent, voice-first platform:

| Challenge                             | Our Solution                                             |
| :------------------------------------ | :------------------------------------------------------- |
| 🏪 Middlemen extract huge margins     | Direct farmer-to-buyer marketplace with escrow           |
| 📊 No price transparency              | Real-time data from 160+ Karnataka APMC mandis           |
| 🎤 Language & literacy barriers       | Voice-first interaction in Kannada + 10 Indian languages |
| 📸 Manual, subjective quality grading | AI computer vision grading (YOLOv8 + ViT-B/16)           |
| 🔍 Difficulty finding the right buyer | Graph-based intelligent matchmaking engine               |
| 🚚 Inefficient logistics              | DPLE route optimizer with multi-pickup clustering        |
| ❓ Lack of crop advisory              | RAG-powered agronomy knowledge base                      |

---

## ✨ Platform Highlights

<table>
<tr>
<td width="50%" valign="top">

### 🤖 Multi-Agent AI System

A **LangGraph**-powered supervisor that routes every query to the right specialist:

- **Supervisor** — Intelligent query classification (90%+ accuracy target)
- **Agronomy** — Crop advisory, pest & disease diagnosis
- **Pricing (DPLE)** — Real-time AISP pricing with logistics margin
- **Commerce** — Transaction & marketplace operations
- **Voice** — Kannada/Hindi/English voice interaction
- **Quality Assessment** — CV-based quality grading
- **Buyer Matching** — 5-factor graph-based matchmaking
- **ADCL** — Weekly assured demand crop list generator
- **Knowledge** — RAG-based information retrieval
- **Browser/Scraping** — APMC, eNAM & weather data

</td>
<td width="50%" valign="top">

### 🎤 Voice-First for Every Farmer

Built from the ground up for farmers who prefer speaking over typing:

- **Kannada STT** — IndicWhisper + Groq Whisper fallback
- **Natural TTS** — Edge-TTS voice synthesis in 10+ languages
- **Live Streaming** — WebSocket + Pipecat real-time voice pipeline
- **Silero VAD** — Intelligent voice activity detection
- **Language Support** — Kannada, Hindi, Tamil, Telugu, Malayalam + 6 more
- **WhatsApp Bot** — Planned: text, voice, image & location handling

</td>
</tr>
<tr>
<td valign="top">

### 🧠 Intelligent RAG Pipeline

Advanced retrieval-augmented generation for farm knowledge:

- **pgvector** — PostgreSQL semantic vector search
- **RAPTOR** — Hierarchical multi-level document summaries
- **Graph RAG** — Neo4j knowledge graph for buyer-seller relationships
- **BGE-M3** — Multilingual cross-lingual embeddings
- **Knowledge Base** — FSSAI, AGMARK, PM-KISAN, govt. schemes
- **Cohere Re-ranking** — Precision result ordering

</td>
<td valign="top">

### 📊 Real-Time Market Intelligence

Agricultural data pipeline that never sleeps:

- **160+ Karnataka APMC mandis** live price tracking
- **eNAM** national market integration
- **IMD Weather** integration for crop forecasting
- **Seasonal pattern** recognition & trend prediction
- **Digital Twin Engine** — Departure vs. arrival quality comparison
- **Dispute Diff Engine** — SSIM-based quality degradation detection

</td>
</tr>
</table>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            Clients                                   │
│      📱 Flutter App      💬 WhatsApp Bot      🖥 Web Dashboard       │
└──────────┬────────────────────┬───────────────────┬─────────────────┘
           │                    │                   │
           ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FastAPI Gateway  (src/api/)                             │
│       REST Endpoints  ·  WebSocket  ·  JWT Auth  ·  Rate Limiting   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangGraph Multi-Agent Supervisor                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Agronomy  │ │ Pricing  │ │Commerce  │ │  Voice   │ │ Quality  │  │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤  │
│  │ Matching │ │ Research │ │Knowledge │ │  ADCL   │ │Logistics │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└────────────────┬─────────────────┬───────────────────┬──────────────┘
                 │                 │                   │
                 ▼                 ▼                   ▼
┌────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  PostgreSQL + pgvector│  │ Neo4j Graph DB  │  │   Redis Cache        │
│  Users · Listings  │  │  Buyer-Seller    │  │   Sessions · Match   │
│  Orders · Prices   │  │  Relationships   │  │   Results · Pricing  │
└────────────────────┘  └──────────────────┘  └──────────────────────┘
```

### ⚙️ Tech Stack

| Layer                | Technology                                     | Purpose                             |
| :------------------- | :--------------------------------------------- | :---------------------------------- |
| **API**              | FastAPI 0.115+ · Uvicorn                       | Async REST + WebSocket server       |
| **AI Orchestration** | LangGraph · LangChain                          | Multi-agent supervisor graph        |
| **LLM — Production** | AWS Bedrock · Claude Sonnet                    | High-quality inference              |
| **LLM — Fast/Dev**   | Groq · Llama-3.1 / Mixtral                     | ~80ms routing & lightweight tasks   |
| **Primary DB**       | PostgreSQL (RDS / Supabase) + pgvector         | All business data + semantic search |
| **Graph DB**         | Neo4j                                          | Buyer-seller relationship graphs    |
| **Cache**            | Redis                                          | Sessions, match results, price data |
| **Voice**            | Pipecat · IndicWhisper · Edge-TTS · Silero VAD | Kannada/Hindi/English STT & TTS     |
| **Computer Vision**  | YOLOv8 · ViT-B/16 · ONNX Runtime               | Real-time crop quality grading      |
| **Scraping**         | Scrapling · Camoufox · Playwright              | APMC, eNAM, weather data            |
| **Logistics**        | OR-Tools CVRP · HDBSCAN                        | Route optimization, multi-pickup    |
| **Monitoring**       | LangSmith · Prometheus · Grafana               | Tracing, metrics, dashboards        |
| **Package Manager**  | `uv`                                           | 10–100× faster than pip             |

---

## 🧠 AI Agent System

| Agent                  | Role              | Key Capabilities                                                 |
| :--------------------- | :---------------- | :--------------------------------------------------------------- |
| **Supervisor**         | Query router      | Classifies & routes with 90%+ accuracy                           |
| **Agronomy**           | Farm advisor      | Crop health, pest/disease diagnosis, irrigation advice           |
| **Pricing (DPLE)**     | Market pricing    | AISP = Farmer Ask + Logistics + Margin (4–8%) + Risk Buffer (2%) |
| **Commerce**           | Marketplace       | Listings, orders, escrow, APMC mandi prices                      |
| **Voice**              | Accessibility     | 10+ Indian languages, all farmer flows voice-driven              |
| **Quality Assessment** | CV grading        | A+/A/B/C grading, HITL trigger, digital twin linkage             |
| **Buyer Matching**     | Demand-supply     | 5-factor scoring, GPS clustering, reverse matching               |
| **Digital Twin**       | Quality assurance | Departure/arrival quality diff, dispute resolution               |
| **Logistics Router**   | Routing           | Multi-pickup CVRP, deadhead cost calculation                     |
| **ADCL**               | Crop planning     | Weekly assured demand crop list from order data                  |
| **Knowledge**          | RAG retrieval     | Government schemes, agronomy docs, FSSAI/AGMARK                  |
| **Browser/Scraper**    | Data collection   | Stealth APMC + eNAM scraping, weather APIs                       |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — the fast Python package manager

### 1. Clone & Install `uv`

```bash
git clone https://github.com/cropfresh/Cropfresh_Ai.git
cd Cropfresh_Ai

# Install uv — Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
# Create virtual environment
uv venv --python 3.11

# Activate — Windows:
.\.venv\Scripts\Activate.ps1
# Activate — macOS/Linux:
source .venv/bin/activate

# Install core + voice support
uv sync --extra voice
```

| Option             | Command                         | Includes                          |
| :----------------- | :------------------------------ | :-------------------------------- |
| 🎯 Core only       | `uv sync`                       | FastAPI, LangGraph, Groq          |
| 🎤 + Voice         | `uv sync --extra voice`         | + IndicWhisper, Edge-TTS, Pipecat |
| 🧠 + ML            | `uv sync --extra ml`            | + PyTorch, Transformers           |
| 👁️ + Vision        | `uv sync --extra vision`        | + YOLOv8, OpenCV, ONNX            |
| 🌐 + Web Scraping  | `uv sync --extra web`           | + Playwright, Scrapling, MCP      |
| 🔭 + Observability | `uv sync --extra observability` | + OpenTelemetry, Prometheus       |
| 🎯 Everything      | `uv sync --all-extras`          | All of the above                  |

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

```env
# ── LLM Providers (at least one required) ──────────────────
GROQ_API_KEY=gsk_xxxxx
AWS_ACCESS_KEY_ID=xxxxx
AWS_SECRET_ACCESS_KEY=xxxxx
AWS_REGION=ap-south-1

# ── Database ────────────────────────────────────────────────
DATABASE_URL=postgresql://user:pass@host:5432/cropfresh

# ── Cache ───────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ── Graph DB (optional but recommended) ─────────────────────
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxxxx
```

### 4. Run

```bash
uv run uvicorn src.api.main:app --reload
```

🎉 Open **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive API playground.

---

## 🔌 API Reference

### Voice

| Endpoint                   | Method      | Description                                 |
| :------------------------- | :---------- | :------------------------------------------ |
| `/api/v1/voice/process`    | `POST`      | Voice audio → AI response → voice audio     |
| `/api/v1/voice/transcribe` | `POST`      | Audio → text (Kannada / Hindi / English)    |
| `/api/v1/voice/synthesize` | `POST`      | Text → natural Kannada/Hindi/English speech |
| `/ws/voice/{user_id}`      | `WebSocket` | Real-time duplex voice streaming            |

### Chat & RAG

| Endpoint              | Method | Description                       |
| :-------------------- | :----- | :-------------------------------- |
| `/api/v1/chat`        | `POST` | Multi-agent text chat             |
| `/api/v1/chat/stream` | `POST` | Streaming chat (SSE token events) |
| `/api/v1/rag/query`   | `POST` | Agricultural knowledge base query |
| `/api/v1/rag/search`  | `POST` | Semantic vector search            |

### Marketplace — Listings

| Endpoint                       | Method                     | Description                        |
| :----------------------------- | :------------------------- | :--------------------------------- |
| `/api/v1/listings`             | `POST`                     | Create a produce listing           |
| `/api/v1/listings`             | `GET`                      | Search/filter listings             |
| `/api/v1/listings/farmer/{id}` | `GET`                      | All listings for a farmer          |
| `/api/v1/listings/{id}`        | `GET` · `PATCH` · `DELETE` | Get / update / soft-cancel listing |
| `/api/v1/listings/{id}/grade`  | `POST`                     | Attach AI quality grade            |

### Marketplace — Orders

| Endpoint                      | Method  | Description                          |
| :---------------------------- | :------ | :----------------------------------- |
| `/api/v1/orders`              | `POST`  | Create order from matched listing    |
| `/api/v1/orders`              | `GET`   | List orders (filter by farmer/buyer) |
| `/api/v1/orders/{id}`         | `GET`   | Order details + AISP breakdown       |
| `/api/v1/orders/{id}/status`  | `PATCH` | Advance through 11-state machine     |
| `/api/v1/orders/{id}/dispute` | `POST`  | Raise dispute with arrival evidence  |
| `/api/v1/orders/{id}/settle`  | `POST`  | Settle order & release escrow        |
| `/api/v1/orders/{id}/aisp`    | `GET`   | Full AISP price breakdown            |

### Authentication & Health

| Endpoint                | Method          | Description                      |
| :---------------------- | :-------------- | :------------------------------- |
| `/api/v1/auth/register` | `POST`          | Farmer/buyer OTP registration    |
| `/api/v1/auth/login`    | `POST`          | JWT token issuance               |
| `/api/v1/auth/profile`  | `GET` · `PATCH` | User profile CRUD                |
| `/health`               | `GET`           | Liveness probe                   |
| `/health/ready`         | `GET`           | Readiness check (DB, Redis, LLM) |
| `/metrics`              | `GET`           | Prometheus metrics               |

---

## 📁 Project Structure

```
cropfresh-ai/
│
├── 📄 Root
│   ├── pyproject.toml         # uv project config & all extras
│   ├── docker-compose.yml     # API + Neo4j + Redis local stack
│   ├── Makefile               # make dev / make test / make lint …
│   ├── PLAN.md                # Master living plan (start here)
│   ├── ROADMAP.md             # 6-phase milestone roadmap
│   └── CHANGELOG.md           # Version history
│
├── src/                       # Application Source
│   ├── api/                   # FastAPI routers, middleware, schemas
│   ├── agents/                # All AI agents
│   │   ├── supervisor_agent.py
│   │   ├── agronomy_agent.py
│   │   ├── pricing_agent.py   # DPLE engine
│   │   ├── commerce_agent.py
│   │   ├── voice_agent.py
│   │   ├── quality_assessment/
│   │   ├── buyer_matching/
│   │   ├── digital_twin/
│   │   ├── logistics_router/
│   │   ├── price_prediction/
│   │   ├── crop_listing/
│   │   ├── adcl/              # Assured Demand Crop List
│   │   └── whatsapp_bot/
│   ├── rag/                   # RAG pipeline + vector search
│   ├── voice/                 # STT · TTS · VAD · WebSocket
│   ├── scrapers/              # APMC · eNAM · IMD weather
│   ├── db/                    # PostgreSQL models + migrations
│   ├── tools/                 # Agent tools (search, calculator, APIs)
│   ├── pipelines/             # Data processing pipelines
│   ├── resilience/            # Retry · circuit breaker · health checks
│   └── shared/                # Logging · constants · Kannada utils
│
├── ai/                        # AI & ML Infrastructure
│   ├── evals/                 # RAGAS evaluation framework
│   ├── rag/                   # Knowledge base + RAPTOR
│   └── models/                # Model registry
│
├── config/                    # DB migrations, Neo4j schemas, seeds
├── infra/                     # Docker · Cloud Run · Monitoring
├── tests/                     # Unit · Integration · E2E · Load
├── scripts/                   # Automation utilities
└── docs/                      # Architecture · ADRs · API specs · PRD
```

---

## 🧪 Development

```bash
# Start development server (hot reload)
make dev

# Run all tests
make test

# Lint & format
make lint
make format

# Type checking
make typecheck

# Run agent evaluations
make eval

# Start local stack (API + Neo4j + Redis)
make docker-up
```

See the [Makefile](Makefile) for all available commands.

---

## 📊 Key Metrics

| Metric                    | Target              | Status                 |
| :------------------------ | :------------------ | :--------------------- |
| 🎯 Agent Routing Accuracy | > 90%               | 🟡 ~87% (mock data)    |
| ⚡ Voice P95 Latency      | < 2s                | 🔴 ~4.5s (in progress) |
| 💰 API Cost per Query     | < ₹0.25             | 🔴 ~₹0.44 (optimizing) |
| 🧪 Test Coverage          | > 60% (Phase 5)     | ✅ ~57% (340 tests)    |
| 🏪 REST Endpoints         | —                   | ✅ 21 endpoints live   |
| 👩‍🌾 Farmer Pilot           | 50 active (Phase 6) | 🔲 Planned             |
| 📈 Price Improvement      | > 20% vs. middlemen | 🔲 Phase 6             |

---

## 🗺️ Roadmap

| Phase                               | Period       | Focus                                                         | Status      |
| :---------------------------------- | :----------- | :------------------------------------------------------------ | :---------- |
| **1 — Foundation & Core Agents**    | Feb–Mar 2026 | Multi-agent system, voice, pricing, matching, quality grading | ✅ Complete |
| **2 — Business Services**           | Mar–Apr 2026 | PostgreSQL schema, Listings API, Order lifecycle, Auth        | 🟢 Active   |
| **3 — Intelligence & Digital Twin** | Apr–May 2026 | Digital Twin, DPLE logistics, ADCL agent, APMC scraper        | 🔲 Planned  |
| **4 — Mobile & WhatsApp**           | Apr–Jun 2026 | Flutter app, WhatsApp bot, 10+ language voice                 | 🔲 Planned  |
| **5 — Testing & Evaluation**        | Jun–Jul 2026 | RAGAS framework, E2E tests, 60%+ coverage                     | 🔲 Planned  |
| **6 — Beta Launch**                 | Jul–Aug 2026 | 50-farmer Karnataka pilot, < 2s P95 latency                   | 🔲 Planned  |

See [ROADMAP.md](ROADMAP.md) for full deliverables and success criteria per phase.

---

## 📖 Documentation

| Document                                                  | Description                           |
| :-------------------------------------------------------- | :------------------------------------ |
| [PLAN.md](PLAN.md)                                        | Master living plan — **start here**   |
| [ROADMAP.md](ROADMAP.md)                                  | 6-phase milestone roadmap             |
| [Architecture](docs/architecture/ARCHITECTURE.md)         | System design overview                |
| [Tech Stack](docs/architecture/tech-stack.md)             | Every technology + why chosen         |
| [API Design](docs/architecture/api-design.md)             | API conventions & auth                |
| [Database Schema](docs/architecture/database-schema.md)   | PostgreSQL + pgvector + Neo4j schemas |
| [Agent Registry](docs/agents/REGISTRY.md)                 | All AI agents — specs & prompts       |
| [PRD](docs/planning/PRD.md)                               | Product Requirements Document         |
| [Coding Standards](docs/architecture/coding-standards.md) | Python style guide                    |
| [Security](docs/architecture/security.md)                 | Auth, encryption, DPDP compliance     |
| [DEPLOYMENT.md](DEPLOYMENT.md)                            | Production deployment guide           |

---

## 🤝 Contributing

1. Read [PLAN.md](PLAN.md) to understand current priorities
2. Pick an open issue or create one from the [templates](.github/ISSUE_TEMPLATE/)
3. Follow the [coding standards](docs/architecture/coding-standards.md) — enforced by `ruff` + `mypy`
4. Submit a PR using the [PR template](.github/pull_request_template.md)

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

<br/>

**Built with ❤️ for Indian farmers**

_Empowering Karnataka's agricultural community through the power of AI_

🌾 &nbsp; 🤖 &nbsp; 🇮🇳

<br/>

</div>
