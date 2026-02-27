<div align="center">

# 🌾 CropFresh AI

### *India's Intelligent Agricultural Marketplace*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Connecting Karnataka farmers directly with buyers using AI agents, voice-first Kannada interaction, and real-time market intelligence — eliminating middlemen, one crop at a time.*

[**Getting Started**](#-quick-start) · [**Architecture**](#-architecture) · [**Features**](#-features) · [**Documentation**](#-documentation) · [**Contributing**](#-contributing)

</div>

---

## 🌟 The Problem We Solve

> **Indian farmers lose 30-40% of their crop value to intermediaries.** Most have no access to real-time market data, quality assessment tools, or direct buyer connections — especially in their local language.

**CropFresh AI** changes this with:

| Challenge | Our Solution |
|-----------|-------------|
| 🏪 Middlemen take huge cuts | Direct farmer-to-buyer marketplace |
| 📊 No price transparency | Real-time APMC data + AI price prediction |
| 🎤 Language barriers | Voice-first Kannada interaction |
| 📋 Manual crop listing | AI-powered auto-listing from photos |
| ❓ Subjective quality grading | Computer vision quality assessment |
| 🔍 Finding the right buyer | Graph-based intelligent matching |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 AI Agent System
Multi-agent architecture powered by **LangGraph** with specialized agents:
- **Supervisor** — Routes queries to the right specialist
- **Agronomy** — Agricultural knowledge & crop advisory
- **Pricing** — Real-time market price analysis
- **Commerce** — Transaction & marketplace operations
- **Voice** — Kannada voice interaction
- **Knowledge** — RAG-based information retrieval
- **Browser** — Web scraping & data collection

</td>
<td width="50%">

### 🎤 Voice-First Design
Built for farmers who prefer speaking over typing:
- **Kannada STT** — Whisper-powered speech recognition
- **Kannada TTS** — Edge-TTS natural voice synthesis
- **WebRTC** — Real-time audio streaming
- **11 Indian languages** — Hindi, Telugu, Tamil, Malayalam, and more
- **WhatsApp Bot** — Coming soon

</td>
</tr>
<tr>
<td>

### 🔍 Intelligent RAG Pipeline
Advanced retrieval-augmented generation:
- **Qdrant** vector search with hybrid retrieval
- **RAPTOR** hierarchical document summaries
- **Graph RAG** with Neo4j knowledge graph
- **Contextual chunking** & advanced re-ranking
- **Agricultural knowledge base** (FSSAI, AGMARK, govt. schemes)

</td>
<td>

### 📊 Market Intelligence
Real-time agricultural data pipeline:
- **160+ Karnataka APMC mandis** price tracking
- **eNAM** national market integration
- **IMD weather** data for crop forecasting
- **Historical price analysis** & trend prediction
- **Seasonal pattern** recognition

</td>
</tr>
</table>

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Clients                               │
│   📱 Flutter App    💬 WhatsApp Bot    🖥️ Web Dashboard       │
└───────────┬──────────────┬──────────────┬────────────────────┘
            │              │              │
            ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                  FastAPI Gateway (src/api/)                    │
│   REST Routers  ·  WebSocket  ·  Auth  ·  Rate Limiting       │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│              LangGraph Multi-Agent System                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  Supervisor Agent                         │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────────┐  │ │
│  │  │Agronomy │ │ Pricing │ │Commerce │ │    Voice      │  │ │
│  │  ├─────────┤ ├─────────┤ ├─────────┤ ├───────────────┤  │ │
│  │  │Browser  │ │Research │ │Knowledge│ │   General     │  │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────┬──────────────┬──────────────┬────────────────────┘
            │              │              │
            ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                       Data Layer                              │
│   🐘 Supabase        🔮 Qdrant          🕸️ Neo4j             │
│   (PostgreSQL)       (Vectors)          (Graph)              │
└──────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async REST + WebSocket server |
| **AI Framework** | LangGraph + LangChain | Multi-agent orchestration |
| **LLM** | Groq (Llama/Mixtral) | Fast, cost-effective inference |
| **Vector DB** | Qdrant Cloud | Semantic search & RAG retrieval |
| **Graph DB** | Neo4j | Buyer-seller matching & relationships |
| **Primary DB** | Supabase (PostgreSQL) | Users, listings, orders, prices |
| **Voice** | Whisper + Edge-TTS | Kannada speech-to-text & text-to-speech |
| **Scraping** | Crawl4AI + Playwright | APMC, eNAM, weather data collection |
| **Package Manager** | uv | 10-100x faster than pip |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone & Setup

```bash
git clone https://github.com/cropfresh/Cropfresh_Ai.git
cd Cropfresh_Ai

# Install uv (if not already installed)
# Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
# Create virtual environment & install
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1    # Windows
# source .venv/bin/activate     # macOS/Linux

uv sync --extra voice           # Core + Voice features
```

| Install Option | Command | What You Get |
|---------------|---------|-------------|
| 🎯 Core only | `uv sync` | FastAPI, LangGraph, Groq |
| 🎤 + Voice | `uv sync --extra voice` | + STT/TTS (Kannada) |
| 🧠 + ML | `uv sync --extra ml` | + PyTorch, Transformers |
| 👁️ + Vision | `uv sync --extra vision` | + YOLOv11, OpenCV |
| 🌐 + Web | `uv sync --extra web` | + Playwright, Crawl4AI |
| 🔭 + Observability | `uv sync --extra observability` | + OpenTelemetry |
| 🎯 Everything | `uv sync --all-extras` | All of the above |

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

```env
# Required
GROQ_API_KEY=gsk_xxxxx

# Optional (for full features)
QDRANT_URL=https://xxx.qdrant.io
QDRANT_API_KEY=xxxxx
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxxxx
NEO4J_URI=neo4j+s://xxx
```

### 4. Run

```bash
uv run uvicorn src.api.main:app --reload
```

🎉 **Visit** [http://localhost:8000/docs](http://localhost:8000/docs) — Interactive API documentation

---

## 🎤 API Endpoints

### Voice API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/voice/process` | POST | Voice-in → AI → Voice-out |
| `/api/v1/voice/transcribe` | POST | Audio → Text (Kannada/Hindi/English) |
| `/api/v1/voice/synthesize` | POST | Text → Natural speech |
| `/ws/voice/{user_id}` | WebSocket | Real-time voice streaming |

### Chat & RAG API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Multi-agent chat (text) |
| `/api/v1/rag/query` | POST | Knowledge base query |

### Health
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service health check |

---

## 📁 Project Structure

```
cropfresh-ai/
│
├── 📄 Root Files
│   ├── AGENTS.md              # AI agent instructions
│   ├── PLAN.md                # Master living plan
│   ├── CHANGELOG.md           # Version history
│   ├── Makefile               # Quick commands (make dev, make test)
│   ├── docker-compose.yml     # Local: API + Qdrant + Neo4j + n8n
│   └── pyproject.toml         # Python config (uv)
│
├── 📚 docs/                   # All Documentation
│   ├── planning/              #   PRD, personas, roadmap, market analysis
│   ├── architecture/          #   System design, tech stack, data flow
│   ├── decisions/             #   Architecture Decision Records (ADRs)
│   ├── features/              #   Feature specs (F001-F009)
│   ├── agents/                #   Per-agent specs, prompts, evals
│   └── api/                   #   API docs, endpoint specs, JSON schemas
│
├── 📊 tracking/               # Development Progress
│   ├── GOALS.md               #   OKRs & milestones
│   ├── sprints/               #   Sprint planning & review
│   ├── daily/ & weekly/       #   Development logs
│   ├── agent-performance/     #   Per-agent metrics over time
│   └── retros/                #   Sprint retrospectives
│
├── 🐍 src/                    # Application Source Code
│   ├── api/                   #   FastAPI (routers, models, services, middleware)
│   ├── agents/                #   AI agents (supervisor + 10 specialists)
│   │   ├── crop_listing/      #     Crop Listing Agent (photos → listing)
│   │   ├── price_prediction/  #     Price Prediction Agent
│   │   ├── quality_assessment/#     Quality Grading Agent (CV)
│   │   ├── buyer_matching/    #     Buyer-Seller Matching Agent
│   │   └── whatsapp_bot/      #     WhatsApp Conversation Agent
│   ├── scrapers/              #   APMC, eNAM, weather data collectors
│   ├── pipelines/             #   Data processing pipelines
│   ├── voice/                 #   Voice processing (STT/TTS/VAD/WebRTC)
│   ├── tools/                 #   Agent tools (search, calculator, APIs)
│   ├── workflows/             #   n8n workflow definitions
│   └── shared/                #   Logging, exceptions, constants, Kannada utils
│
├── 🧠 ai/                    # AI & ML Infrastructure
│   ├── data/                  #   Training data (raw/processed/eval)
│   ├── models/                #   Trained models + registry
│   ├── notebooks/             #   Jupyter experiments
│   ├── evals/                 #   Agent evaluation framework
│   │   ├── judges/            #     LLM-as-Judge evaluators
│   │   ├── metrics/           #     Accuracy, latency, cost metrics
│   │   └── regression/        #     Regression detection
│   └── rag/                   #   RAG pipeline + knowledge base
│
├── 🧪 tests/                  # Test Infrastructure
│   ├── e2e/                   #   End-to-end flow tests
│   ├── integration/           #   Service integration tests
│   └── load/                  #   Performance & load tests (Locust)
│
├── 🐳 infra/                  # Deployment & Monitoring
│   ├── docker/                #   Dockerfiles (API, scraper, n8n)
│   ├── gcp/                   #   Cloud Run configs
│   └── monitoring/            #   Dashboards & alert rules
│
├── ⚙️ config/                 # Database & Service Configs
│   ├── supabase/              #   SQL migrations + seed data
│   ├── qdrant/                #   Vector collection definitions
│   ├── neo4j/                 #   Graph constraints + seed
│   ├── firebase/              #   Auth & storage rules
│   └── n8n/                   #   Workflow credentials
│
├── 📱 mobile/                 # Flutter App (Sprint 4)
├── 🔧 scripts/                # Automation Utilities
└── 🔄 .github/                # CI/CD Workflows & Templates
```

---

## 🧪 Development

```bash
# Development server (hot reload)
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

# Docker (API + Qdrant + Neo4j + n8n)
make docker-up
```

See the [Makefile](Makefile) for all available commands.

---

## 🗺️ Roadmap

| Phase | Timeline | Focus | Status |
|-------|----------|-------|--------|
| **1. Foundation** | Feb 2026 | Project setup, RAG, voice | ✅ Complete |
| **2. Data Pipeline** | Mar 2026 | APMC scraping, Supabase, indexing | 🔄 In Progress |
| **3. First Agent** | Apr 2026 | Crop Listing Agent MVP | ⬜ Planned |
| **4. Market Intelligence** | May 2026 | Price prediction, weather | ⬜ Planned |
| **5. Mobile & Voice** | Jun 2026 | Flutter app, WhatsApp bot | ⬜ Planned |
| **6. Beta Launch** | Jul 2026 | 50 farmers in Karnataka | ⬜ Planned |

See [PLAN.md](PLAN.md) and [docs/planning/roadmap.md](docs/planning/roadmap.md) for detailed roadmap.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Master living plan — **start here** |
| [Architecture](docs/architecture/ARCHITECTURE.md) | System design overview |
| [Tech Stack](docs/architecture/tech-stack.md) | Every technology + why chosen |
| [API Design](docs/architecture/api-design.md) | API conventions & auth |
| [Database Schema](docs/architecture/database-schema.md) | Supabase + Qdrant + Neo4j schemas |
| [Agent Registry](docs/agents/REGISTRY.md) | All AI agents listed |
| [PRD](docs/planning/PRD.md) | Product Requirements Document |
| [Coding Standards](docs/architecture/coding-standards.md) | Python style guide |
| [Security](docs/architecture/security.md) | Auth, encryption, DPDP compliance |

---

## 🎯 Key Metrics (Targets)

| Metric | Target | Description |
|--------|--------|-------------|
| 🎯 Agent Accuracy | >90% | Across all task types |
| ⚡ Response Latency | <2s | End-to-end response time |
| 💰 Cost per Query | <₹0.50 | API + LLM cost per interaction |
| 👨‍🌾 Farmer Adoption | 50+ | Beta test in Karnataka |
| 📈 Price Improvement | >20% | vs. middleman prices |

---

## 🤝 Contributing

1. Check [PLAN.md](PLAN.md) for current priorities
2. Pick an issue or create one from [templates](.github/ISSUE_TEMPLATE/)
3. Follow [coding standards](docs/architecture/coding-standards.md)
4. Submit a PR using the [PR template](.github/pull_request_template.md)

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for Indian farmers**

*Empowering Karnataka's agricultural community through AI*

🌾 🤖 🇮🇳

</div>
