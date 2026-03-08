<div align="center">

<br/>

# 🌾 CropFresh AI

### _AI-Powered Farm-to-Market Platform for Indian Farmers_

**Connecting Karnataka farmers directly with buyers — eliminating middlemen with AI agents, voice-first Kannada interaction, and real-time market intelligence.**

<br/>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AWS Bedrock](https://img.shields.io/badge/AWS_Bedrock-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

</div>

---

## 🔗 Quick Links

|                  |                                                     |
| :--------------- | :-------------------------------------------------- |
| 🚀 **Live API**  | https://xjivm4x5cn.ap-south-1.awsapprunner.com/docs |
| 📘 **Track**     | AI for Bharat — Agriculture & Rural Supply Chain    |
| 👤 **Team Lead** | Shrikantha M — CropFresh                            |

---

## ⚡ TL;DR

|                    |                                                                                                               |
| :----------------- | :------------------------------------------------------------------------------------------------------------ |
| **Problem**        | Small Indian farmers lose 30–40% of crop value to middlemen, with zero price transparency or buyer access     |
| **Solution**       | An AI agent + marketplace that tells farmers _where, when, and at what price to sell_ — in their own language |
| **Core AI**        | Amazon Bedrock (Claude) orchestrates 12 specialized AI agents via LangGraph                                   |
| **Voice**          | Kannada-first voice interaction — speak a question, get an answer in 2 seconds                                |
| **Infrastructure** | AWS App Runner · RDS PostgreSQL · EC2 GPU · Bedrock · Secrets Manager (production, `ap-south-1`)              |
| **Impact**         | Better prices for farmers, reliable supply for buyers — via voice, web, or WhatsApp                           |

---

## 🌾 The Problem

Indian farmers — especially small and marginal holders — are locked into multi-layer mandi systems:

- **30–40% of crop value** is captured by intermediaries, not the farmer
- **Zero price transparency** — farmers don't know today's mandi rate in the next district
- **No buyer discovery** — matching is done by phone calls and local agents
- **Language & literacy barriers** — most digital tools are English-only
- **Subjective quality grading** — disputes at delivery with no objective evidence

This hurts **farmers** (low income, uncertainty) and **buyers** (fragmented, unreliable supply chains).

---

## 💡 Solution — What CropFresh AI Does

CropFresh is an AI-powered farm-to-market platform built on AWS that:

- Lets **farmers** list produce via voice, get AI-suggested fair price bands, and connect with verified buyers
- Helps **buyers** discover and rank farmers by crop, volume, quality, and proximity using graph-based AI
- Provides a **conversational AI agent** answering real questions like:
  - _"ಈ ವಾರ ಮಾರಾಟ ಮಾಡಬಹುದಾದ ಅತ್ಯುತ್ತಮ ಬೆಳೆ ಯಾವುದು?"_ (What's the best crop to sell this week? — in Kannada)
  - _"What is the tomato price in Mysore APMC today?"_
  - _"Find me the best buyer for 500 kg of Grade-A onions within 50 km"_

> **In one line: CropFresh is an AI agent that helps farmers sell smarter and buyers procure reliably.**

---

## ✨ Key Features

**🌱 For Farmers**

- Voice-first onboarding and crop listing in Kannada, Hindi, or English
- AI computer vision crop quality grading (A+/A/B/C) from photos — no human subjectivity
- Fair price band recommendations from 160+ Karnataka APMC mandis in real time
- AI-ranked buyer suggestions with plain-language explanations (_"Buyer A is better — ₹2/kg more after transport"_)
- End-to-end escrow-protected order lifecycle — farmer gets paid on delivery confirmation

**🏪 For Buyers**

- Search farmers by crop type, volume, quality grade, GPS radius, and harvest window
- Graph-based AI matching with 5-factor confidence scores
- Order status tracking through 11 defined states: created → confirmed → dispatched → delivered → settled

**🤖 AI Agent (Farmer & Buyer)**

- Natural language Q&A powered by Amazon Bedrock (Claude Sonnet)
- Agronomy advisor — pest & disease diagnosis from voice or photo
- Weekly ADCL (Assured Demand Crop List) — tells farmers which crops have confirmed buyer demand
- All recommendations come with plain-language explanations — building trust, not black boxes

---

## 🧠 Why AI is Essential Here

AI is not a wrapper — it is the core of every farmer interaction:

| Function               | Without AI                          | With CropFresh AI                                                                                           |
| :--------------------- | :---------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Price guidance**     | Farmer guesses or calls local agent | DPLE engine computes fair price: farmer ask + logistics + 4–8% margin + 2% risk buffer, from live APMC data |
| **Buyer matching**     | Phone chain, 2–3 days               | Graph-based 5-factor match in < 1 second                                                                    |
| **Quality grading**    | Human visual — subjective, disputed | YOLOv8 + ViT computer vision, objective A+/A/B/C grade with photographic evidence                           |
| **Advisory**           | Wait for extension officer          | RAG-powered agronomy agent answers pest/disease/irrigation queries instantly                                |
| **Dispute resolution** | Word vs. word                       | Digital Twin Engine compares departure photo vs. arrival photo with SSIM diff score                         |
| **Language access**    | English-only tools                  | Voice in Kannada, Hindi, Tamil, Telugu + 7 more Indian languages                                            |

---

## ☁️ AWS Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│     📱 Flutter App (planned)   💬 WhatsApp Bot   🖥 Web / API       │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ HTTPS / WebSocket
┌─────────────────────────────▼────────────────────────────────────────┐
│            AWS App Runner — cropfresh-ai  (ap-south-1)               │
│            FastAPI + 12 LangGraph AI Agents  (~$25/mo)               │
│            Edge-TTS (cloud TTS) · Sentence Transformers (embeddings) │
└────────┬──────────────┬──────────────────────────────────────────────┘
         │              │ Internal API (POST /infer/*)
         │   ┌──────────▼──────────────────────────────────────┐
         │   │  EC2 g4dn.xlarge — GPU Inference Server (:8001) │
         │   │  IndicParler TTS · faster-whisper STT (Kannada) │
         │   │  YOLOv8 + DINOv2 ONNX Vision · Silero VAD       │
         │   └─────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                    │
│   🐘 RDS PostgreSQL + pgvector   🔮 Qdrant Cloud   ⚡ Redis Cloud    │
│   🕸 Neo4j (Graph relationships)  🔐 AWS Secrets Manager              │
└───────────────────────────────────────────────────────────────────────┘

 Amazon Bedrock (Claude Sonnet) — called by agents for LLM inference
 Amazon ECR — container image registry
 Amazon CloudWatch — logs + metrics
```

**How a request travels end-to-end:**

1. **Farmer speaks** — Flutter / web captures audio and sends a WebSocket frame
2. **App Runner receives** it and routes to the `VoiceAgent` via the LangGraph Supervisor
3. **Supervisor classifies** the intent (price query / listing / advisory / ...) and delegates to the right specialist agent
4. **Specialist agents** call their tools: APMC price scraper, PostgreSQL queries, Bedrock LLM, Qdrant RAG, Neo4j graph
5. **Amazon Bedrock** (Claude Sonnet) generates the natural language response with explanation
6. **EC2 GPU** converts the response text → Kannada audio via IndicParler TTS
7. **Audio streamed back** to the farmer — full round-trip target: < 3 seconds

---

## 🤖 AI Agents — Full Registry

| Agent                  | Role                                                | AWS / AI Service Used               |
| :--------------------- | :-------------------------------------------------- | :---------------------------------- |
| **Supervisor**         | Intent classification & routing                     | Groq Llama-3.1 (fast routing)       |
| **Agronomy**           | Crop advisory, pest & disease                       | Amazon Bedrock + RAG (Qdrant)       |
| **Pricing (DPLE)**     | Fair price = farmer ask + logistics + margin + risk | Live APMC data + rule engine        |
| **Commerce**           | Listings, orders, escrow, mandi lookup              | RDS PostgreSQL                      |
| **Voice**              | Kannada/Hindi/English STT→AI→TTS pipeline           | faster-whisper · Bedrock · IndicTTS |
| **Quality Assessment** | Photo → A+/A/B/C grade                              | YOLOv8 + ViT-B/16 (EC2 GPU ONNX)    |
| **Buyer Matching**     | 5-factor GPS + demand graph match                   | Neo4j + Redis cache                 |
| **Digital Twin**       | Departure vs arrival quality diff                   | SSIM + OpenCV (dispute engine)      |
| **Logistics Router**   | Multi-pickup route optimization                     | OR-Tools CVRP + HDBSCAN             |
| **ADCL**               | Weekly assured demand crop list                     | 90-day order aggregation            |
| **Knowledge / RAG**    | Govt. schemes, agronomy, FSSAI                      | Qdrant + RAPTOR + BGE-M3 embeddings |
| **Browser / Scraper**  | Live APMC · eNAM · IMD weather                      | Scrapling + Camoufox (stealth)      |

---

## ☁️ AWS Services — How Each Is Used

| AWS Service                        | How CropFresh Uses It                                                                |
| :--------------------------------- | :----------------------------------------------------------------------------------- |
| **Amazon Bedrock** (Claude Sonnet) | Core LLM for agent responses, buyer recommendations, and plain-language explanations |
| **AWS App Runner**                 | Runs the FastAPI backend — auto-scales, no server management, ~$25/mo                |
| **Amazon RDS (PostgreSQL)**        | All business data: users, listings, orders, prices, quality grades                   |
| **Amazon EC2 g4dn.xlarge**         | GPU inference: IndicParler TTS, faster-whisper STT, YOLOv8 vision ONNX               |
| **Amazon ECR**                     | Docker image registry — stores the API container built by GitHub Actions             |
| **AWS Secrets Manager**            | All API keys and credentials — zero secrets in code or environment files             |
| **Amazon CloudWatch**              | Lambda-style structured logs from App Runner + API metrics                           |

---

## 🛠️ Full Tech Stack

| Layer                  | Technology                        | Purpose                                 |
| :--------------------- | :-------------------------------- | :-------------------------------------- |
| **API**                | FastAPI 0.115+ · Uvicorn          | Async REST + WebSocket server           |
| **AI Orchestration**   | LangGraph · LangChain             | Multi-agent supervisor graph            |
| **LLM — Production**   | Amazon Bedrock · Claude Sonnet    | High-quality inference + explanations   |
| **LLM — Fast Routing** | Groq · Llama-3.1 / Mixtral        | ~80ms intent classification             |
| **Primary DB**         | RDS PostgreSQL + pgvector         | Business data + semantic vector search  |
| **Graph DB**           | Neo4j                             | Buyer-seller relationship graphs        |
| **Vector DB**          | Qdrant Cloud                      | RAG knowledge base retrieval            |
| **Cache**              | Redis Cloud                       | Sessions, match results, price data     |
| **Voice — STT**        | faster-whisper · IndicWhisper     | Kannada / Hindi / English transcription |
| **Voice — TTS**        | IndicParler TTS · Edge-TTS        | Natural Kannada/Hindi/English speech    |
| **Voice — Pipeline**   | Pipecat · Silero VAD · WebSocket  | Real-time duplex voice streaming        |
| **Computer Vision**    | YOLOv8 · ViT-B/16 · ONNX Runtime  | Crop quality grading from photos        |
| **Logistics**          | OR-Tools CVRP · HDBSCAN           | Route optimization, multi-pickup        |
| **Scraping**           | Scrapling · Camoufox · Playwright | APMC, eNAM, IMD weather data            |
| **CI/CD**              | GitHub Actions → ECR → App Runner | Automated build + deploy pipeline       |
| **Package Manager**    | `uv`                              | 10–100× faster than pip                 |

---

## 📁 Project Structure

```bash
Cropfresh_Ai/
├── src/                        # Application source
│   ├── api/                    # FastAPI routers, schemas, middleware
│   ├── agents/                 # All 12 AI agents
│   │   ├── supervisor_agent.py
│   │   ├── voice_agent.py      # 58KB — full 10-language voice pipeline
│   │   ├── pricing_agent.py    # DPLE fair price engine
│   │   ├── quality_assessment/ # YOLOv8 + ViT grading
│   │   ├── buyer_matching/     # Neo4j graph matching
│   │   ├── digital_twin/       # SSIM dispute engine
│   │   ├── logistics_router/   # CVRP route optimizer
│   │   ├── crop_listing/       # Photo → structured listing
│   │   ├── adcl/               # Assured Demand Crop List
│   │   └── whatsapp_bot/       # WhatsApp integration (planned)
│   ├── rag/                    # RAG pipeline + Qdrant + RAPTOR
│   ├── voice/                  # STT · TTS · VAD · WebSocket streaming
│   ├── scrapers/               # APMC · eNAM · IMD weather scrapers
│   ├── db/                     # PostgreSQL models + migrations
│   ├── tools/                  # Agent tools (price lookup, calculator)
│   ├── resilience/             # Retry · circuit breakers · health checks
│   └── shared/                 # Logging · constants · Kannada NLP utils
├── ai/
│   ├── evals/                  # RAGAS evaluation framework (golden dataset)
│   └── rag/                    # Knowledge base (FSSAI, AGMARK, govt. schemes)
├── config/                     # DB migrations, Neo4j schema, seeds
├── infra/
│   ├── docker/                 # CPU + GPU Dockerfiles
│   └── monitoring/             # Prometheus + Grafana dashboards
├── tests/                      # Unit · Integration · E2E · Load (Locust)
├── .github/workflows/          # CI/CD: build → ECR → App Runner
├── docs/                       # Architecture · ADRs · API specs · PRD
├── pyproject.toml              # uv project config + all extras
├── docker-compose.yml          # Local stack: API + Neo4j + Redis
├── Makefile                    # make dev / test / lint / eval
└── DEPLOYMENT.md               # AWS step-by-step deployment tracker
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- AWS account with Bedrock access (for production LLM)
- PostgreSQL database (or use the Docker Compose local stack)

### 1. Clone & Install `uv`

```bash
git clone https://github.com/cropfresh/Cropfresh_Ai.git
cd Cropfresh_Ai

# Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Environment & Install

```bash
uv venv --python 3.11

# Activate — Windows:
.\.venv\Scripts\Activate.ps1
# Activate — macOS/Linux:
source .venv/bin/activate

# Install core + voice support (recommended for demo):
uv sync --extra voice
```

| Option        | Command                  | What You Get                        |
| :------------ | :----------------------- | :---------------------------------- |
| 🎯 Core only  | `uv sync`                | FastAPI + LangGraph + Groq          |
| 🎤 + Voice    | `uv sync --extra voice`  | + IndicWhisper · Edge-TTS · Pipecat |
| 🧠 + ML       | `uv sync --extra ml`     | + PyTorch · Transformers            |
| 👁️ + Vision   | `uv sync --extra vision` | + YOLOv8 · OpenCV · ONNX            |
| 🌐 + Scraping | `uv sync --extra web`    | + Playwright · Scrapling            |
| 🎯 Everything | `uv sync --all-extras`   | Full production stack               |

### 3. Configure Environment

```bash
cp .env.example .env
# Fill in your credentials:
```

```env
# ── LLM (at least one required) ────────────────────────────
GROQ_API_KEY=gsk_xxxxx                       # fast routing + dev
AWS_ACCESS_KEY_ID=xxxxx                      # Amazon Bedrock (production)
AWS_SECRET_ACCESS_KEY=xxxxx
AWS_REGION=ap-south-1

# ── Database ────────────────────────────────────────────────
DATABASE_URL=postgresql://user:pass@host:5432/cropfresh

# ── Vector Search ───────────────────────────────────────────
QDRANT_URL=https://xxxxx.eu-central-1.aws.cloud.qdrant.io
QDRANT_API_KEY=xxxxx

# ── Cache ───────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ── Graph DB (buyer-seller matching) ────────────────────────
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxxxx

# ── Auth ────────────────────────────────────────────────────
JWT_SECRET=your-256-bit-secret
```

### 4. Run Local Stack

```bash
# Option A — Docker (API + PostgreSQL + Neo4j + Redis, zero config):
make docker-up

# Option B — Direct:
uv run uvicorn src.api.main:app --reload
```

🎉 **API Playground:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Quick Demo — Test the AI Agent

```bash
# Health check
curl http://localhost:8000/health/ready

# Ask the AI agent a farming question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the tomato price in Mysore APMC today?", "user_id": "demo"}'

# Voice: text → Kannada speech
curl -X POST http://localhost:8000/api/v1/voice/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "ನಮಸ್ಕಾರ, ನಾನು ಕ್ರಾಪ್‌ಫ್ರೆಶ್ AI", "language": "kn"}' \
  --output demo_kannada.wav
```

---

## 🧪 Development Commands

```bash
make dev          # Hot-reload dev server
make test         # Run full test suite (340+ tests)
make lint         # Ruff lint + format check
make typecheck    # mypy strict type check
make eval         # RAGAS agent evaluation
make docker-up    # Start local full stack
```

---

## 📊 Performance & Metrics

| Metric                    | Target          | Current Status                              |
| :------------------------ | :-------------- | :------------------------------------------ |
| 🎯 Agent Routing Accuracy | > 90%           | ~87% (mock eval)                            |
| ⚡ Voice Round-trip P95   | < 2 s           | ~4.5 s (GPU still provisioning)             |
| 💰 API Cost per Query     | < ₹0.25         | ~₹0.44 (being optimized)                    |
| 🧪 Test Coverage          | > 60% (Phase 5) | **~57%** (340 tests / 15 files)             |
| 🏪 REST Endpoints Live    | —               | **21 endpoints** (listings + orders + auth) |
| 📦 Docker Image Build     | < 5 min         | **2m 52s** on GitHub Actions                |

---

## 💰 Cost & Responsible AI

**Cost optimization:**

- **AWS App Runner** (serverless containers) — no idle cost, scales to zero: ~$25/mo
- **Groq** (fast routing layer) — ~80ms latency at a fraction of Bedrock cost for lightweight intent classification
- **EC2 GPU** used only for inference, not always-on training; can be replaced with SageMaker endpoints on demand
- **Bedrock** calls scoped to production-quality responses only — cheap Groq/Llama handles classification

**Responsible AI & guardrails:**

- Farmer data is **never used for LLM training** — a hard contractual commitment
- All PII (phone numbers, Aadhaar) stored with **AES-256 encryption** and accessed only via JWT-scoped roles
- **Explainable recommendations** — every AI suggestion (price, buyer, advisory) includes a plain-language reason so farmers can verify and trust the output
- Quality dispute evidence is **photographic + algorithmic** (SSIM diff), not just agent opinion
- Planned: Amazon Bedrock **Guardrails** for content filtering before Phase 6 launch
- Compliant with **India DPDP Act 2023** data minimisation principles

---

## 🗺️ Roadmap

| Phase                               | Period       | Focus                                                               | Status      |
| :---------------------------------- | :----------- | :------------------------------------------------------------------ | :---------- |
| **1 — Foundation & Core Agents**    | Feb–Mar 2026 | Multi-agent system, voice, pricing, quality grading, buyer matching | ✅ Complete |
| **2 — Business Services**           | Mar–Apr 2026 | PostgreSQL schema, Listings API, Order lifecycle, OTP Auth          | 🟢 Active   |
| **3 — Intelligence & Digital Twin** | Apr–May 2026 | Digital Twin disputes, DPLE logistics, ADCL, APMC live scraper      | 🔲 Planned  |
| **4 — Mobile & WhatsApp**           | Apr–Jun 2026 | Flutter app, WhatsApp bot, 10+ language voice production            | 🔲 Planned  |
| **5 — Testing & Evaluation**        | Jun–Jul 2026 | RAGAS framework, E2E coverage ≥ 60%, security hardening             | 🔲 Planned  |
| **6 — Beta Launch**                 | Jul–Aug 2026 | 50-farmer Karnataka pilot, NPS > 40, < 1% error rate                | 🔲 Planned  |

---

## ✅ How to Evaluate This Project — For Judges

| Step                         | What to Do                                                        | What Demonstrates                                   |
| :--------------------------- | :---------------------------------------------------------------- | :-------------------------------------------------- |
| **1. Hit the live API**      | `GET https://xjivm4x5cn.ap-south-1.awsapprunner.com/health/ready` | Real AWS deployment, not localhost                  |
| **2. Try the AI agent**      | `POST /api/v1/chat` with a farming question                       | Amazon Bedrock + LangGraph agent working end-to-end |
| **3. Watch the demo video**  | Full farmer → listing → AI price → buyer match → explanation      | Complete user flow with Kannada voice               |
| **4. Review the agent code** | `src/agents/supervisor_agent.py` + specialist agents              | Multi-agent complexity, not a single GPT wrapper    |
| **5. Check the AI value**    | "Why AI is Essential" section above                               | Non-trivial AI — 12 agents, CV, RAG, graph matching |
| **6. Review AWS usage**      | `DEPLOYMENT.md` + this README                                     | Bedrock, App Runner, RDS, EC2 GPU, Secrets Manager  |
| **7. Evaluate impact**       | Problem statement + roadmap                                       | Real problem, real farmers, measurable metrics      |

**Hackathon criteria alignment:**

- 🎨 **Creativity** — Voice-first Kannada AI for farmers; Digital Twin quality dispute engine
- 🔧 **Technical complexity** — 12 LangGraph agents, CV pipeline, RAG, graph DB, CVRP routing, Bedrock orchestration
- 🌍 **Impact** — 93M+ small & marginal farmers in India; attacks a ₹17T agriculture market
- 📱 **Usability** — Farmer can interact entirely by voice in their native language; no typing required

---

## 📖 Documentation

| Document                                                | Description                                     |
| :------------------------------------------------------ | :---------------------------------------------- |
| [PLAN.md](PLAN.md)                                      | Master living plan — start here                 |
| [ROADMAP.md](ROADMAP.md)                                | 6-phase milestone roadmap with success criteria |
| [DEPLOYMENT.md](DEPLOYMENT.md)                          | AWS deployment step-by-step tracker             |
| [Architecture](docs/architecture/ARCHITECTURE.md)       | System design deep-dive                         |
| [Tech Stack](docs/architecture/tech-stack.md)           | Every technology and why it was chosen          |
| [API Design](docs/architecture/api-design.md)           | API conventions, auth, versioning               |
| [Database Schema](docs/architecture/database-schema.md) | PostgreSQL + pgvector + Neo4j                   |
| [Agent Registry](docs/agents/REGISTRY.md)               | All 12 AI agents — specs & prompts              |
| [PRD](docs/planning/PRD.md)                             | Product Requirements Document                   |
| [Security](docs/architecture/security.md)               | Auth, encryption, DPDP compliance               |

---

## 👥 Team

| Name             | Role                                           |
| :--------------- | :--------------------------------------------- |
| **Shrikantha M** | Product, Architecture, Backend, AI Integration |

---

## 🙏 Acknowledgements

- AWS for Bedrock access and serverless infrastructure
- The LangChain / LangGraph team for the multi-agent framework
- AI4Bharat for IndicWhisper and IndicParler TTS models
- Karnataka government for open APMC mandi price data
- Every farmer who gave feedback on voice interaction prototypes

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

<br/>

**Built with ❤️ for India's 93 million small and marginal farmers**

_Empowering Karnataka's agricultural community through the power of AI_

🌾 &nbsp; 🤖 &nbsp; 🇮🇳

<br/>

</div>
