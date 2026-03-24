# CropFresh AI

AI-powered agricultural marketplace backend for Indian farmers and buyers.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

CropFresh AI connects Karnataka farmers with buyers using multi-agent orchestration, shared market data services, and voice-first interaction. The repo includes chat, listings, orders, ADCL, RAG, and realtime voice surfaces.

---

## Current Status

As of March 17, 2026:

- Sprint 06 is still the active sprint for ADCL productionization.
- Sprint 07 is the next sprint and is dedicated to duplex voice productionization, Bedrock removal, local-language quality, and live-test cleanup.
- The canonical realtime voice route is `/api/v1/voice/ws/duplex`.
- Current voice latency is still roughly `3-4s` end to end.
- Bedrock references still exist in parts of the codebase, but Bedrock is no longer the intended provider direction.

Quick links:

| Resource | Link |
|----------|------|
| Live API docs | `https://xjivm4x5cn.ap-south-1.awsapprunner.com/docs` |
| Project status | `tracking/PROJECT_STATUS.md` |
| Voice sprint handoff | `tracking/sprints/sprint-07-voice-duplex-productionization.md` |
| Voice websocket protocol | `docs/api/websocket-voice.md` |

---

## What the Platform Does

- Helps farmers list produce, check prices, and ask market questions in natural language
- Helps buyers discover produce, manage orders, and inspect pricing logic
- Aggregates official-first market data through a shared Karnataka rate hub
- Generates district-level weekly demand recommendations through ADCL
- Supports realtime voice interaction through a duplex websocket path

---

## Architecture Snapshot

| Layer | Current Direction |
|-------|-------------------|
| API | FastAPI with REST and websocket routes |
| Agent orchestration | Supervisor plus domain agents |
| LLM provider layer | Groq, vLLM, Together |
| Voice | Duplex websocket, hybrid STT, Edge/local Indic TTS |
| Data | PostgreSQL/Aurora, Redis, Qdrant, Neo4j |
| Deployment | AWS App Runner, RDS, EC2 GPU, Secrets Manager |

Migration note: AWS infrastructure remains in use after Bedrock is removed from model-serving paths.

---

## Voice Stack

The production-facing voice story is:

- `/api/v1/voice/process` for one-shot REST voice processing
- `/api/v1/voice/ws` as a compatibility websocket path
- `/api/v1/voice/ws/duplex` as the canonical realtime voice contract

Current voice realities:

- JSON text frames with base64 audio payloads are still the active transport
- Silero VAD is used for speech detection and interruption handling
- the duplex route currently hardcodes `groq` for LLM and STT plus `edge` for TTS
- Pipecat remains experimental and is not the production-default path

Sprint 07 upgrades this path with stage-level latency instrumentation, Bedrock-free provider policy, better local-language naturalness, and cleaner live-test pages.

---

## Quick Start

### 1. Install dependencies

```bash
git clone <your-repo-url> cropfresh-ai
cd cropfresh-ai
uv sync
```

### 2. Configure `.env`

Recommended starting point:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...

PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=cropfresh
PG_USER=cropfresh_app
PG_PASSWORD=change_me

REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Alternative local model setup:

```env
LLM_PROVIDER=vllm
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_BASE_URL=http://localhost:8001/v1
```

### 3. Start local services

```bash
docker compose up -d qdrant redis neo4j
```

### 4. Run the API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### 5. Try the app

- Swagger: `http://localhost:8000/docs`
- Duplex demo page: `http://localhost:8000/static/premium_voice.html`
- Voice lab page: `http://localhost:8000/static/voice_agent.html`

---

## Recommended Docs

| Need | Document |
|------|----------|
| Current truth | `tracking/PROJECT_STATUS.md` |
| Next voice sprint | `tracking/sprints/sprint-07-voice-duplex-productionization.md` |
| Voice websocket contract | `docs/api/websocket-voice.md` |
| Voice architecture | `docs/features/voice-pipeline.md` |
| Full API reference | `docs/api/endpoints-reference.md` |
| Environment setup | `docs/guides/environment-variables.md` |

---

## Production Notes

- The recommended provider direction is `groq`, `vllm`, or `together`.
- Bedrock should be treated as a legacy code path scheduled for removal, not as the intended production model provider.
- Current voice latency is still above the target; Sprint 07 focuses on first-audio timing, interruption latency, and local-language quality.
- AWS infrastructure docs remain valid for App Runner, Aurora, VPC, IAM, and Secrets Manager.

---

## Acknowledgements

- AWS for the deployment infrastructure
- LangChain and LangGraph for orchestration primitives
- AI4Bharat and related Indic speech/model work
- Karnataka open-data sources for mandi and agriculture signals

---

## License

MIT - see [LICENSE](LICENSE)
