# CropFresh AI - Getting Started

> **Last Updated:** 2026-03-17
> **Audience:** New contributors and local developers who want the app running quickly

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required |
| uv | Latest | Preferred package manager |
| Docker Desktop | Latest | For local infra services |
| Git | Latest | Repository workflows |
| One LLM option | Groq API key or local vLLM | Recommended starting point |

AWS infrastructure is still used by the project, but new local setup should not start with Bedrock.

---

## 1. Clone and Install

```bash
git clone <your-repo-url> cropfresh-ai
cd cropfresh-ai
uv sync
```

If you need the voice extras immediately:

```bash
uv sync --extra voice
```

---

## 2. Configure `.env`

```bash
cp .env.example .env
```

### Recommended Option A: Groq

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

### Recommended Option B: Local vLLM

```env
LLM_PROVIDER=vllm
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_BASE_URL=http://localhost:8001/v1
```

Migration note: Bedrock references still exist in code as of March 17, 2026, but Bedrock is no longer the recommended starting path and Sprint 07 removes it from runtime policy and docs.

See `docs/guides/environment-variables.md` for the complete settings list.

---

## 3. Start Local Infrastructure

### Minimal services for most development

```bash
docker compose up -d qdrant redis neo4j
```

### Notes

- If you are working on listings, orders, or ADCL, point `PG_*` to a real PostgreSQL or Aurora instance.
- If you are only exercising chat, docs, or some voice flows, the local service mix above is enough to start.

---

## 4. Populate the Knowledge Base

```bash
uv run python scripts/populate_qdrant.py
```

---

## 5. Run the API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

Useful URLs:

- Swagger: `http://localhost:8000/docs`
- Static live-test pages: `http://localhost:8000/static/voice_agent.html`
- Duplex demo page: `http://localhost:8000/static/premium_voice.html`

---

## 6. Smoke Test the Stack

### Chat API

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is the tomato price in Mysore?\"}"
```

### Voice REST

```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@sample.wav" \
  -F "user_id=demo" \
  -F "language=auto"
```

### Focused tests

```bash
uv run pytest tests/unit/test_voice_agent.py -q
uv run pytest tests/unit/test_pipecat_pipeline.py -q
```

---

## 7. What to Read Next

| Topic | Document |
|-------|----------|
| Current project state | `tracking/PROJECT_STATUS.md` |
| Voice sprint handoff | `tracking/sprints/sprint-07-voice-duplex-productionization.md` |
| Voice websocket contract | `docs/api/websocket-voice.md` |
| Voice architecture | `docs/features/voice-pipeline.md` |
| API reference | `docs/api/endpoints-reference.md` |
