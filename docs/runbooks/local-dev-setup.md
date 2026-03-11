# CropFresh AI — Local Development Runbook

> Step-by-step runbook for setting up the complete local development environment.

---

## Pre-Check

```bash
# Verify tools installed
python --version    # 3.11+
uv --version        # any
docker --version    # any
git --version       # any
```

---

## 1. Clone Repository

```bash
git clone <your-repo-url> cropfresh-ai
cd cropfresh-ai
```

## 2. Install Dependencies

```bash
uv sync
```

## 3. Start Infrastructure

```bash
# All services
docker compose up -d qdrant redis neo4j

# Verify
docker compose ps
# All should show "healthy"
```

## 4. Configure Environment

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   GROQ_API_KEY or AWS credentials
#   QDRANT_HOST=localhost
#   REDIS_URL=redis://localhost:6379/0
```

## 5. Populate Knowledge Base

```bash
uv run python scripts/populate_qdrant.py
# Should print: "Ingested X documents into agri_knowledge"
```

## 6. Seed Graph DB (Optional)

```bash
uv run python scripts/seed_graph.py
```

## 7. Run Server

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

## 8. Verify

```bash
# Health check
curl http://localhost:8000/health

# Expected: {"status": "healthy", "agents": 15, ...}
```

## 9. Open Swagger

Visit: **http://localhost:8000/docs**

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `EMBEDDING_MODEL too large` | Set `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` in `.env` |
| Qdrant connection refused | Check `docker ps` — qdrant container running |
| Redis timeout | Start Redis: `docker compose up -d redis` |
| Pipecat on Windows | Add to code: `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` |
| NVIDIA CUDA not available | Set `EMBEDDING_DEVICE=cpu` in `.env` |

---

## Teardown

```bash
# Stop all services but keep data
docker compose down

# Stop and remove all data
docker compose down -v
```
