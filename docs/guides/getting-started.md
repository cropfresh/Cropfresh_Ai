# CropFresh AI — Getting Started Guide

> **For beginners:** This guide walks you through setting up, running, and testing CropFresh AI from scratch.

---

## Prerequisites

| Requirement | Version | Installation |
|-------------|---------|-------------|
| Python | 3.11+ | [python.org](https://python.org) |
| uv (package manager) | Latest | `pip install uv` |
| Docker Desktop | Latest | [docker.com](https://docker.com) |
| Git | Latest | [git-scm.com](https://git-scm.com) |

---

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url> cropfresh-ai
cd cropfresh-ai

# Install Python dependencies with uv
uv sync
```

---

## Step 2: Configure Environment

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Required for LLM (choose one)
GROQ_API_KEY=gsk_...                    # Groq Cloud (free tier available)
# OR
AWS_ACCESS_KEY_ID=...                    # Amazon Bedrock
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Optional but recommended
QDRANT_HOST=localhost                    # Docker: localhost, Cloud: your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=...                       # Only for Qdrant Cloud
REDIS_URL=redis://localhost:6379/0
```

See [`docs/guides/environment-variables.md`](environment-variables.md) for all options.

---

## Step 3: Start Infrastructure

### Option A: Docker Compose (Recommended)

```bash
# Start all services (Qdrant, Redis, Neo4j)
docker compose up -d qdrant redis neo4j

# Verify services are healthy
docker compose ps
```

### Option B: Minimal (Qdrant only)

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

---

## Step 4: Populate Knowledge Base

```bash
uv run python scripts/populate_qdrant.py
```

This loads agricultural knowledge documents into Qdrant for the RAG pipeline.

---

## Step 5: Run the Server

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Connected to Qdrant
INFO:     Agent system initialized (15 agents)
INFO:     ✅ Ready to serve
```

---

## Step 6: Test It

### Open Swagger Docs

Visit: **http://localhost:8000/docs**

### Quick Chat Test

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the price of tomato in Mysore?"}'
```

### Run Tests

```bash
uv run pytest -v
```

---

## What's Next?

| Task | Document |
|------|----------|
| Understand the architecture | [`docs/architecture/system-architecture.md`](../architecture/system-architecture.md) |
| Learn about agents | [`docs/agents/REGISTRY.md`](../agents/REGISTRY.md) |
| Create a new agent | [`docs/agents/agent-design-guide.md`](../agents/agent-design-guide.md) |
| API reference | [`docs/api/endpoints-reference.md`](../api/endpoints-reference.md) |
| Day-to-day workflow | [`docs/guides/development-workflow.md`](development-workflow.md) |
