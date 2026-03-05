---
description: How to set up and run the CropFresh AI service with UV package manager
---

# Setup and Run CropFresh AI Service

## Prerequisites

- Python 3.11+
- UV package manager
- Redis (optional — falls back to in-memory if unavailable)

## Cloud Services (Already Configured)

- **Qdrant Cloud**: Vector database for RAG
- **Supabase**: PostgreSQL for structured data
- **Neo4j AuraDB**: Graph database for relationships
- **Groq**: LLM API (Llama-3.3-70B)
- **Redis**: Session/memory cache (optional)

## Steps

// turbo-all

### 1. Navigate to project directory

```bash
cd "d:\Cropfresh-dev\Cropfresh Ai\Cropfresh_Ai"
```

### 2. Create virtual environment with UV

```bash
uv venv --python 3.11
```

### 3. Activate virtual environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install core dependencies

```bash
uv sync
```

### 5. Copy and configure environment

```bash
copy .env.example .env
# Edit .env with your API keys:
# - GROQ_API_KEY          (required — LLM provider)
# - QDRANT_HOST            (required — RAG vector DB)
# - QDRANT_API_KEY         (required)
# - SUPABASE_URL           (optional — structured data)
# - SUPABASE_KEY           (optional)
# - NEO4J_URI              (optional — graph DB for research)
# - NEO4J_PASSWORD         (optional)
# - REDIS_URL              (optional — session persistence, default: in-memory)
```

### 6. Test database connections

```bash
uv run python scripts/test_qdrant_cloud.py
uv run python scripts/test_supabase.py
uv run python scripts/test_neo4j.py
```

### 7. Populate knowledge base

```bash
uv run python scripts/populate_qdrant.py
```

### 8. Run the service

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

## Verification

After running, check the startup logs for:

```
✅ LLM provider ready
✅ KnowledgeAgent initialized
✅ Agent system: 15 agents registered
🚀 CropFresh AI Service ready
```

Then visit:

- Health: http://localhost:8000/health
- Ready: http://localhost:8000/health/ready (checks LLM + Qdrant + supervisor)
- Docs: http://localhost:8000/docs
- Chat: http://localhost:8000/api/v1/chat

## Agent System Architecture

On startup, `src/agents/agent_registry.py` creates and wires **15 agents**:

```
┌─────────────────── Supervisor ───────────────────┐
│  Routes queries to the best agent via LLM intent │
│  classification (falls back to keyword matching)  │
├──────────────────────────────────────────────────-┤
│  Core:  agronomy | commerce | platform | general  │
│  Price: pricing_agent | price_prediction_agent    │
│  Market: buyer_matching | quality_assessment      │
│         crop_listing                              │
│  Intel: adcl_agent | logistics_agent              │
│  Web:   research | web_scraping | browser         │
│  RAG:   knowledge_agent                           │
├───────────────────────────────────────────────────┤
│  Shared: LLM | KnowledgeBase | StateManager      │
│          ToolRegistry (7+ tools)                  │
└───────────────────────────────────────────────────┘
```

## Testing

### Run agent registry tests (15 tests)

```bash
uv run pytest tests/unit/test_agent_registry.py -v
```

### Run all unit tests

```bash
uv run pytest tests/unit/ -v --tb=short
```

### Test specific agents

```bash
uv run pytest tests/unit/test_adcl_agent.py -v
uv run pytest tests/unit/test_pricing_agent.py -v
uv run pytest tests/unit/test_quality_assessment.py -v
uv run pytest tests/unit/test_logistics_router.py -v
```

### Test Multi-Agent System

```bash
uv run python scripts/test_multi_agent.py
```

### Test RAG

```bash
uv run python scripts/test_rag.py
```

## Database Architecture

```
┌───────────────────────────────────────────────────┐
│            CropFresh AI Agent System              │
├───────────┬───────────┬─────────────┬─────────────┤
│ Qdrant    │ Supabase  │ Neo4j       │ Redis       │
│ (Vector)  │ (RDBMS)   │ (Graph)     │ (Cache)     │
├───────────┼───────────┼─────────────┼─────────────┤
│ RAG KB    │ Users     │ Research    │ Sessions    │
│ Embeddings│ Listings  │ Entity      │ Agent State │
│ Search    │ Orders    │ Relations   │ Match Cache │
└───────────┴───────────┴─────────────┴─────────────┘
```

## Common Issues

### Agent system shows 0 agents?

Check startup logs for warnings. Common causes:

- Missing `src/agents/agent_registry.py` — the factory file
- Import errors in individual agents (check each warning line)
- The system falls back to a bare supervisor if registry fails

### SSL Certificate Error with Neo4j?

```bash
uv add pip-system-certs
```

### GROQ_API_KEY missing?

Edit `.env` and add your Groq API key from https://console.groq.com
Without it, agents use rule-based fallbacks (no LLM generation).

### Supabase tables not found?

Run the SQL schema in Supabase Dashboard → SQL Editor

### Redis unavailable?

Not a blocker — AgentStateManager auto-falls back to in-memory.
Sessions won't persist across restarts without Redis.
