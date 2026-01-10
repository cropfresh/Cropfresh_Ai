---
description: How to set up and run the CropFresh AI service with UV package manager
---

# Setup and Run CropFresh AI Service

## Prerequisites
- Python 3.11+
- UV package manager

## Cloud Services (Already Configured)
- **Qdrant Cloud**: Vector database for RAG
- **Supabase**: PostgreSQL for structured data
- **Neo4j AuraDB**: Graph database for relationships
- **Groq**: LLM API

## Steps

// turbo-all

### 1. Navigate to project directory
```bash
cd d:\Cropfresh Ai\cropfresh-service-ai
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
# - GROQ_API_KEY
# - QDRANT_HOST, QDRANT_API_KEY
# - SUPABASE_URL, SUPABASE_KEY
# - NEO4J_URI, NEO4J_PASSWORD
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
After running, visit:
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs
- RAG: http://localhost:8000/api/v1/query
- Chat: http://localhost:8000/api/v1/chat

## Testing

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
┌─────────────────────────────────────┐
│       CropFresh AI Agent            │
├───────────┬───────────┬─────────────┤
│ Qdrant    │ Supabase  │ Neo4j       │
│ (Vector)  │ (RDBMS)   │ (Graph)     │
├───────────┼───────────┼─────────────┤
│ RAG       │ Users     │ Farmers     │
│ Knowledge │ Chat Hist │ Crops       │
│ Search    │ Orders    │ Supply Chain│
└───────────┴───────────┴─────────────┘
```

## Common Issues

### SSL Certificate Error with Neo4j?
```bash
uv add pip-system-certs
```

### GROQ_API_KEY missing?
Edit `.env` and add your Groq API key from https://console.groq.com

### Supabase tables not found?
Run the SQL schema in Supabase Dashboard → SQL Editor
