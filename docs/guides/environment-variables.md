# CropFresh AI — Environment Variables

> All environment variables from `src/api/config.py` (`Settings` class).
> Copy `.env.example` → `.env` and fill in your values.

---

## LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM backend: `groq`, `bedrock`, `together`, `vllm` |
| `GROQ_API_KEY` | — | Groq Cloud API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `AWS_ACCESS_KEY_ID` | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-haiku-20240307-v1:0` | Bedrock model |
| `TOGETHER_API_KEY` | — | Together.ai API key |
| `TOGETHER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Together model |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM server URL |

---

## Databases

### Qdrant (Vector DB)

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant gRPC port |
| `QDRANT_API_KEY` | — | API key (Cloud only) |
| `QDRANT_COLLECTION` | `agri_knowledge` | Default collection |

### PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DATABASE` | `cropfresh` | Database name |
| `PG_USER` | `cropfresh` | Database user |
| `PG_PASSWORD` | — | Database password |

### Neo4j (Graph DB)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |

### Redis (Cache)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |

### Supabase

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase anon key |

---

## API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind host |
| `API_PORT` | `8000` | Bind port |
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOWED_ORIGINS` | `["*"]` | CORS allowed origins (never `*` in production) |
| `API_KEY` | — | API key for `X-API-Key` auth |
| `API_KEY_HEADER` | `X-API-Key` | Auth header name |

---

## Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `EMBEDDING_DEVICE` | `cpu` | Device: `cpu` or `cuda` |

---

## Voice

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `DEFAULT_LANGUAGE` | `kn` | Default voice language |
| `VAD_THRESHOLD` | `0.5` | Silero VAD sensitivity |

---

## Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGSMITH_API_KEY` | — | LangSmith tracing key |
| `LANGSMITH_PROJECT` | `cropfresh-ai` | LangSmith project name |
| `OTEL_ENDPOINT` | — | OpenTelemetry collector URL |
| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus metrics |

---

## Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_VOICE` | `true` | Enable voice endpoints |
| `ENABLE_RAG` | `true` | Enable RAG pipeline |
| `ENABLE_GRAPH_RAG` | `false` | Enable Neo4j graph retrieval |
| `ENABLE_SCRAPING` | `true` | Enable live scraping |
| `ENABLE_AUTONOMOUS` | `false` | Enable autonomous goal agent |
