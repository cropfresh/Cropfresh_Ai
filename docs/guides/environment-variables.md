# CropFresh AI - Environment Variables

> **Last Updated:** 2026-03-17
> **Source of truth:** `src/api/config.py` (`Settings`)

Copy `.env.example` to `.env`, then override only what you need for your environment.

---

## Recommended LLM Setup

The code still defaults to `bedrock` today, but the recommended operating direction is now:

- `LLM_PROVIDER=groq`
- or `LLM_PROVIDER=vllm`
- or `LLM_PROVIDER=together`

Migration note: Bedrock references still exist in code as of March 17, 2026, but Sprint 07 removes Bedrock from runtime policy, docs, and fallbacks.

| Variable | Default in code | Recommended use |
|----------|-----------------|-----------------|
| `LLM_PROVIDER` | `bedrock` | Set to `groq`, `vllm`, or `together` |
| `LLM_MODEL` | `claude-sonnet-4` | Pick a model that matches the selected provider |
| `GROQ_API_KEY` | empty | Required when `LLM_PROVIDER=groq` |
| `TOGETHER_API_KEY` | empty | Required when `LLM_PROVIDER=together` |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Required when `LLM_PROVIDER=vllm` |
| `AWS_REGION` | `ap-south-1` | Legacy Bedrock and AWS infra region setting |
| `AWS_PROFILE` | empty | Optional AWS profile for local AWS SDK use |
| `BEDROCK_ROUTER_MODEL` | `claude-haiku` | Legacy Bedrock-only routing setting |

`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are not top-level app settings here, but the AWS SDK may still read them if the legacy Bedrock path is active.

---

## Data Stores

### PostgreSQL / Aurora

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_HOST` | empty | Database host |
| `PG_PORT` | `5432` | Database port |
| `PG_DATABASE` | `cropfresh` | Database name |
| `PG_USER` | `cropfresh_app` | Database user |
| `PG_PASSWORD` | empty | Database password |
| `PG_USE_IAM_AUTH` | `false` | Use IAM auth for Aurora when enabled |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |

### Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | empty | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `agri_knowledge` | Default collection |

### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | empty | Neo4j connection string |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | empty | Neo4j password |

### Legacy Migration Fields

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | empty | Legacy migration-period field |
| `SUPABASE_KEY` | empty | Legacy migration-period field |

---

## API and Runtime Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind host |
| `API_PORT` | `8080` | App port when not overridden by the launch command |
| `ENVIRONMENT` | `development` | `development`, `staging`, or `production` |
| `DEBUG` | `false` | Enables debug behavior |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_KEY` | empty | `X-API-Key` secret for non-dev routes |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

---

## Models and Voice Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Default STT model size hint |
| `YOLO_MODEL_PATH` | `models/yolov11m.pt` | Vision model path |

Note: VAD thresholds and duplex transport settings are not centralized in `Settings` yet. They are still configured in code and should be documented with the voice sprint work.

---

## Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENDPOINT` | empty | OpenTelemetry collector endpoint |
| `LANGSMITH_TRACING` | `true` | Enables LangSmith tracing |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith endpoint |
| `LANGSMITH_API_KEY` | empty | LangSmith API key |
| `LANGSMITH_PROJECT` | `Cropfresh Ai` | LangSmith project name |

---

## Feature and Scraping Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_ADAPTIVE_ROUTER` | `true` | Enable the adaptive query router |
| `USE_REDIS_CACHE` | `true` | Enable Redis-backed response caching |
| `ENABLE_RERANKER` | `true` | Enable reranking |
| `SCRAPING_RATE_LIMIT` | `30` | Request budget for scraping |
| `SCRAPING_CACHE_TTL` | `300` | Scraper cache TTL in seconds |

---

## 2026-03-24 Update - CI and Scraper Workflow Variables

### Scheduled Scraper Workflow

- The GitHub Actions scraper workflow now reuses the Aurora variables above when it runs `python -m src.scrapers`.
- `PG_HOST` is the on/off switch for persistence in that workflow. If it is empty, the scraper CLI runs in no-persistence smoke mode.
- The workflow currently defaults `PG_DATABASE=postgres`, `PG_USER=postgres`, `PG_PORT=5432`, and `PG_USE_IAM_AUTH=false` unless you override them in the workflow environment.
- `AWS_REGION` is also reused by the scraper CLI when IAM auth is enabled for Aurora.

### Opt-In Realtime Smoke Tests

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_REALTIME_E2E` | unset | Set to `1` to run the websocket realtime smoke tests against a live server |
| `VOICE_REALTIME_E2E_URL` | `ws://127.0.0.1:8000/api/v1/voice/ws` | Override the live websocket target for those smoke tests |
