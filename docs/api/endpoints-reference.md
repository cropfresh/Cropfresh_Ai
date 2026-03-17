# CropFresh AI â€” API Endpoints Reference

> **Last Updated:** 2026-03-17
> **Base URL:** `http://localhost:8000` (dev) / `https://api.cropfresh.in` (prod)
> **Auth:** API Key (`X-API-Key` header) â€” skipped in dev mode
> **Content-Type:** `application/json`

---

## API Router Architecture

```mermaid
graph TD
    MAIN["FastAPI App<br/>src/api/main.py"] --> R1
    MAIN --> R2
    MAIN --> R3
    MAIN --> R4
    MAIN --> R5
    MAIN --> R6
    MAIN --> R7
    MAIN --> R8
    MAIN --> R9
    MAIN --> R10

    R1["routes/chat.py<br/>/api/v1/chat"]
    R2["routes/rag.py<br/>/api/v1/rag"]
    R3["routes/prices.py<br/>/api/v1/prices"]
    R4["routes/data.py<br/>/api/v1/data"]
    R5["routes/adcl.py<br/>/api/v1/adcl"]
    R6["routers/auth.py<br/>/api/v1/auth"]
    R7["routers/listings.py<br/>/api/v1/listings"]
    R8["routers/orders.py<br/>/api/v1/orders"]
    R9["rest/voice.py<br/>/api/v1/voice"]
    R10["websocket/voice_ws.py<br/>/ws/voice"]
```

---

## 1. Chat API (`/api/v1/chat`)

**Source:** `src/api/routes/chat.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Multi-turn conversation with agent routing |
| `/api/v1/chat/stream` | POST | SSE streaming responses |
| `/api/v1/chat/session` | POST | Create new session |
| `/api/v1/chat/agents` | GET | List available agents |

### POST `/api/v1/chat`

Query the multi-agent system. The SupervisorAgent routes to the best agent.

**Request:**
```json
{
  "query": "What is the price of tomato in Mysore?",
  "session_id": "uuid-string (optional)",
  "user_id": "farmer_123 (optional)",
  "agent_name": "commerce_agent (optional â€” force routing)",
  "language": "kn (optional)"
}
```

**Response:**
```json
{
  "response": "Tomato price in Mysore mandi is â‚¹25/kg.",
  "agent_name": "commerce_agent",
  "confidence": 0.92,
  "session_id": "uuid-string",
  "sources": ["Agmarknet APMC data", "knowledge_base"],
  "tools_used": ["agmarknet"],
  "suggested_actions": ["Check weekly trends", "Set price alert"]
}
```

### POST `/api/v1/chat/stream`

Same request as `/chat`, returns Server-Sent Events (SSE):

```
data: {"token": "Tomato"}
data: {"token": " price"}
data: {"token": " in"}
...
data: {"done": true, "agent_name": "commerce_agent"}
```

---

## 2. RAG API (`/api/v1/rag`)

**Source:** `src/api/routes/rag.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/rag/query` | POST | Query knowledge base with RAG pipeline |
| `/api/v1/rag/search` | POST | Semantic search (retrieval only) |
| `/api/v1/rag/ingest` | POST | Ingest documents into knowledge base |

### POST `/api/v1/rag/query`

Full RAG pipeline: query â†’ retrieve â†’ rerank â†’ generate.

**Request:**
```json
{
  "query": "How to grow tomatoes in Karnataka?",
  "top_k": 5,
  "categories": ["agronomy"],
  "use_graph": true,
  "use_reranker": true
}
```

---

## 3. Prices API (`/api/v1/prices`)

**Source:** `src/api/routes/prices.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/prices/current` | GET | Current APMC mandi prices |
| `/api/v1/prices/predict` | POST | Price predictions |

---

## 4. ADCL API (`/api/v1/adcl`)

**Source:** `src/api/routes/adcl.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/adcl/weekly` | GET | District-scoped weekly crop-demand report |

### GET `/api/v1/adcl/weekly`

**Query params:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `district` | string | Yes | District to score, e.g. `Kolar` |
| `force_live` | boolean | No | Bypass cached weekly report and rebuild from live data |
| `farmer_id` | string | No | Optional farmer context for future overlays |
| `language` | string | No | Optional response-language hint for summaries |

**Response shape:**
```json
{
  "week_start": "2026-03-16",
  "district": "Kolar",
  "generated_by": "adcl_service",
  "generated_at": "2026-03-17T10:00:00+00:00",
  "freshness": {
    "generated_at": "2026-03-17T10:00:00+00:00",
    "price": {"Tomato": "2026-03-17"},
    "imd": {"Tomato": "2026-03-18T06:00:00+00:00"},
    "enam": {}
  },
  "source_health": {
    "orders": {"status": "healthy", "order_count": 14},
    "rate_hub": {"status": "healthy", "sources": []},
    "imd": {"status": "healthy"},
    "enam": {"status": "gated"}
  },
  "metadata": {
    "force_live": true,
    "crop_count": 3,
    "green_count": 2,
    "recommendation_coverage": 0.667
  },
  "crops": [
    {
      "commodity": "Tomato",
      "green_label": true,
      "recommendation": "High demand and favorable price outlook",
      "demand_trend": "rising",
      "price_trend": "rising",
      "seasonal_fit": "good",
      "sow_season_fit": "current_window",
      "buyer_count": 4,
      "total_demand_kg": 650.0,
      "predicted_price_per_kg": 19.0,
      "evidence": [],
      "freshness": {},
      "source_health": {}
    }
  ]
}
```

---

## 5. Auth API (`/api/v1/auth`)

**Source:** `src/api/routers/auth.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/register` | POST | Register new farmer/buyer |
| `/api/v1/auth/login` | POST | Login (Firebase JWT) |
| `/api/v1/auth/profile` | GET | Get user profile |
| `/api/v1/auth/profile` | PUT | Update user profile |

---

## 6. Listings API (`/api/v1/listings`)

**Source:** `src/api/routers/listings.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/listings` | POST | Create crop listing |
| `/api/v1/listings` | GET | List user's listings |
| `/api/v1/listings/{id}` | GET | Get listing by ID |
| `/api/v1/listings/{id}` | PUT | Update listing |
| `/api/v1/listings/{id}` | DELETE | Cancel listing |

### POST `/api/v1/listings`

**Request:**
```json
{
  "commodity": "Tomato",
  "quantity_kg": 100,
  "asking_price_per_kg": 25.0,
  "grade": "A",
  "harvest_date": "2026-03-10",
  "location": "Kolar",
  "description": "Fresh farm tomatoes, hand-picked"
}
```

---

## 7. Orders API (`/api/v1/orders`)

**Source:** `src/api/routers/orders.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/orders` | POST | Create order from listing |
| `/api/v1/orders` | GET | List user's orders |
| `/api/v1/orders/{id}` | GET | Get order details |
| `/api/v1/orders/{id}/status` | PUT | Update order status |
| `/api/v1/orders/{id}/dispute` | POST | Report dispute |

---

## 8. Voice REST API (`/api/v1/voice`)

**Source:** `src/api/rest/voice.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/voice/process` | POST | Full voice pipeline (audio â†’ text â†’ agent â†’ audio) |
| `/api/v1/voice/transcribe` | POST | Audio â†’ text only (STT) |
| `/api/v1/voice/synthesize` | POST | Text â†’ audio only (TTS) |

### POST `/api/v1/voice/process`

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file (WAV/WebM) | âœ… | Audio file |
| `user_id` | string | âŒ | User identifier |
| `session_id` | string | âŒ | Session for multi-turn |
| `language` | string | âŒ | Language code (default: `kn`) |

**Response:**
```json
{
  "transcription": "à²Ÿà³Šà²®à³à²¯à²¾à²Ÿà³Š à²¬à³†à²²à³† à²Žà²·à³à²Ÿà³",
  "detected_language": "kn",
  "response_text": "à²Ÿà³Šà²®à³à²¯à²¾à²Ÿà³Š à²¬à³†à²²à³† â‚¹25/kg",
  "audio_url": "/tmp/response_audio.wav",
  "intent": "CHECK_PRICE",
  "entities": {"commodity": "Tomato"},
  "session_id": "uuid-string"
}
```

---

## 9. Voice WebSocket (`/ws/voice/{user_id}`)

**Source:** `src/api/websocket/voice_ws.py`

Real-time bidirectional audio streaming with VAD.

```
ws://localhost:8000/ws/voice/{user_id}?language=kn&session_id=uuid
```

See [`docs/api/websocket-voice.md`](websocket-voice.md) for full protocol.

---

## 10. Health API

**Source:** `src/api/main.py` (inline) + `src/api/routers/health.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/ready` | GET | Full readiness check (DB connections) |
| `/metrics` | GET | Prometheus metrics (when enabled) |

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Human-readable error message",
  "error_code": "AGENT_ERROR (or VALIDATION_ERROR, AUTH_ERROR, etc)",
  "status_code": 400
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (missing fields, invalid input) |
| 401 | Unauthorized (invalid API key) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not found |
| 429 | Rate limited |
| 500 | Internal server error |
| 503 | Service unavailable (DB connection lost) |

### 2026-03-17 Update — Multi-Source Rate Hub

The Prices API now supports the shared rate hub in addition to the legacy mocked endpoints.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/prices/latest` | GET | Legacy transparent aggregation mock kept for backward compatibility |
| `/api/v1/prices/history` | GET | Legacy history mock |
| `/api/v1/prices/summary` | GET | Legacy summary mock |
| `/api/v1/prices/query` | POST | Official-first multi-source Karnataka rate query |
| `/api/v1/prices/source-health` | GET | Health, freshness, and pending-source metadata for rate connectors |

#### POST `/api/v1/prices/query`

**Request:**
```json
{
  "rate_kinds": ["mandi_wholesale"],
  "commodity": "tomato",
  "state": "Karnataka",
  "district": null,
  "market": "Kolar",
  "date": "2026-03-17",
  "include_reference": true,
  "force_live": false,
  "comparison_depth": "all_sources"
}
```

**Response fields:**
- `query_target`
- `canonical_rates`
- `comparison_quotes`
- `source_health`
- `warnings`
- `pending_sources`
- `fetched_at`

#### GET `/api/v1/prices/source-health`

Returns enabled source health snapshots plus metadata-only pending sources such as eNAM official API access, NCDEX, and app-only sources that are not executed automatically.

