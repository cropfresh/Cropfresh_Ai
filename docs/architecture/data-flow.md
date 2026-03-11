# CropFresh AI — Data Flow Diagrams

> **Last Updated:** 2026-03-11
> All diagrams reflect the **actual codebase** as of this date.

---

## 1. Agent Routing Flow

Every user query (text or voice) flows through the `SupervisorAgent` which decides which specialized agent handles it.

```mermaid
flowchart TD
    U["👤 User Query<br/>(text / voice / WhatsApp)"] --> API["FastAPI Endpoint<br/>/api/v1/chat or /ws/voice"]
    API --> SM["StateManager<br/>Load/create session"]
    SM --> SUP["SupervisorAgent.process()"]

    SUP --> R{"Has LLM?"}
    R -->|Yes| LLM["LLM Routing<br/>(temp=0.1, max_tokens=200)<br/>Returns JSON with agent_name + confidence"]
    R -->|No| RB["Rule-Based Routing<br/>Keyword matching<br/>across 14 agent categories"]

    LLM --> D{"Parse JSON?"}
    D -->|Success| SEL
    D -->|Fail| RB

    RB --> SEL["Select Target Agent"]
    SEL --> AGENT["Target Agent.process()<br/>(query, context, execution)"]
    AGENT --> MULTI{"Multi-agent<br/>needed?"}
    MULTI -->|Yes| SEC["Secondary Agents<br/>merge_responses()"]
    MULTI -->|No| RESP["AgentResponse"]
    SEC --> RESP
    RESP --> SAVE["Save to session<br/>Extract entities<br/>Update current_agent"]
    SAVE --> U2["👤 Response to User"]
```

### Routing Decision Table

| Agent | Primary Keywords | Confidence Threshold |
|-------|-----------------|---------------------|
| `agronomy_agent` | grow, plant, pest, disease, soil, seed, irrigation | 0.3–0.9 (score-based) |
| `commerce_agent` | price, sell, buy, mandi, market, rate, profit | 0.3–0.9 |
| `platform_agent` | register, login, app, account, order, payment | 0.3–0.9 |
| `web_scraping_agent` | live, current, today, real-time, fetch, scrape | 0.3–0.9 |
| `browser_agent` | login to, submit, navigate, download, dashboard | 0.3–0.9 |
| `research_agent` | research, investigate, comprehensive, compare | 0.3–0.9 |
| `buyer_matching_agent` | find buyer, match buyer, sell my produce | 0.85 (exact match) |
| `quality_assessment_agent` | quality check, grade, defect, shelf life | 0.84 (exact match) |
| `adcl_agent` | recommend, sow, what to grow, demand | 0.83 (exact match) |
| `crop_listing_agent` | list my crop, create listing, my listings | 0.83 (exact match) |
| `logistics_agent` | delivery, transport, route, vehicle, shipping | 0.82 (exact match) |
| `price_prediction_agent` | predict, forecast, trend, future price | 0.3–0.9 |
| `knowledge_agent` | explain, tell me about, information, what is | 0.3–0.9 |
| `general_agent` | hello, hi, thanks, help, who are you | 0.3–0.9 (fallback) |

---

## 2. Voice Pipeline Flow

The voice pipeline supports **10 Indian languages** and handles both STT and TTS with multi-turn conversation flows.

```mermaid
flowchart LR
    A["🎤 Farmer speaks<br/>(audio bytes)"] --> STT

    subgraph "Speech-to-Text"
        STT["MultiProviderSTT<br/>src/voice/stt.py"]
        STT --> W1["Faster Whisper<br/>(local, CPU)"]
        STT --> W2["Groq Whisper<br/>(cloud fallback)"]
        STT --> W3["IndicWhisper<br/>(Indian languages)"]
    end

    STT --> TR["TranscriptionResult<br/>text + language + confidence"]
    TR --> EE["VoiceEntityExtractor<br/>src/voice/entity_extractor/"]

    EE --> EX["ExtractionResult<br/>intent (VoiceIntent enum)<br/>entities (crop, quantity, price, location)"]

    EX --> VA["VoiceAgent._generate_response()<br/>src/agents/voice_agent.py"]

    VA --> MT{"Multi-turn<br/>flow?"}
    MT -->|Yes| PEND["Check pending_intent<br/>Collect missing fields<br/>(crop → quantity → price)"]
    MT -->|No| HANDLER["Intent Handler<br/>(12 handlers)"]

    PEND --> ASK["Ask follow-up question<br/>in user's language"]
    HANDLER --> RESP["Response text<br/>(template-based or LLM)"]

    RESP --> TTS

    subgraph "Text-to-Speech"
        TTS["TTS Provider"]
        TTS --> E1["Edge-TTS<br/>(10 languages)"]
        TTS --> E2["IndicTTS<br/>(fallback)"]
    end

    TTS --> OUT["🔊 Audio response<br/>to farmer"]
```

### Supported Languages

| Code | Language | STT Provider | TTS Voice |
|------|----------|-------------|-----------|
| `kn` | Kannada | IndicWhisper | Edge-TTS `kn-IN-SapnaNeural` |
| `hi` | Hindi | Whisper/Groq | Edge-TTS `hi-IN-SwaraNeural` |
| `en` | English | Whisper/Groq | Edge-TTS `en-IN-NeerjaNeural` |
| `ta` | Tamil | IndicWhisper | Edge-TTS `ta-IN-PallaviNeural` |
| `te` | Telugu | IndicWhisper | Edge-TTS `te-IN-ShrutiNeural` |
| `mr` | Marathi | IndicWhisper | Edge-TTS `mr-IN-AarohiNeural` |
| `bn` | Bengali | IndicWhisper | Edge-TTS `bn-IN-TanishaaNeural` |
| `gu` | Gujarati | IndicWhisper | Edge-TTS `gu-IN-DhwaniNeural` |
| `pa` | Punjabi | IndicWhisper | Edge-TTS `pa-IN-VaaniNeural` |
| `ml` | Malayalam | IndicWhisper | Edge-TTS `ml-IN-SobhanaNeural` |

### Voice Intents

| Intent | Required Fields | Multi-Turn |
|--------|----------------|-----------|
| `CREATE_LISTING` | crop, quantity, asking_price | ✅ Yes |
| `CHECK_PRICE` | crop | No |
| `TRACK_ORDER` | order_id (optional) | No |
| `MY_LISTINGS` | — | No |
| `FIND_BUYER` | commodity, quantity_kg | ✅ Yes |
| `REGISTER` | name, phone, district | ✅ Yes |
| `CHECK_WEATHER` | location | No |
| `GET_ADVISORY` | crop | No |
| `QUALITY_CHECK` | commodity | No |
| `WEEKLY_DEMAND` | location | No |
| `DISPUTE_STATUS` | dispute_id | No |
| `GREETING` | — | No |
| `HELP` | — | No |

---

## 3. Price Discovery Flow

```mermaid
flowchart TD
    Q["👤 'What is the price of tomato in Mysore?'"] --> SUP["SupervisorAgent routes to<br/>commerce_agent (keyword: price, mandi)"]
    SUP --> CA["CommerceAgent.process()"]
    CA --> RAG["Retrieve from KnowledgeBase<br/>(Qdrant: agri_knowledge collection)"]
    CA --> TOOL["Use tool: agmarknet<br/>Scrape live APMC prices"]

    RAG --> CTX["Context: historical price data<br/>+ farming guides"]
    TOOL --> LIVE["Live data: today's mandi prices<br/>from Agmarknet/eNAM"]

    CTX & LIVE --> LLM["LLM generates response<br/>(Groq Llama-3.3-70B or Bedrock Claude)"]
    LLM --> RESP["💬 'Tomato price in Mysore mandi<br/>today is ₹25/kg.<br/>Recommendation: Hold for 2 days.'"]
```

---

## 4. Crop Listing Flow

```mermaid
flowchart TD
    F["🧑‍🌾 Farmer: 'I have 100 kg tomato at ₹25/kg'"] --> VA["VoiceAgent / Chat API"]
    VA --> EE["Entity Extraction<br/>crop=tomato, quantity=100, price=25"]
    EE --> CL["CropListingAgent.process()"]

    CL --> DB["Create listing in PostgreSQL<br/>src/db/postgres_client.py"]
    DB --> LISTING["Listing {id, farmer_id, commodity,<br/>quantity_kg, asking_price}"]

    LISTING --> BM["BuyerMatchingAgent<br/>GPS clustering + preference matrix"]
    BM --> MATCH["Matched buyers<br/>(grade-fit, price-fit, distance)"]

    MATCH --> NOTIFY["Buyer notifications<br/>workflows/buyer-notification.json"]
    NOTIFY --> BUYER["📩 Buyer receives alert"]

    LISTING --> RESP["✅ 'Listing created.<br/>ID: LST-001. 3 buyers matched.'"]
    RESP --> F
```

---

## 5. RAG Pipeline Flow

The RAG system uses 21 modules for production-grade retrieval.

```mermaid
flowchart TD
    Q["User Query"] --> QP["QueryProcessor<br/>src/rag/query_processor.py"]

    QP --> HYDE["HyDE<br/>(Hypothetical<br/>Document)"]
    QP --> MQ["Multi-Query<br/>(3 perspectives)"]
    QP --> SB["Step-Back<br/>(abstract concepts)"]
    QP --> DEC["Decomposition<br/>(sub-questions)"]

    HYDE & MQ & SB & DEC --> HS["HybridSearch<br/>src/rag/hybrid_search.py"]

    HS --> BM25["BM25 Sparse Search"]
    HS --> DENSE["Dense Vector Search<br/>(BGE-M3 / MiniLM embeddings)"]
    BM25 & DENSE --> RRF["Reciprocal Rank Fusion<br/>(combine sparse + dense)"]

    RRF --> GR["GraphRetriever<br/>src/rag/graph_retriever.py<br/>(Neo4j entity relationships)"]

    GR --> RR["Reranker<br/>src/rag/reranker.py<br/>(Cross-Encoder)"]

    RR --> GRADE["Grader<br/>src/rag/grader.py<br/>(relevance scoring)"]

    GRADE --> LLM["LLM Generation<br/>(Groq / Bedrock)"]

    LLM --> OBS["Observability<br/>src/rag/observability.py<br/>(LangSmith traces)"]

    OBS --> RESP["📋 Grounded Response<br/>+ Sources + Confidence"]

    subgraph "Indexing (Offline)"
        DOC["Documents"] --> CHUNK["ContextualChunker<br/>src/rag/contextual_chunker.py"]
        CHUNK --> RAP["RAPTOR<br/>src/rag/raptor.py<br/>(GMM hierarchical tree)"]
        RAP --> EMB["Embeddings<br/>src/rag/embeddings.py<br/>(BGE-M3)"]
        EMB --> QDB[("Qdrant<br/>agri_knowledge<br/>collection")]
    end
```

---

## 6. Session & Memory Flow

```mermaid
flowchart TD
    Q["User Query"] --> SM["AgentStateManager<br/>src/memory/state_manager.py"]

    SM --> GET{"Session exists?"}
    GET -->|Yes| LOAD["Load ConversationContext<br/>from Redis / in-memory"]
    GET -->|No| CREATE["Create new session<br/>(UUID, 24h TTL)"]

    LOAD & CREATE --> MSG["Add user message to history<br/>(windowed: max 50)"]
    MSG --> EXT["Extract entities<br/>(commodity, quantity, district, price)<br/>Regex-based, no LLM call"]
    EXT --> CTX["Build context dict:<br/>• user_profile<br/>• entities<br/>• current_agent<br/>• conversation_summary"]
    CTX --> AGENT["Agent processes with context"]
    AGENT --> SAVE["Save assistant message<br/>Extract entities from response<br/>Update current_agent"]
    SAVE --> REDIS[("Redis / In-Memory<br/>session:{uuid}")]
```

### Entity Extraction Patterns

| Entity | Regex Pattern | Example Match |
|--------|--------------|---------------|
| `commodity` | tomato, potato, onion, carrot, okra... (+ Hindi/Kannada) | "tamatar" → "Tomato" |
| `quantity_kg` | `\d+\.?\d* (kg|kilo)` | "100 kg" → 100.0 |
| `quantity_quintal` | `\d+\.?\d* (quintal|q)` | "2 quintal" → 200 kg |
| `district` | Kolar, Mysuru, Belagavi, Bangalore... | "mysore" → "Mysore" |
| `price_per_kg` | `₹\d+\.?\d*/kg` | "₹25/kg" → 25.0 |

---

## 7. WebSocket Voice Streaming Flow

```mermaid
sequenceDiagram
    participant Client as 📱 Client (WebRTC)
    participant WS as WebSocket Handler<br/>/ws/voice/{user_id}
    participant VAD as Silero VAD
    participant STT as STT Provider
    participant Agent as SupervisorAgent
    participant TTS as TTS Provider

    Client->>WS: Connect (user_id, language, session_id)
    WS->>WS: Accept + create session

    loop Real-time streaming
        Client->>WS: Audio frame (binary)
        WS->>VAD: Voice Activity Detection
        VAD-->>WS: speech_start / speech_end

        alt Speech complete
            WS->>STT: Transcribe audio buffer
            STT-->>WS: TranscriptionResult
            WS->>Agent: process_with_session(text, session_id)
            Agent-->>WS: AgentResponse
            WS->>TTS: Synthesize response
            TTS-->>WS: Audio bytes
            WS->>Client: Audio response (binary)
            WS->>Client: JSON metadata (transcript, intent, entities)
        end
    end

    Client->>WS: Disconnect
    WS->>WS: Cleanup session
```

---

## Related Documentation

| Document | Path |
|----------|------|
| System Architecture | [`docs/architecture/system-architecture.md`](system-architecture.md) |
| Agent Registry | [`docs/agents/REGISTRY.md`](../agents/REGISTRY.md) |
| Voice Pipeline | [`docs/features/voice-pipeline.md`](../features/voice-pipeline.md) |
| RAG Pipeline | [`docs/features/rag-pipeline.md`](../features/rag-pipeline.md) |
