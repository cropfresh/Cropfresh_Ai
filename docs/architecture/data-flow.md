# CropFresh AI — Data Flow Diagrams

> **Last Updated:** 2026-03-14
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

## 5. Advanced Agentic RAG Pipeline (ADR-010)

The RAG system uses a **LangGraph state machine** to orchestrate a self-correcting, anti-hallucination pipeline with 8 nodes and conditional routing.

### 5.1 High-Level Flow

```mermaid
flowchart TD
    Q["👤 User Query"] --> DECOMP["QueryDecomposer<br/>ai/rag/retrieval/query_decomposer.py<br/>(split multi-part Qs)"]

    DECOMP --> REWRITE["QueryRewriter<br/>ai/rag/query_rewriter.py"]

    subgraph "Query Enhancement"
        REWRITE --> HYDE["HyDE<br/>(hypothetical doc)"]
        REWRITE --> SB["Step-Back<br/>(abstract query)"]
        REWRITE --> MQ["Multi-Query<br/>(3 perspectives)"]
    end

    HYDE & SB & MQ --> RETRIEVE

    subgraph "Advanced Retrieval"
        RETRIEVE["AdvancedRetriever<br/>ai/rag/retrieval/advanced_retriever.py"]
        RETRIEVE --> CTX_ENRICH["ContextualEnricher<br/>(Anthropic-style<br/>chunk context)"]
        CTX_ENRICH --> HS["HybridSearch<br/>BM25 + Dense + RRF"]
        HS --> TIME["TimeAwareRetriever<br/>(freshness decay)"]
    end

    TIME --> GRADE["DocumentGrader<br/>ai/rag/grader.py<br/>(continuous 0-1 scoring)"]

    GRADE -->|"relevant docs ≥ 0.5"| GEN["SpeculativeDraftEngine<br/>ai/rag/agentic/speculative.py<br/>(parallel drafts)"]
    GRADE -->|"no relevant docs"| WEB["WebSearch Fallback<br/>ai/rag/graph/nodes_safety.py"]
    WEB --> GRADE

    GEN --> CITE["CitationEngine<br/>ai/rag/citation_engine.py<br/>(inline [1],[2] refs)"]

    CITE --> EVAL["SelfEvaluator<br/>ai/rag/agentic/evaluator.py<br/>(faithfulness × relevance)"]

    EVAL -->|"confidence ≥ 0.75"| GATE["ConfidenceGate<br/>ai/rag/confidence_gate.py<br/>(safety + Kannada)"]
    EVAL -->|"confidence < 0.75<br/>retry ≤ 2"| REWRITE

    GATE --> RESP["📋 Grounded Answer<br/>+ Citations [1],[2]<br/>+ Confidence score"]
    GATE -->|"unsafe query"| DECLINE["🚫 Decline response<br/>(medical/legal/financial)"]

    style DECOMP fill:#4CAF50,color:white
    style GRADE fill:#FF9800,color:white
    style EVAL fill:#2196F3,color:white
    style GATE fill:#f44336,color:white
```

### 5.2 LangGraph State Machine Nodes

| Node | File | Purpose |
|------|------|---------|
| **rewrite** | `ai/rag/graph/nodes.py` | HyDE / step-back / multi-query expansion |
| **retrieve** | `ai/rag/graph/nodes.py` | Qdrant hybrid search with deduplication |
| **grade** | `ai/rag/graph/nodes.py` | Continuous 0–1 scoring + time-decay penalty |
| **generate** | `ai/rag/graph/nodes.py` | Speculative parallel drafts + best selection |
| **cite** | `ai/rag/graph/nodes.py` | Inline citation insertion |
| **evaluate** | `ai/rag/graph/nodes_safety.py` | RAGAS-style faithfulness × relevance gate |
| **gate** | `ai/rag/graph/nodes_safety.py` | Safety classifier (medical/legal/financial) |
| **web_search** | `ai/rag/graph/nodes_safety.py` | Fallback when grading finds no relevant docs |

### 5.3 Conditional Routing Logic

```mermaid
stateDiagram-v2
    [*] --> rewrite
    rewrite --> retrieve
    retrieve --> grade

    grade --> generate : relevant_ratio ≥ 0.5
    grade --> web_search : relevant_ratio < 0.5

    web_search --> grade : retry

    generate --> cite
    cite --> evaluate

    evaluate --> gate : confidence ≥ 0.75
    evaluate --> rewrite : confidence < 0.75 AND retry_count ≤ 2

    gate --> [*]
```

### 5.4 Anti-Hallucination Techniques

| Technique | Module | Impact |
|-----------|--------|--------|
| Contextual Enrichment | `retrieval/contextual_enricher.py` | Retrieval failure ↓ 15% → 5% |
| Query Decomposition | `retrieval/query_decomposer.py` | Multi-part queries handled correctly |
| Time-Aware Scoring | `retrieval/time_aware.py` | Stale market data penalized (24h decay) |
| Document Grading | `grader.py` | Continuous 0–1 scoring filters weak docs |
| Self-Evaluation | `evaluator.py` | Low-confidence answers retried (max 2×) |
| Citation Engine | `citation_engine.py` | Every claim traced to source doc |
| Confidence Gate | `confidence_gate.py` | Unsafe queries declined in user's language |

### 5.5 Evaluation & CI Guardrail

```mermaid
flowchart LR
    GD["Golden Dataset<br/>30 queries × 6 categories"] --> EVAL["RAGEvaluator<br/>(RAGAS metrics)"]
    EVAL --> F["Faithfulness<br/>≥ 0.80"]
    EVAL --> R["Relevancy<br/>≥ 0.70"]
    EVAL --> P["Precision<br/>≥ 0.60"]
    EVAL --> H["Hallucination<br/>≤ 0.20"]
    F & R & P & H --> GATE{"CI Gate"}
    GATE -->|"All pass"| GREEN["✅ Pipeline passes"]
    GATE -->|"Any fail"| RED["❌ Pipeline blocked"]
```

### 5.6 Indexing Pipeline (Offline)

```mermaid
flowchart LR
    DOC["Documents"] --> CHUNK["ContextualChunker<br/>(section-aware splitting)"]
    CHUNK --> ENRICH["ContextualEnricher<br/>(prepend doc context)"]
    ENRICH --> RAP["RAPTOR<br/>(hierarchical tree)"]
    RAP --> EMB["BGE-M3 Embeddings"]
    EMB --> QDB[("Qdrant<br/>agri_knowledge")]
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
