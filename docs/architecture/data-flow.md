# CropFresh AI - Data Flow Diagrams

> **Last Updated:** 2026-03-17
> Diagrams reflect the intended current runtime story, with the duplex websocket as the canonical voice path.

---

## 1. Agent Routing Flow

```mermaid
flowchart TD
    U["User request<br/>(chat, REST voice, or duplex voice)"] --> API["FastAPI entrypoint<br/>/api/v1/chat<br/>/api/v1/voice/process<br/>/api/v1/voice/ws/duplex"]
    API --> SM["Session and app state"]
    SM --> SUP["SupervisorAgent"]
    SUP --> DECIDE{"Route"}
    DECIDE --> AGRI["Agronomy / advisory"]
    DECIDE --> COM["Commerce / price / listings"]
    DECIDE --> ADCL["ADCL / weekly demand"]
    DECIDE --> GEN["General fallback"]
    AGRI --> TOOLS["Tools and data services"]
    COM --> TOOLS
    ADCL --> TOOLS
    GEN --> TOOLS
    TOOLS --> RESP["Structured response"]
    RESP --> U2["User-visible text or audio"]
```

Key routing notes:

- Chat and voice both end up in the same supervisor and domain-agent ecosystem.
- Voice-specific extraction happens before the supervisor sees the resolved text.
- Redis-backed session state supports multi-turn continuity.

---

## 2. Duplex Voice Flow

```mermaid
sequenceDiagram
    participant Client as Voice client
    participant WS as /api/v1/voice/ws/duplex
    participant VAD as Silero VAD
    participant STT as STT provider
    participant Agent as VoiceAgent + supervisor
    participant LLM as LLM provider layer
    participant TTS as Streaming TTS

    Client->>WS: connect(user_id, language)
    WS-->>Client: ready

    loop audio chunks
        Client->>WS: {"type":"audio_chunk","audio_base64":"..."}
        WS->>VAD: process_chunk()
        alt speech end
            WS->>STT: transcribe(buffer)
            STT-->>WS: text + language
            WS-->>Client: pipeline_state
            WS->>Agent: process text
            Agent->>LLM: generate response
            LLM-->>TTS: response text
            TTS-->>WS: response_audio chunks
            WS-->>Client: response_sentence
            WS-->>Client: response_audio
            WS-->>Client: response_end
        end
    end

    Client->>WS: {"type":"bargein"}
    WS-->>Client: bargein
```

Current transport notes:

- JSON text frames only
- base64 audio payloads in both directions
- stage-level latency metadata still needs Sprint 07 instrumentation

---

## 3. One-Shot Voice REST Flow

```mermaid
flowchart LR
    A["Audio file"] --> REST["POST /api/v1/voice/process"]
    REST --> STT["STT runtime"]
    STT --> VA["VoiceAgent"]
    VA --> LLM["LLM provider layer"]
    LLM --> TTS["TTS runtime"]
    TTS --> RESP["VoiceProcessResponse<br/>text + audio_base64"]
```

This path is useful for controlled testing, server-side integrations, and regression checks against the websocket flow.

---

## 4. Price Discovery Flow

```mermaid
flowchart TD
    Q["User asks for mandi price"] --> SUP["Supervisor routes to commerce agent"]
    SUP --> TOOL["Shared rate hub"]
    SUP --> KB["Knowledge / RAG context"]
    TOOL --> LIVE["Official-first source fan-out"]
    KB --> CONTEXT["Historical and explanatory context"]
    LIVE --> MERGE["Evidence merge"]
    CONTEXT --> MERGE
    MERGE --> LLM["Groq / vLLM / Together"]
    LLM --> RESP["Grounded answer with recommendation"]
```

Provider policy note: Bedrock should no longer be described as the intended active provider in this flow.

---

## 5. Session and Memory Flow

```mermaid
flowchart TD
    REQ["Incoming request"] --> STATE["State manager / app state"]
    STATE --> EXISTS{"Session exists?"}
    EXISTS -->|Yes| LOAD["Load context"]
    EXISTS -->|No| CREATE["Create session"]
    LOAD --> CTX["Conversation context"]
    CREATE --> CTX
    CTX --> EXTRACT["Entity extraction"]
    EXTRACT --> AGENT["Agent execution"]
    AGENT --> SAVE["Persist messages and entities"]
    SAVE --> OUT["Response returned"]
```

The same session concept is shared across chat, REST voice, and websocket voice flows, even though the transport contracts differ.

---

## 6. Voice Improvement Focus for Sprint 07

```mermaid
flowchart LR
    A["Current latency ~3-4s"] --> B["Add stage timings"]
    B --> C["Tune VAD and interruption"]
    C --> D["Start TTS earlier on safe partial text"]
    D --> E["Benchmark local-language quality"]
    E --> F["Ship duplex-first production contract"]
```

Target framing:

- 20ms is an audio-frame budget, not the full response SLA
- sub-second first audio is the practical near-term goal
- local-language naturalness is as important as raw latency

---

## Related Docs

| Document | Path |
|----------|------|
| System architecture | `docs/architecture/system-architecture.md` |
| Voice pipeline | `docs/features/voice-pipeline.md` |
| Websocket protocol | `docs/api/websocket-voice.md` |
