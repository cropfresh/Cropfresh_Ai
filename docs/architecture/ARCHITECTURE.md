# System Architecture — CropFresh AI

## Overview
CropFresh AI is a multi-agent agricultural marketplace platform built on a modern async Python stack.

```
┌─────────────────────────────────────────────────────┐
│                    Clients                           │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐    │
│  │ Flutter  │  │ WhatsApp │  │ Web Dashboard  │    │
│  │   App    │  │   Bot    │  │   (Admin)      │    │
│  └────┬─────┘  └────┬─────┘  └───────┬────────┘    │
│       │              │                │              │
└───────┼──────────────┼────────────────┼──────────────┘
        │              │                │
        ▼              ▼                ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Gateway (src/api/)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │
│  │  REST    │  │ WebSocket│  │  Middleware   │      │
│  │ Routers  │  │ Handler  │  │ Auth/Rate/Log│      │
│  └────┬─────┘  └────┬─────┘  └──────────────┘      │
└───────┼──────────────┼──────────────────────────────┘
        │              │
        ▼              ▼
┌─────────────────────────────────────────────────────┐
│          LangGraph Multi-Agent System                │
│  ┌──────────────────────────────────────────────┐   │
│  │          Supervisor Agent                     │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │   │
│  │  │Agronomy│ │Pricing │ │Commerce│ │Voice  │ │   │
│  │  │ Agent  │ │ Agent  │ │ Agent  │ │Agent  │ │   │
│  │  └────────┘ └────────┘ └────────┘ └───────┘ │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │   │
│  │  │Browser │ │Research│ │Platform│ │General│ │   │
│  │  │ Agent  │ │ Agent  │ │ Agent  │ │Agent  │ │   │
│  │  └────────┘ └────────┘ └────────┘ └───────┘ │   │
│  └──────────────────────────────────────────────┘   │
└───────┼──────────────┼──────────────┼───────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│              Data Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │
│  │ Supabase │  │  Qdrant  │  │    Neo4j     │      │
│  │ Postgres │  │ Vectors  │  │   Graph DB   │      │
│  └──────────┘  └──────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────┘
```

## Key Components
- **API Layer** (`src/api/`): FastAPI with REST + WebSocket endpoints
- **Agent System** (`src/agents/`): LangGraph multi-agent with supervisor routing
- **RAG Pipeline** (`ai/rag/`): Qdrant-backed retrieval-augmented generation
- **Data Collection** (`src/scrapers/`): APMC, eNAM, weather data scrapers
- **Voice Processing** (`src/voice/`): Kannada STT/TTS with WebRTC
- **Shared Utilities** (`src/shared/`): Logging, resilience, memory, orchestration
