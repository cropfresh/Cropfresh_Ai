# CropFresh AI - Voice Pipeline

> **Last Updated:** 2026-03-17
> **Primary Sources:** `src/api/websocket/voice_pkg/`, `src/voice/`, `src/agents/voice_agent.py`

---

## Overview

The current voice stack is no longer a Pipecat-first design. The production-facing path is the duplex websocket flow built around:

- `/api/v1/voice/ws/duplex` for realtime voice
- `/api/v1/voice/ws` as a compatibility session flow
- `/api/v1/voice/process` for one-shot REST voice in -> voice out
- `VoiceAgent` for multi-turn intent handling and response generation
- Hybrid STT plus TTS providers behind the voice runtime

Pipecat still exists in the repo, but it is experimental and should not be described as the primary production path.

---

## Current Voice Stack

```mermaid
flowchart LR
    MIC["Audio input"] --> WS["Duplex websocket"]
    WS --> VAD["Silero VAD"]
    VAD --> STT["Hybrid STT"]
    STT --> AGENT["VoiceAgent"]
    AGENT --> LLM["LLM provider layer"]
    LLM --> TTS["Edge or local Indic TTS"]
    TTS --> OUT["Audio output"]
```

### Transport Layer

| Path | Role | Status |
|------|------|--------|
| `/api/v1/voice/ws/duplex` | Canonical realtime duplex path | Active |
| `/api/v1/voice/ws` | Compatibility session path | Active |
| `/api/v1/voice/process` | One-shot REST pipeline | Active |
| Pipecat / WebRTC slices | Experimental alternate path | Not production-default |

### Core Components

| Component | Current Use |
|-----------|-------------|
| Silero VAD | Speech start/end detection and interruption handling |
| `MultiProviderSTT` | Used by the session-oriented websocket and REST flows |
| Duplex STT path | Currently initialized with Groq in the duplex websocket |
| `VoiceAgent` | Intent detection, entity extraction, and multi-turn business logic |
| Edge TTS | Active default in duplex initialization |
| Local Indic TTS | Available candidate path for better local-language naturalness |

---

## STT Strategy

The repo currently mixes local and hosted STT approaches.

| Provider | Current Role | Notes |
|----------|--------------|-------|
| Faster Whisper | Local CPU/GPU transcription | Good fallback and offline option |
| Groq Whisper | Low-latency hosted transcription | Currently used by the duplex path |
| Indic models | Better regional-language accuracy | Needs tighter benchmarking and selection rules |

Sprint 07 should keep the hybrid approach, but make provider selection explicit instead of implicit.

---

## TTS Strategy

| Provider | Current Role | Notes |
|----------|--------------|-------|
| Edge TTS | Safe default and current duplex default | Reliable but can sound robotic in local languages |
| Local Indic TTS | Higher naturalness candidate | Needs warm-worker and latency validation |

### Language Quality Focus

The next voice sprint should benchmark at least:

- `kn` (Kannada)
- `hi` (Hindi)
- `te` (Telugu)
- `ta` (Tamil)

The goal is not just correctness. The evaluation needs to score naturalness, intelligibility, and barge-in recovery so the assistant sounds less robotic.

---

## Latency Reality

As of 2026-03-17, voice is still roughly `3-4s` end to end for the current stack. That means:

- "20ms" is realistic only as an audio-frame or playback budget
- first-audio timing is not yet instrumented end to end
- sentence buffering and provider warm-up still add noticeable delay

### Sprint 07 Target Framing

| Metric | Target |
|--------|--------|
| Audio chunk cadence | 20ms budget |
| Speech end -> first audio | < 800ms P50, < 1200ms P95 |
| Full short spoken reply | < 2.0s P95 |
| Barge-in cancel reaction | < 150ms |

These are production-grade responsiveness goals, not a claim that the entire STT -> LLM -> TTS loop can complete in 20ms.

---

## Known Gaps

- Duplex websocket still uses JSON plus base64 audio instead of binary transport.
- Stage-level timing is not yet exposed in websocket events or the live test UI.
- The duplex route hardcodes provider choices at startup.
- Pipecat tests are currently failing in focused runs and need cleanup before any stronger claims are made.
- Bedrock references still exist in code, but Bedrock is no longer the intended provider direction.

---

## Sprint 07 Direction

1. Keep `/api/v1/voice/ws/duplex` as the canonical production contract.
2. Add stage-level timing and Prometheus-visible metrics.
3. Remove Bedrock from the intended provider policy and fallback order.
4. Benchmark local-language TTS and STT on a fixed utterance set.
5. Consolidate live testing around the supported static pages.

Migration note: Bedrock references still exist in code as of March 17, 2026, but Sprint 07 removes them from runtime policy, docs, and fallbacks.
