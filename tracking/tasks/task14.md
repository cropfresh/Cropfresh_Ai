# Task 14: Complete Pipecat WebSocket Voice Streaming Pipeline

> **Priority:** 🟡 P2 | **Phase:** 4 | **Effort:** 3–4 days  
> **Files:** `src/voice/pipecat_bot.py`, `src/voice/webrtc_transport.py`, `src/voice/vad.py`  
> **Score Target:** 9/10 — Sub-2-second voice round-trip latency

---

## 📌 Problem Statement

The Pipecat voice pipeline exists but WebSocket streaming is untested on Windows. Need to build a fully working real-time voice pipeline with VAD, streaming STT/TTS, and interruption handling.

---

## 🔬 Research Findings (Pipecat 2025)

### Pipecat Architecture
```
Audio Input → VAD → STT (streaming) → LLM/Agent → TTS (streaming) → Audio Output
         ↕                                                      ↕
    WebSocket Transport (bi-directional audio frames)
```

### Key Pipecat Concepts
- **Frames**: Atomic units of data flowing through pipeline
- **Processors**: Modular handlers (STT, TTS, LLM, VAD)
- **Transport**: WebSocket for browser clients, WebRTC for mobile
- **Smart Turn Detection**: Beyond simple VAD — linguistic + acoustic cues
- **Interruption Handling**: User can interrupt bot mid-sentence

### Target Latency Budget
| Component | Budget | Current |
|-----------|--------|---------|
| STT | <500ms | ~800ms |
| Agent routing | <200ms | ~200ms |
| TTS | <500ms | ~700ms |
| Network + transport | <200ms | ~300ms |
| **Total P95** | **<1,500ms** | **~2,000ms** |

---

## 🏗️ Implementation Spec

### Pipecat Pipeline Setup
```python
class CropFreshVoiceBot:
    """
    Production Pipecat voice bot for CropFresh.
    
    Pipeline:
    1. WebSocket transport → receive audio frames
    2. Silero VAD → detect speech start/end
    3. IndicWhisper STT → transcribe (streaming)
    4. Entity Extraction → detect intent
    5. Agent Routing → process via appropriate agent
    6. Edge TTS → synthesize response (streaming)
    7. WebSocket transport → send audio frames
    """
    
    async def create_pipeline(self, websocket, language: str = "auto"):
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask
        
        transport = WebSocketTransport(websocket)
        vad = SileroVAD(threshold=0.5, min_speech_duration=0.3)
        stt = IndicWhisperSTTProcessor(language=language)
        agent = CropFreshAgentProcessor(self.voice_agent)
        tts = EdgeTTSProcessor(voice=self._get_voice(language))
        
        pipeline = Pipeline([
            transport.input(),
            vad,
            stt,
            agent,
            tts,
            transport.output(),
        ])
        
        task = PipelineTask(pipeline)
        await task.run()
```

### WebSocket Endpoint
```python
# src/api/websocket/voice_ws.py
@router.websocket("/ws/voice/{user_id}")
async def voice_websocket(websocket: WebSocket, user_id: str):
    await websocket.accept()
    bot = CropFreshVoiceBot(voice_agent=get_voice_agent())
    try:
        await bot.create_pipeline(websocket)
    except WebSocketDisconnect:
        logger.info(f"Voice session ended for user {user_id}")
```

### VAD Configuration (Silero)
```python
class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.
    
    Config:
    - threshold: 0.5 (sensitivity)
    - min_speech_duration: 300ms (avoid false triggers)
    - max_speech_duration: 30s (prevent infinite recording)
    - silence_duration: 800ms (end-of-utterance detection)
    """
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | WebSocket streaming works on Windows | 20% |
| 2 | Silero VAD detects speech start/end accurately | 20% |
| 3 | End-to-end voice round-trip < 2 seconds (P95) | 20% |
| 4 | Interruption handling (user cuts in mid-response) | 15% |
| 5 | Browser test page connects and streams audio | 15% |
| 6 | Graceful reconnection on WebSocket drop | 10% |
