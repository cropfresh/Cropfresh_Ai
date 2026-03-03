# Task 27: Fix WebSocket Handler — Wire `MultiProviderSTT` + `EdgeTTS`

> **Priority:** 🟠 P1 | **Phase:** Voice Fix | **Effort:** 3 hours
> **Files:** `src/api/websocket/voice_ws.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

The WebSocket voice handler (`/api/v1/voice/ws`) has no wired STT/TTS. It accepts audio but doesn't transcribe it or send back audio responses. The VAD (Silero) is referenced but never initialized in the WS flow.

---

## 🏗️ Implementation Spec

### Session setup in WS handler

```python
async def handle_voice_session(websocket: WebSocket):
    stt = MultiProviderSTT(use_faster_whisper=True, use_indicconformer=False)
    tts = EdgeTTSProvider()
    vad = SileroVAD()

    try:
        await vad.initialize()  # downloads ONNX model if not cached
    except Exception as e:
        logger.warning(f"VAD unavailable: {e} — using manual chunking")
        vad = None
```

### After STT transcript received

```python
# Generate response via VoiceAgent
agent = VoiceAgent(stt=stt, tts=tts)
response_text = await agent.process_text(transcript, session_id=session_id)

# Synthesize audio response
synthesis = await tts.synthesize(response_text, language=detected_language)
audio_b64 = base64.b64encode(synthesis.audio).decode()

await websocket.send_json({
    "type": "response_audio",
    "audio_base64": audio_b64,
    "format": synthesis.format,
    "transcript": response_text,
})
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                    | Weight |
| --- | ------------------------------------------------------------ | ------ |
| 1   | WS connects at `/api/v1/voice/ws` without error              | 20%    |
| 2   | Sending PCM audio → `transcription` message returned         | 30%    |
| 3   | Sending PCM audio → `response_audio` base64 message returned | 30%    |
| 4   | Frontend WS tab shows transcript + plays audio               | 20%    |
