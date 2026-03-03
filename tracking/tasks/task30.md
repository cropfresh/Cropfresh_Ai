# Task 30: End-to-End Verification + Dynamic Health Check

> **Priority:** 🟠 P1 | **Phase:** Voice Fix | **Effort:** 2 hours
> **Files:** `src/api/rest/voice.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

The `/api/v1/voice/health` endpoint returns a static hardcoded response. It should dynamically test each component (STT, TTS, VAD) and report actual availability. This is the final verification step after Tasks 21–29.

---

## 🏗️ Implementation Spec

### Update `/health` in `src/api/rest/voice.py`

```python
@router.get("/health")
async def voice_health():
    """Dynamic health check — tests each voice component."""
    stt = get_stt()
    tts = get_tts()

    stt_providers = stt.get_available_providers()
    tts_provider = tts.__class__.__name__
    languages = stt.get_supported_languages()

    # Check VAD
    try:
        from src.voice.vad import SileroVAD
        vad_ok = SileroVAD()._initialized  # True if pre-downloaded at startup
    except Exception:
        vad_ok = False

    return {
        "status": "healthy" if stt_providers else "degraded",
        "stt_providers": stt_providers,
        "tts_provider": tts_provider,
        "vad_available": vad_ok,
        "languages": languages[:5],
        "version": "0.9.2",
    }
```

### End-to-End Test Checklist

```
1. GET  /api/v1/voice/health
   → status: healthy, stt_providers: [...], tts_provider: EdgeTTSProvider

2. GET  /api/v1/voice/languages
   → stt_languages: [list], tts_languages: [list]

3. POST /api/v1/voice/synthesize
   body: {"text": "ನಮಸ್ಕಾರ", "language": "kn"}
   → audio_base64: <non-empty>, format: "mp3"

4. POST /api/v1/voice/process  (with test WAV file)
   → transcription: <non-empty>, response_audio: <non-empty>

5. WS   /api/v1/voice/ws
   → connect → stream 1s of silence + speech PCM
   → receive: {type: "transcription", text: ...}
   → receive: {type: "response_audio", audio_base64: ...}
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                                 | Weight |
| --- | ------------------------------------------------------------------------- | ------ |
| 1   | `GET /api/v1/voice/health` → `status: healthy`, `stt_providers` non-empty | 20%    |
| 2   | `GET /api/v1/voice/languages` → HTTP 200, non-empty language lists        | 15%    |
| 3   | `POST /api/v1/voice/synthesize` → audio base64 in response                | 20%    |
| 4   | `POST /api/v1/voice/process` → transcription + response_audio             | 25%    |
| 5   | Frontend Tools Inspector shows all green ✅ status chips                  | 10%    |
| 6   | WebSocket connects and returns audio response                             | 10%    |
