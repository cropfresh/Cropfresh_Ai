# Task 25: Wire `EdgeTTSProvider` as Default in REST Router + VoiceAgent

> **Priority:** 🔴 P0 | **Phase:** Voice Fix | **Effort:** 1.5 hours
> **Files:** `src/api/rest/voice.py`, `src/agents/voice_agent.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

Even after adding `EdgeTTSProvider` (Task 24), it won't be used unless wired into the REST API dependency injection and `VoiceAgent`. Currently both try `IndicTTS` which always fails.

---

## 🏗️ Implementation Spec

### In `src/api/rest/voice.py` — `get_tts()` dependency

```python
def get_tts():
    """Get TTS provider — EdgeTTS primary, IndicTTS on GPU."""
    try:
        tts = IndicTTS()
        tts._load_model()  # fast check
        return tts
    except Exception:
        logger.warning("IndicTTS unavailable, using EdgeTTSProvider")
        return EdgeTTSProvider()
```

### In `src/agents/voice_agent.py` — `__init__` type hint fix

```python
def __init__(
    self,
    stt: MultiProviderSTT | IndicWhisperSTT | None = None,  # accepts any STT
    tts: EdgeTTSProvider | IndicTTS | None = None,            # accepts any TTS
    ...
):
    self.stt = stt or MultiProviderSTT(use_faster_whisper=True, use_indicconformer=False)
    self.tts = tts or EdgeTTSProvider()
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                  | Weight |
| --- | ---------------------------------------------------------- | ------ |
| 1   | `VoiceAgent()` initializes without errors on CPU           | 30%    |
| 2   | `POST /api/v1/voice/synthesize` returns audio base64       | 40%    |
| 3   | `/health` endpoint reports `tts_provider: EdgeTTSProvider` | 30%    |
