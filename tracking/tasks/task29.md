# Task 29: Implement `GroqWhisperSTT` Cloud Fallback

> **Priority:** 🟠 P1 | **Phase:** Voice Fix | **Effort:** 2 hours
> **Files:** `src/voice/stt.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

If `faster-whisper` fails (model download, memory), there is no cloud fallback for STT. Groq offers `whisper-large-v3-turbo` via their API at near-zero cost with very fast turnaround (~300ms). The `groq` SDK is already installed.

---

## 🏗️ Implementation Spec

### New class in `src/voice/stt.py`

```python
class GroqWhisperSTT:
    """
    Groq Whisper cloud STT fallback.
    Uses whisper-large-v3-turbo model via Groq API.
    Reads GROQ_API_KEY from environment.
    """

    MODEL = "whisper-large-v3-turbo"

    def __init__(self):
        import os
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set — GroqWhisperSTT unavailable")
        self._client = Groq(api_key=api_key)
        logger.info("GroqWhisperSTT initialized")

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes via Groq Whisper API.
        Writes to temp WAV, sends to API, returns result.
        """
        import tempfile
        import asyncio

        # Write to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            # Write WAV header + PCM data
            _write_wav(tmp, audio_data, sample_rate)

        try:
            # Run in thread (Groq client is sync)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_sync, wav_path, language
            )
            return result
        finally:
            import os
            os.unlink(wav_path)

    def _transcribe_sync(self, wav_path: str, language: str) -> TranscriptionResult:
        with open(wav_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                file=(wav_path, f.read()),
                model=self.MODEL,
                language=language if language != "auto" else None,
            )
        return TranscriptionResult(
            text=response.text,
            language=language,
            confidence=0.9,  # Groq doesn't return confidence scores
            provider="groq_whisper",
        )
```

### Register in `MultiProviderSTT` as 3rd provider

```python
# In _initialize_providers():
try:
    self._providers.append(GroqWhisperSTT())
    logger.info("GroqWhisperSTT registered as fallback")
except Exception as e:
    logger.warning(f"GroqWhisperSTT unavailable: {e}")
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                                         | Weight |
| --- | --------------------------------------------------------------------------------- | ------ |
| 1   | `GroqWhisperSTT` initializes when `GROQ_API_KEY` is set                           | 25%    |
| 2   | `transcribe(audio_bytes, "en")` returns `TranscriptionResult` with non-empty text | 40%    |
| 3   | `MultiProviderSTT.get_available_providers()` includes `groq_whisper`              | 20%    |
| 4   | Fallback activates if faster-whisper raises an exception                          | 15%    |
