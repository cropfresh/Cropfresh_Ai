# Task 24: Add `EdgeTTSProvider` Class to `tts.py`

> **Priority:** 🔴 P0 | **Phase:** Voice Fix | **Effort:** 2 hours
> **Files:** `src/voice/tts.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

`IndicTTS` attempts to load `ai4bharat/indic-parler-tts` from HuggingFace on every initialization — a 600MB+ model not cached on dev. TTS always fails. `edge-tts` is already installed and provides free, high-quality TTS in 11 Indian languages with no model download required.

---

## 🏗️ Implementation Spec

### New class in `src/voice/tts.py`

```python
class EdgeTTSProvider:
    """
    Edge TTS provider — practical TTS using Microsoft Edge TTS.
    No model download required. Supports 11 Indian languages.
    Acts as a drop-in replacement for IndicTTS interface.
    """

    EDGE_VOICES = {
        "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
        "kn": {"male": "kn-IN-GaganNeural", "female": "kn-IN-SapnaNeural"},
        "te": {"male": "te-IN-MohanNeural", "female": "te-IN-ShrutiNeural"},
        "ta": {"male": "ta-IN-ValluvarNeural", "female": "ta-IN-PallaviNeural"},
        "ml": {"male": "ml-IN-MidhunNeural", "female": "ml-IN-SobhanaNeural"},
        "mr": {"male": "mr-IN-ManoharNeural", "female": "mr-IN-AarohiNeural"},
        "gu": {"male": "gu-IN-NiranjanNeural", "female": "gu-IN-DhwaniNeural"},
        "bn": {"male": "bn-IN-BashkarNeural", "female": "bn-IN-TanishaaNeural"},
        "en": {"male": "en-IN-PrabhatNeural", "female": "en-IN-NeerjaNeural"},
    }

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        gender: str = "female",
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize text using Edge TTS, returns SynthesisResult."""
        import edge_tts
        import io

        voices = self.EDGE_VOICES.get(language, self.EDGE_VOICES["en"])
        voice = voices.get(gender, voices["female"])

        communicate = edge_tts.Communicate(text, voice)
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        return SynthesisResult(
            audio=audio_bytes,
            sample_rate=24000,
            format="mp3",
            duration_seconds=len(audio_bytes) / (24000 * 2),
            language=language,
            voice=voice,
        )
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                              | Weight |
| --- | ---------------------------------------------------------------------- | ------ |
| 1   | `EdgeTTSProvider` class importable from `src/voice/tts.py`             | 20%    |
| 2   | `synthesize("Hello", "en")` returns `SynthesisResult` with audio bytes | 40%    |
| 3   | `synthesize("नमस्ते", "hi")` returns non-empty audio bytes             | 20%    |
| 4   | `synthesize("ನಮಸ್ಕಾರ", "kn")` returns non-empty audio bytes            | 20%    |
