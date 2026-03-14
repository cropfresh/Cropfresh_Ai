"""
Edge TTS Provider
=================
Provider for Microsoft Edge TTS fallback support.
"""

import asyncio
import io
from loguru import logger

from .models import SynthesisResult
from .utils import normalize_edge_rate


class EdgeTTSProvider:
    """
    Edge TTS provider — practical TTS using Microsoft Edge TTS.

    No model download required. Supports 11 Indian languages.
    Acts as a drop-in replacement for IndicTTS interface.
    """

    EDGE_VOICES = {
        "hi": {"male": "hi-IN-MadhurNeural",   "female": "hi-IN-SwaraNeural"},
        "kn": {"male": "kn-IN-GaganNeural",    "female": "kn-IN-SapnaNeural"},
        "te": {"male": "te-IN-MohanNeural",    "female": "te-IN-ShrutiNeural"},
        "ta": {"male": "ta-IN-ValluvarNeural", "female": "ta-IN-PallaviNeural"},
        "ml": {"male": "ml-IN-MidhunNeural",   "female": "ml-IN-SobhanaNeural"},
        "mr": {"male": "mr-IN-ManoharNeural",  "female": "mr-IN-AarohiNeural"},
        "gu": {"male": "gu-IN-NiranjanNeural", "female": "gu-IN-DhwaniNeural"},
        "bn": {"male": "bn-IN-BashkarNeural",  "female": "bn-IN-TanishaaNeural"},
        "en": {"male": "en-IN-PrabhatNeural",  "female": "en-IN-NeerjaNeural"},
    }

    SAMPLE_RATE = 24000

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        gender: str = "female",
        voice: str = "default",
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> SynthesisResult:
        import edge_tts

        voices = self.EDGE_VOICES.get(language, self.EDGE_VOICES["en"])
        if voice in ("default", "female"):
            resolved_voice = voices["female"]
        elif voice == "male":
            resolved_voice = voices["male"]
        elif voice in voices.values():
            resolved_voice = voice
        else:
            resolved_voice = voices["female"]

        rate_str = normalize_edge_rate(speed)

        if not text or not text.strip():
            logger.warning("EdgeTTS received empty text after trimming, skipping.")
            return SynthesisResult(
                audio=b"",
                sample_rate=self.SAMPLE_RATE,
                format="mp3",
                duration_seconds=0.0,
                language=language,
                voice=resolved_voice,
                provider="edge-tts",
            )

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, 4):
            try:
                communicate = edge_tts.Communicate(text, resolved_voice, rate=rate_str)
                audio_buffer = io.BytesIO()

                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_buffer.write(chunk["data"])

                audio_bytes = audio_buffer.getvalue()

                if len(audio_bytes) == 0:
                    raise RuntimeError("Edge TTS returned empty audio")

                duration_seconds = len(audio_bytes) / (self.SAMPLE_RATE * 2)

                logger.info(
                    f"EdgeTTS synthesized {len(text)} chars → {len(audio_bytes)} bytes "
                    f"(lang={language}, voice={resolved_voice}, attempt={attempt})"
                )

                return SynthesisResult(
                    audio=audio_bytes,
                    sample_rate=self.SAMPLE_RATE,
                    format="mp3",
                    duration_seconds=duration_seconds,
                    language=language,
                    voice=resolved_voice,
                    provider="edge-tts",
                )

            except (asyncio.CancelledError, ConnectionError, OSError, RuntimeError) as exc:
                last_exc = exc
                wait = attempt * 0.5
                logger.warning(
                    f"EdgeTTS attempt {attempt}/3 failed ({type(exc).__name__}: {exc}), "
                    f"retrying in {wait}s…"
                )
                await asyncio.sleep(wait)

        logger.error(f"EdgeTTS failed after 3 attempts: {last_exc}")
        raise RuntimeError(f"EdgeTTS synthesis failed after 3 attempts: {last_exc}")

    def get_supported_languages(self) -> list[str]:
        return list(self.EDGE_VOICES.keys())

    def get_available_voices(self, language: str) -> dict:
        return self.EDGE_VOICES.get(language, self.EDGE_VOICES["en"])
