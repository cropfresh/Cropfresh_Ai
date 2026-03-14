"""
Groq Whisper cloud STT fallback provider.

Uses ``whisper-large-v3-turbo`` via Groq API — near-zero cost, ~300ms latency.
"""

import asyncio
import os
import tempfile
import wave

from loguru import logger

from src.voice.stt_pkg.models import TranscriptionResult


class GroqWhisperSTT:
    """
    Groq Whisper cloud STT fallback.

    Activated automatically by MultiProviderSTT when GROQ_API_KEY is set.
    Raises ValueError if key is absent (caller should catch and skip).
    """

    MODEL = "whisper-large-v3-turbo"

    def __init__(self) -> None:
        from groq import Groq  # type: ignore[import]

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set — GroqWhisperSTT unavailable")
        self._client = Groq(api_key=api_key)
        logger.info("GroqWhisperSTT initialized (model={})", self.MODEL)

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe audio bytes via Groq Whisper API."""
        from src.voice.audio_utils import AudioFormat, AudioProcessor

        fmt = AudioProcessor().detect_format(audio_data)
        ext = ".wav"
        if fmt == AudioFormat.WEBM:
            ext = ".webm"
        elif fmt == AudioFormat.MP3:
            ext = ".mp3"
        elif fmt == AudioFormat.OGG:
            ext = ".ogg"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            wav_path = tmp.name
            if fmt != AudioFormat.RAW:
                tmp.write(audio_data)
            else:
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_sync, wav_path, language,
            )
            return result
        finally:
            os.unlink(wav_path)

    def _transcribe_sync(self, wav_path: str, language: str) -> TranscriptionResult:
        """Blocking Groq API call — run via run_in_executor."""
        with open(wav_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                file=(wav_path, f.read()),
                model=self.MODEL,
                language=language if language not in ("auto", "") else None,
                response_format="verbose_json",
            )

        text = getattr(response, "text", "") or ""
        detected_lang_str = getattr(response, "language", "").lower()

        lang_map = {
            "english": "en", "kannada": "kn", "telugu": "te",
            "hindi": "hi", "tamil": "ta", "malayalam": "ml",
            "marathi": "mr", "gujarati": "gu", "bengali": "bn",
        }

        final_lang = language if language != "auto" else "en"
        if language == "auto" and detected_lang_str:
            final_lang = lang_map.get(detected_lang_str, "en")

        logger.info("GroqWhisperSTT transcribed {} chars (detected: {})", len(text), final_lang)
        return TranscriptionResult(
            text=text,
            language=final_lang,
            confidence=0.9,
            duration_seconds=0.0,
            provider="groq_whisper",
        )

    def get_supported_languages(self) -> list[str]:
        """Groq Whisper supports the same languages as OpenAI Whisper."""
        return [
            "en", "hi", "kn", "te", "ta", "ml", "mr", "gu", "pa", "bn",
            "ur", "ne", "si", "or", "as",
        ]
