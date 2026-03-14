"""
Faster Whisper STT provider — local, CPU-friendly, ~4x faster than original Whisper.
"""

import os
import tempfile

from loguru import logger

from src.voice.audio_utils import AudioProcessor
from src.voice.stt_pkg.models import TranscriptionResult


class FasterWhisperSTT:
    """
    High-performance Speech-to-Text using Faster Whisper.

    Features:
    - <1s latency for short audio
    - 99 language support
    - Automatic language detection
    - Low memory footprint (CTranslate2 backend)
    """

    MODEL_TINY = "tiny"
    MODEL_BASE = "base"
    MODEL_SMALL = "small"
    MODEL_MEDIUM = "medium"
    MODEL_LARGE = "large-v3"
    DEFAULT_MODEL = MODEL_SMALL

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        self._model = None
        self._audio_processor = AudioProcessor()
        self._initialized = False

        logger.info(f"FasterWhisperSTT initialized with model: {model_size}")

    async def _ensure_initialized(self):
        """Lazy load the model on first use"""
        if self._initialized:
            return
        try:
            await self._load_model()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper: {e}")
            raise

    async def _load_model(self):
        """Load Faster Whisper model"""
        logger.info(f"Loading Faster Whisper model: {self.model_size}")
        try:
            from faster_whisper import WhisperModel

            if self.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            if self.compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            else:
                compute_type = self.compute_type

            self._model = WhisperModel(
                self.model_size, device=device, compute_type=compute_type,
            )
            self._device = device
            logger.info(f"Faster Whisper loaded on {device} ({compute_type})")

        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio to text using Faster Whisper."""
        await self._ensure_initialized()

        if self._model is None:
            raise RuntimeError("Faster Whisper model not loaded")

        preprocessed = self._audio_processor.preprocess_for_stt(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(preprocessed)
            temp_path = f.name

        try:
            language_arg = None if language == "auto" else language
            segments, info = self._model.transcribe(
                temp_path, language=language_arg,
                beam_size=5, vad_filter=True,
            )

            text_parts = [segment.text for segment in segments]
            full_text = " ".join(text_parts).strip()
            detected_language = info.language if language == "auto" else language

            logger.info(
                f"Faster Whisper transcribed: {full_text[:50]}... "
                f"(lang: {detected_language})"
            )

            return TranscriptionResult(
                text=full_text,
                language=detected_language,
                confidence=info.language_probability if language == "auto" else 0.95,
                duration_seconds=info.duration,
                provider="faster-whisper",
            )
        finally:
            os.unlink(temp_path)
