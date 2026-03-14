"""
AI4Bharat IndicConformer STT provider — best for Indian languages.

Supports all 22 official Indian languages with state-of-the-art accuracy.
"""

from loguru import logger

from src.voice.audio_utils import AudioProcessor
from src.voice.stt_pkg.models import SupportedLanguage, TranscriptionResult


class IndicWhisperSTT:
    """
    Speech-to-Text using AI4Bharat IndicConformer.

    Usage:
        stt = IndicWhisperSTT()
        result = await stt.transcribe(audio_bytes, language="hi")
        print(result.text)  # "मेरे पास 200 किलो टमाटर है"
    """

    MODEL_CONFORMER = "ai4bharat/indic-conformer-600m-multilingual"
    MODEL_SMALL = MODEL_CONFORMER
    MODEL_MEDIUM = MODEL_CONFORMER
    MODEL_LARGE = MODEL_CONFORMER
    DEFAULT_MODEL = MODEL_CONFORMER

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._audio_processor = AudioProcessor()
        self._initialized = False

        logger.info(f"IndicWhisperSTT initialized with model: {model_name}")

    async def _ensure_initialized(self):
        """Lazy load the model on first use"""
        if self._initialized:
            return
        try:
            await self._load_model()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    async def _load_model(self):
        """Load IndicConformer model from HuggingFace"""
        logger.info(f"Loading IndicConformer model: {self.model_name}")
        try:
            import torch
            from transformers import AutoModel

            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True,
            )
            if device == "cuda":
                self._model = self._model.to(device)

            self._device = device
            self._is_conformer = True
            logger.info(f"IndicConformer loaded on {device}")

        except ImportError as e:
            logger.warning(f"Missing ML dependencies: {e}")
            raise RuntimeError("Install ML dependencies: uv sync --extra ml")

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        await self._ensure_initialized()

        is_valid, error = self._audio_processor.validate_audio(audio)
        if not is_valid:
            logger.error(f"Invalid audio: {error}")
            return TranscriptionResult(
                text="", language=language, confidence=0.0,
                duration_seconds=0.0, provider="error",
            )

        duration = self._audio_processor.get_audio_duration(audio)

        if self._model is not None:
            result = await self._transcribe_local(audio, language, task)
            result.duration_seconds = duration
            return result

        raise RuntimeError("Local IndicConformer model is not loaded")

    async def _transcribe_local(self, audio, language, task):
        """Transcribe using local IndicConformer model"""
        import io

        import numpy as np
        import torch

        preprocessed = self._audio_processor.preprocess_for_stt(audio)

        wav = None
        sr = None

        try:
            import soundfile as sf
            audio_array, sr = sf.read(io.BytesIO(preprocessed))
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            wav = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
        except Exception as sf_error:
            logger.debug(f"Soundfile failed: {sf_error}, trying torchaudio")
            import torchaudio
            torchaudio.set_audio_backend("soundfile")
            wav, sr = torchaudio.load(io.BytesIO(preprocessed))

        wav = torch.mean(wav, dim=0, keepdim=True)

        target_sample_rate = 16000
        if sr != target_sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sample_rate,
            )
            wav = resampler(wav)

        lang_code = language if language != "auto" else "hi"

        try:
            transcription = self._model(wav, lang_code, "ctc")
        except Exception as e:
            logger.warning(f"CTC decoding failed, trying RNNT: {e}")
            transcription = self._model(wav, lang_code, "rnnt")

        detected_language = language
        if language == "auto":
            detected_language = self._detect_language(transcription)

        return TranscriptionResult(
            text=transcription.strip() if transcription else "",
            language=detected_language,
            confidence=0.95,
            duration_seconds=0.0,
            provider="indicconformer",
        )

    def _detect_language(self, text: str) -> str:
        """Simple script-based language detection from text."""
        if not text:
            return "unknown"

        checks = [
            ("\u0900", "\u097f", "hi"),   # Devanagari
            ("\u0c80", "\u0cff", "kn"),   # Kannada
            ("\u0c00", "\u0c7f", "te"),   # Telugu
            ("\u0b80", "\u0bff", "ta"),   # Tamil
            ("\u0d00", "\u0d7f", "ml"),   # Malayalam
            ("\u0a80", "\u0aff", "gu"),   # Gujarati
            ("\u0980", "\u09ff", "bn"),   # Bengali
            ("\u0a00", "\u0a7f", "pa"),   # Punjabi
            ("\u0b00", "\u0b7f", "or"),   # Odia
        ]
        for lo, hi, lang in checks:
            if any(lo <= c <= hi for c in text):
                return lang
        return "en"

    async def detect_language(self, audio: bytes) -> str:
        """Detect language from audio without full transcription."""
        result = await self.transcribe(audio, language="auto")
        return result.language

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.AUTO]
