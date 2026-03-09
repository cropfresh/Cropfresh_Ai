"""
Speech-to-Text (STT) Module for CropFresh Voice Agent

Uses AI4Bharat IndicWhisper for Indian language transcription.
Supports: Hindi, Kannada, Telugu, Tamil, + 10 more Indian languages.

Features:
- 20-50% better WER than OpenAI Whisper for Indian languages
- Automatic language detection
- Fallback to Groq Whisper API
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from .audio_utils import AudioProcessor


class SupportedLanguage(Enum):
    """Supported languages for STT"""

    HINDI = "hi"
    KANNADA = "kn"
    TELUGU = "te"
    TAMIL = "ta"
    MALAYALAM = "ml"
    MARATHI = "mr"
    GUJARATI = "gu"
    BENGALI = "bn"
    PUNJABI = "pa"
    ODIA = "or"
    ENGLISH = "en"
    AUTO = "auto"  # Auto-detect


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""

    text: str
    language: str
    confidence: float
    duration_seconds: float
    provider: str  # "faster-whisper", "indicconformer", or "groq"

    @property
    def is_successful(self) -> bool:
        return len(self.text) > 0 and self.confidence > 0.5


class FasterWhisperSTT:
    """
    High-performance Speech-to-Text using Faster Whisper.

    Faster Whisper provides ~4x faster transcription than original Whisper
    with CTranslate2 backend. Ideal for real-time applications.

    Features:
    - <1s latency for short audio
    - 99 language support
    - Automatic language detection
    - Low memory footprint

    Usage:
        stt = FasterWhisperSTT()
        result = await stt.transcribe(audio_bytes, language="hi")
    """

    # Model sizes (speed vs accuracy tradeoff)
    MODEL_TINY = "tiny"
    MODEL_BASE = "base"
    MODEL_SMALL = "small"
    MODEL_MEDIUM = "medium"
    MODEL_LARGE = "large-v3"

    # Default model (good balance)
    DEFAULT_MODEL = MODEL_SMALL

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """
        Initialize Faster Whisper STT.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: 'cuda', 'cpu', or 'auto'
            compute_type: 'float16', 'int8', or 'auto'
        """
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

            # Determine device and compute type
            if self.device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            if self.compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            else:
                compute_type = self.compute_type

            # Load model
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )

            self._device = device
            logger.info(f"Faster Whisper loaded on {device} ({compute_type})")

        except ImportError:
            logger.error(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper: {e}")
            raise

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """
        Transcribe audio to text using Faster Whisper.

        Args:
            audio: Audio bytes (WAV, MP3, etc.)
            language: Language code or "auto" for detection

        Returns:
            TranscriptionResult with transcribed text
        """
        await self._ensure_initialized()

        if self._model is None:
            raise RuntimeError("Faster Whisper model not loaded")

        # Preprocess audio
        preprocessed = self._audio_processor.preprocess_for_stt(audio)

        # Save to temp file (faster-whisper needs file path)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(preprocessed)
            temp_path = f.name

        try:
            # Transcribe
            language_arg = None if language == "auto" else language

            segments, info = self._model.transcribe(
                temp_path,
                language=language_arg,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection
            )

            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)

            full_text = " ".join(text_parts).strip()
            detected_language = info.language if language == "auto" else language

            logger.info(
                f"Faster Whisper transcribed: {full_text[:50]}... (lang: {detected_language})"
            )

            return TranscriptionResult(
                text=full_text,
                language=detected_language,
                confidence=info.language_probability if language == "auto" else 0.95,
                duration_seconds=info.duration,
                provider="faster-whisper",
            )

        finally:
            # Cleanup temp file
            os.unlink(temp_path)


class IndicWhisperSTT:
    """
    Speech-to-Text using AI4Bharat IndicConformer.

    IndicConformer supports all 22 official Indian languages with
    state-of-the-art accuracy. Falls back to Groq Whisper API if needed.

    Usage:
        stt = IndicWhisperSTT()
        result = await stt.transcribe(audio_bytes, language="hi")
        print(result.text)  # "मेरे पास 200 किलो टमाटर है"
    """

    # Model options - IndicConformer (22 Indian languages)
    MODEL_CONFORMER = "ai4bharat/indic-conformer-600m-multilingual"

    # Legacy model names (for backwards compatibility - will use Conformer)
    MODEL_SMALL = MODEL_CONFORMER
    MODEL_MEDIUM = MODEL_CONFORMER
    MODEL_LARGE = MODEL_CONFORMER

    # Default model
    DEFAULT_MODEL = MODEL_CONFORMER

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
    ):
        """
        Initialize IndicWhisper STT.

        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or 'auto'
        """
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
            # Import torch and transformers
            import torch
            from transformers import AutoModel

            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # Load IndicConformer model (requires trust_remote_code for custom architecture)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Move to device if CUDA available
            if device == "cuda":
                self._model = self._model.to(device)

            self._device = device
            self._is_conformer = True
            logger.info(f"IndicConformer loaded on {device}")

        except ImportError as e:
            logger.warning(f"Missing ML dependencies: {e}")
            raise RuntimeError("Install ML dependencies: uv sync --extra ml")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio bytes (WAV, MP3, OGG, or WEBM)
            language: Language code ('hi', 'kn', 'te', 'ta', 'auto')
            task: 'transcribe' or 'translate' (to English)

        Returns:
            TranscriptionResult with text, language, confidence
        """
        await self._ensure_initialized()

        # Validate audio
        is_valid, error = self._audio_processor.validate_audio(audio)
        if not is_valid:
            logger.error(f"Invalid audio: {error}")
            return TranscriptionResult(
                text="",
                language=language,
                confidence=0.0,
                duration_seconds=0.0,
                provider="error",
            )

        # Get duration
        duration = self._audio_processor.get_audio_duration(audio)

        # Use local model only
        if self._model is not None:
            result = await self._transcribe_local(audio, language, task)
            result.duration_seconds = duration
            return result

        raise RuntimeError(
            "Local IndicConformer model is not loaded and no fallbacks are allowed"
        )

    async def _transcribe_local(
        self,
        audio: bytes,
        language: str,
        task: str,
    ) -> TranscriptionResult:
        """Transcribe using local IndicConformer model"""
        import torch
        import io
        import numpy as np

        # Preprocess audio (converts to WAV format if FFmpeg available)
        preprocessed = self._audio_processor.preprocess_for_stt(audio)

        # Try loading audio with soundfile first (more compatible)
        wav = None
        sr = None

        try:
            import soundfile as sf

            audio_array, sr = sf.read(io.BytesIO(preprocessed))
            # Convert to tensor
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # Mono
            wav = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
            logger.debug("Audio loaded with soundfile")
        except Exception as sf_error:
            logger.debug(f"Soundfile failed: {sf_error}, trying torchaudio")
            # Fallback to torchaudio with explicit backend
            try:
                import torchaudio

                # Try with soundfile backend first (avoids torchcodec)
                torchaudio.set_audio_backend("soundfile")
                wav, sr = torchaudio.load(io.BytesIO(preprocessed))
                logger.debug("Audio loaded with torchaudio soundfile backend")
            except Exception as ta_error:
                logger.warning(f"Failed to load audio: {ta_error}")
                raise

        # Ensure mono
        wav = torch.mean(wav, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        target_sample_rate = 16000
        if sr != target_sample_rate:
            import torchaudio

            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sample_rate
            )
            wav = resampler(wav)

        # Use auto-detect if language is "auto", default to Hindi
        lang_code = language if language != "auto" else "hi"

        # Transcribe using IndicConformer (supports CTC and RNNT decoders)
        # Using CTC for faster inference
        try:
            transcription = self._model(wav, lang_code, "ctc")
        except Exception as e:
            logger.warning(f"CTC decoding failed, trying RNNT: {e}")
            transcription = self._model(wav, lang_code, "rnnt")

        # Detect language if auto
        detected_language = language
        if language == "auto":
            detected_language = self._detect_language(transcription)

        return TranscriptionResult(
            text=transcription.strip() if transcription else "",
            language=detected_language,
            confidence=0.95,  # IndicConformer has high accuracy
            duration_seconds=0.0,  # Will be set by caller
            provider="indicconformer",
        )

    # Groq fallback method removed completely for 100% local operation

    def _detect_language(self, text: str) -> str:
        """Simple language detection from text"""
        if not text:
            return "unknown"

        # Check for Devanagari (Hindi, Marathi)
        if any("\u0900" <= c <= "\u097f" for c in text):
            # Marathi has some specific patterns, default to Hindi
            return "hi"

        # Check for Kannada
        if any("\u0c80" <= c <= "\u0cff" for c in text):
            return "kn"

        # Check for Telugu
        if any("\u0c00" <= c <= "\u0c7f" for c in text):
            return "te"

        # Check for Tamil
        if any("\u0b80" <= c <= "\u0bff" for c in text):
            return "ta"

        # Check for Malayalam
        if any("\u0d00" <= c <= "\u0d7f" for c in text):
            return "ml"

        # Check for Gujarati
        if any("\u0a80" <= c <= "\u0aff" for c in text):
            return "gu"

        # Check for Bengali
        if any("\u0980" <= c <= "\u09ff" for c in text):
            return "bn"

        # Check for Punjabi (Gurmukhi)
        if any("\u0a00" <= c <= "\u0a7f" for c in text):
            return "pa"

        # Check for Odia
        if any("\u0b00" <= c <= "\u0b7f" for c in text):
            return "or"

        # Default to English
        return "en"

    async def detect_language(self, audio: bytes) -> str:
        """
        Detect language from audio without full transcription.

        Args:
            audio: Audio bytes

        Returns:
            Language code
        """
        result = await self.transcribe(audio, language="auto")
        return result.language

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return [
            lang.value for lang in SupportedLanguage if lang != SupportedLanguage.AUTO
        ]


class GroqWhisperSTT:
    """
    Groq Whisper cloud STT fallback.

    Uses ``whisper-large-v3-turbo`` via Groq API — near-zero cost, ~300ms latency.
    Activated automatically by MultiProviderSTT when GROQ_API_KEY is set.

    Reads GROQ_API_KEY from environment at construction time.
    Raises ValueError if key is absent (caller should catch and skip).
    """

    MODEL = "whisper-large-v3-turbo"

    def __init__(self) -> None:
        import os
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
        """
        Transcribe audio bytes via Groq Whisper API.

        Writes audio data to a temporary file with the correct extension,
        calls the sync Groq client in a thread executor, then cleans up.
        """
        import asyncio
        import os
        import tempfile
        import wave

        from .audio_utils import AudioProcessor, AudioFormat

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
            # If it's a known compressed format or already WAV, write bytes directly.
            # Only wrap in WAV header if it's RAW PCM.
            if fmt != AudioFormat.RAW:
                tmp.write(audio_data)
            else:
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit PCM
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_sync, wav_path, language
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
            )
        text = getattr(response, "text", "") or ""
        logger.info("GroqWhisperSTT transcribed {} chars", len(text))
        return TranscriptionResult(
            text=text,
            language=language if language != "auto" else "en",
            confidence=0.9,  # Groq doesn't return confidence scores
            duration_seconds=0.0,  # Groq API doesn't return duration
            provider="groq_whisper",
        )

    def get_supported_languages(self) -> list[str]:
        """Groq Whisper supports the same languages as OpenAI Whisper."""
        return [
            "en",
            "hi",
            "kn",
            "te",
            "ta",
            "ml",
            "mr",
            "gu",
            "pa",
            "bn",
            "ur",
            "ne",
            "si",
            "or",
            "as",
        ]


class MultiProviderSTT:
    """
    Multi-provider Speech-to-Text with automatic fallback.

    Provides maximum reliability by trying multiple STT providers in order:
    1. Faster Whisper (fastest, local, CPU-friendly — default primary)
    2. IndicConformer (best for Indian languages, needs GPU + cached model)
    3. Groq Whisper API (cloud fallback, added in Task 29)

    Usage:
        stt = MultiProviderSTT()
        result = await stt.transcribe(audio_bytes, language="hi")
    """

    def __init__(
        self,
        use_faster_whisper: bool = True,  # primary on CPU
        use_indicconformer: bool = False,  # disabled by default (needs GPU + cached model)
        faster_whisper_model: str = "small",
    ):
        """
        Initialize multi-provider STT.

        Args:
            use_faster_whisper: Enable Faster Whisper (primary, CPU-friendly)
            use_indicconformer: Enable IndicConformer (Indian languages, requires GPU)
            faster_whisper_model: Model size for Faster Whisper
        """
        self._providers = []
        self._provider_names = []

        # Priority 1: faster-whisper (local, CPU-friendly)
        if use_faster_whisper:
            try:
                self._providers.append(
                    FasterWhisperSTT(model_size=faster_whisper_model)
                )
                self._provider_names.append("faster-whisper")
                logger.info("MultiProviderSTT: Faster Whisper enabled")
            except Exception as e:
                logger.warning(f"Could not init Faster Whisper: {e}")

        # Priority 2: IndicConformer (GPU, AI4Bharat model)
        if use_indicconformer:
            try:
                self._providers.append(IndicWhisperSTT())
                self._provider_names.append("indicconformer")
                logger.info("MultiProviderSTT: AI4Bharat IndicConformer enabled")
            except Exception as e:
                logger.warning(f"Could not init IndicConformer: {e}")

        # Priority 3: GroqWhisper (cloud fallback — activates when GROQ_API_KEY is set)
        try:
            self._providers.append(GroqWhisperSTT())
            self._provider_names.append("groq_whisper")
            logger.info("MultiProviderSTT: GroqWhisperSTT registered as cloud fallback")
        except ValueError:
            pass  # No GROQ_API_KEY — skip silently
        except Exception as e:
            logger.warning(f"GroqWhisperSTT unavailable: {e}")

        if not self._providers:
            raise RuntimeError(
                "No STT providers available. Set GROQ_API_KEY for cloud fallback, "
                "or install faster-whisper for local inference."
            )

        logger.info(
            f"MultiProviderSTT initialized with {len(self._providers)} providers: {self._provider_names}"
        )

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """
        Transcribe audio using the best available provider.

        Tries providers in order until one succeeds.

        Args:
            audio: Audio bytes
            language: Language code or "auto"

        Returns:
            TranscriptionResult from the first successful provider
        """
        errors = []

        # Try each provider in order
        for i, provider in enumerate(self._providers):
            try:
                logger.debug(f"Trying STT provider: {self._provider_names[i]}")
                result = await provider.transcribe(audio, language)

                if result.is_successful:
                    logger.info(
                        f"STT success with {self._provider_names[i]}: {result.text[:50]}..."
                    )
                    return result
                else:
                    logger.warning(
                        f"STT provider {self._provider_names[i]} returned empty/low-confidence result"
                    )
                    errors.append(f"{self._provider_names[i]}: empty result")

            except Exception as e:
                logger.warning(f"STT provider {self._provider_names[i]} failed: {e}")
                errors.append(f"{self._provider_names[i]}: {str(e)}")
                continue

        # All local providers failed
        error_msg = "; ".join(errors)
        logger.error(f"All local STT providers failed: {error_msg}")

        # Return empty result rather than crash
        return TranscriptionResult(
            text="",
            language=language if language != "auto" else "hi",
            confidence=0.0,
            duration_seconds=0.0,
            provider="none",
        )

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names"""
        return self._provider_names.copy()

    def get_supported_languages(self) -> list[str]:
        """Get supported language codes from first available provider.

        Delegates to the first provider that exposes get_supported_languages().
        Falls back to all known CropFresh languages if no provider has the method.
        """
        for provider in self._providers:
            if hasattr(provider, "get_supported_languages"):
                return provider.get_supported_languages()
        # Fallback: return all known CropFresh languages
        return [
            lang.value for lang in SupportedLanguage if lang != SupportedLanguage.AUTO
        ]
