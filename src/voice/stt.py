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
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
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
            
            logger.info(f"Faster Whisper transcribed: {full_text[:50]}... (lang: {detected_language})")
            
            return TranscriptionResult(
                text=full_text,
                language=detected_language,
                confidence=info.language_probability if language == "auto" else 0.95,
                duration_seconds=info.duration,
                provider="faster-whisper"
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
        use_groq_fallback: bool = True,
        groq_api_key: Optional[str] = None,
    ):
        """
        Initialize IndicWhisper STT.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or 'auto'
            use_groq_fallback: Fall back to Groq API if local fails
            groq_api_key: API key for Groq (reads from env if not provided)
        """
        self.model_name = model_name
        self.device = device
        self.use_groq_fallback = use_groq_fallback
        self._groq_api_key = groq_api_key
        
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
            logger.warning(f"Failed to load local model: {e}")
            if self.use_groq_fallback:
                logger.info("Will use Groq API fallback")
                self._initialized = True
            else:
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
                provider="error"
            )
        
        # Get duration
        duration = self._audio_processor.get_audio_duration(audio)
        
        # Try local model first
        if self._model is not None:
            try:
                result = await self._transcribe_local(audio, language, task)
                result.duration_seconds = duration
                return result
            except Exception as e:
                logger.warning(f"Local transcription failed: {e}")
                if not self.use_groq_fallback:
                    raise
        
        # Fallback to Groq API
        if self.use_groq_fallback:
            return await self._transcribe_groq(audio, language, duration)
        
        raise RuntimeError("No transcription backend available")
    
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
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
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
            provider="indicconformer"
        )
    
    async def _transcribe_groq(
        self,
        audio: bytes,
        language: str,
        duration: float,
    ) -> TranscriptionResult:
        """Transcribe using Groq Whisper API"""
        import os
        import httpx
        import base64
        
        # Try instance key first, then settings, then env
        api_key = self._groq_api_key
        if not api_key:
            try:
                from src.config import get_settings
                api_key = get_settings().groq_api_key
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set for fallback")

        
        logger.info("Using Groq Whisper API fallback")
        
        # Preprocess audio
        preprocessed = self._audio_processor.preprocess_for_stt(audio)
        
        # Use Groq transcription endpoint
        async with httpx.AsyncClient() as client:
            # Create multipart form data
            files = {
                "file": ("audio.wav", preprocessed, "audio/wav"),
                "model": (None, "whisper-large-v3"),
            }
            
            if language != "auto":
                files["language"] = (None, language)
            
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                timeout=60.0,
            )
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.text}")
                raise RuntimeError(f"Groq API error: {response.status_code}")
            
            result = response.json()
        
        text = result.get("text", "").strip()
        detected_lang = result.get("language", language)
        if detected_lang == "auto":
            detected_lang = self._detect_language(text)
        
        return TranscriptionResult(
            text=text,
            language=detected_lang,
            confidence=0.85,
            duration_seconds=duration,
            provider="groq"
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection from text"""
        if not text:
            return "unknown"
        
        # Check for Devanagari (Hindi, Marathi)
        if any('\u0900' <= c <= '\u097F' for c in text):
            # Marathi has some specific patterns, default to Hindi
            return "hi"
        
        # Check for Kannada
        if any('\u0C80' <= c <= '\u0CFF' for c in text):
            return "kn"
        
        # Check for Telugu
        if any('\u0C00' <= c <= '\u0C7F' for c in text):
            return "te"
        
        # Check for Tamil
        if any('\u0B80' <= c <= '\u0BFF' for c in text):
            return "ta"
        
        # Check for Malayalam
        if any('\u0D00' <= c <= '\u0D7F' for c in text):
            return "ml"
        
        # Check for Gujarati
        if any('\u0A80' <= c <= '\u0AFF' for c in text):
            return "gu"
        
        # Check for Bengali
        if any('\u0980' <= c <= '\u09FF' for c in text):
            return "bn"
        
        # Check for Punjabi (Gurmukhi)
        if any('\u0A00' <= c <= '\u0A7F' for c in text):
            return "pa"
        
        # Check for Odia
        if any('\u0B00' <= c <= '\u0B7F' for c in text):
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
        return [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.AUTO]


class MultiProviderSTT:
    """
    Multi-provider Speech-to-Text with automatic fallback.
    
    Provides maximum reliability by trying multiple STT providers in order:
    1. Faster Whisper (fastest, local)
    2. IndicConformer (best for Indian languages)  
    3. Groq Whisper API (cloud fallback)
    
    Usage:
        stt = MultiProviderSTT()
        result = await stt.transcribe(audio_bytes, language="hi")
    """
    
    def __init__(
        self,
        use_faster_whisper: bool = True,
        use_indicconformer: bool = True,
        use_groq: bool = True,
        faster_whisper_model: str = "small",
        groq_api_key: Optional[str] = None,
    ):
        """
        Initialize multi-provider STT.
        
        Args:
            use_faster_whisper: Enable Faster Whisper (primary)
            use_indicconformer: Enable IndicConformer (Indian languages)
            use_groq: Enable Groq Whisper API (cloud fallback)
            faster_whisper_model: Model size for Faster Whisper
            groq_api_key: API key for Groq
        """
        self._providers = []
        self._provider_names = []
        
        # Add providers in priority order
        if use_faster_whisper:
            try:
                self._providers.append(FasterWhisperSTT(model_size=faster_whisper_model))
                self._provider_names.append("faster-whisper")
                logger.info("MultiProviderSTT: Faster Whisper enabled")
            except Exception as e:
                logger.warning(f"Could not init Faster Whisper: {e}")
        
        if use_indicconformer:
            try:
                self._providers.append(IndicWhisperSTT(use_groq_fallback=False))
                self._provider_names.append("indicconformer")
                logger.info("MultiProviderSTT: IndicConformer enabled")
            except Exception as e:
                logger.warning(f"Could not init IndicConformer: {e}")
        
        if use_groq:
            # Groq is handled via IndicWhisperSTT with fallback enabled
            self._groq_enabled = True
            self._groq_api_key = groq_api_key
            logger.info("MultiProviderSTT: Groq fallback enabled")
        else:
            self._groq_enabled = False
        
        if not self._providers and not self._groq_enabled:
            raise RuntimeError("No STT providers available")
        
        logger.info(f"MultiProviderSTT initialized with {len(self._providers)} providers: {self._provider_names}")
    
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
                    logger.info(f"STT success with {self._provider_names[i]}: {result.text[:50]}...")
                    return result
                else:
                    logger.warning(f"STT provider {self._provider_names[i]} returned empty/low-confidence result")
                    errors.append(f"{self._provider_names[i]}: empty result")
                    
            except Exception as e:
                logger.warning(f"STT provider {self._provider_names[i]} failed: {e}")
                errors.append(f"{self._provider_names[i]}: {str(e)}")
                continue
        
        # Final fallback: Groq API
        if self._groq_enabled:
            try:
                logger.info("Trying Groq Whisper API as final fallback")
                groq_stt = IndicWhisperSTT(use_groq_fallback=True, groq_api_key=self._groq_api_key)
                # Force Groq by ensuring local model fails
                groq_stt._model = None
                groq_stt._initialized = True
                result = await groq_stt.transcribe(audio, language)
                
                if result.is_successful:
                    logger.info(f"STT success with Groq: {result.text[:50]}...")
                    return result
                    
            except Exception as e:
                logger.error(f"Groq fallback also failed: {e}")
                errors.append(f"groq: {str(e)}")
        
        # All providers failed
        error_msg = "; ".join(errors)
        logger.error(f"All STT providers failed: {error_msg}")
        
        # Return empty result rather than crash
        return TranscriptionResult(
            text="",
            language=language if language != "auto" else "hi",
            confidence=0.0,
            duration_seconds=0.0,
            provider="none"
        )
    
    def get_available_providers(self) -> list[str]:
        """Get list of available provider names"""
        providers = self._provider_names.copy()
        if self._groq_enabled:
            providers.append("groq")
        return providers
