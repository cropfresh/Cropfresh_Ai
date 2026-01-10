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
    provider: str  # "indicwhisper" or "groq"
    
    @property
    def is_successful(self) -> bool:
        return len(self.text) > 0 and self.confidence > 0.5


class IndicWhisperSTT:
    """
    Speech-to-Text using AI4Bharat IndicWhisper.
    
    IndicWhisper is fine-tuned on Indian languages and provides
    20-50% better Word Error Rate compared to vanilla OpenAI Whisper.
    
    Usage:
        stt = IndicWhisperSTT()
        result = await stt.transcribe(audio_bytes, language="hi")
        print(result.text)  # "मेरे पास 200 किलो टमाटर है"
    """
    
    # Model options
    MODEL_SMALL = "ai4bharat/indicwhisper-small"
    MODEL_MEDIUM = "ai4bharat/indicwhisper-medium"
    MODEL_LARGE = "ai4bharat/indicwhisper-large-v2"
    
    # Default model (balance between speed and accuracy)
    DEFAULT_MODEL = MODEL_MEDIUM
    
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
        """Load IndicWhisper model from HuggingFace"""
        logger.info(f"Loading IndicWhisper model: {self.model_name}")
        
        try:
            # Import torch and transformers
            import torch
            from transformers import (
                WhisperProcessor,
                WhisperForConditionalGeneration,
            )
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load processor and model
            self._processor = WhisperProcessor.from_pretrained(self.model_name)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            
            self._device = device
            logger.info(f"IndicWhisper loaded on {device}")
            
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
        """Transcribe using local IndicWhisper model"""
        import torch
        import numpy as np
        import soundfile as sf
        import io
        
        # Preprocess audio
        preprocessed = self._audio_processor.preprocess_for_stt(audio)
        
        # Load audio as numpy array
        audio_array, sample_rate = sf.read(io.BytesIO(preprocessed))
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Process with Whisper processor
        input_features = self._processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self._device)
        
        # Generate transcription
        with torch.no_grad():
            # Set language if specified
            forced_decoder_ids = None
            if language != "auto":
                lang_id = self._processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
                forced_decoder_ids = [(1, lang_id)]
            
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
            )
        
        # Decode
        transcription = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        # Detect language if auto
        detected_language = language
        if language == "auto":
            detected_language = self._detect_language(transcription)
        
        return TranscriptionResult(
            text=transcription.strip(),
            language=detected_language,
            confidence=0.9,  # Local model confidence
            duration_seconds=0.0,  # Will be set by caller
            provider="indicwhisper"
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
        
        api_key = self._groq_api_key or os.getenv("GROQ_API_KEY")
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
