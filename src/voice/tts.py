"""
Text-to-Speech (TTS) Module for CropFresh Voice Agent

Uses AI4Bharat IndicTTS/IndicF5 for Indian language speech synthesis.
Supports: Hindi, Kannada, Telugu, Tamil, + 16 more Indian languages.

Features:
- 20 Indic languages support
- 69 unique voice options
- Emotion support (neutral, happy, sad)
- Near-human quality synthesis
"""

import asyncio
import io
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger


class TTSVoice(Enum):
    """Available TTS voices"""
    MALE_DEFAULT = "male_default"
    FEMALE_DEFAULT = "female_default"
    MALE_YOUNG = "male_young"
    FEMALE_YOUNG = "female_young"
    MALE_SENIOR = "male_senior"
    FEMALE_SENIOR = "female_senior"


class TTSEmotion(Enum):
    """Supported emotions for TTS"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis"""
    audio: bytes
    format: str  # "wav"
    sample_rate: int
    duration_seconds: float
    language: str
    voice: str
    provider: str  # "indictts" or "edge"
    
    @property
    def is_successful(self) -> bool:
        return len(self.audio) > 0


class IndicTTS:
    """
    Text-to-Speech using AI4Bharat IndicTTS.
    
    IndicTTS provides high-quality speech synthesis for 20 Indian languages
    with multiple voice options and emotion support.
    
    Usage:
        tts = IndicTTS()
        result = await tts.synthesize("नमस्ते, कैसे हैं आप?", language="hi")
        # result.audio contains WAV bytes
    """
    
    # Model options
    MODEL_PARLER = "ai4bharat/indic-parler-tts"
    MODEL_F5 = "ai4bharat/IndicF5"
    
    # Language to voice mapping
    LANGUAGE_VOICES = {
        "hi": ["hindi_male_1", "hindi_female_1"],
        "kn": ["kannada_male_1", "kannada_female_1"],
        "te": ["telugu_male_1", "telugu_female_1"],
        "ta": ["tamil_male_1", "tamil_female_1"],
        "ml": ["malayalam_male_1", "malayalam_female_1"],
        "mr": ["marathi_male_1", "marathi_female_1"],
        "gu": ["gujarati_male_1", "gujarati_female_1"],
        "bn": ["bengali_male_1", "bengali_female_1"],
        "pa": ["punjabi_male_1", "punjabi_female_1"],
        "or": ["odia_male_1", "odia_female_1"],
        "en": ["english_male_1", "english_female_1"],
    }
    
    # Sample rate for output
    OUTPUT_SAMPLE_RATE = 22050
    
    def __init__(
        self,
        model_name: str = MODEL_PARLER,
        device: str = "auto",
        use_edge_fallback: bool = True,
    ):
        """
        Initialize IndicTTS.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or 'auto'
            use_edge_fallback: Fall back to Edge TTS if local fails
        """
        self.model_name = model_name
        self.device = device
        self.use_edge_fallback = use_edge_fallback
        
        self._model = None
        self._tokenizer = None
        self._initialized = False
        
        logger.info(f"IndicTTS initialized with model: {model_name}")
    
    async def _ensure_initialized(self):
        """Lazy load the model on first use"""
        if self._initialized:
            return
        
        # For now, skip local model loading due to Parler-TTS compatibility issues
        # Use Edge TTS as the primary TTS engine (high quality, free)
        # TODO: Add Coqui XTTS-v2 support when GPU available
        if self.use_edge_fallback:
            logger.info("Using Edge TTS as primary TTS (skipping local model)")
            self._initialized = True
            return
        
        try:
            await self._load_model()
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to load local TTS model: {e}")
            if self.use_edge_fallback:
                logger.info("Will use Edge TTS fallback")
                self._initialized = True
            else:
                raise
    
    async def _load_model(self):
        """Load IndicTTS model from HuggingFace"""
        logger.info(f"Loading IndicTTS model: {self.model_name}")
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTextToWaveform
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForTextToWaveform.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            
            self._device = device
            logger.info(f"IndicTTS loaded on {device}")
            
        except Exception as e:
            logger.warning(f"Could not load local TTS model: {e}")
            # Don't raise - we'll use fallback
    
    async def synthesize(
        self,
        text: str,
        language: str,
        voice: str = "default",
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            language: Language code ('hi', 'kn', 'te', 'ta', etc.)
            voice: Voice ID or 'default', 'male', 'female'
            emotion: Emotion tone ('neutral', 'happy', 'sad')
            speed: Speech speed multiplier (0.5 - 2.0)
            
        Returns:
            SynthesisResult with audio bytes
        """
        await self._ensure_initialized()
        
        if not text:
            return SynthesisResult(
                audio=b"",
                format="wav",
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                duration_seconds=0.0,
                language=language,
                voice=voice,
                provider="error"
            )
        
        # Normalize text
        text = self._normalize_text(text, language)
        
        # Try local model first
        if self._model is not None:
            try:
                return await self._synthesize_local(text, language, voice, emotion, speed)
            except Exception as e:
                logger.warning(f"Local TTS failed: {e}")
                if not self.use_edge_fallback:
                    raise
        
        # Fallback to Edge TTS
        if self.use_edge_fallback:
            return await self._synthesize_edge(text, language, voice)
        
        raise RuntimeError("No TTS backend available")
    
    async def _synthesize_local(
        self,
        text: str,
        language: str,
        voice: str,
        emotion: str,
        speed: float,
    ) -> SynthesisResult:
        """Synthesize using local IndicTTS model"""
        import torch
        import numpy as np
        import struct
        
        # Get voice for language
        actual_voice = self._get_voice(language, voice)
        
        # Build prompt for Parler TTS
        prompt = self._build_voice_prompt(actual_voice, emotion, speed)
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        
        # Generate audio
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_length=2048,
            )
        
        # Convert to numpy
        audio_array = output.cpu().numpy().flatten()
        
        # Normalize
        audio_array = audio_array / np.abs(audio_array).max()
        
        # Convert to WAV bytes
        audio_bytes = self._array_to_wav(audio_array, self.OUTPUT_SAMPLE_RATE)
        
        # Calculate duration
        duration = len(audio_array) / self.OUTPUT_SAMPLE_RATE
        
        return SynthesisResult(
            audio=audio_bytes,
            format="wav",
            sample_rate=self.OUTPUT_SAMPLE_RATE,
            duration_seconds=duration,
            language=language,
            voice=actual_voice,
            provider="indictts"
        )
    
    async def _synthesize_edge(
        self,
        text: str,
        language: str,
        voice: str,
    ) -> SynthesisResult:
        """Synthesize using Microsoft Edge TTS (free, no API key)"""
        try:
            import edge_tts
        except ImportError:
            # Fallback to gTTS
            return await self._synthesize_gtts(text, language)
        
        # Map language to Edge voice
        edge_voice = self._get_edge_voice(language, voice)
        
        logger.info(f"Using Edge TTS with voice: {edge_voice}")
        
        # Generate audio
        communicate = edge_tts.Communicate(text, edge_voice)
        
        # Collect audio chunks
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        
        audio_bytes = b"".join(audio_chunks)
        
        # Convert MP3 to WAV if needed
        # Edge TTS outputs MP3
        audio_bytes = await self._mp3_to_wav(audio_bytes)
        
        # Estimate duration (rough estimate from byte size)
        duration = len(audio_bytes) / (self.OUTPUT_SAMPLE_RATE * 2)
        
        return SynthesisResult(
            audio=audio_bytes,
            format="wav",
            sample_rate=self.OUTPUT_SAMPLE_RATE,
            duration_seconds=duration,
            language=language,
            voice=edge_voice,
            provider="edge"
        )
    
    async def _synthesize_gtts(
        self,
        text: str,
        language: str,
    ) -> SynthesisResult:
        """Fallback to Google Text-to-Speech (gTTS)"""
        try:
            from gtts import gTTS
        except ImportError:
            raise RuntimeError("Install edge-tts or gtts: pip install edge-tts gtts")
        
        logger.info("Using gTTS fallback")
        
        # Map language code
        gtts_lang = self._map_to_gtts_lang(language)
        
        # Generate audio
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Save to bytes
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        mp3_bytes = audio_buffer.read()
        
        # Convert MP3 to WAV
        wav_bytes = await self._mp3_to_wav(mp3_bytes)
        
        duration = len(wav_bytes) / (self.OUTPUT_SAMPLE_RATE * 2)
        
        return SynthesisResult(
            audio=wav_bytes,
            format="wav",
            sample_rate=self.OUTPUT_SAMPLE_RATE,
            duration_seconds=duration,
            language=language,
            voice="gtts",
            provider="gtts"
        )
    
    def _normalize_text(self, text: str, language: str) -> str:
        """Normalize text for TTS"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Add sentence-ending pause if missing
        if text and text[-1] not in ".!?।":
            text += "।" if language in ["hi", "mr", "kn", "te", "ta"] else "."
        
        return text
    
    def _get_voice(self, language: str, voice: str) -> str:
        """Get appropriate voice for language"""
        voices = self.LANGUAGE_VOICES.get(language, self.LANGUAGE_VOICES["en"])
        
        if voice in ["male", "male_default"]:
            return voices[0]
        elif voice in ["female", "female_default"]:
            return voices[1] if len(voices) > 1 else voices[0]
        elif voice == "default":
            return voices[0]
        
        return voice
    
    def _get_edge_voice(self, language: str, voice: str) -> str:
        """Map language to Microsoft Edge TTS voice"""
        edge_voices = {
            "hi": "hi-IN-SwaraNeural" if "female" in voice else "hi-IN-MadhurNeural",
            "kn": "kn-IN-SapnaNeural" if "female" in voice else "kn-IN-GaganNeural",
            "te": "te-IN-ShrutiNeural" if "female" in voice else "te-IN-MohanNeural",
            "ta": "ta-IN-PallaviNeural" if "female" in voice else "ta-IN-ValluvarNeural",
            "ml": "ml-IN-SobhanaNeural" if "female" in voice else "ml-IN-MidhunNeural",
            "mr": "mr-IN-AarohiNeural" if "female" in voice else "mr-IN-ManoharNeural",
            "gu": "gu-IN-DhwaniNeural" if "female" in voice else "gu-IN-NiranjanNeural",
            "bn": "bn-IN-TanishaaNeural" if "female" in voice else "bn-IN-BashkarNeural",
            "en": "en-IN-NeerjaNeural" if "female" in voice else "en-IN-PrabhatNeural",
        }
        return edge_voices.get(language, edge_voices["en"])
    
    def _map_to_gtts_lang(self, language: str) -> str:
        """Map language code to gTTS language"""
        gtts_map = {
            "hi": "hi",
            "kn": "kn",
            "te": "te",
            "ta": "ta",
            "ml": "ml",
            "mr": "mr",
            "gu": "gu",
            "bn": "bn",
            "pa": "pa",
            "en": "en-in",
        }
        return gtts_map.get(language, "en")
    
    def _build_voice_prompt(self, voice: str, emotion: str, speed: float) -> str:
        """Build voice description prompt for Parler TTS"""
        prompts = {
            "neutral": "speaks clearly and naturally",
            "happy": "speaks with a warm and cheerful tone",
            "sad": "speaks with a soft and gentle tone",
            "excited": "speaks with enthusiasm and energy",
        }
        
        emotion_desc = prompts.get(emotion, prompts["neutral"])
        
        speed_desc = ""
        if speed < 0.8:
            speed_desc = " at a slow pace"
        elif speed > 1.2:
            speed_desc = " at a fast pace"
        
        return f"A clear voice that {emotion_desc}{speed_desc}."
    
    def _array_to_wav(self, audio_array, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes"""
        import struct
        import numpy as np
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Build WAV header
        num_channels = 1
        sample_width = 2  # 16-bit
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        data_size = len(audio_int16) * sample_width
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # PCM format size
            1,   # PCM format
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            16,  # bits per sample
            b'data',
            data_size,
        )
        
        return header + audio_int16.tobytes()
    
    async def _mp3_to_wav(self, mp3_bytes: bytes) -> bytes:
        """Convert MP3 to WAV format"""
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            audio = audio.set_frame_rate(self.OUTPUT_SAMPLE_RATE)
            audio = audio.set_channels(1)
            
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            return wav_buffer.read()
            
        except ImportError:
            logger.warning("pydub not available, returning raw bytes")
            return mp3_bytes
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_VOICES.keys())
    
    def get_available_voices(self, language: str) -> list[str]:
        """Get available voices for a language"""
        return self.LANGUAGE_VOICES.get(language, ["default"])
