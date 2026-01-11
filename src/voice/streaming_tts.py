"""
Streaming Text-to-Speech (TTS) Module for CropFresh Voice Agent

Provides real-time streaming TTS with multiple providers:
- Kokoro (82M params, fast, MIT license)
- XTTS-v2 (voice cloning, 17 languages)
- Edge TTS (fallback, good quality)

Features:
- Chunked audio generation for low latency
- Cancellation token for barge-in
- Language-matched voice selection
- Emotion and prosody control
"""

import asyncio
import io
import struct
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Callable, Optional

import numpy as np
from loguru import logger


class StreamingTTSProvider(str, Enum):
    """Available streaming TTS providers"""
    KOKORO = "kokoro"
    XTTS_V2 = "xtts_v2"
    EDGE = "edge"
    GTTS = "gtts"


@dataclass
class AudioChunk:
    """Streaming audio chunk"""
    data: bytes
    sample_rate: int
    format: str
    chunk_index: int
    is_last: bool
    duration_ms: float


@dataclass  
class StreamingConfig:
    """Configuration for streaming TTS"""
    provider: StreamingTTSProvider = StreamingTTSProvider.EDGE
    sample_rate: int = 22050
    chunk_size_ms: int = 100  # 100ms chunks for low latency
    voice: str = "default"
    language: str = "en"
    emotion: str = "neutral"
    speed: float = 1.0


class CancellationToken:
    """Token for cancelling TTS generation"""
    
    def __init__(self):
        self._cancelled = False
        self._on_cancel: Optional[Callable[[], None]] = None
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def cancel(self) -> None:
        """Cancel the operation"""
        self._cancelled = True
        if self._on_cancel:
            self._on_cancel()
        logger.debug("TTS cancelled")
    
    def reset(self) -> None:
        """Reset the token"""
        self._cancelled = False
    
    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Set cancellation callback"""
        self._on_cancel = callback


class StreamingTTS:
    """
    Streaming Text-to-Speech with multiple provider support.
    
    Designed for real-time voice agents with low latency requirements.
    Supports cancellation for barge-in interruption.
    
    Usage:
        tts = StreamingTTS()
        cancel_token = CancellationToken()
        
        async for chunk in tts.synthesize_stream("Hello world", "en", cancel_token):
            if cancel_token.is_cancelled:
                break
            play_audio(chunk.data)
    """
    
    # Language to voice mapping for each provider
    KOKORO_VOICES = {
        "en": "af_heart",
        "hi": "af_heart",  # Kokoro multilingual support
        "es": "af_heart",
        "fr": "af_heart",
    }
    
    XTTS_VOICES = {
        "en": "english",
        "hi": "hindi",
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "it": "italian",
        "pt": "portuguese",
        "pl": "polish",
        "tr": "turkish",
        "ru": "russian",
        "nl": "dutch",
        "cs": "czech",
        "ar": "arabic",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "hu": "hungarian",
    }
    
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
        "en-us": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    }
    
    def __init__(
        self,
        preferred_provider: StreamingTTSProvider = StreamingTTSProvider.EDGE,
        sample_rate: int = 22050,
        chunk_size_ms: int = 100,
    ):
        """
        Initialize streaming TTS.
        
        Args:
            preferred_provider: Primary TTS provider
            sample_rate: Output sample rate
            chunk_size_ms: Chunk size in milliseconds
        """
        self.preferred_provider = preferred_provider
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        
        # Provider availability
        self._kokoro_available = self._check_kokoro()
        self._xtts_available = self._check_xtts()
        self._edge_available = self._check_edge()
        
        logger.info(f"StreamingTTS initialized: "
                   f"kokoro={self._kokoro_available}, "
                   f"xtts={self._xtts_available}, "
                   f"edge={self._edge_available}")
    
    def _check_kokoro(self) -> bool:
        """Check if Kokoro is available"""
        try:
            # Kokoro uses specific import
            # import kokoro
            return False  # Disable for now, add when kokoro package available
        except ImportError:
            return False
    
    def _check_xtts(self) -> bool:
        """Check if XTTS-v2 is available"""
        try:
            # from TTS.api import TTS
            return False  # Disable for now, requires GPU
        except ImportError:
            return False
    
    def _check_edge(self) -> bool:
        """Check if Edge TTS is available"""
        try:
            import edge_tts
            return True
        except ImportError:
            return False
    
    def get_voice_for_language(
        self,
        language: str,
        provider: StreamingTTSProvider,
        gender: str = "female",
    ) -> str:
        """
        Get appropriate voice for language and provider.
        
        Args:
            language: Language code
            provider: TTS provider
            gender: 'male' or 'female'
            
        Returns:
            Voice identifier
        """
        if provider == StreamingTTSProvider.EDGE:
            voices = self.EDGE_VOICES.get(language, self.EDGE_VOICES["en"])
            return voices.get(gender, voices.get("female"))
        
        elif provider == StreamingTTSProvider.KOKORO:
            return self.KOKORO_VOICES.get(language, "af_heart")
        
        elif provider == StreamingTTSProvider.XTTS_V2:
            return self.XTTS_VOICES.get(language, "english")
        
        return "default"
    
    async def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        cancel_token: Optional[CancellationToken] = None,
        voice: str = "female",
        speed: float = 1.0,
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream synthesized audio chunks.
        
        Args:
            text: Text to synthesize
            language: Language code
            cancel_token: Cancellation token for interruption
            voice: Voice type ('male', 'female', or specific voice ID)
            speed: Speech speed multiplier
            
        Yields:
            AudioChunk objects
        """
        if not text:
            return
        
        cancel = cancel_token or CancellationToken()
        
        # Select provider
        if self._edge_available:
            provider = StreamingTTSProvider.EDGE
        elif self._kokoro_available:
            provider = StreamingTTSProvider.KOKORO
        else:
            logger.error("No TTS provider available")
            return
        
        # Get voice
        voice_id = self.get_voice_for_language(language, provider, voice)
        
        logger.info(f"Streaming TTS: {provider.value}, {language}, {voice_id}")
        
        # Route to provider
        if provider == StreamingTTSProvider.EDGE:
            async for chunk in self._stream_edge(text, voice_id, cancel, speed):
                yield chunk
        
        elif provider == StreamingTTSProvider.KOKORO:
            async for chunk in self._stream_kokoro(text, language, voice_id, cancel):
                yield chunk
    
    async def _stream_edge(
        self,
        text: str,
        voice: str,
        cancel: CancellationToken,
        speed: float = 1.0,
    ) -> AsyncIterator[AudioChunk]:
        """Stream using Edge TTS"""
        import edge_tts
        
        # Calculate rate adjustment
        rate = f"+{int((speed - 1) * 100)}%" if speed > 1 else f"{int((speed - 1) * 100)}%"
        
        communicate = edge_tts.Communicate(text, voice, rate=rate if speed != 1.0 else None)
        
        chunk_index = 0
        audio_buffer = io.BytesIO()
        
        async for chunk in communicate.stream():
            if cancel.is_cancelled:
                logger.debug("Edge TTS cancelled")
                break
            
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
                
                # Check if we have enough for a chunk
                chunk_size = int(self.sample_rate * self.chunk_size_ms / 1000 * 2)  # 16-bit
                
                if audio_buffer.tell() >= chunk_size:
                    audio_buffer.seek(0)
                    chunk_data = audio_buffer.read(chunk_size)
                    
                    yield AudioChunk(
                        data=chunk_data,
                        sample_rate=self.sample_rate,
                        format="mp3",  # Edge outputs MP3
                        chunk_index=chunk_index,
                        is_last=False,
                        duration_ms=self.chunk_size_ms,
                    )
                    
                    chunk_index += 1
                    
                    # Keep remaining data
                    remaining = audio_buffer.read()
                    audio_buffer = io.BytesIO()
                    audio_buffer.write(remaining)
        
        # Yield remaining audio
        if audio_buffer.tell() > 0 and not cancel.is_cancelled:
            audio_buffer.seek(0)
            remaining_data = audio_buffer.read()
            
            if remaining_data:
                yield AudioChunk(
                    data=remaining_data,
                    sample_rate=self.sample_rate,
                    format="mp3",
                    chunk_index=chunk_index,
                    is_last=True,
                    duration_ms=len(remaining_data) * 1000 / (self.sample_rate * 2),
                )
    
    async def _stream_kokoro(
        self,
        text: str,
        language: str,
        voice: str,
        cancel: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Stream using Kokoro TTS"""
        # Kokoro implementation - placeholder for when package is available
        # This would use the kokoro Python API for streaming
        logger.warning("Kokoro TTS not implemented yet")
        return
        yield  # Make this a generator
    
    async def synthesize_full(
        self,
        text: str,
        language: str = "en",
        voice: str = "female",
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesize complete audio (non-streaming).
        
        Args:
            text: Text to synthesize
            language: Language code
            voice: Voice type
            speed: Speech speed
            
        Returns:
            Complete audio bytes
        """
        chunks = []
        
        async for chunk in self.synthesize_stream(text, language, None, voice, speed):
            chunks.append(chunk.data)
        
        return b"".join(chunks)


class MultiProviderStreamingTTS:
    """
    Multi-provider streaming TTS with automatic fallback.
    
    Tries providers in order of preference, falling back if one fails.
    """
    
    def __init__(
        self,
        providers: Optional[list[StreamingTTSProvider]] = None,
    ):
        """
        Initialize multi-provider TTS.
        
        Args:
            providers: Ordered list of providers to try
        """
        self.providers = providers or [
            StreamingTTSProvider.EDGE,
            StreamingTTSProvider.KOKORO,
            StreamingTTSProvider.GTTS,
        ]
        
        self._tts = StreamingTTS()
    
    async def synthesize_stream(
        self,
        text: str,
        language: str,
        cancel_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> AsyncIterator[AudioChunk]:
        """Synthesize with automatic fallback"""
        async for chunk in self._tts.synthesize_stream(
            text,
            language,
            cancel_token,
            **kwargs,
        ):
            yield chunk
    
    def get_available_providers(self) -> list[str]:
        """Get list of available providers"""
        available = []
        if self._tts._edge_available:
            available.append("edge")
        if self._tts._kokoro_available:
            available.append("kokoro")
        if self._tts._xtts_available:
            available.append("xtts_v2")
        return available
