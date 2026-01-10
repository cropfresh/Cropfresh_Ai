"""
Audio Processing Utilities for CropFresh Voice Agent

Provides audio format conversion, validation, and preprocessing.
"""

import io
import struct
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

from loguru import logger


class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    WEBM = "webm"
    RAW = "raw"


@dataclass
class AudioMetadata:
    """Audio file metadata"""
    format: AudioFormat
    sample_rate: int
    channels: int
    duration_seconds: float
    size_bytes: int


class AudioProcessor:
    """
    Audio processing utilities for voice agent.
    
    Handles:
    - Format detection and validation
    - Sample rate conversion
    - Channel normalization
    - Audio preprocessing for STT
    """
    
    # Target format for STT (IndicWhisper)
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    
    # Supported formats
    SUPPORTED_FORMATS = {
        b"RIFF": AudioFormat.WAV,
        b"\xff\xfb": AudioFormat.MP3,
        b"\xff\xfa": AudioFormat.MP3,
        b"ID3": AudioFormat.MP3,
        b"OggS": AudioFormat.OGG,
        b"\x1a\x45\xdf\xa3": AudioFormat.WEBM,
    }
    
    def __init__(self):
        """Initialize audio processor"""
        self._ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available for advanced processing"""
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            logger.warning("FFmpeg not available, using basic audio processing")
            return False
    
    def detect_format(self, audio_data: bytes) -> AudioFormat:
        """
        Detect audio format from bytes.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Detected AudioFormat
        """
        if len(audio_data) < 12:
            return AudioFormat.RAW
        
        # Check magic bytes
        for magic, fmt in self.SUPPORTED_FORMATS.items():
            if audio_data.startswith(magic):
                return fmt
        
        return AudioFormat.RAW
    
    def get_metadata(self, audio_data: bytes) -> AudioMetadata:
        """
        Extract metadata from audio data.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            AudioMetadata with format, sample rate, etc.
        """
        fmt = self.detect_format(audio_data)
        
        if fmt == AudioFormat.WAV:
            return self._parse_wav_header(audio_data)
        
        # For other formats, return basic info
        return AudioMetadata(
            format=fmt,
            sample_rate=0,  # Unknown without FFmpeg
            channels=0,
            duration_seconds=0.0,
            size_bytes=len(audio_data)
        )
    
    def _parse_wav_header(self, audio_data: bytes) -> AudioMetadata:
        """Parse WAV file header for metadata"""
        try:
            if len(audio_data) < 44:
                raise ValueError("WAV file too short")
            
            # Parse RIFF header
            riff = audio_data[0:4]
            if riff != b"RIFF":
                raise ValueError("Not a valid WAV file")
            
            # Parse fmt chunk
            fmt_chunk = audio_data[12:16]
            if fmt_chunk != b"fmt ":
                raise ValueError("Invalid WAV format chunk")
            
            # Audio format (1 = PCM)
            audio_format = struct.unpack("<H", audio_data[20:22])[0]
            channels = struct.unpack("<H", audio_data[22:24])[0]
            sample_rate = struct.unpack("<I", audio_data[24:28])[0]
            
            # Calculate duration
            byte_rate = struct.unpack("<I", audio_data[28:32])[0]
            data_size = len(audio_data) - 44
            duration = data_size / byte_rate if byte_rate > 0 else 0.0
            
            return AudioMetadata(
                format=AudioFormat.WAV,
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration,
                size_bytes=len(audio_data)
            )
        except Exception as e:
            logger.error(f"Failed to parse WAV header: {e}")
            return AudioMetadata(
                format=AudioFormat.WAV,
                sample_rate=0,
                channels=0,
                duration_seconds=0.0,
                size_bytes=len(audio_data)
            )
    
    def preprocess_for_stt(self, audio_data: bytes) -> bytes:
        """
        Preprocess audio for Speech-to-Text.
        
        Converts to:
        - 16kHz sample rate
        - Mono channel
        - 16-bit PCM WAV
        
        Args:
            audio_data: Input audio bytes
            
        Returns:
            Preprocessed WAV bytes
        """
        fmt = self.detect_format(audio_data)
        
        if fmt == AudioFormat.WAV:
            metadata = self._parse_wav_header(audio_data)
            
            # If already in correct format, return as-is
            if (metadata.sample_rate == self.TARGET_SAMPLE_RATE and 
                metadata.channels == self.TARGET_CHANNELS):
                return audio_data
        
        # Use FFmpeg for conversion if available
        if self._ffmpeg_available:
            return self._convert_with_ffmpeg(audio_data)
        
        # If no FFmpeg and format is WAV, try basic conversion
        if fmt == AudioFormat.WAV:
            return self._basic_wav_conversion(audio_data)
        
        # Cannot convert without FFmpeg
        logger.warning("Cannot convert audio format without FFmpeg, returning original")
        return audio_data
    
    def _convert_with_ffmpeg(self, audio_data: bytes) -> bytes:
        """Convert audio using FFmpeg subprocess"""
        import subprocess
        import tempfile
        import os
        
        try:
            # Write input to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f_in:
                f_in.write(audio_data)
                input_path = f_in.name
            
            # Output temp file
            output_path = input_path + ".wav"
            
            # Run FFmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", str(self.TARGET_SAMPLE_RATE),
                "-ac", str(self.TARGET_CHANNELS),
                "-f", "wav",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
                return audio_data
            
            # Read output
            with open(output_path, "rb") as f_out:
                converted = f_out.read()
            
            # Cleanup
            os.unlink(input_path)
            os.unlink(output_path)
            
            return converted
            
        except Exception as e:
            logger.error(f"FFmpeg conversion error: {e}")
            return audio_data
    
    def _basic_wav_conversion(self, audio_data: bytes) -> bytes:
        """Basic WAV conversion without FFmpeg (limited)"""
        # For basic conversion, we just return the original
        # Full conversion requires soundfile/librosa
        return audio_data
    
    def validate_audio(self, audio_data: bytes, max_duration_seconds: float = 60.0) -> tuple[bool, str]:
        """
        Validate audio for processing.
        
        Args:
            audio_data: Audio bytes to validate
            max_duration_seconds: Maximum allowed duration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not audio_data:
            return False, "Empty audio data"
        
        if len(audio_data) < 44:
            return False, "Audio data too short"
        
        fmt = self.detect_format(audio_data)
        if fmt == AudioFormat.RAW:
            return False, "Unsupported audio format"
        
        if fmt == AudioFormat.WAV:
            metadata = self._parse_wav_header(audio_data)
            if metadata.duration_seconds > max_duration_seconds:
                return False, f"Audio too long: {metadata.duration_seconds:.1f}s (max: {max_duration_seconds}s)"
        
        return True, ""
    
    def get_audio_duration(self, audio_data: bytes) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            Duration in seconds, or 0 if unknown
        """
        metadata = self.get_metadata(audio_data)
        return metadata.duration_seconds
