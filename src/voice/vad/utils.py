"""
VAD Utilities
=============
Helper functions for voice activity detection processing.
"""

import struct
import numpy as np


def create_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Create silent audio bytes"""
    num_samples = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()


def bytes_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw PCM bytes to WAV format"""
    num_channels = 1
    sample_width = 2  # 16-bit
    byte_rate = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    data_size = len(audio_bytes)
    
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
    
    return header + audio_bytes
