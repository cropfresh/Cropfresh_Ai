"""
Voice Activity Detection (Proxy)
================================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.voice.vad`.
"""

from src.voice.vad import (
    VADState,
    VADEvent,
    SpeechSegment,
    SileroVAD,
    BargeinDetector,
    create_silence,
    bytes_to_wav,
)

__all__ = [
    "VADState",
    "VADEvent",
    "SpeechSegment",
    "SileroVAD",
    "BargeinDetector",
    "create_silence",
    "bytes_to_wav",
]
