"""
Voice Activity Detection (VAD) Package
======================================
Real-time voice activity detection using Silero VAD.
"""

from .bargein import BargeinDetector
from .models import SpeechSegment, VADEvent, VADState
from .silero import SileroVAD
from .utils import bytes_to_wav, create_silence

__all__ = [
    "VADState",
    "VADEvent",
    "SpeechSegment",
    "SileroVAD",
    "BargeinDetector",
    "create_silence",
    "bytes_to_wav",
]
