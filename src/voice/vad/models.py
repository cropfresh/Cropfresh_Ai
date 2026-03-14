"""
VAD Models
==========
Data structures and enumerations for Voice Activity Detection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VADState(Enum):
    """Voice activity detection states"""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


@dataclass
class VADEvent:
    """Event from VAD processing"""
    state: VADState
    timestamp_ms: float
    probability: float
    audio_chunk: Optional[bytes] = None


@dataclass
class SpeechSegment:
    """Complete speech segment with audio data"""
    audio: bytes
    start_ms: float
    end_ms: float
    duration_ms: float
    sample_rate: int = 16000
