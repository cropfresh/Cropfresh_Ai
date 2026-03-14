"""
TTS Models
==========
Data structures for Text-to-Speech synthesis in CropFresh Voice Agent.
"""

from dataclasses import dataclass
from enum import Enum


class TTSVoice(Enum):
    """Available TTS voices."""
    MALE_DEFAULT = "male_default"
    FEMALE_DEFAULT = "female_default"
    MALE_YOUNG = "male_young"
    FEMALE_YOUNG = "female_young"
    MALE_SENIOR = "male_senior"
    FEMALE_SENIOR = "female_senior"


class TTSEmotion(Enum):
    """Supported emotions for TTS."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis."""
    audio: bytes
    format: str  # "wav" or "mp3"
    sample_rate: int
    duration_seconds: float
    language: str
    voice: str
    provider: str  # "indictts" or "edge-tts"
    
    @property
    def is_successful(self) -> bool:
        return len(self.audio) > 0
