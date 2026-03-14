"""
STT data models and enumerations.
"""

from dataclasses import dataclass
from enum import Enum


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
        # We relax the confidence threshold because for short Indic phrases,
        # Whisper's language detection confidence can be very low even when text is correct.
        return len(self.text.strip()) > 0
