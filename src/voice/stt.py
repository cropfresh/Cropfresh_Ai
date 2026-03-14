"""
Speech-to-Text (STT) Module — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.voice.stt_pkg` package.
! Import from `src.voice.stt_pkg` directly in new code.
"""

from src.voice.stt_pkg.faster_whisper import FasterWhisperSTT
from src.voice.stt_pkg.groq_whisper import GroqWhisperSTT
from src.voice.stt_pkg.indic_whisper import IndicWhisperSTT
from src.voice.stt_pkg.models import SupportedLanguage, TranscriptionResult
from src.voice.stt_pkg.multi_provider import MultiProviderSTT

__all__ = [
    "FasterWhisperSTT",
    "GroqWhisperSTT",
    "IndicWhisperSTT",
    "MultiProviderSTT",
    "SupportedLanguage",
    "TranscriptionResult",
]
