"""
Text-to-Speech (TTS) Package for CropFresh Voice Agent
======================================================
Uses AI4Bharat IndicTTS/IndicF5 for Indian language speech synthesis.
Supports: Hindi, Kannada, Telugu, Tamil, + 16 more Indian languages.
"""

from .models import TTSVoice, TTSEmotion, SynthesisResult
from .indic import IndicTTS
from .edge import EdgeTTSProvider
from .utils import normalize_edge_rate

__all__ = [
    "TTSVoice",
    "TTSEmotion",
    "SynthesisResult",
    "IndicTTS",
    "EdgeTTSProvider",
    "normalize_edge_rate",
]
