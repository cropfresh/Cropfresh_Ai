"""
VoiceAgent â€” backward-compatible wrapper around `src.agents.voice`.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.voice.agent import VoiceAgent as _VoiceAgent
from src.agents.voice.models import VoiceResponse, VoiceSession
from src.agents.voice.templates import REQUIRED_FIELDS, RESPONSE_TEMPLATES
from src.voice.entity_extractor import VoiceEntityExtractor


class _CompatSTT:
    """Minimal STT stub for legacy no-arg VoiceAgent construction."""

    async def transcribe(self, audio: bytes, language: str = "auto"):
        del audio, language
        raise RuntimeError("STT dependency was not provided to VoiceAgent")

    def get_supported_languages(self) -> list[str]:
        return ["hi", "kn", "en", "ta", "te", "mr", "bn", "gu", "pa", "ml"]


class _CompatTTS:
    """Minimal TTS stub for legacy no-arg VoiceAgent construction."""

    async def synthesize(self, text: str, language: str):
        del text, language
        return SimpleNamespace(audio=b"")


class VoiceAgent(_VoiceAgent):
    """Backward-compatible wrapper with optional dependency defaults."""

    def __init__(self, stt=None, tts=None, entity_extractor=None, **kwargs):
        super().__init__(
            stt=stt or _CompatSTT(),
            tts=tts or _CompatTTS(),
            entity_extractor=entity_extractor or VoiceEntityExtractor(),
            **kwargs,
        )


__all__ = [
    "VoiceAgent",
    "VoiceResponse",
    "VoiceSession",
    "RESPONSE_TEMPLATES",
    "REQUIRED_FIELDS",
]
