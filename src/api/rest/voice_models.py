"""Pydantic models for voice REST routes."""

from __future__ import annotations

from pydantic import BaseModel


class VoiceProcessResponse(BaseModel):
    """Response from voice processing."""

    transcription: str
    language: str
    intent: str
    entities: dict
    response_text: str
    response_audio_base64: str
    session_id: str
    confidence: float


class TranscribeResponse(BaseModel):
    """Response from transcription."""

    text: str
    language: str
    confidence: float
    duration_seconds: float
    provider: str


class SynthesizeRequest(BaseModel):
    """Request for text-to-speech."""

    text: str
    language: str = "hi"
    voice: str = "default"
    emotion: str = "neutral"


class SynthesizeResponse(BaseModel):
    """Response from synthesis."""

    audio_base64: str
    format: str
    duration_seconds: float


class LanguagesResponse(BaseModel):
    """Supported STT/TTS languages."""

    stt_languages: list[str]
    tts_languages: list[str]
