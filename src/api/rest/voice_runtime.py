"""Shared runtime helpers for voice REST routes."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from loguru import logger

from src.api.runtime.voice_agent import build_voice_agent

_fallback_voice_agent = None


def resolve_voice_agent(request: Request) -> Any:
    """Return the shared app voice agent or build a local fallback."""
    agent = getattr(request.app.state, "voice_agent", None)
    if agent is not None:
        return agent

    global _fallback_voice_agent
    if _fallback_voice_agent is None:
        _fallback_voice_agent = build_voice_agent(
            llm_provider=getattr(request.app.state, "llm", None),
            listing_service=getattr(request.app.state, "listing_service", None),
            order_service=getattr(request.app.state, "order_service", None),
            adcl_agent=getattr(request.app.state, "adcl_service", None),
        )
        logger.warning("Voice REST is using a fallback voice agent")
    return _fallback_voice_agent


def resolve_stt(request: Request) -> Any:
    """Resolve STT from the shared voice agent."""
    return resolve_voice_agent(request).stt


def resolve_tts(request: Request) -> Any:
    """Resolve TTS from the shared voice agent."""
    return resolve_voice_agent(request).tts
