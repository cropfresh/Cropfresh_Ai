"""
Voice Agent Package.

Re-exports for backward compatibility with code importing from
src.agents.voice_agent.
"""

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.models import VoiceResponse, VoiceSession
from src.agents.voice.templates import REQUIRED_FIELDS, RESPONSE_TEMPLATES

__all__ = [
    "VoiceAgent",
    "VoiceResponse",
    "VoiceSession",
    "RESPONSE_TEMPLATES",
    "REQUIRED_FIELDS",
]
