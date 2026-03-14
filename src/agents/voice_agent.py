"""
VoiceAgent — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.agents.voice` package.
! Import from `src.agents.voice` directly in new code.
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
