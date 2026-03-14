"""
Voice agent data models — session and response.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VoiceSession:
    """Voice conversation session"""
    session_id: str
    user_id: str
    language: str
    history: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def add_turn(self, user_text: str, bot_response: str):
        """Add a conversation turn"""
        self.history.append({
            "user": user_text,
            "bot": bot_response,
        })
        # Keep last 5 turns
        if len(self.history) > 5:
            self.history = self.history[-5:]


@dataclass
class VoiceResponse:
    """Complete voice agent response"""
    # Transcription
    transcription: str
    detected_language: str

    # Intent & Entities
    intent: str
    entities: dict[str, Any]

    # Response
    response_text: str
    response_audio: bytes

    # Metadata
    session_id: str
    confidence: float

    @property
    def is_successful(self) -> bool:
        return len(self.transcription) > 0 and len(self.response_audio) > 0
