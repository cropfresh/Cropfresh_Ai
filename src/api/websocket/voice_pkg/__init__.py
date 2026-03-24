"""
Voice WebSocket Package — Re-exports for backward compatibility.
"""

from src.api.websocket.voice_pkg.duplex import (
    process_duplex_speech,
    process_duplex_text_response,
)
from src.api.websocket.voice_pkg.router import get_sessions, router
from src.api.websocket.voice_pkg.session import (
    MessageType,
    SessionManager,
    VoiceSession,
)

__all__ = [
    "MessageType",
    "SessionManager",
    "VoiceSession",
    "process_duplex_speech",
    "process_duplex_text_response",
    "router",
    "get_sessions",
]
