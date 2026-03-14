"""
Voice WebSocket Package — Re-exports for backward compatibility.
"""

from src.api.websocket.voice_pkg.session import (
    MessageType,
    SessionManager,
    VoiceSession,
)
from src.api.websocket.voice_pkg.duplex import process_duplex_speech
from src.api.websocket.voice_pkg.router import router, get_sessions

__all__ = [
    "MessageType",
    "SessionManager",
    "VoiceSession",
    "process_duplex_speech",
    "router",
    "get_sessions",
]
