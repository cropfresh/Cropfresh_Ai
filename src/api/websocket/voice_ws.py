"""
WebSocket Voice Endpoint — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.api.websocket.voice_pkg` package.
! Import from `src.api.websocket.voice_pkg` directly in new code.
"""

from src.api.websocket.voice_pkg.router import router
from src.api.websocket.voice_pkg.session import (
    MessageType,
    SessionManager,
    VoiceSession,
)

__all__ = [
    "MessageType",
    "SessionManager",
    "VoiceSession",
    "router",
]
