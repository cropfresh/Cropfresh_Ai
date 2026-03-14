"""
WebRTC Transport Layer (Proxy)
==============================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.voice.webrtc`.
"""

from src.voice.webrtc import (
    AIORTC_AVAILABLE,
    AudioChunk,
    AudioReceiveTrack,
    AudioSendTrack,
    ConnectionState,
    WebRTCConfig,
    WebRTCSignaling,
    WebRTCTransport,
)

__all__ = [
    "ConnectionState",
    "WebRTCConfig",
    "AudioChunk",
    "AudioReceiveTrack",
    "AudioSendTrack",
    "AIORTC_AVAILABLE",
    "WebRTCTransport",
    "WebRTCSignaling",
]
