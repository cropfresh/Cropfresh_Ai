"""
WebRTC Transport Layer (Proxy)
==============================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.voice.webrtc`.
"""

from src.voice.webrtc import (
    ConnectionState,
    WebRTCConfig,
    AudioChunk,
    AudioReceiveTrack,
    AudioSendTrack,
    AIORTC_AVAILABLE,
    WebRTCTransport,
    WebRTCSignaling,
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
