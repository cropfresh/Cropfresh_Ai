"""
WebRTC Transport Models
=======================
Data structures and enumerations for WebRTC streaming.
"""

from dataclasses import dataclass, field
from enum import Enum


class ConnectionState(Enum):
    """WebRTC connection states"""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class WebRTCConfig:
    """WebRTC configuration"""
    ice_servers: list[dict] = field(default_factory=lambda: [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ])
    audio_codec: str = "opus"
    sample_rate: int = 16000
    channels: int = 1
    enable_dtx: bool = True  # Discontinuous transmission


@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: bytes
    sample_rate: int
    channels: int
    timestamp_ms: float
    samples: int
