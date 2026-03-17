"""
Duplex Pipeline Models
======================
Data structures and enums for the duplex voice pipeline.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class PipelineState(str, Enum):
    """Current state of the duplex pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class PipelineEvent:
    """Event emitted by the pipeline for UI updates."""
    state: PipelineState
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioOutputChunk:
    """Chunk of synthesized audio to send to the client."""
    audio_base64: str
    format: str = "mp3"
    sample_rate: int = 24000
    chunk_index: int = 0
    is_last: bool = False
    text: str = ""  # The sentence that was synthesized
    timing: dict[str, float | None] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    transcription: str = ""
    language: str = "en"
    response_text: str = ""
    audio_chunks_sent: int = 0
    was_interrupted: bool = False
    latency_ms: float = 0.0
    timing: dict[str, float | None] = field(default_factory=dict)
