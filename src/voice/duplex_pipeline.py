"""
Duplex Pipeline Proxy
=====================
This proxy preserves existing imports after the duplex pipeline
was extracted into `src.voice.duplex` modules.
"""

from src.voice.duplex import (
    AudioOutputChunk,
    DuplexPipeline,
    PipelineEvent,
    PipelineResult,
    PipelineState,
)

__all__ = [
    "PipelineState",
    "PipelineEvent",
    "AudioOutputChunk",
    "PipelineResult",
    "DuplexPipeline",
]
