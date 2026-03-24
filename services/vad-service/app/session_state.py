"""Session-scoped runtime state for the VAD service."""

from __future__ import annotations

from dataclasses import dataclass

from .segmenter import StreamingVadSegmenter


@dataclass(slots=True)
class VadSessionState:
    """Per-session state for acoustic and semantic endpointing."""

    segmenter: StreamingVadSegmenter
    semantic_hold_started_at: float | None = None
