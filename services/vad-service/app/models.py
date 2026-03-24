"""Typed VAD models for the Sprint 08 service."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentState(str, Enum):
    """Streaming states emitted to downstream consumers."""

    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


@dataclass(slots=True)
class SegmenterSettings:
    """Thresholds and timing rules for the streaming segmenter."""

    sample_rate: int
    frame_samples: int
    rms_threshold: float
    speech_onset_threshold: float
    speech_offset_threshold: float
    min_speech_ms: int
    silence_padding_ms: int

    @property
    def frame_duration_ms(self) -> float:
        return self.frame_samples * 1000 / self.sample_rate


@dataclass(slots=True)
class VadFrameResult:
    """One segmenter decision for one incoming audio frame."""

    sequence: int
    state: SegmentState
    probability: float
    rms: float
    end_of_segment: bool
    segment_id: str | None
