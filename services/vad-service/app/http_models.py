"""HTTP request and response models for the Sprint 08 VAD service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeFrameRequest(BaseModel):
    """One PCM16 frame submitted over the HTTP compatibility surface."""

    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    pcm16_base64: str = Field(min_length=1)


class AnalyzeFrameResponse(BaseModel):
    """Serialized segmenter output for one analyzed frame."""

    session_id: str
    sequence: int
    state: str
    probability: float
    rms: float
    segment_id: str | None
    end_of_segment: bool


class EvaluateSegmentRequest(BaseModel):
    """Semantic completeness request for one acoustically-finished segment."""

    session_id: str = Field(min_length=1)
    transcript: str = Field(min_length=1)
    language: str = Field(min_length=1, max_length=8)


class EvaluateSegmentResponse(BaseModel):
    """Serialized semantic hold/flush decision for one session."""

    session_id: str
    transcript: str
    language: str
    should_flush: bool
    reason: str
    semantic_hold_ms: int
    used_llm: bool
    timed_out: bool


class ResetSessionResponse(BaseModel):
    """Result for clearing one in-memory stream session."""

    session_id: str
    cleared: bool
