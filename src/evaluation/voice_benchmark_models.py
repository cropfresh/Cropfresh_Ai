"""Models for the Sprint 09 multilingual voice benchmark."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

EndpointingExpectation = Literal["flush", "hold_then_flush"]


class VoiceBenchmarkEntry(BaseModel):
    """One fixed utterance used for Sprint 09 voice evaluation."""

    id: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=8)
    prompt_text: str = Field(min_length=1)
    expected_endpointing: EndpointingExpectation
    expected_notes: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class VoiceBenchmarkObservation(BaseModel):
    """Observed runtime metrics captured during a live or manual benchmark run."""

    benchmark_id: str = Field(min_length=1)
    first_audio_ms: int | None = Field(default=None, ge=0)
    bargein_reaction_ms: int | None = Field(default=None, ge=0)
    interruption_recovery_ms: int | None = Field(default=None, ge=0)
    notes: str | None = None


class VoiceBenchmarkArtifact(BaseModel):
    """Per-turn artifact written after evaluating one benchmark utterance."""

    benchmark_id: str
    language: str
    prompt_text: str
    expected_endpointing: EndpointingExpectation
    actual_endpointing: EndpointingExpectation
    matched: bool
    decision_reason: str
    semantic_hold_ms: int
    first_audio_ms: int | None = None
    bargein_reaction_ms: int | None = None
    interruption_recovery_ms: int | None = None
    timing_targets_met: bool | None = None
    expected_notes: str
    observation_notes: str | None = None


class VoiceBenchmarkReport(BaseModel):
    """Summary plus per-turn artifacts for a benchmark run."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_path: str
    rubric_path: str
    total_cases: int
    matched_cases: int
    languages: list[str]
    artifacts: list[VoiceBenchmarkArtifact]
