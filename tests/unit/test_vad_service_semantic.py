# ruff: noqa: E402

"""Focused tests for Sprint 09 semantic endpointing in the VAD service."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = ROOT / "services" / "vad-service"

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import VadServiceSettings
from app.runtime import VadServiceRuntime


class SilentScorer:
    """Protocol-compatible scorer placeholder for runtime construction."""

    def score_pcm16(self, pcm16: bytes, sample_rate: int) -> float:
        del pcm16, sample_rate
        return 0.0


class FakeClock:
    """Mutable monotonic clock for semantic hold tests."""

    def __init__(self) -> None:
        self.current = 0.0

    def advance(self, seconds: float) -> None:
        self.current += seconds

    def __call__(self) -> float:
        return self.current


@pytest.mark.asyncio
async def test_semantic_segment_holds_known_pause_phrase_until_more_audio_arrives() -> None:
    clock = FakeClock()
    settings = VadServiceSettings(
        semantic_endpointing_enabled=True,
        semantic_hold_max_ms=800,
    )
    runtime = VadServiceRuntime(settings, scorer=SilentScorer(), clock=clock)

    first = await runtime.evaluate_segment(
        session_id="session-1",
        transcript="one second",
        language="en",
    )
    clock.advance(0.2)
    second = await runtime.evaluate_segment(
        session_id="session-1",
        transcript="one second",
        language="en",
    )

    assert first.should_flush is False
    assert first.reason == "heuristic_hold"
    assert second.should_flush is False
    assert second.semantic_hold_ms >= 200


@pytest.mark.asyncio
async def test_semantic_segment_forces_flush_after_hold_timeout() -> None:
    clock = FakeClock()
    settings = VadServiceSettings(
        semantic_endpointing_enabled=True,
        semantic_hold_max_ms=300,
    )
    runtime = VadServiceRuntime(settings, scorer=SilentScorer(), clock=clock)

    _ = await runtime.evaluate_segment(
        session_id="session-2",
        transcript="one second",
        language="en",
    )
    clock.advance(0.35)
    decision = await runtime.evaluate_segment(
        session_id="session-2",
        transcript="one second",
        language="en",
    )

    assert decision.should_flush is True
    assert decision.reason == "semantic_hold_timeout"
    assert decision.semantic_hold_ms >= 300
