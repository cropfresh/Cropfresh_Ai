"""Unit tests for duplex semantic endpointing rules."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.voice.semantic_endpointing import evaluate_semantic_flush, is_likely_incomplete


class StubSemanticProvider:
    def __init__(self, content: str) -> None:
        self.content = content

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        del messages, kwargs
        return SimpleNamespace(content=self.content)


@pytest.mark.asyncio
async def test_semantic_endpointing_holds_known_filler_phrase() -> None:
    decision = await evaluate_semantic_flush(
        transcript="one second",
        language="en",
        llm_provider=None,
        enabled=True,
        timeout_ms=150,
        max_hold_ms=800,
    )

    assert decision.should_flush is False
    assert decision.reason == "heuristic_hold"


def test_semantic_incomplete_heuristic_handles_kannada_pause_phrase() -> None:
    assert is_likely_incomplete("ಒಂದು ನಿಮಿಷ", "kn") is True


@pytest.mark.asyncio
async def test_semantic_endpointing_flushes_when_llm_marks_complete() -> None:
    decision = await evaluate_semantic_flush(
        transcript="what is tomato price in kolar today",
        language="en",
        llm_provider=StubSemanticProvider("COMPLETE"),
        enabled=True,
        timeout_ms=150,
        max_hold_ms=800,
    )

    assert decision.should_flush is True
    assert decision.reason == "llm_flush"
    assert decision.used_llm is True


@pytest.mark.asyncio
async def test_semantic_endpointing_holds_when_llm_marks_incomplete() -> None:
    decision = await evaluate_semantic_flush(
        transcript="tomato price and",
        language="en",
        llm_provider=StubSemanticProvider("HOLD"),
        enabled=True,
        timeout_ms=150,
        max_hold_ms=800,
    )

    assert decision.should_flush is False
    assert decision.reason == "heuristic_hold"
