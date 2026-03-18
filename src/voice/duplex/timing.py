"""
Duplex Turn Timing
==================
Helpers for recording stage timings in the duplex voice pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

TimingSnapshot = dict[str, float | None]


def _elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000, 2)


@dataclass
class TurnTiming:
    """Tracks coarse stage timings for a single duplex turn."""

    turn_started_at: float = field(default_factory=time.perf_counter)
    transcription_started_at: float | None = None
    transcription_completed_at: float | None = None
    thinking_started_at: float | None = None
    first_sentence_at: float | None = None
    speaking_started_at: float | None = None
    first_audio_at: float | None = None
    interrupt_requested_at: float | None = None
    interrupted_at: float | None = None
    completed_at: float | None = None

    def mark_transcription_started(self) -> None:
        self.transcription_started_at = self.transcription_started_at or time.perf_counter()

    def mark_transcription_completed(self) -> None:
        self.transcription_completed_at = time.perf_counter()

    def mark_thinking_started(self) -> None:
        self.thinking_started_at = self.thinking_started_at or time.perf_counter()

    def mark_first_sentence(self) -> None:
        self.first_sentence_at = self.first_sentence_at or time.perf_counter()

    def mark_speaking_started(self) -> None:
        self.speaking_started_at = self.speaking_started_at or time.perf_counter()

    def mark_first_audio(self) -> None:
        self.first_audio_at = self.first_audio_at or time.perf_counter()

    def mark_interrupt_requested(self, requested_at: float | None = None) -> None:
        self.interrupt_requested_at = self.interrupt_requested_at or requested_at or time.perf_counter()

    def mark_interrupted(self) -> None:
        self.interrupted_at = self.interrupted_at or time.perf_counter()

    def mark_completed(self) -> None:
        self.completed_at = time.perf_counter()

    def snapshot(self) -> TimingSnapshot:
        now = self.completed_at or time.perf_counter()
        return {
            "transcription_ms": _elapsed_ms(
                self.transcription_started_at,
                self.transcription_completed_at,
            ),
            "llm_first_sentence_ms": _elapsed_ms(
                self.thinking_started_at,
                self.first_sentence_at,
            ),
            "tts_first_audio_ms": _elapsed_ms(
                self.speaking_started_at,
                self.first_audio_at,
            ),
            "first_audio_ms": _elapsed_ms(
                self.turn_started_at,
                self.first_audio_at,
            ),
            "total_ms": _elapsed_ms(self.turn_started_at, now),
            "interrupted_ms": _elapsed_ms(
                self.turn_started_at,
                self.interrupted_at,
            ),
            "bargein_reaction_ms": _elapsed_ms(
                self.interrupt_requested_at,
                self.interrupted_at,
            ),
        }
