"""Unit tests for the Sprint 08 VAD segmenter and runtime."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = ROOT / "services" / "vad-service"

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import VadServiceSettings
from app.models import SegmentState, SegmenterSettings
from app.runtime import VadServiceRuntime
from app.segmenter import StreamingVadSegmenter, compute_normalized_rms


class FakeScorer:
    """Deterministic scorer used to verify runtime behavior."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls = 0

    def score_pcm16(self, pcm16: bytes, sample_rate: int) -> float:
        self.calls += 1
        return self.probability


def build_segmenter() -> StreamingVadSegmenter:
    """Create a segmenter with Sprint 08 settings."""
    return StreamingVadSegmenter(
        SegmenterSettings(
            sample_rate=16000,
            frame_samples=512,
            rms_threshold=0.015,
            speech_onset_threshold=0.5,
            speech_offset_threshold=0.35,
            min_speech_ms=250,
            silence_padding_ms=300,
        )
    )


def test_click_noise_shorter_than_minimum_speech_is_rejected() -> None:
    """Short acoustic spikes should never advance the segmenter into speech."""
    segmenter = build_segmenter()
    events = [
        segmenter.process_frame(sequence=index, probability=0.7, rms=0.2)
        for index in range(1, 5)
    ]

    assert all(event.state is SegmentState.SILENCE for event in events)
    assert all(event.segment_id is None for event in events)


def test_segmenter_emits_speech_end_after_trailing_silence_padding() -> None:
    """A confirmed utterance should end only after the configured silence padding."""
    segmenter = build_segmenter()

    results = [
        segmenter.process_frame(sequence=index, probability=0.8, rms=0.25)
        for index in range(1, 9)
    ]
    silence_results = [
        segmenter.process_frame(sequence=8 + index, probability=0.1, rms=0.01)
        for index in range(1, 11)
    ]

    assert results[-1].state is SegmentState.SPEECH_START
    assert results[-1].segment_id is not None
    assert all(result.state is SegmentState.SPEECH for result in silence_results[:-1])
    assert silence_results[-1].state is SegmentState.SPEECH_END
    assert silence_results[-1].end_of_segment is True


def test_runtime_skips_scorer_for_quiet_frames_before_speech() -> None:
    """The energy gate should avoid Silero work while the stream is still idle."""
    quiet_pcm = (b"\x00\x00" * 512)
    scorer = FakeScorer(probability=0.8)
    runtime = VadServiceRuntime(VadServiceSettings(), scorer=scorer)
    segmenter = runtime.create_segmenter()

    result = runtime.analyze_frame(
        segmenter=segmenter,
        sequence=1,
        sample_rate=16000,
        pcm16=quiet_pcm,
    )

    assert result.state is SegmentState.SILENCE
    assert scorer.calls == 0
    assert compute_normalized_rms(quiet_pcm) == 0.0
