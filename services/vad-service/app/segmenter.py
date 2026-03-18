"""Streaming acoustic segmentation for the Sprint 08 VAD service."""

from __future__ import annotations

from math import ceil
from uuid import uuid4

import numpy as np

from .models import SegmentState, SegmenterSettings, VadFrameResult


def compute_normalized_rms(pcm16: bytes) -> float:
    """Return normalized RMS for little-endian PCM16 audio."""
    if not pcm16:
        return 0.0

    samples = np.frombuffer(pcm16[: len(pcm16) - (len(pcm16) % 2)], dtype=np.int16)
    if samples.size == 0:
        return 0.0

    normalized = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(normalized * normalized)))


class StreamingVadSegmenter:
    """Dual-threshold acoustic segmenter aligned with the Sprint 08 contract."""

    def __init__(self, settings: SegmenterSettings) -> None:
        self.settings = settings
        self._candidate_speech_frames = 0
        self._trailing_silence_frames = 0
        self._is_speaking = False
        self._active_segment_id: str | None = None

        self._min_speech_frames = ceil(settings.min_speech_ms / settings.frame_duration_ms)
        self._silence_padding_frames = ceil(
            settings.silence_padding_ms / settings.frame_duration_ms
        )

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def process_frame(self, sequence: int, probability: float, rms: float) -> VadFrameResult:
        """Advance the segmenter by one frame using pre-computed probability and RMS."""
        probability = max(0.0, min(1.0, probability))

        if not self._is_speaking:
            if probability >= self.settings.speech_onset_threshold:
                self._candidate_speech_frames += 1
                if self._candidate_speech_frames >= self._min_speech_frames:
                    self._is_speaking = True
                    self._trailing_silence_frames = 0
                    self._active_segment_id = self._active_segment_id or str(uuid4())
                    return VadFrameResult(
                        sequence=sequence,
                        state=SegmentState.SPEECH_START,
                        probability=probability,
                        rms=rms,
                        end_of_segment=False,
                        segment_id=self._active_segment_id,
                    )
            else:
                self._candidate_speech_frames = 0

            return VadFrameResult(
                sequence=sequence,
                state=SegmentState.SILENCE,
                probability=probability,
                rms=rms,
                end_of_segment=False,
                segment_id=None,
            )

        if probability >= self.settings.speech_offset_threshold:
            self._trailing_silence_frames = 0
            return VadFrameResult(
                sequence=sequence,
                state=SegmentState.SPEECH,
                probability=probability,
                rms=rms,
                end_of_segment=False,
                segment_id=self._active_segment_id,
            )

        self._trailing_silence_frames += 1
        if self._trailing_silence_frames >= self._silence_padding_frames:
            completed_segment_id = self._active_segment_id
            self._candidate_speech_frames = 0
            self._trailing_silence_frames = 0
            self._is_speaking = False
            self._active_segment_id = None
            return VadFrameResult(
                sequence=sequence,
                state=SegmentState.SPEECH_END,
                probability=probability,
                rms=rms,
                end_of_segment=True,
                segment_id=completed_segment_id,
            )

        return VadFrameResult(
            sequence=sequence,
            state=SegmentState.SPEECH,
            probability=probability,
            rms=rms,
            end_of_segment=False,
            segment_id=self._active_segment_id,
        )
