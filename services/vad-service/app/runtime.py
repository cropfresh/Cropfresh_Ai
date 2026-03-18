"""Runtime wiring for the Sprint 08 VAD service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.shared.logger import setup_logger

from .config import VadServiceSettings
from .models import SegmenterSettings, VadFrameResult
from .segmenter import StreamingVadSegmenter, compute_normalized_rms
from .silero_engine import SileroOnnxScorer


class VadProbabilityScorer(Protocol):
    """Protocol for frame scorers used by the runtime."""

    def score_pcm16(self, pcm16: bytes, sample_rate: int) -> float:
        """Return a speech probability for one PCM16 frame."""


@dataclass(slots=True)
class RuntimeStatus:
    """Serializable service health state."""

    ready: bool
    bootstrap_error: str | None


class VadServiceRuntime:
    """Shared runtime for both the HTTP and gRPC surfaces."""

    def __init__(
        self,
        settings: VadServiceSettings,
        scorer: VadProbabilityScorer | None = None,
    ) -> None:
        self.settings = settings
        self.logger = setup_logger(settings.log_level)
        self.scorer = scorer
        self.bootstrap_error: str | None = None

    async def bootstrap(self) -> None:
        """Attempt to load the Silero scorer unless a test scorer was injected."""
        if self.scorer is not None:
            return

        scorer = SileroOnnxScorer(self.settings)
        try:
            await scorer.load()
            self.scorer = scorer
            self.bootstrap_error = None
        except Exception as exc:  # noqa: BLE001 - readiness should degrade instead of crashing startup
            self.bootstrap_error = str(exc)
            self.logger.warning("VAD service started in degraded mode: {}", exc)

    def create_segmenter(self) -> StreamingVadSegmenter:
        """Create an isolated segmenter for one inbound audio stream."""
        return StreamingVadSegmenter(
            SegmenterSettings(
                sample_rate=self.settings.sample_rate,
                frame_samples=self.settings.frame_samples,
                rms_threshold=self.settings.rms_threshold,
                speech_onset_threshold=self.settings.speech_onset_threshold,
                speech_offset_threshold=self.settings.speech_offset_threshold,
                min_speech_ms=self.settings.min_speech_ms,
                silence_padding_ms=self.settings.silence_padding_ms,
            )
        )

    def analyze_frame(
        self,
        *,
        segmenter: StreamingVadSegmenter,
        sequence: int,
        sample_rate: int,
        pcm16: bytes,
    ) -> VadFrameResult:
        """Apply the energy gate, score the frame, and update the stream segmenter."""
        if sample_rate != self.settings.sample_rate:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        rms = compute_normalized_rms(pcm16)
        if rms < self.settings.rms_threshold and not segmenter.is_speaking:
            probability = 0.0
        else:
            if self.scorer is None:
                raise RuntimeError("Silero scorer is not ready")
            probability = self.scorer.score_pcm16(pcm16, sample_rate)

        return segmenter.process_frame(sequence=sequence, probability=probability, rms=rms)

    def health_payload(self) -> dict[str, object]:
        """Return a liveness payload that stays truthful even in degraded mode."""
        return {
            "status": "healthy",
            "service": "vad-service",
            "version": self.settings.service_version,
            "grpc_enabled": self.settings.enable_grpc,
            "bootstrap_error": self.bootstrap_error,
        }

    def readiness(self) -> RuntimeStatus:
        """Return the current readiness state for probes and tests."""
        return RuntimeStatus(ready=self.scorer is not None, bootstrap_error=self.bootstrap_error)
