"""Runtime wiring for the Sprint 08 VAD service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from src.shared.logger import setup_logger
from src.shared.voice_semantic import (
    SemanticEndpointDecision,
    SupportsGenerate,
    evaluate_semantic_flush,
)

from .config import VadServiceSettings
from .models import SegmenterSettings, VadFrameResult
from .segmenter import StreamingVadSegmenter, compute_normalized_rms
from .session_state import VadSessionState
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
        semantic_provider: SupportsGenerate | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.logger = setup_logger(settings.log_level)
        self.scorer = scorer
        self.semantic_provider = semantic_provider
        self.clock = clock or time.monotonic
        self.bootstrap_error: str | None = None
        self._sessions: dict[str, VadSessionState] = {}

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

    def get_session_state(self, session_id: str) -> VadSessionState:
        """Return the cached runtime state for one session, creating it on first use."""
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("session_id is required")

        session_state = self._sessions.get(normalized_session_id)
        if session_state is None:
            session_state = VadSessionState(segmenter=self.create_segmenter())
            self._sessions[normalized_session_id] = session_state
        return session_state

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

    def analyze_session_frame(
        self,
        *,
        session_id: str,
        sequence: int,
        sample_rate: int,
        pcm16: bytes,
    ) -> VadFrameResult:
        """Analyze one frame using the session-scoped segmenter cache."""
        session_state = self.get_session_state(session_id)
        return self.analyze_frame(
            segmenter=session_state.segmenter,
            sequence=sequence,
            sample_rate=sample_rate,
            pcm16=pcm16,
        )

    async def evaluate_segment(
        self,
        *,
        session_id: str,
        transcript: str,
        language: str,
    ) -> SemanticEndpointDecision:
        """Evaluate whether an acoustically-ended segment should flush downstream."""
        session_state = self.get_session_state(session_id)
        decision = await evaluate_semantic_flush(
            transcript=transcript,
            language=language,
            llm_provider=self.semantic_provider,
            enabled=self.settings.semantic_endpointing_enabled,
            timeout_ms=self.settings.semantic_timeout_ms,
            max_hold_ms=self.settings.semantic_hold_max_ms,
        )

        if decision.should_flush:
            session_state.semantic_hold_started_at = None
            return decision

        hold_started_at = session_state.semantic_hold_started_at
        if hold_started_at is None:
            hold_started_at = self.clock()
        session_state.semantic_hold_started_at = hold_started_at
        elapsed_ms = int((self.clock() - hold_started_at) * 1000)
        if elapsed_ms >= self.settings.semantic_hold_max_ms:
            session_state.semantic_hold_started_at = None
            return SemanticEndpointDecision(
                transcript=decision.transcript,
                detected_language=decision.detected_language,
                should_flush=True,
                reason="semantic_hold_timeout",
                semantic_hold_ms=elapsed_ms,
                used_llm=decision.used_llm,
                timed_out=decision.timed_out,
            )

        decision.semantic_hold_ms = elapsed_ms
        return decision

    def reset_session(self, session_id: str) -> bool:
        """Clear one cached segmenter and return whether it existed."""
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            return False

        return self._sessions.pop(normalized_session_id, None) is not None

    def health_payload(self) -> dict[str, object]:
        """Return a liveness payload that stays truthful even in degraded mode."""
        return {
            "status": "healthy",
            "service": "vad-service",
            "version": self.settings.service_version,
            "grpc_enabled": self.settings.enable_grpc,
            "bootstrap_error": self.bootstrap_error,
            "tracked_streams": len(self._sessions),
        }

    def readiness(self) -> RuntimeStatus:
        """Return the current readiness state for probes and tests."""
        return RuntimeStatus(ready=self.scorer is not None, bootstrap_error=self.bootstrap_error)
