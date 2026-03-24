"""FastAPI surface for the Sprint 08 VAD service."""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from .config import VadServiceSettings
from .http_models import (
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    EvaluateSegmentRequest,
    EvaluateSegmentResponse,
    ResetSessionResponse,
)
from .runtime import VadServiceRuntime


def create_app(runtime: VadServiceRuntime | None = None) -> FastAPI:
    """Create the FastAPI app for health, readiness, and config inspection."""
    settings = VadServiceSettings()
    service_runtime = runtime or VadServiceRuntime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service_runtime.bootstrap()
        yield

    app = FastAPI(
        title="CropFresh VAD Service",
        version=settings.service_version,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return service_runtime.health_payload()

    @app.get("/ready")
    async def ready() -> JSONResponse:
        readiness = service_runtime.readiness()
        payload = {
            "status": "ready" if readiness.ready else "not_ready",
            "ready": readiness.ready,
            "bootstrap_error": readiness.bootstrap_error,
        }
        return JSONResponse(status_code=200 if readiness.ready else 503, content=payload)

    @app.get("/v1/vad/config")
    async def config() -> dict[str, object]:
        return {
            "sample_rate": settings.sample_rate,
            "frame_samples": settings.frame_samples,
            "rms_threshold": settings.rms_threshold,
            "speech_onset_threshold": settings.speech_onset_threshold,
            "speech_offset_threshold": settings.speech_offset_threshold,
            "min_speech_ms": settings.min_speech_ms,
            "silence_padding_ms": settings.silence_padding_ms,
            "semantic_endpointing_enabled": settings.semantic_endpointing_enabled,
            "semantic_timeout_ms": settings.semantic_timeout_ms,
            "semantic_hold_max_ms": settings.semantic_hold_max_ms,
            "grpc_enabled": settings.enable_grpc,
        }

    @app.post("/v1/vad/analyze", response_model=AnalyzeFrameResponse)
    async def analyze_frame(payload: AnalyzeFrameRequest) -> AnalyzeFrameResponse:
        try:
            pcm16 = base64.b64decode(payload.pcm16_base64, validate=True)
        except binascii.Error as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pcm16_base64 must be valid base64-encoded PCM16 audio",
            ) from exc

        try:
            result = service_runtime.analyze_session_frame(
                session_id=payload.session_id,
                sequence=payload.sequence,
                sample_rate=payload.sample_rate,
                pcm16=pcm16,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        return AnalyzeFrameResponse(
            session_id=payload.session_id,
            sequence=result.sequence,
            state=result.state.value,
            probability=result.probability,
            rms=result.rms,
            segment_id=result.segment_id,
            end_of_segment=result.end_of_segment,
        )

    @app.post("/v1/vad/segments/evaluate", response_model=EvaluateSegmentResponse)
    async def evaluate_segment(payload: EvaluateSegmentRequest) -> EvaluateSegmentResponse:
        decision = await service_runtime.evaluate_segment(
            session_id=payload.session_id,
            transcript=payload.transcript,
            language=payload.language,
        )
        return EvaluateSegmentResponse(
            session_id=payload.session_id,
            transcript=decision.transcript,
            language=decision.detected_language,
            should_flush=decision.should_flush,
            reason=decision.reason,
            semantic_hold_ms=decision.semantic_hold_ms,
            used_llm=decision.used_llm,
            timed_out=decision.timed_out,
        )

    @app.delete("/v1/vad/sessions/{session_id}", response_model=ResetSessionResponse)
    async def reset_session(session_id: str) -> ResetSessionResponse:
        return ResetSessionResponse(
            session_id=session_id,
            cleared=service_runtime.reset_session(session_id),
        )

    return app
