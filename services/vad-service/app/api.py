"""FastAPI surface for the Sprint 08 VAD service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import VadServiceSettings
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
            "grpc_enabled": settings.enable_grpc,
        }

    return app
