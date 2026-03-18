"""API tests for the Sprint 08 VAD service health and readiness routes."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = ROOT / "services" / "vad-service"

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api import create_app
from app.config import VadServiceSettings
from app.runtime import VadServiceRuntime


class FailingRuntime(VadServiceRuntime):
    """Runtime test double that forces degraded readiness."""

    async def bootstrap(self) -> None:
        self.scorer = None
        self.bootstrap_error = "missing model"


def test_ready_returns_503_until_scorer_is_bootstrapped() -> None:
    """Readiness should stay false while the model artifact is missing."""
    runtime = FailingRuntime(VadServiceSettings())
    app = create_app(runtime)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_health_and_config_routes_expose_service_contract() -> None:
    """Health and config should stay available even when readiness is degraded."""
    runtime = FailingRuntime(VadServiceSettings())
    app = create_app(runtime)

    with TestClient(app) as client:
        health_response = client.get("/health")
        config_response = client.get("/v1/vad/config")

    assert health_response.status_code == 200
    assert health_response.json()["service"] == "vad-service"
    assert config_response.status_code == 200
    assert config_response.json()["frame_samples"] == 512
