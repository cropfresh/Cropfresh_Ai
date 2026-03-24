# ruff: noqa: E402

"""API tests for the Sprint 08 VAD service health and readiness routes."""

from __future__ import annotations

import sys
from base64 import b64encode
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = ROOT / "services" / "vad-service"

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api import create_app
from app.config import VadServiceSettings
from app.runtime import VadServiceRuntime


class FixedScorer:
    """Deterministic scorer used for HTTP contract tests."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def score_pcm16(self, pcm16: bytes, sample_rate: int) -> float:
        return self.probability


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


def test_analyze_route_streams_session_state_and_reset_endpoint_clears_it() -> None:
    """HTTP analyze should expose the session-scoped segmenter and reset support."""
    runtime = VadServiceRuntime(VadServiceSettings(), scorer=FixedScorer(probability=0.8))
    app = create_app(runtime)
    pcm16_base64 = b64encode(b"\x01\x10" * 512).decode("ascii")

    with TestClient(app) as client:
        last_response = None
        for sequence in range(1, 9):
            last_response = client.post(
                "/v1/vad/analyze",
                json={
                    "session_id": "bridge-session",
                    "sequence": sequence,
                    "sample_rate": 16000,
                    "pcm16_base64": pcm16_base64,
                },
            )
        reset_response = client.delete("/v1/vad/sessions/bridge-session")

    assert last_response is not None
    assert last_response.status_code == 200
    assert last_response.json()["state"] == "speech_start"
    assert last_response.json()["segment_id"]
    assert reset_response.status_code == 200
    assert reset_response.json()["cleared"] is True


def test_analyze_route_rejects_invalid_base64_payloads() -> None:
    """The HTTP compatibility surface should validate PCM payload encoding."""
    runtime = VadServiceRuntime(VadServiceSettings(), scorer=FixedScorer(probability=0.8))
    app = create_app(runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/vad/analyze",
            json={
                "session_id": "bad-frame",
                "sequence": 1,
                "sample_rate": 16000,
                "pcm16_base64": "%%%not-base64%%%",
            },
        )

    assert response.status_code == 400
    assert "base64" in response.json()["detail"]


def test_semantic_segment_endpoint_holds_filler_pause_when_feature_flag_is_enabled() -> None:
    """Semantic endpointing should expose pause-aware hold decisions over HTTP."""
    runtime = VadServiceRuntime(
        VadServiceSettings(
            semantic_endpointing_enabled=True,
        ),
        scorer=FixedScorer(probability=0.8),
    )
    app = create_app(runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/vad/segments/evaluate",
            json={
                "session_id": "semantic-session",
                "transcript": "one second",
                "language": "en",
            },
        )

    assert response.status_code == 200
    assert response.json()["should_flush"] is False
    assert response.json()["reason"] == "heuristic_hold"
