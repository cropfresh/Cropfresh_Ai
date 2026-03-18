from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routers.vision import router


class FakeQualityAgent:
    def __init__(self, fallback_mode: bool = False):
        self.vision_pipeline = SimpleNamespace(
            fallback_mode=fallback_mode,
            model_dir="models/vision/",
        )
        self.calls = []

    async def assess(
        self,
        listing_id: str,
        commodity: str,
        description: str = "",
        image_b64: str | None = None,
        require_upgrade_review: bool = False,
    ):
        self.calls.append(
            {
                "listing_id": listing_id,
                "commodity": commodity,
                "description": description,
                "image_b64": image_b64,
                "require_upgrade_review": require_upgrade_review,
            }
        )
        assessment = SimpleNamespace(
            listing_id=listing_id,
            commodity=commodity,
            grade="A",
            confidence=0.91,
            defects_detected=["bruise"],
            defect_count=1,
            hitl_required=False,
            shelf_life_days=5,
            assessment_id="qa-123",
        )
        return SimpleNamespace(
            assessment=assessment,
            method="vision" if not self.vision_pipeline.fallback_mode else "rule_based",
            digital_twin_linked=True,
        )


@pytest.mark.asyncio
async def test_vision_health_reports_service_state() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.listing_service = SimpleNamespace(quality_agent=FakeQualityAgent())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/vision/health")

    assert response.status_code == 200
    body = response.json()
    assert body["service_ready"] is True
    assert body["vision_ready"] is True
    assert body["assessment_mode"] == "vision"


@pytest.mark.asyncio
async def test_vision_assess_returns_ui_contract() -> None:
    app = FastAPI()
    app.include_router(router)
    fake_agent = FakeQualityAgent()
    app.state.listing_service = SimpleNamespace(quality_agent=fake_agent)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/vision/assess",
            json={
                "listing_id": "lst-42",
                "commodity": "Tomato",
                "description": "Fresh and firm",
                "image_b64": "aGVsbG8=",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["listing_id"] == "lst-42"
    assert body["assessment_mode"] == "vision"
    assert body["grade_attach_preview"]["grade"] == "A"
    assert fake_agent.calls[0]["commodity"] == "Tomato"


@pytest.mark.asyncio
async def test_vision_assess_returns_503_when_agent_missing() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.listing_service = SimpleNamespace(quality_agent=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/vision/assess", json={"commodity": "Tomato"})

    assert response.status_code == 503
