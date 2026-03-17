from datetime import date, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.agents.adcl.models import WeeklyReport
from src.api.routes.adcl import router


class FakeADCLService:
    def __init__(self):
        self.calls = []

    async def generate_weekly_report(self, district: str, force_live: bool = False, farmer_id=None, language=None):
        self.calls.append((district, force_live, farmer_id, language))
        return WeeklyReport(
            week_start=date(2026, 3, 16),
            district=district,
            generated_at=datetime(2026, 3, 17, 10, 0, 0),
            crops=[],
            freshness={"generated_at": "2026-03-17T10:00:00Z"},
            source_health={"orders": {"status": "healthy"}},
        )


@pytest.mark.asyncio
async def test_adcl_weekly_route_returns_canonical_payload() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.adcl_service = FakeADCLService()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/adcl/weekly", params={"district": "Kolar", "force_live": "true"})

    assert response.status_code == 200
    body = response.json()
    assert body["district"] == "Kolar"
    assert "freshness" in body
    assert "source_health" in body
    assert app.state.adcl_service.calls[0][:2] == ("Kolar", True)
