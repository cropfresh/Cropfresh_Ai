from datetime import date, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routes.prices import router
from src.rates.enums import AuthorityTier, FetchMode, RateKind
from src.rates.models import (
    CanonicalRate,
    MultiSourceRateResult,
    PendingSource,
    SourceHealthSnapshot,
)


class FakeRateService:
    def __init__(self):
        self.queries = []

    async def query(self, query):
        self.queries.append(query)
        return MultiSourceRateResult(
            query_target=query.model_dump(mode="json"),
            canonical_rates=[
                CanonicalRate(
                    rate_kind=RateKind.MANDI_WHOLESALE,
                    source="krama_daily",
                    authority_tier=AuthorityTier.OFFICIAL,
                    commodity="tomato",
                    location_label="Kolar",
                    price_date=date(2026, 3, 17),
                    unit="INR/quintal",
                    modal_price=1200.0,
                    comparison_count=2,
                )
            ],
            comparison_quotes=[],
            source_health=self.get_source_health(),
            warnings=[],
            pending_sources=[
                PendingSource(
                    source="enam_official_api",
                    rate_kind=RateKind.MANDI_WHOLESALE,
                    reason="pending access",
                    source_url="https://example.com/enam",
                )
            ],
            fetched_at=datetime(2026, 3, 17, 10, 0, 0),
        )

    def get_source_health(self):
        return [
            SourceHealthSnapshot(
                source="krama_daily",
                supports_live=True,
                fetch_mode=FetchMode.LIVE,
                last_successful_fetch=datetime(2026, 3, 17, 10, 0, 0),
            )
        ]


@pytest.mark.asyncio
async def test_prices_query_endpoint_returns_multi_source_payload(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.redis = None
    app.state.llm = None
    fake_service = FakeRateService()

    async def fake_get_rate_service(**kwargs):
        return fake_service

    monkeypatch.setattr("src.api.routes.prices.get_rate_service", fake_get_rate_service)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/prices/query",
            json={"rate_kinds": ["mandi_wholesale"], "commodity": "tomato", "market": "Kolar"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["canonical_rates"][0]["source"] == "krama_daily"
    assert fake_service.queries[0].commodity == "tomato"


@pytest.mark.asyncio
async def test_prices_source_health_endpoint_exposes_sources(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.redis = None
    app.state.llm = None
    fake_service = FakeRateService()

    async def fake_get_rate_service(**kwargs):
        return fake_service

    monkeypatch.setattr("src.api.routes.prices.get_rate_service", fake_get_rate_service)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/prices/source-health")

    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["source"] == "krama_daily"
    assert body["pending_sources"]
