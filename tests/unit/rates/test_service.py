from datetime import date, datetime

import pytest

from src.rates.enums import AuthorityTier, FetchMode, RateKind
from src.rates.models import NormalizedRateRecord, SourceHealthSnapshot
from src.rates.query_builder import normalize_rate_query
from src.rates.service import RateService


class FakeConnector:
    def __init__(self, source_id: str, price_value: float = 101.5, should_fail: bool = False):
        self.source_id = source_id
        self.rate_kind = RateKind.FUEL
        self.authority_tier = AuthorityTier.REFERENCE_OFFICIAL
        self.fetch_mode = FetchMode.LIVE
        self.ttl_minutes = 60
        self.supports_live = True
        self.uses_browser = False
        self.calls = 0
        self._price_value = price_value
        self._should_fail = should_fail

    async def fetch(self, query):
        self.calls += 1
        if self._should_fail:
            raise RuntimeError(f"{self.source_id} unavailable")
        return [
            NormalizedRateRecord(
                rate_kind=RateKind.FUEL,
                commodity="petrol",
                state=query.state,
                location_label=query.state,
                price_date=query.target_date,
                unit="INR/litre",
                price_value=self._price_value,
                modal_price=self._price_value,
                source=self.source_id,
                authority_tier=self.authority_tier,
                source_url=f"https://example.com/{self.source_id}",
                fetched_at=datetime.utcnow(),
            )
        ]


def _service_with_connectors(connectors: list[FakeConnector]) -> RateService:
    service = RateService(repository=None)
    service.connectors = connectors
    service._health = {
        connector.source_id: SourceHealthSnapshot(
            source=connector.source_id,
            supports_live=connector.supports_live,
            fetch_mode=connector.fetch_mode,
        )
        for connector in connectors
    }
    return service


@pytest.mark.asyncio
async def test_rate_service_uses_cache_between_queries() -> None:
    connector = FakeConnector("petroldieselprice")
    service = _service_with_connectors([connector])
    query = normalize_rate_query(rate_kinds=["fuel"], state="Karnataka", date=date(2026, 3, 17))

    first = await service.query(query)
    second = await service.query(query)

    assert connector.calls == 1
    assert first.canonical_rates[0].source == "petroldieselprice"
    assert second.canonical_rates[0].source == "petroldieselprice"


@pytest.mark.asyncio
async def test_rate_service_opens_circuit_after_repeated_failures() -> None:
    connector = FakeConnector("petroldieselprice", should_fail=True)
    service = _service_with_connectors([connector])
    query = normalize_rate_query(rate_kinds=["fuel"], state="Karnataka", force_live=True)

    for _ in range(3):
        await service.query(query)

    health = service.get_source_health()[0]
    calls_before = connector.calls
    await service.query(query)

    assert health.status == "unavailable"
    assert health.circuit_open_until is not None
    assert connector.calls == calls_before
