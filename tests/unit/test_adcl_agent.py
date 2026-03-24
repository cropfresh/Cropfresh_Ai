"""Focused unit tests for the canonical ADCL service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from src.agents.adcl.models import WeeklyReport
from src.agents.adcl.service import ADCLService


@dataclass
class FakeRate:
    modal_price: float
    unit: str = "INR/quintal"
    price_date: str = date.today().isoformat()

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {
            "modal_price": self.modal_price,
            "unit": self.unit,
            "price_date": self.price_date,
            "source": "krama_daily",
        }


@dataclass
class FakeHealth:
    status: str = "healthy"

    def model_dump(self, mode: str = "json") -> dict[str, str]:
        del mode
        return {"source": "krama_daily", "status": self.status}


class FakeRateResult:
    def __init__(self, price: float) -> None:
        self.canonical_rates = [FakeRate(modal_price=price)]


class FakeRateService:
    def __init__(self, price: float = 1800.0) -> None:
        self.price = price
        self.queries = []

    async def query(self, query):
        self.queries.append(query)
        return FakeRateResult(self.price)

    def get_source_health(self):
        return [FakeHealth()]


class FakeDB:
    def __init__(self, orders=None, history=None, cached_report=None) -> None:
        self.orders = orders or []
        self.history = history or {}
        self.cached_report = cached_report
        self.saved_reports: list[dict[str, object]] = []

    async def get_recent_orders(self, district: str, days: int = 90):
        del district, days
        return self.orders

    async def get_price_history(self, commodity: str, district: str, days: int = 30):
        del district, days
        return self.history.get(commodity, [])

    async def get_latest_adcl_report(self, district: str, week_start=None):
        del district, week_start
        return self.cached_report

    async def insert_adcl_report(self, report: dict[str, object]) -> None:
        self.saved_reports.append(report)
        self.cached_report = report


def _order(commodity: str, buyer_id: str, quantity: float, days_ago: int) -> dict[str, object]:
    return {
        "commodity": commodity,
        "buyer_id": buyer_id,
        "quantity_kg": quantity,
        "created_at": (date.today() - timedelta(days=days_ago)).isoformat(),
    }


def _price_history(commodity: str) -> dict[str, list[dict[str, object]]]:
    return {
        commodity: [
            {"commodity": commodity, "date": date.today().isoformat(), "modal_price": 1700.0},
            {"commodity": commodity, "date": (date.today() - timedelta(days=7)).isoformat(), "modal_price": 1600.0},
        ]
    }


@pytest.mark.asyncio
async def test_generate_weekly_report_returns_empty_live_only_report_when_no_orders() -> None:
    db = FakeDB()
    service = ADCLService(db=db, rate_service=FakeRateService())

    report = await service.generate_weekly_report(district="Kolar", force_live=True)

    assert isinstance(report, WeeklyReport)
    assert report.district == "Kolar"
    assert report.crops == []
    assert report.source_health["orders"]["status"] == "unavailable"
    assert db.saved_reports, "service should persist even empty district reports"


@pytest.mark.asyncio
async def test_generate_weekly_report_builds_canonical_crop_payload() -> None:
    db = FakeDB(
        orders=[
            _order("Tomato", "buyer-1", 400.0, 5),
            _order("Tomato", "buyer-2", 250.0, 12),
            _order("Onion", "buyer-3", 100.0, 15),
        ],
        history=_price_history("Tomato") | _price_history("Onion"),
    )
    service = ADCLService(db=db, rate_service=FakeRateService(price=1900.0))

    report = await service.generate_weekly_report(district="Bangalore", force_live=True)

    assert report.crops, "live orders should produce at least one crop recommendation"
    crop = report.crops[0]
    payload = crop.to_dict()
    assert payload["commodity"]
    assert "green_label" in payload
    assert "recommendation" in payload
    assert "evidence" in payload
    assert "freshness" in payload
    assert "source_health" in payload
    assert db.saved_reports, "report should be persisted"


@pytest.mark.asyncio
async def test_generate_weekly_report_uses_cached_report_when_force_live_is_false() -> None:
    cached_report = {
        "week_start": date.today().isoformat(),
        "district": "Mysore",
        "generated_by": "adcl_service",
        "generated_at": "2026-03-17T10:00:00+00:00",
        "summary_en": "Cached report",
        "summary_hi": "",
        "summary_kn": "",
        "freshness": {"generated_at": "2026-03-17T10:00:00+00:00"},
        "source_health": {"orders": {"status": "healthy"}},
        "metadata": {},
        "crops": [],
    }
    db = FakeDB(cached_report=cached_report)
    service = ADCLService(db=db, rate_service=FakeRateService())

    report = await service.generate_weekly_report(district="Mysore", force_live=False)

    assert report.summary_en == "Cached report"
    assert db.saved_reports == []


@pytest.mark.asyncio
async def test_is_recommended_crop_uses_generated_report() -> None:
    db = FakeDB(
        orders=[
            _order("Okra", "buyer-1", 500.0, 3),
            _order("Okra", "buyer-2", 450.0, 10),
        ],
        history=_price_history("Okra"),
    )
    service = ADCLService(db=db, rate_service=FakeRateService())

    result = await service.is_recommended_crop("Okra", district="Bangalore")

    assert result is True


@pytest.mark.asyncio
async def test_get_weekly_list_keeps_voice_compatibility() -> None:
    db = FakeDB(
        orders=[
            _order("Tomato", "buyer-1", 500.0, 3),
            _order("Onion", "buyer-2", 200.0, 9),
        ],
        history=_price_history("Tomato") | _price_history("Onion"),
    )
    service = ADCLService(db=db, rate_service=FakeRateService())

    crops = await service.get_weekly_list(location="Bangalore")

    assert "tomato" in crops
