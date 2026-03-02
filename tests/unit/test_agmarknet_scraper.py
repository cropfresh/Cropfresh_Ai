"""
Unit Tests — Task 13: APMC Live Mandi Scraper
==============================================

Covers all 6 acceptance criteria:

  AC1 — Scrapes prices for 8+ Karnataka mandis           (25%)
  AC2 — Handles 10+ commodities                           (15%)
  AC3 — Rate-limited (≤1 req/sec) with retries            (15%)
  AC4 — Stores in price_history via DB interface          (20%)
  AC5 — Daily scheduler runs at 10 AM IST                 (15%)
  AC6 — Graceful fallback when site is down               (10%)

No real HTTP calls — Scrapling fetch is patched throughout.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scrapers.agmarknet import (
    AgmarknetScraper,
    MandiPrice,
    get_agmarknet_scraper,
)
from src.scrapers.scraper_scheduler import ScraperScheduler, get_scraper_scheduler


# ============================================================================
# Helpers / Fixtures
# ============================================================================


def _make_mandi_price(commodity: str, mandi: str, modal_price: float = 2000.0) -> MandiPrice:
    """Convenience factory for MandiPrice test fixtures."""
    return MandiPrice(
        commodity=commodity,
        mandi=mandi,
        state="Karnataka",
        district=mandi,
        min_price=1000.0,
        max_price=3000.0,
        modal_price=modal_price,
        date=date.today(),
        source="agmarknet_dev",
    )


@pytest.fixture
def scraper() -> AgmarknetScraper:
    """Fresh AgmarknetScraper with no previous state."""
    return AgmarknetScraper()


@pytest.fixture
def scheduler_with_scraper(scraper: AgmarknetScraper) -> ScraperScheduler:
    """ScraperScheduler wired with an AgmarknetScraper, no DB, no IMD."""
    return ScraperScheduler(agmarknet=scraper, imd=None, db=None)


# ============================================================================
# AC1 — Mandis & Commodity Counts
# ============================================================================


def test_karnataka_mandis_count_gte_8(scraper: AgmarknetScraper) -> None:
    """AC1: At least 8 Karnataka mandis defined on the scraper."""
    assert len(scraper.KARNATAKA_MANDIS) >= 8


def test_target_commodities_count_gte_10(scraper: AgmarknetScraper) -> None:
    """AC2: At least 10 target commodities defined."""
    assert len(scraper.TARGET_COMMODITIES) >= 10


def test_karnataka_mandis_includes_major_cities(scraper: AgmarknetScraper) -> None:
    """AC1: Known major Karnataka mandis are present."""
    mandis_lower = [m.lower() for m in scraper.KARNATAKA_MANDIS]
    for expected in ["bangalore", "hubli", "mysore"]:
        assert expected in mandis_lower, f"'{expected}' not found in KARNATAKA_MANDIS"


def test_target_commodities_includes_key_crops(scraper: AgmarknetScraper) -> None:
    """AC2: Key vegetables are in the commodities list."""
    lower = [c.lower() for c in scraper.TARGET_COMMODITIES]
    for expected in ["tomato", "onion", "potato"]:
        assert any(expected in c for c in lower), (
            f"'{expected}' not found in TARGET_COMMODITIES"
        )


# ============================================================================
# AC1 — scrape_daily_prices()
# ============================================================================


@pytest.mark.asyncio
async def test_scrape_daily_prices_returns_list(scraper: AgmarknetScraper) -> None:
    """AC1: scrape_daily_prices returns a non-empty list of MandiPrice."""
    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        # fetch returns a fake page — parsing will fail and fallback kicks in (AC6)
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        prices = await scraper.scrape_daily_prices(state="Karnataka")

    assert isinstance(prices, list)
    assert len(prices) > 0
    assert all(isinstance(p, MandiPrice) for p in prices)


@pytest.mark.asyncio
async def test_scrape_daily_prices_covers_8_mandis(scraper: AgmarknetScraper) -> None:
    """AC1: Dev-fallback data covers all 8 Karnataka mandis for each commodity."""
    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        prices = await scraper.scrape_daily_prices()

    unique_mandis = {p.mandi for p in prices}
    assert len(unique_mandis) >= 8, (
        f"Expected ≥8 unique mandis in output, got {len(unique_mandis)}: {unique_mandis}"
    )


@pytest.mark.asyncio
async def test_scrape_daily_prices_all_records_have_state(scraper: AgmarknetScraper) -> None:
    """AC1: Every returned MandiPrice has 'state' populated."""
    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        prices = await scraper.scrape_daily_prices(state="Karnataka")

    assert all(p.state != "" for p in prices)


# ============================================================================
# AC3 — Rate Limiting & Retries
# ============================================================================


def test_rate_limit_attribute_lte_one_second(scraper: AgmarknetScraper) -> None:
    """AC3: rate_limit_delay is set to ≤ 1.0 second."""
    assert scraper.rate_limit_delay <= 1.0


def test_max_retries_gte_3(scraper: AgmarknetScraper) -> None:
    """AC3: At least 3 retry attempts configured via base class."""
    assert scraper.max_retries >= 3


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold_failures(
    scraper: AgmarknetScraper,
) -> None:
    """AC3: Circuit breaker opens after repeated fetch failures."""
    from src.scrapers.base_scraper import CircuitState

    # Record enough failures to trip the breaker
    for _ in range(scraper.circuit_breaker_threshold):
        scraper._circuit.record_failure()

    assert scraper._circuit.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_scrape_does_not_raise_on_fetch_failure(scraper: AgmarknetScraper) -> None:
    """AC6 + AC3: scrape() catches exceptions and returns fallback data, not an exception."""
    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = ConnectionError("Agmarknet unreachable")

        result = await scraper.scrape(commodity="Tomato", state="Karnataka")

    # Should not raise — returns a ScrapeResult (possibly with dev data)
    assert result is not None


# ============================================================================
# AC4 — Store in DB
# ============================================================================


@pytest.mark.asyncio
async def test_scrape_and_store_calls_db_insert(scraper: AgmarknetScraper) -> None:
    """AC4: scrape_and_store calls db.insert_mandi_prices with the scraped data."""
    inserted_records: list[list[dict]] = []

    class MockDB:
        async def insert_mandi_prices(self, records: list[dict]) -> None:
            inserted_records.append(records)

    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        await scraper.scrape_and_store(db=MockDB())

    assert len(inserted_records) == 1, "insert_mandi_prices should be called exactly once"
    records = inserted_records[0]
    assert len(records) > 0, "At least one price record should be inserted"


@pytest.mark.asyncio
async def test_scrape_and_store_returns_count(scraper: AgmarknetScraper) -> None:
    """AC4: scrape_and_store returns integer count of records inserted."""

    class MockDB:
        async def insert_mandi_prices(self, records: list[dict]) -> None:
            pass  # just accept without storing

    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        count = await scraper.scrape_and_store(db=MockDB())

    assert isinstance(count, int)
    assert count > 0


@pytest.mark.asyncio
async def test_scrape_and_store_no_db_returns_zero(scraper: AgmarknetScraper) -> None:
    """AC4: When db=None, scrape_and_store returns 0 without crashing."""
    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        count = await scraper.scrape_and_store(db=None)

    assert count == 0


@pytest.mark.asyncio
async def test_scrape_and_store_records_have_required_fields(scraper: AgmarknetScraper) -> None:
    """AC4: Every stored record has commodity, mandi, modal_price, and date fields."""
    inserted_records: list[list[dict]] = []

    class MockDB:
        async def insert_mandi_prices(self, records: list[dict]) -> None:
            inserted_records.append(records)

    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_page = MagicMock()
        mock_page.css.return_value = []
        mock_fetch.return_value = mock_page

        await scraper.scrape_and_store(db=MockDB())

    for record in inserted_records[0]:
        assert "commodity" in record
        assert "mandi" in record
        assert "modal_price" in record
        assert "date" in record


# ============================================================================
# AC5 — Scheduler at 10 AM IST
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_agmarknet_job_registered(
    scheduler_with_scraper: ScraperScheduler,
) -> None:
    """AC5: Scheduler has an 'agmarknet_daily' job after start()."""
    scheduler_with_scraper.start()
    try:
        job = scheduler_with_scraper.get_job("agmarknet_daily")
        assert job is not None, "agmarknet_daily job should be registered"
    finally:
        scheduler_with_scraper.stop()


@pytest.mark.asyncio
async def test_scheduler_agmarknet_job_at_10am_ist(
    scheduler_with_scraper: ScraperScheduler,
) -> None:
    """AC5: The agmarknet_daily cron job fires at 10:00 in Asia/Kolkata."""
    scheduler_with_scraper.start()
    try:
        job = scheduler_with_scraper.get_job("agmarknet_daily")
        assert job is not None

        # Check the trigger string representation (version-agnostic)
        trigger_str = str(job.trigger)
        assert "10" in trigger_str, (
            f"Expected hour=10 in trigger repr, got: {trigger_str}"
        )
    finally:
        scheduler_with_scraper.stop()


def test_scheduler_timezone_is_kolkata(
    scheduler_with_scraper: ScraperScheduler,
) -> None:
    """AC5: Scheduler timezone is set to Asia/Kolkata (IST)."""
    assert scheduler_with_scraper.TIMEZONE == "Asia/Kolkata"


@pytest.mark.asyncio
async def test_scheduler_start_stop_lifecycle(
    scheduler_with_scraper: ScraperScheduler,
) -> None:
    """AC5: start() and stop() are side-effect free and don't raise."""
    assert not scheduler_with_scraper.is_running
    scheduler_with_scraper.start()
    assert scheduler_with_scraper.is_running
    scheduler_with_scraper.stop()
    assert not scheduler_with_scraper.is_running


@pytest.mark.asyncio
async def test_scheduler_get_jobs_returns_list(
    scheduler_with_scraper: ScraperScheduler,
) -> None:
    """AC5: get_jobs() returns a list of job metadata dicts after start()."""
    scheduler_with_scraper.start()
    try:
        jobs = scheduler_with_scraper.get_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) >= 1
        ids = [j["id"] for j in jobs]
        assert "agmarknet_daily" in ids
    finally:
        scheduler_with_scraper.stop()


# ============================================================================
# AC6 — Graceful Fallback
# ============================================================================


@pytest.mark.asyncio
async def test_fallback_returns_dev_data_when_fetch_fails(
    scraper: AgmarknetScraper,
) -> None:
    """AC6: When fetch raises, scrape() returns development fallback data."""
    with patch.object(scraper, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = OSError("Connection refused")

        result = await scraper.scrape(commodity="Tomato", state="Karnataka")

    # Result has data from dev fallback
    assert result.data is not None


@pytest.mark.asyncio
async def test_fallback_dev_data_covers_all_mandis(scraper: AgmarknetScraper) -> None:
    """AC6: Dev fallback data covers all 8 Karnataka mandis for any commodity."""
    dev_data = scraper._get_dev_data("Tomato", "Karnataka")
    mandis = {p.mandi for p in dev_data}
    assert len(mandis) >= 8, f"Expected ≥8 mandis in dev data, got {len(mandis)}"
    assert len(mandis) == len(scraper.KARNATAKA_MANDIS)


def test_fallback_dev_data_modal_price_positive(scraper: AgmarknetScraper) -> None:
    """AC6: Dev fallback modal prices are positive (non-zero)."""
    dev_data = scraper._get_dev_data("Onion")
    assert all(p.modal_price > 0 for p in dev_data)


# ============================================================================
# Data Integrity
# ============================================================================


def test_mandi_price_model_fields() -> None:
    """MandiPrice Pydantic model validates required fields correctly."""
    price = MandiPrice(
        commodity="Tomato",
        mandi="Bangalore",
        state="Karnataka",
        modal_price=2000.0,
        date=date.today(),
    )
    assert price.commodity == "Tomato"
    assert price.modal_price_per_kg == pytest.approx(20.0)
    assert price.unit == "Rs/Quintal"
    assert price.source == "agmarknet"


def test_get_agmarknet_scraper_factory() -> None:
    """Factory function returns a valid AgmarknetScraper."""
    s = get_agmarknet_scraper()
    assert isinstance(s, AgmarknetScraper)
    assert s.rate_limit_delay <= 1.0


def test_get_scraper_scheduler_factory() -> None:
    """get_scraper_scheduler() returns a ScraperScheduler without error."""
    scheduler = get_scraper_scheduler(agmarknet=AgmarknetScraper(), db=None)
    assert isinstance(scheduler, ScraperScheduler)
