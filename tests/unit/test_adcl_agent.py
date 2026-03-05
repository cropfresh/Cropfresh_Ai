"""
Unit tests for Task 12/35: ADCL Agent (Adaptive Demand Crop List).

Covers:
  - Demand aggregation (count, weight, trend) [AC 1]
  - Percentile-rank normalisation             [AC 1 — upgraded]
  - Demand trend (renamed from price_trend)   [AC 1 — upgraded]
  - Sow-season calendar logic                 [AC 2 — upgraded]
  - Green-label scoring logic                 [AC 2]
  - Price forecast integration                [AC 3]
  - Multi-language summaries (en, hi, kn)     [AC 4]
  - Weekly report structure and DB stub       [AC 1, 2, 5]
"""

# * TEST MODULE — ADCL AGENT
# NOTE: Arrange-Act-Assert (AAA) pattern. No external I/O.

from __future__ import annotations

import pytest
from datetime import date, timedelta

from src.agents.adcl.demand import aggregate_demand
from src.agents.adcl.engine import ADCLAgent, get_adcl_agent
from src.agents.adcl.models import ADCLCrop, WeeklyReport
from src.agents.adcl.scoring import score_and_label
from src.agents.adcl.seasonal import SeasonalCalendar
from src.agents.adcl.summary import SummaryGenerator


# * ═══════════════════════════════════════════════════════════════
# * Shared Fixtures
# * ═══════════════════════════════════════════════════════════════

@pytest.fixture
def today() -> date:
    return date.today()


@pytest.fixture
def sample_orders(today: date) -> list[dict]:
    """Realistic orders for 3 commodities over 90 days."""
    def _ago(n: int) -> str:
        return (today - timedelta(days=n)).isoformat()

    return [
        # Tomato — high demand, rising (more in last 30d)
        {"commodity": "tomato", "quantity_kg": 500.0, "buyer_id": "b1", "created_at": _ago(5)},
        {"commodity": "tomato", "quantity_kg": 400.0, "buyer_id": "b2", "created_at": _ago(15)},
        {"commodity": "tomato", "quantity_kg": 200.0, "buyer_id": "b3", "created_at": _ago(50)},
        {"commodity": "tomato", "quantity_kg": 150.0, "buyer_id": "b4", "created_at": _ago(80)},
        # Onion — stable
        {"commodity": "onion",  "quantity_kg": 300.0, "buyer_id": "b5", "created_at": _ago(10)},
        {"commodity": "onion",  "quantity_kg": 310.0, "buyer_id": "b6", "created_at": _ago(55)},
        # Capsicum — falling (low recent demand)
        {"commodity": "capsicum", "quantity_kg": 80.0, "buyer_id": "b7", "created_at": _ago(8)},
        {"commodity": "capsicum", "quantity_kg": 220.0, "buyer_id": "b8", "created_at": _ago(70)},
    ]


@pytest.fixture
def calendar() -> SeasonalCalendar:
    return SeasonalCalendar()


# * ═══════════════════════════════════════════════════════════════
# * AC 1 — Demand Aggregation
# * ═══════════════════════════════════════════════════════════════

def test_aggregate_demand_counts_buyers_and_weight(
    sample_orders: list[dict], today: date
) -> None:
    """Buyer count and total weight aggregated correctly."""
    records = aggregate_demand(sample_orders, reference_date=today)
    tomato = next(r for r in records if r["commodity"] == "tomato")
    assert tomato["buyer_count"] == 4
    assert tomato["total_demand_kg"] == pytest.approx(1250.0)


def test_aggregate_demand_trend_rising(
    sample_orders: list[dict], today: date
) -> None:
    """Tomato: recent 30d demand much higher than prior → 'rising'."""
    records = aggregate_demand(sample_orders, reference_date=today)
    tomato = next(r for r in records if r["commodity"] == "tomato")
    #! Fixed: now asserts against "demand_trend" instead of "price_trend"
    assert tomato["demand_trend"] == "rising"


def test_aggregate_demand_trend_falling(
    sample_orders: list[dict], today: date
) -> None:
    """Capsicum: recent demand lower than prior period → 'falling'."""
    records = aggregate_demand(sample_orders, reference_date=today)
    capsicum = next(r for r in records if r["commodity"] == "capsicum")
    assert capsicum["demand_trend"] == "falling"


def test_aggregate_demand_empty_orders() -> None:
    """Empty order list returns empty result."""
    assert aggregate_demand([]) == []


def test_aggregate_demand_demand_score_normalised(
    sample_orders: list[dict], today: date
) -> None:
    """Highest-demand crop has score 1.0; others in [0, 1]."""
    records = aggregate_demand(sample_orders, reference_date=today)
    scores = [r["demand_score"] for r in records]
    assert max(scores) == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_aggregate_demand_percentile_multiple_high_scores(today: date) -> None:
    """Percentile ranking allows multiple crops to exceed 0.6 threshold."""
    def _ago(n: int) -> str:
        return (today - timedelta(days=n)).isoformat()

    # 5 crops with somewhat similar volumes — at least 3 should score >= 0.5
    orders = [
        {"commodity": "tomato",   "quantity_kg": 500.0, "buyer_id": "b1", "created_at": _ago(5)},
        {"commodity": "onion",    "quantity_kg": 480.0, "buyer_id": "b2", "created_at": _ago(5)},
        {"commodity": "potato",   "quantity_kg": 450.0, "buyer_id": "b3", "created_at": _ago(5)},
        {"commodity": "cabbage",  "quantity_kg": 420.0, "buyer_id": "b4", "created_at": _ago(5)},
        {"commodity": "capsicum", "quantity_kg": 100.0, "buyer_id": "b5", "created_at": _ago(5)},
    ]
    records = aggregate_demand(orders, reference_date=today)
    high_scores = [r for r in records if r["demand_score"] >= 0.5]
    assert len(high_scores) >= 3, "Percentile ranking should let multiple crops score high"


def test_aggregate_demand_output_has_demand_trend_key(
    sample_orders: list[dict], today: date,
) -> None:
    """Verify output uses 'demand_trend' key (not old 'price_trend')."""
    records = aggregate_demand(sample_orders, reference_date=today)
    for rec in records:
        assert "demand_trend" in rec, "Should use 'demand_trend' not 'price_trend'"
        assert "price_trend" not in rec, "Old 'price_trend' key should not exist"


# * ═══════════════════════════════════════════════════════════════
# * AC 2 — Seasonal Calendar (harvest + sow)
# * ═══════════════════════════════════════════════════════════════

def test_seasonal_fit_in_season(calendar: SeasonalCalendar) -> None:
    """Tomato in November → in_season (harvest)."""
    assert calendar.get_fit("tomato", month=11) == "in_season"


def test_seasonal_fit_off_season(calendar: SeasonalCalendar) -> None:
    """Mango in December → off_season (harvest)."""
    assert calendar.get_fit("mango", month=12) == "off_season"


def test_seasonal_fit_year_round_unknown(calendar: SeasonalCalendar) -> None:
    """Unknown crop → year_round (conservative default)."""
    assert calendar.get_fit("magic_bean_xyz", month=6) == "year_round"


def test_sow_fit_ideal(calendar: SeasonalCalendar) -> None:
    """Tomato in August → ideal_sow (sowing season is Jun-Sep)."""
    assert calendar.get_sow_fit("tomato", month=8) == "ideal_sow"


def test_sow_fit_not_sow_season(calendar: SeasonalCalendar) -> None:
    """Tomato in January → not_sow_season (sowing is Jun-Sep)."""
    assert calendar.get_sow_fit("tomato", month=1) == "not_sow_season"


def test_sow_fit_possible_adjacent(calendar: SeasonalCalendar) -> None:
    """Tomato in October → possible_sow (adjacent to Sep sowing window)."""
    assert calendar.get_sow_fit("tomato", month=10) == "possible_sow"


def test_sow_fit_unknown_crop(calendar: SeasonalCalendar) -> None:
    """Unknown crop → ideal_sow (conservative: allow planting)."""
    assert calendar.get_sow_fit("magic_bean_xyz", month=6) == "ideal_sow"


def test_sow_fit_year_round_crop(calendar: SeasonalCalendar) -> None:
    """Banana (year-round) → ideal_sow in any month."""
    assert calendar.get_sow_fit("banana", month=3) == "ideal_sow"
    assert calendar.get_sow_fit("banana", month=9) == "ideal_sow"


# * ═══════════════════════════════════════════════════════════════
# * AC 2 — Green-Label Logic (updated for sow-season + demand_trend)
# * ═══════════════════════════════════════════════════════════════

def _make_demand_record(
    commodity: str,
    demand_score: float,
    demand_trend: str,
    total_demand_kg: float = 500.0,
    buyer_count: int = 3,
) -> dict:
    return {
        "commodity": commodity,
        "demand_score": demand_score,
        "demand_trend": demand_trend,
        "total_demand_kg": total_demand_kg,
        "buyer_count": buyer_count,
    }


def test_green_label_true_all_conditions_met() -> None:
    """High demand + rising + ideal_sow → green_label True."""
    # August → tomato sowing season (ideal_sow)
    records = [_make_demand_record("tomato", demand_score=0.75, demand_trend="rising")]
    crops = score_and_label(records, price_forecasts={}, current_month=8)
    assert crops[0].green_label is True
    assert crops[0].sow_season_fit == "ideal_sow"


def test_green_label_false_not_sow_season() -> None:
    """High demand but not_sow_season → green_label False."""
    # January → tomato is NOT in sowing season
    records = [_make_demand_record("tomato", demand_score=0.80, demand_trend="rising")]
    crops = score_and_label(records, price_forecasts={}, current_month=1)
    assert crops[0].green_label is False
    assert crops[0].sow_season_fit == "not_sow_season"


def test_green_label_false_low_demand() -> None:
    """demand_score <= 0.6 → green_label False even if ideal_sow."""
    records = [_make_demand_record("tomato", demand_score=0.55, demand_trend="stable")]
    crops = score_and_label(records, price_forecasts={}, current_month=8)
    assert crops[0].green_label is False


def test_green_label_false_falling_demand() -> None:
    """Falling demand trend → green_label False even with high demand."""
    records = [_make_demand_record("tomato", demand_score=0.90, demand_trend="falling")]
    crops = score_and_label(records, price_forecasts={}, current_month=8)
    assert crops[0].green_label is False


def test_green_label_crop_has_both_trends() -> None:
    """ADCLCrop has separate demand_trend and price_trend fields."""
    records = [_make_demand_record("tomato", demand_score=0.80, demand_trend="rising")]
    crops = score_and_label(records, price_forecasts={"tomato": 35.0}, current_month=8)
    crop = crops[0]
    assert hasattr(crop, "demand_trend")
    assert hasattr(crop, "price_trend")
    assert crop.demand_trend == "rising"


# * ═══════════════════════════════════════════════════════════════
# * AC 3 — Price Forecast Integration
# * ═══════════════════════════════════════════════════════════════

def test_price_forecast_applied_to_crop() -> None:
    """Price forecast from dict is set on ADCLCrop.predicted_price_per_kg."""
    records = [_make_demand_record("tomato", demand_score=0.80, demand_trend="rising")]
    crops = score_and_label(
        records,
        price_forecasts={"tomato": 35.50},
        current_month=8,
    )
    assert crops[0].predicted_price_per_kg == pytest.approx(35.50)


def test_price_forecast_zero_when_missing() -> None:
    """Missing price forecast → predicted_price_per_kg == 0.0."""
    records = [_make_demand_record("onion", demand_score=0.70, demand_trend="stable")]
    crops = score_and_label(records, price_forecasts={}, current_month=8)
    assert crops[0].predicted_price_per_kg == pytest.approx(0.0)


# * ═══════════════════════════════════════════════════════════════
# * AC 4 — Multi-Language Summaries
# * ═══════════════════════════════════════════════════════════════

def test_summary_en_generated_no_llm() -> None:
    """English summary generated without LLM."""
    gen = SummaryGenerator(llm=None)
    crops = [
        ADCLCrop("tomato", 0.9, 35.0, "rising", "in_season", True, 4, 1000.0),
        ADCLCrop("onion",  0.7, 28.0, "stable", "in_season", True, 3, 800.0),
    ]
    result = gen.generate(crops)
    assert "en" in result
    assert len(result["en"]) > 20
    assert "tomato" in result["en"].lower() or "Tomato" in result["en"]


def test_summary_hi_generated_no_llm() -> None:
    """Hindi summary generated without LLM (contains Devanagari)."""
    gen = SummaryGenerator(llm=None)
    crops = [ADCLCrop("tomato", 0.9, 35.0, "rising", "in_season", True, 4, 1000.0)]
    result = gen.generate(crops)
    assert "hi" in result
    # Devanagari range U+0900–U+097F
    assert any("\u0900" <= ch <= "\u097f" for ch in result["hi"])


def test_summary_kn_generated_no_llm() -> None:
    """Kannada summary generated without LLM (contains Kannada script)."""
    gen = SummaryGenerator(llm=None)
    crops = [ADCLCrop("tomato", 0.9, 35.0, "rising", "in_season", True, 4, 1000.0)]
    result = gen.generate(crops)
    assert "kn" in result
    # Kannada range U+0C80–U+0CFF
    assert any("\u0c80" <= ch <= "\u0cff" for ch in result["kn"])


def test_summary_no_green_crops_no_llm() -> None:
    """Summary when no green-labelled crops → meaningful fallback text."""
    gen = SummaryGenerator(llm=None)
    crops = [ADCLCrop("capsicum", 0.4, 0.0, "falling", "off_season", False, 1, 100.0)]
    result = gen.generate(crops)
    assert "no crops" in result["en"].lower() or "no" in result["en"].lower()


# * ═══════════════════════════════════════════════════════════════
# * AC 1 + 2 + 5 — Weekly Report Structure
# * ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_generate_weekly_report_structure() -> None:
    """WeeklyReport has required fields, non-empty crops list."""
    agent = get_adcl_agent()   # No DB, no price_agent, no LLM
    report = await agent.generate_weekly_report(district="Bangalore")

    assert isinstance(report, WeeklyReport)
    assert isinstance(report.week_start, date)
    assert report.generated_by == "adcl_agent"
    assert len(report.crops) > 0
    assert report.summary_en != ""
    assert report.summary_hi != ""
    assert report.summary_kn != ""


@pytest.mark.asyncio
async def test_generate_weekly_report_no_db_no_crash() -> None:
    """Agent with no DB injection completes without exception."""
    agent = ADCLAgent(db=None, price_agent=None, llm=None)
    report = await agent.generate_weekly_report()
    assert isinstance(report, WeeklyReport)


@pytest.mark.asyncio
async def test_generate_weekly_report_persists_to_db() -> None:
    """Report is persisted to DB via insert_adcl_report when DB is provided."""
    persisted: list[dict] = []

    class MockDB:
        async def get_recent_orders(self, days: int) -> list[dict]:
            today = date.today()
            return [
                {"commodity": "tomato", "quantity_kg": 500.0, "buyer_id": "b1",
                 "created_at": (today - timedelta(days=5)).isoformat()},
                {"commodity": "onion", "quantity_kg": 400.0, "buyer_id": "b2",
                 "created_at": (today - timedelta(days=10)).isoformat()},
            ]

        async def insert_adcl_report(self, data: dict) -> None:
            persisted.append(data)

    agent = ADCLAgent(db=MockDB())
    await agent.generate_weekly_report()
    assert len(persisted) == 1
    assert "week_start" in persisted[0]
    assert "crops" in persisted[0]


@pytest.mark.asyncio
async def test_generate_weekly_report_crop_fields_complete() -> None:
    """Every ADCLCrop in the report has all required fields populated."""
    agent = get_adcl_agent()
    report = await agent.generate_weekly_report()
    for crop in report.crops:
        assert crop.commodity != ""
        assert 0.0 <= crop.demand_score <= 1.0
        assert crop.demand_trend in ("rising", "stable", "falling")
        assert crop.price_trend in ("rising", "stable", "falling")
        assert crop.seasonal_fit in ("in_season", "off_season", "year_round")
        assert crop.sow_season_fit in ("ideal_sow", "possible_sow", "not_sow_season")
        assert isinstance(crop.green_label, bool)
        assert crop.buyer_count >= 0
        assert crop.total_demand_kg >= 0.0


@pytest.mark.asyncio
async def test_generate_weekly_report_to_dict_has_new_fields() -> None:
    """WeeklyReport.to_dict() includes demand_trend and sow_season_fit."""
    agent = get_adcl_agent()
    report = await agent.generate_weekly_report()
    data = report.to_dict()
    assert "crops" in data
    if data["crops"]:
        crop_dict = data["crops"][0]
        assert "demand_trend" in crop_dict
        assert "sow_season_fit" in crop_dict
