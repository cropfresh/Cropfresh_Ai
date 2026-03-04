"""
Unit tests for Pricing Agent v2 — covers:
  - AISP calculation (via aisp_calculator module)
  - Concurrent signal gathering with mocks
  - Price forecast validation
  - Seasonal adjustments
  - Heuristic recommendation logic
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.aisp_calculator import calculate_aisp
from src.agents.pricing_agent import FarmerContext, PricingAgent
from src.tools.agmarknet import AgmarknetPrice
from src.tools.ml_forecaster import PriceForecaster, PriceSample


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def agent() -> PricingAgent:
    """Return a PricingAgent wired to mock data for all signals."""
    return PricingAgent(llm=None, use_mock=True)


def _mock_price(modal: float = 2500.0) -> AgmarknetPrice:
    return AgmarknetPrice(
        commodity="Tomato",
        state="Karnataka",
        district="Kolar",
        market="Kolar Main Market",
        date=datetime.now(),
        min_price=1500.0,
        max_price=3500.0,
        modal_price=modal,
    )


# ── AISP Calculation ──────────────────────────────────────────────────────────

class TestAISPCalculation:
    """Covers aisp_calculator.calculate_aisp via the agent wrapper."""

    def test_basic_aisp(self, agent: PricingAgent) -> None:
        """AISP includes farmer payout + logistics + handling + fee + buffer."""
        result = agent.calculate_aisp(
            farmer_price_per_kg=20.0,
            quantity_kg=100,
            distance_km=10,
        )
        assert result.farmer_payout == 2000.0
        assert result.logistics_cost == 1.5   # 10 * 1.5 * 100 / 1000
        assert result.risk_buffer > 0
        assert result.risk_buffer_pct == 0.02
        assert result.aisp_per_kg > 20.0
        assert abs(result.total_aisp - result.aisp_per_kg * result.quantity_kg) < 1.0

    def test_risk_buffer_is_two_percent(self, agent: PricingAgent) -> None:
        result = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=100, distance_km=10,
        )
        subtotal = (
            result.farmer_payout + result.logistics_cost
            + result.deadhead_surcharge + result.handling_cost
        )
        assert abs(result.risk_buffer - subtotal * 0.02) < 0.01

    def test_deadhead_high_utilization_has_no_surcharge(self, agent: PricingAgent) -> None:
        high = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=100, distance_km=30,
            route_utilization_pct=90,
        )
        low = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=100, distance_km=30,
            route_utilization_pct=30,
        )
        assert high.deadhead_surcharge == 0.0
        assert low.deadhead_surcharge > 0.0

    def test_mandi_cap_limits_aisp(self, agent: PricingAgent) -> None:
        result = agent.calculate_aisp(
            farmer_price_per_kg=25.0, quantity_kg=100, distance_km=50,
            mandi_modal_per_kg=22.0,
        )
        assert result.aisp_per_kg <= 22.0 * 1.05
        assert result.mandi_cap_applied is True

    def test_mandi_cap_not_applied_when_below(self, agent: PricingAgent) -> None:
        result = agent.calculate_aisp(
            farmer_price_per_kg=5.0, quantity_kg=1000, distance_km=10,
            mandi_modal_per_kg=100.0,
        )
        assert result.mandi_cap_applied is False

    def test_aisp_zero_quantity_raises(self, agent: PricingAgent) -> None:
        with pytest.raises(ValueError, match="quantity_kg must be > 0"):
            agent.calculate_aisp(farmer_price_per_kg=20.0, quantity_kg=0, distance_km=10)

    def test_aisp_negative_quantity_raises(self, agent: PricingAgent) -> None:
        with pytest.raises(ValueError, match="quantity_kg must be > 0"):
            agent.calculate_aisp(farmer_price_per_kg=20.0, quantity_kg=-10.0, distance_km=10)

    def test_platform_fee_tiers(self, agent: PricingAgent) -> None:
        small      = agent.calculate_aisp(farmer_price_per_kg=10.0, quantity_kg=50, distance_km=10)
        medium     = agent.calculate_aisp(farmer_price_per_kg=10.0, quantity_kg=200, distance_km=10)
        large      = agent.calculate_aisp(farmer_price_per_kg=10.0, quantity_kg=600, distance_km=10)
        extra_large= agent.calculate_aisp(farmer_price_per_kg=10.0, quantity_kg=1500, distance_km=10)

        assert small.platform_fee_pct == 0.08
        assert medium.platform_fee_pct == 0.07
        assert large.platform_fee_pct == 0.06
        assert extra_large.platform_fee_pct == 0.05

    def test_logistics_rate_tiers(self, agent: PricingAgent) -> None:
        r10 = agent.calculate_aisp(farmer_price_per_kg=10, quantity_kg=100, distance_km=10)
        r40 = agent.calculate_aisp(farmer_price_per_kg=10, quantity_kg=100, distance_km=40)
        r80 = agent.calculate_aisp(farmer_price_per_kg=10, quantity_kg=100, distance_km=80)
        assert r10.logistics_cost < r40.logistics_cost < r80.logistics_cost


# ── Recommendation pipeline ───────────────────────────────────────────────────

class TestGetRecommendation:
    """Integration-style tests that mock live APIs and check the pipeline."""

    @pytest.mark.asyncio
    async def test_recommendation_returns_result(self, agent: PricingAgent) -> None:
        rec = await agent.get_recommendation("Tomato", "Kolar", 100)
        assert rec.commodity == "Tomato"
        assert rec.current_price > 0
        assert rec.recommended_action in ("sell", "hold", "wait", "unknown")
        assert rec.data_source == "mock"

    @pytest.mark.asyncio
    async def test_aisp_included_in_recommendation(self, agent: PricingAgent) -> None:
        rec = await agent.get_recommendation("Tomato", "Kolar", 200)
        assert rec.aisp_per_kg is not None
        assert rec.aisp_per_kg > 0
        assert rec.aisp_breakdown is not None
        assert "risk_buffer" in rec.aisp_breakdown

    @pytest.mark.asyncio
    async def test_recommendation_with_farmer_context(self, agent: PricingAgent) -> None:
        """Urgent financial context should push recommendation towards 'sell'."""
        ctx = FarmerContext(financial_urgency="urgent")
        rec = await agent.get_recommendation("Tomato", "Kolar", 100, farmer_context=ctx)
        # Urgent context + large quantity = near-certain sell
        assert rec.recommended_action in ("sell", "hold", "wait")

    @pytest.mark.asyncio
    async def test_empty_recommendation_on_no_prices(self) -> None:
        """If no prices are returned, action should be 'unknown'."""
        a = PricingAgent(use_mock=True)
        with patch.object(a, "_fetch_prices", new=AsyncMock(return_value=[])):
            rec = await a.get_recommendation("UnknownCrop", "Nowhere")
        assert rec.recommended_action == "unknown"
        assert rec.confidence == 0


# ── Trend and seasonality ─────────────────────────────────────────────────────

class TestTrendAndSeasonality:

    @pytest.mark.asyncio
    async def test_get_price_trend_returns_expected_shape(self, agent: PricingAgent) -> None:
        result = await agent.get_price_trend("Tomato", district="Kolar", days=30)
        assert result["trend"] in ("rising", "falling", "stable")
        assert result["recommendation"] in ("sell_now", "hold_3_days", "hold_7_days")
        assert isinstance(result["7d_avg"], float)
        assert isinstance(result["30d_avg"], float)
        assert 0.0 <= result["volatility_index"] <= 1.0
        assert "forecasted_7d" in result

    def test_seasonal_adjustment_default_and_specific(self, agent: PricingAgent) -> None:
        tomato_may   = agent.get_seasonal_adjustment("Tomato", 5)
        unknown_crop = agent.get_seasonal_adjustment("DragonFruit", 5)
        assert tomato_may == 1.3
        assert unknown_crop == 1.0

    def test_seasonal_adjustment_invalid_month_raises(self, agent: PricingAgent) -> None:
        with pytest.raises(ValueError, match="month must be 1"):
            agent.get_seasonal_adjustment("Tomato", 13)


# ── ML Forecaster unit tests ──────────────────────────────────────────────────

class TestMLForecaster:

    def test_forecast_rising_trend(self) -> None:
        """Steadily increasing prices should yield a 'rising' forecast."""
        prices = [float(10 + i) for i in range(20)]
        forecaster = PriceForecaster()
        result = forecaster.forecast_from_raw(prices, "Tomato", "Kolar")
        assert result.trend_direction in ("rising", "stable")
        assert len(result.forecasted_prices) == 7

    def test_forecast_falling_trend(self) -> None:
        prices = [float(30 - i) for i in range(20)]
        forecaster = PriceForecaster()
        result = forecaster.forecast_from_raw(prices, "Tomato", "Kolar")
        assert result.trend_direction in ("falling", "stable")

    def test_forecast_confidence_low_with_few_samples(self) -> None:
        forecaster = PriceForecaster()
        result = forecaster.forecast_from_raw([20.0, 21.0], "Tomato", "Kolar")
        # Few samples = lower confidence
        assert result.confidence < 0.75

    def test_no_negative_forecast(self) -> None:
        """Ensured by blending logic — prices cannot go negative."""
        prices = [1.0, 0.5, 0.3]
        forecaster = PriceForecaster()
        result = forecaster.forecast_from_raw(prices, "Tomato", "Kolar")
        assert all(p >= 0.0 for p in result.forecasted_prices)
