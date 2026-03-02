"""
Unit tests for Pricing Agent — AISP calculation with
deadhead factor, risk buffer, and mandi cap.
"""

import pytest

from src.agents.pricing_agent import PricingAgent


@pytest.fixture
def agent() -> PricingAgent:
    return PricingAgent(llm=None, use_mock=True)


class TestAISPCalculation:
    """Tests for the calculate_aisp method."""

    def test_basic_aisp(self, agent: PricingAgent):
        """AISP includes farmer payout + logistics + handling + fee + buffer."""
        result = agent.calculate_aisp(
            farmer_price_per_kg=20.0,
            quantity_kg=100,
            distance_km=10,
        )

        assert result.farmer_payout == 2000.0
        assert result.logistics_cost == 1.5  # 10km * 1.5 ₹/km/kg * 100kg / 1000
        assert result.risk_buffer > 0
        assert result.risk_buffer_pct == 0.02
        assert result.aisp_per_kg > 20.0
        assert abs(result.total_aisp - (result.aisp_per_kg * result.quantity_kg)) < 1.0

    def test_risk_buffer_is_two_percent(self, agent: PricingAgent):
        """Risk buffer should be 2% of subtotal."""
        result = agent.calculate_aisp(
            farmer_price_per_kg=10.0,
            quantity_kg=100,
            distance_km=10,
        )

        subtotal = (
            result.farmer_payout
            + result.logistics_cost
            + result.deadhead_surcharge
            + result.handling_cost
        )
        expected_buffer = subtotal * 0.02
        assert abs(result.risk_buffer - expected_buffer) < 0.01

    def test_deadhead_high_utilization_has_no_surcharge(self, agent: PricingAgent):
        """High route utilization (>80%) should have zero deadhead surcharge."""
        high_utilization = agent.calculate_aisp(
            farmer_price_per_kg=10.0,
            quantity_kg=100,
            distance_km=30,
            route_utilization_pct=90,
        )
        low_utilization = agent.calculate_aisp(
            farmer_price_per_kg=10.0,
            quantity_kg=100,
            distance_km=30,
            route_utilization_pct=30,
        )

        assert high_utilization.deadhead_surcharge == 0.0
        assert low_utilization.deadhead_surcharge > 0.0

    def test_mandi_cap_limits_aisp(self, agent: PricingAgent):
        """AISP should not exceed mandi modal x 1.05 when cap is provided."""
        result = agent.calculate_aisp(
            farmer_price_per_kg=25.0,
            quantity_kg=100,
            distance_km=50,
            mandi_modal_per_kg=22.0,
        )

        assert result.aisp_per_kg <= 22.0 * 1.05
        assert result.mandi_cap_applied is True

    def test_mandi_cap_not_applied_when_below(self, agent: PricingAgent):
        """No cap applied if AISP is naturally below mandi price."""
        result = agent.calculate_aisp(
            farmer_price_per_kg=5.0,
            quantity_kg=1000,
            distance_km=10,
            mandi_modal_per_kg=100.0,
        )

        assert result.mandi_cap_applied is False

    def test_aisp_zero_quantity_raises(self, agent: PricingAgent):
        """AC: quantity_kg <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="quantity_kg must be greater than 0"):
            agent.calculate_aisp(
                farmer_price_per_kg=20.0,
                quantity_kg=0,
                distance_km=10,
            )

    def test_aisp_negative_quantity_raises(self, agent: PricingAgent):
        """Negative quantity must raise ValueError."""
        with pytest.raises(ValueError, match="quantity_kg must be greater than 0"):
            agent.calculate_aisp(
                farmer_price_per_kg=20.0,
                quantity_kg=-10.0,
                distance_km=10,
            )

    def test_platform_fee_tiers(self, agent: PricingAgent):
        """Platform fee % decreases with volume."""
        small = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=50, distance_km=10,
        )
        medium = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=200, distance_km=10,
        )
        large = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=600, distance_km=10,
        )
        extra_large = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=1500, distance_km=10,
        )

        assert small.platform_fee_pct == 0.08
        assert medium.platform_fee_pct == 0.07
        assert large.platform_fee_pct == 0.06
        assert extra_large.platform_fee_pct == 0.05

    def test_logistics_rate_tiers(self, agent: PricingAgent):
        """Logistics rate increases with distance."""
        r10 = agent.calculate_aisp(farmer_price_per_kg=10, quantity_kg=100, distance_km=10)
        r40 = agent.calculate_aisp(farmer_price_per_kg=10, quantity_kg=100, distance_km=40)
        r80 = agent.calculate_aisp(farmer_price_per_kg=10, quantity_kg=100, distance_km=80)

        assert r10.logistics_cost < r40.logistics_cost < r80.logistics_cost


class TestGetRecommendation:
    """Tests for the get_recommendation method."""

    @pytest.mark.asyncio
    async def test_recommendation_returns_result(self, agent: PricingAgent):
        """Recommendation should return a valid result with mock data."""
        rec = await agent.get_recommendation("Tomato", "Kolar", 100)

        assert rec.commodity == "Tomato"
        assert rec.current_price > 0
        assert rec.recommended_action in ("sell", "hold", "wait", "unknown")
        assert rec.data_source == "mock"

    @pytest.mark.asyncio
    async def test_aisp_included_in_recommendation(self, agent: PricingAgent):
        """Recommendation includes AISP breakdown."""
        rec = await agent.get_recommendation("Tomato", "Kolar", 200)

        assert rec.aisp_per_kg is not None
        assert rec.aisp_per_kg > 0
        assert rec.aisp_breakdown is not None
        assert "risk_buffer" in rec.aisp_breakdown


class TestTrendAndSeasonality:
    """Tests for trend and seasonal helpers."""

    @pytest.mark.asyncio
    async def test_get_price_trend_returns_expected_shape(self, agent: PricingAgent):
        """Trend output includes averages, trend, volatility, and recommendation."""
        result = await agent.get_price_trend("Tomato", district="Kolar", days=30)
        assert result["trend"] in ("rising", "falling", "stable")
        assert result["recommendation"] in ("sell_now", "hold_3_days", "hold_7_days")
        assert isinstance(result["7d_avg"], float)
        assert isinstance(result["30d_avg"], float)
        assert 0.0 <= result["volatility_index"] <= 1.0

    def test_seasonal_adjustment_default_and_specific(self, agent: PricingAgent):
        """Seasonal adjustment should return crop-specific and default values."""
        tomato_may = agent.get_seasonal_adjustment("Tomato", 5)
        unknown_crop = agent.get_seasonal_adjustment("DragonFruit", 5)
        assert tomato_may == 1.3
        assert unknown_crop == 1.0
