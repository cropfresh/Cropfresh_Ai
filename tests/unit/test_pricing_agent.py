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
        assert result.logistics_cost == 200.0  # 10km → ₹2/kg tier
        assert result.risk_buffer > 0
        assert result.risk_buffer_pct == 0.02
        assert result.aisp_per_kg > 20.0
        assert result.total_aisp == result.aisp_per_kg * result.quantity_kg

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

    def test_deadhead_surcharge_applied_over_threshold(self, agent: PricingAgent):
        """Deadhead surcharge applies when distance > 15km."""
        short = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=100, distance_km=10,
        )
        long = agent.calculate_aisp(
            farmer_price_per_kg=10.0, quantity_kg=100, distance_km=30,
        )

        assert short.deadhead_surcharge == 0.0
        assert long.deadhead_surcharge == 0.50 * 100  # ₹0.50/kg * 100kg

    def test_mandi_cap_limits_aisp(self, agent: PricingAgent):
        """AISP should not exceed mandi modal price when cap is provided."""
        result = agent.calculate_aisp(
            farmer_price_per_kg=20.0,
            quantity_kg=100,
            distance_km=50,
            mandi_modal_per_kg=22.0,
        )

        assert result.aisp_per_kg <= 22.0
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

        assert small.platform_fee_pct == 0.08
        assert medium.platform_fee_pct == 0.06
        assert large.platform_fee_pct == 0.04

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
