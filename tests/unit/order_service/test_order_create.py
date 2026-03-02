"""
Unit tests — OrderService.create_order and AISP calculation.

* TEST MODULE: Order creation, listing validation, AISP breakdown
"""

from unittest.mock import AsyncMock

import pytest

from src.api.services.order_service import (
    AISP_FARMER_RATIO,
    CreateOrderRequest,
    OrderResponse,
    OrderService,
)

from tests.unit.order_service.constants import (
    BUYER_UUID,
    LISTING_UUID,
    ORDER_UUID,
    SAMPLE_LISTING,
)


# * ═══════════════════════════════════════════════════════════════
# * CreateOrder Tests
# * ═══════════════════════════════════════════════════════════════

class TestCreateOrder:
    """Tests for the create_order flow."""

    @pytest.mark.asyncio
    async def test_create_order_returns_order_response(
        self, service, create_request, mock_db
    ):
        """Happy path: returns OrderResponse with all required fields."""
        result = await service.create_order(create_request)
        assert isinstance(result, OrderResponse)
        assert result.id == ORDER_UUID
        assert result.listing_id == LISTING_UUID
        assert result.buyer_id == BUYER_UUID
        assert result.quantity_kg == 100.0

    @pytest.mark.asyncio
    async def test_create_order_sets_escrow_held(self, service, create_request):
        """Escrow must be 'held' immediately at order creation."""
        result = await service.create_order(create_request)
        assert result.escrow_status == "held"

    @pytest.mark.asyncio
    async def test_create_order_status_confirmed(self, service, create_request):
        """Initial order_status must be 'confirmed'."""
        result = await service.create_order(create_request)
        assert result.order_status == "confirmed"

    @pytest.mark.asyncio
    async def test_create_order_aisp_breakdown_present(self, service, create_request):
        """AISP breakdown must be non-zero and internally consistent."""
        result = await service.create_order(create_request)
        aisp = result.aisp
        assert aisp.aisp_total > 0
        component_sum = (
            aisp.farmer_payout
            + aisp.logistics_cost
            + aisp.platform_margin
            + aisp.risk_buffer
        )
        assert abs(component_sum - aisp.aisp_total) < 0.02

    @pytest.mark.asyncio
    async def test_create_order_marks_listing_matched(
        self, service, create_request, mock_db
    ):
        """After order creation, listing status must be updated to 'matched'."""
        await service.create_order(create_request)
        mock_db.update_listing.assert_awaited_once_with(
            LISTING_UUID, {"status": "matched"}
        )

    @pytest.mark.asyncio
    async def test_create_order_persists_to_db(
        self, service, create_request, mock_db
    ):
        """create_order must call db.create_order with correct fields."""
        await service.create_order(create_request)
        mock_db.create_order.assert_awaited_once()
        call_data = mock_db.create_order.call_args[0][0]
        assert call_data["listing_id"] == LISTING_UUID
        assert call_data["buyer_id"] == create_request.buyer_id
        assert call_data["quantity_kg"] == 100.0

    @pytest.mark.asyncio
    async def test_create_order_inactive_listing_raises_value_error(
        self, mock_db, create_request
    ):
        """Creating an order for a non-active listing must raise ValueError."""
        mock_db.get_listing = AsyncMock(return_value={
            **SAMPLE_LISTING, "status": "expired"
        })
        svc = OrderService(db=mock_db)
        with pytest.raises(ValueError, match="not available"):
            await svc.create_order(create_request)

    @pytest.mark.asyncio
    async def test_create_order_listing_not_found_raises_value_error(
        self, mock_db, create_request
    ):
        """Creating an order when listing not found must raise ValueError."""
        mock_db.get_listing = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)
        with pytest.raises(ValueError, match="not found"):
            await svc.create_order(create_request)

    @pytest.mark.asyncio
    async def test_create_order_with_override_price(
        self, service, mock_db, create_request
    ):
        """Override price should be used in AISP calculation."""
        req = CreateOrderRequest(
            listing_id=LISTING_UUID,
            buyer_id=BUYER_UUID,
            quantity_kg=50.0,
            override_price_per_kg=40.0,
        )
        result = await service.create_order(req)
        assert result.aisp.aisp_per_kg == 40.0

    @pytest.mark.asyncio
    async def test_create_order_no_db_returns_uuid(
        self, no_db_service, create_request
    ):
        """Without DB, create_order must raise ValueError (listing not found)."""
        req = CreateOrderRequest(
            listing_id=LISTING_UUID,
            buyer_id=BUYER_UUID,
            quantity_kg=50.0,
        )
        with pytest.raises(ValueError, match="not found"):
            await no_db_service.create_order(req)


# * ═══════════════════════════════════════════════════════════════
# * AISP Ratio Fallback
# * ═══════════════════════════════════════════════════════════════

class TestAISPRatioFallback:
    """Tests for ratio-based AISP when no PricingAgent is injected."""

    @pytest.mark.asyncio
    async def test_aisp_ratios_sum_to_total(
        self, service, create_request, mock_db
    ):
        """Ratio components must sum to the total transaction value."""
        result = await service.create_order(create_request)
        aisp = result.aisp
        component_sum = (
            aisp.farmer_payout
            + aisp.logistics_cost
            + aisp.platform_margin
            + aisp.risk_buffer
        )
        assert abs(component_sum - aisp.aisp_total) < 0.02

    @pytest.mark.asyncio
    async def test_aisp_farmer_payout_is_80_percent(
        self, service, create_request
    ):
        """Farmer payout must be 80% of total value."""
        result = await service.create_order(create_request)
        aisp = result.aisp
        expected_farmer = round(aisp.aisp_total * AISP_FARMER_RATIO, 2)
        assert abs(aisp.farmer_payout - expected_farmer) < 0.02

    @pytest.mark.asyncio
    async def test_aisp_per_kg_matches_price(self, service, mock_db):
        """aisp_per_kg must equal the listing price per kg."""
        req = CreateOrderRequest(
            listing_id=LISTING_UUID,
            buyer_id=BUYER_UUID,
            quantity_kg=50.0,
        )
        result = await service.create_order(req)
        assert result.aisp.aisp_per_kg == 30.0

    @pytest.mark.asyncio
    async def test_aisp_total_is_quantity_times_price(
        self, service, create_request
    ):
        """AISP total must be quantity_kg × price_per_kg."""
        result = await service.create_order(create_request)
        expected_total = 100.0 * 30.0
        assert result.aisp.aisp_total == expected_total


# * ═══════════════════════════════════════════════════════════════
# * AISP with PricingAgent
# * ═══════════════════════════════════════════════════════════════

class TestAISPWithPricingAgent:
    """Tests for PricingAgent delegation."""

    @pytest.mark.asyncio
    async def test_pricing_agent_calculate_aisp_is_called(
        self, full_service, create_request, mock_pricing_agent
    ):
        """When PricingAgent has calculate_aisp, it must be called."""
        result = await full_service.create_order(create_request)
        mock_pricing_agent.calculate_aisp.assert_awaited_once()
        assert result.aisp.aisp_total == 3000.0

    @pytest.mark.asyncio
    async def test_pricing_agent_failure_falls_back_to_ratios(
        self, mock_db, create_request
    ):
        """If PricingAgent.calculate_aisp raises, fall back to ratio calculation."""
        failing_agent = AsyncMock()
        failing_agent.calculate_aisp = AsyncMock(
            side_effect=RuntimeError("Agent down")
        )
        svc = OrderService(db=mock_db, pricing_agent=failing_agent)
        result = await svc.create_order(create_request)
        assert result.aisp.aisp_total > 0
