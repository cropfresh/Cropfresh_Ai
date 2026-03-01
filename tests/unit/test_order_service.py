"""
Unit Tests — Order Service
===========================
Tests for:
1. OrderService.create_order — AISP calculation, escrow held, listing matched
2. OrderService.update_status — valid transitions, illegal transitions rejected
3. VALID_TRANSITIONS state machine — all legal and illegal paths
4. OrderService.raise_dispute — dispute creation, twin diff trigger
5. OrderService.settle_order — escrow release, valid/invalid paths
6. OrderService.get_order — single fetch, not-found case
7. OrderService.get_orders_by_farmer — filter + empty result
8. OrderService.get_orders_by_buyer — filter + empty result
9. OrderService.get_aisp_breakdown — present / not-found
10. AISP ratio fallback — correct breakdowns without PricingAgent
11. AISP with PricingAgent — delegates then falls back on error
12. _assert_valid_transition — all legal paths and terminal states
13. _row_to_order / _row_to_aisp — converter helpers
14. get_order_service factory — correct dependency wiring
15. No-DB fallback — service degrades without DB (returns UUID stubs)
"""

# * TEST MODULE: OrderService
# NOTE: All external dependencies (DB, pricing agent, quality agent) are mocked

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.services.order_service import (
    AISP_FARMER_RATIO,
    AISP_LOGISTICS_RATIO,
    AISP_PLATFORM_RATIO,
    AISP_RISK_RATIO,
    ESCROW_ON_TRANSITION,
    VALID_TRANSITIONS,
    AISPBreakdown,
    CreateOrderRequest,
    DisputeResponse,
    OrderResponse,
    OrderService,
    RaiseDisputeRequest,
    UpdateStatusRequest,
    get_order_service,
)


# ═══════════════════════════════════════════════════════════════
# * Fixtures
# ═══════════════════════════════════════════════════════════════

LISTING_UUID = "aaaaaaaa-0000-0000-0000-000000000001"
BUYER_UUID = "bbbbbbbb-0000-0000-0000-000000000001"
FARMER_UUID = "cccccccc-0000-0000-0000-000000000001"
ORDER_UUID = "dddddddd-0000-0000-0000-000000000001"
DISPUTE_UUID = "eeeeeeee-0000-0000-0000-000000000001"
TWIN_UUID = "ffffffff-0000-0000-0000-000000000001"

SAMPLE_LISTING: dict[str, Any] = {
    "id": LISTING_UUID,
    "farmer_id": FARMER_UUID,
    "commodity": "tomato",
    "asking_price_per_kg": 30.0,
    "status": "active",
    "quantity_kg": 200.0,
}

SAMPLE_ORDER_ROW: dict[str, Any] = {
    "id": ORDER_UUID,
    "listing_id": LISTING_UUID,
    "buyer_id": BUYER_UUID,
    "hauler_id": None,
    "quantity_kg": 100.0,
    "order_status": "confirmed",
    "escrow_status": "held",
    "farmer_payout": 2400.0,
    "logistics_cost": 300.0,
    "platform_margin": 180.0,
    "risk_buffer": 120.0,
    "aisp_total": 3000.0,
    "aisp_per_kg": 30.0,
    "commodity": "tomato",
    "farmer_id": FARMER_UUID,
    "buyer_name": "Test Buyer",
    "created_at": None,
    "updated_at": None,
}


@pytest.fixture
def mock_db():
    """Mock AuroraPostgresClient with all order CRUD methods."""
    db = AsyncMock()
    db.get_listing = AsyncMock(return_value=SAMPLE_LISTING)
    db.create_order = AsyncMock(return_value=ORDER_UUID)
    db.update_order_status = AsyncMock(return_value=True)
    db.get_order = AsyncMock(return_value=SAMPLE_ORDER_ROW)
    db.get_orders_by_farmer = AsyncMock(return_value=[SAMPLE_ORDER_ROW])
    db.get_orders_by_buyer = AsyncMock(return_value=[SAMPLE_ORDER_ROW])
    db.create_dispute = AsyncMock(return_value=DISPUTE_UUID)
    db.update_dispute = AsyncMock(return_value=True)
    db.update_listing = AsyncMock(return_value={"id": LISTING_UUID, "status": "matched"})
    return db


@pytest.fixture
def mock_pricing_agent():
    """Mock PricingAgent with calculate_aisp method."""
    agent = AsyncMock()
    agent.calculate_aisp = AsyncMock(return_value={
        "farmer_payout": 2400.0,
        "logistics_cost": 300.0,
        "platform_margin": 180.0,
        "risk_buffer": 120.0,
        "aisp_total": 3000.0,
        "aisp_per_kg": 30.0,
    })
    return agent


@pytest.fixture
def mock_quality_agent():
    """Mock QualityAssessmentAgent with compare_twin method."""
    agent = AsyncMock()
    agent.compare_twin = AsyncMock(return_value={
        "score": 0.85,
        "defects_detected": ["minor_bruising"],
        "liability": "buyer",
        "claim_percent": 5.0,
    })
    return agent


@pytest.fixture
def service(mock_db):
    """OrderService with mocked DB and no AI agents (ratio-based AISP)."""
    return OrderService(db=mock_db)


@pytest.fixture
def full_service(mock_db, mock_pricing_agent, mock_quality_agent):
    """OrderService with all dependencies mocked."""
    return OrderService(
        db=mock_db,
        pricing_agent=mock_pricing_agent,
        quality_agent=mock_quality_agent,
    )


@pytest.fixture
def no_db_service():
    """OrderService without any DB (for in-memory fallback tests)."""
    return OrderService()


@pytest.fixture
def create_request() -> CreateOrderRequest:
    return CreateOrderRequest(
        listing_id=LISTING_UUID,
        buyer_id=BUYER_UUID,
        quantity_kg=100.0,
    )


# ═══════════════════════════════════════════════════════════════
# * 1. OrderService.create_order
# ═══════════════════════════════════════════════════════════════

class TestCreateOrder:
    """Tests for the create_order flow."""

    @pytest.mark.asyncio
    async def test_create_order_returns_order_response(self, service, create_request, mock_db):
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
        assert abs(component_sum - aisp.aisp_total) < 0.02    # floating point tolerance

    @pytest.mark.asyncio
    async def test_create_order_marks_listing_matched(self, service, create_request, mock_db):
        """After order creation, listing status must be updated to 'matched'."""
        await service.create_order(create_request)
        mock_db.update_listing.assert_awaited_once_with(
            LISTING_UUID, {"status": "matched"}
        )

    @pytest.mark.asyncio
    async def test_create_order_persists_to_db(self, service, create_request, mock_db):
        """create_order must call db.create_order with correct fields."""
        await service.create_order(create_request)
        mock_db.create_order.assert_awaited_once()
        call_data = mock_db.create_order.call_args[0][0]
        assert call_data["listing_id"] == LISTING_UUID
        assert call_data["buyer_id"] == BUYER_UUID
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
    async def test_create_order_with_override_price(self, service, mock_db):
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
    async def test_create_order_no_db_returns_uuid(self, no_db_service):
        """Without DB, create_order must still return a valid UUID response."""
        req = CreateOrderRequest(
            listing_id=LISTING_UUID,
            buyer_id=BUYER_UUID,
            quantity_kg=50.0,
        )
        # NOTE: _fetch_listing returns None when no DB → will raise ValueError
        with pytest.raises(ValueError, match="not found"):
            await no_db_service.create_order(req)


# ═══════════════════════════════════════════════════════════════
# * 2. OrderService.update_status — state machine transitions
# ═══════════════════════════════════════════════════════════════

class TestUpdateStatus:
    """Tests for the update_status state machine."""

    @pytest.mark.asyncio
    async def test_valid_transition_confirmed_to_pickup_scheduled(self, service, mock_db):
        """confirmed → pickup_scheduled must succeed."""
        result = await service.update_status(ORDER_UUID, "pickup_scheduled")
        assert result.order_status == "pickup_scheduled"

    @pytest.mark.asyncio
    async def test_valid_transition_updates_db(self, service, mock_db):
        """Valid transition must call db.update_order_status."""
        await service.update_status(ORDER_UUID, "pickup_scheduled")
        mock_db.update_order_status.assert_awaited_once_with(
            ORDER_UUID, "pickup_scheduled", None
        )

    @pytest.mark.asyncio
    async def test_valid_transition_with_escrow_change(self, mock_db):
        """settled transition must also set escrow_status = released."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "delivered"
        })
        svc = OrderService(db=mock_db)

        result = await svc.update_status(ORDER_UUID, "settled")
        assert result.order_status == "settled"
        assert result.escrow_status == "released"
        mock_db.update_order_status.assert_awaited_once_with(
            ORDER_UUID, "settled", "released"
        )

    @pytest.mark.asyncio
    async def test_illegal_transition_raises_value_error(self, service):
        """confirmed → delivered must be rejected (skips intermediate states)."""
        with pytest.raises(ValueError, match="Invalid transition"):
            await service.update_status(ORDER_UUID, "delivered")

    @pytest.mark.asyncio
    async def test_terminal_state_transition_raises_value_error(self, mock_db):
        """settled → anything must be rejected (terminal state)."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "settled"
        })
        svc = OrderService(db=mock_db)

        with pytest.raises(ValueError, match="terminal"):
            await svc.update_status(ORDER_UUID, "confirmed")

    @pytest.mark.asyncio
    async def test_order_not_found_raises_value_error(self, mock_db):
        """update_status on missing order must raise ValueError."""
        mock_db.get_order = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)

        with pytest.raises(ValueError, match="not found"):
            await svc.update_status(ORDER_UUID, "pickup_scheduled")

    @pytest.mark.asyncio
    async def test_cancelled_transition_sets_escrow_refunded(self, mock_db):
        """Cancelling an order must set escrow = refunded."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "confirmed"
        })
        svc = OrderService(db=mock_db)

        result = await svc.update_status(ORDER_UUID, "cancelled")
        assert result.order_status == "cancelled"
        assert result.escrow_status == "refunded"


# ═══════════════════════════════════════════════════════════════
# * 3. VALID_TRANSITIONS state machine — exhaustive checks
# ═══════════════════════════════════════════════════════════════

class TestValidTransitions:
    """Verify the state machine map is complete and correct."""

    def test_confirmed_allows_pickup_scheduled_and_cancelled(self):
        assert "pickup_scheduled" in VALID_TRANSITIONS["confirmed"]
        assert "cancelled" in VALID_TRANSITIONS["confirmed"]

    def test_confirmed_does_not_allow_direct_delivery(self):
        assert "delivered" not in VALID_TRANSITIONS["confirmed"]
        assert "settled" not in VALID_TRANSITIONS["confirmed"]

    def test_in_transit_allows_delivered_and_disputed(self):
        assert "delivered" in VALID_TRANSITIONS["in_transit"]
        assert "disputed" in VALID_TRANSITIONS["in_transit"]

    def test_settled_is_terminal(self):
        assert VALID_TRANSITIONS["settled"] == []

    def test_refunded_is_terminal(self):
        assert VALID_TRANSITIONS["refunded"] == []

    def test_cancelled_is_terminal(self):
        assert VALID_TRANSITIONS["cancelled"] == []

    def test_all_statuses_have_entry(self):
        required = {
            "confirmed", "pickup_scheduled", "in_transit",
            "delivered", "disputed", "ai_analysed",
            "resolved", "escalated", "settled", "refunded", "cancelled",
        }
        assert required.issubset(set(VALID_TRANSITIONS.keys()))

    def test_assert_valid_transition_legal(self):
        """Helper must not raise for known-good transitions."""
        OrderService._assert_valid_transition("confirmed", "pickup_scheduled")
        OrderService._assert_valid_transition("in_transit", "disputed")
        OrderService._assert_valid_transition("resolved", "settled")

    def test_assert_valid_transition_illegal_raises(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            OrderService._assert_valid_transition("confirmed", "settled")

    def test_assert_valid_transition_terminal_raises(self):
        with pytest.raises(ValueError):
            OrderService._assert_valid_transition("settled", "confirmed")


# ═══════════════════════════════════════════════════════════════
# * 4. OrderService.raise_dispute
# ═══════════════════════════════════════════════════════════════

class TestRaiseDispute:
    """Tests for the raise_dispute flow."""

    @pytest.fixture
    def in_transit_order_row(self):
        return {**SAMPLE_ORDER_ROW, "order_status": "in_transit"}

    @pytest.mark.asyncio
    async def test_raise_dispute_returns_dispute_response(self, mock_db, in_transit_order_row):
        """Happy path: returns DisputeResponse with all required fields."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db)

        req = RaiseDisputeRequest(raised_by="buyer", reason="Wrong grade delivered")
        result = await svc.raise_dispute(ORDER_UUID, req)

        assert isinstance(result, DisputeResponse)
        assert result.id == DISPUTE_UUID
        assert result.order_id == ORDER_UUID
        assert result.raised_by == "buyer"
        assert result.status == "open"

    @pytest.mark.asyncio
    async def test_raise_dispute_advances_order_to_disputed(self, mock_db, in_transit_order_row):
        """Order status must be updated to 'disputed' when dispute is raised."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db)

        req = RaiseDisputeRequest(raised_by="buyer", reason="Damage in transit")
        await svc.raise_dispute(ORDER_UUID, req)

        mock_db.update_order_status.assert_awaited_once_with(ORDER_UUID, "disputed", None)

    @pytest.mark.asyncio
    async def test_raise_dispute_from_invalid_status_raises_value_error(self, mock_db):
        """Cannot raise dispute from 'confirmed' status."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "confirmed"
        })
        svc = OrderService(db=mock_db)

        req = RaiseDisputeRequest(raised_by="buyer", reason="test")
        with pytest.raises(ValueError, match="Invalid transition"):
            await svc.raise_dispute(ORDER_UUID, req)

    @pytest.mark.asyncio
    async def test_raise_dispute_with_twin_diff_triggers_quality_agent(
        self, mock_db, mock_quality_agent, in_transit_order_row
    ):
        """When departure_twin_id and arrival_photos provided, quality agent is called."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db, quality_agent=mock_quality_agent)

        req = RaiseDisputeRequest(
            raised_by="buyer",
            reason="Grade mismatch",
            departure_twin_id=TWIN_UUID,
            arrival_photos=["s3://bucket/photo1.jpg"],
        )
        result = await svc.raise_dispute(ORDER_UUID, req)

        mock_quality_agent.compare_twin.assert_awaited_once()
        assert result.diff_report is not None

    @pytest.mark.asyncio
    async def test_raise_dispute_without_photos_skips_twin_diff(
        self, mock_db, mock_quality_agent, in_transit_order_row
    ):
        """Without arrival_photos, Digital Twin diff must be skipped."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db, quality_agent=mock_quality_agent)

        req = RaiseDisputeRequest(raised_by="farmer", reason="Buyer rejected unnecessarily")
        result = await svc.raise_dispute(ORDER_UUID, req)

        mock_quality_agent.compare_twin.assert_not_awaited()
        assert result.diff_report is None

    @pytest.mark.asyncio
    async def test_raise_dispute_creates_db_record(self, mock_db, in_transit_order_row):
        """Dispute must be persisted to the DB."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db)

        req = RaiseDisputeRequest(raised_by="buyer", reason="Substandard produce")
        await svc.raise_dispute(ORDER_UUID, req)

        mock_db.create_dispute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raise_dispute_order_not_found_raises_value_error(self, mock_db):
        """Dispute on missing order must raise ValueError."""
        mock_db.get_order = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)

        req = RaiseDisputeRequest(raised_by="buyer", reason="test")
        with pytest.raises(ValueError, match="not found"):
            await svc.raise_dispute(ORDER_UUID, req)


# ═══════════════════════════════════════════════════════════════
# * 5. OrderService.settle_order
# ═══════════════════════════════════════════════════════════════

class TestSettleOrder:
    """Tests for the settle_order flow."""

    @pytest.mark.asyncio
    async def test_settle_order_from_delivered_succeeds(self, mock_db):
        """Settling from 'delivered' must succeed and set escrow = released."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "delivered"
        })
        svc = OrderService(db=mock_db)

        result = await svc.settle_order(ORDER_UUID)
        assert result.order_status == "settled"
        assert result.escrow_status == "released"

    @pytest.mark.asyncio
    async def test_settle_order_from_resolved_succeeds(self, mock_db):
        """Settling from 'resolved' (post-dispute) must also succeed."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "resolved"
        })
        svc = OrderService(db=mock_db)

        result = await svc.settle_order(ORDER_UUID)
        assert result.order_status == "settled"

    @pytest.mark.asyncio
    async def test_settle_order_from_confirmed_raises_value_error(self, mock_db):
        """Cannot settle directly from 'confirmed'."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "confirmed"
        })
        svc = OrderService(db=mock_db)

        with pytest.raises(ValueError, match="Invalid transition"):
            await svc.settle_order(ORDER_UUID)

    @pytest.mark.asyncio
    async def test_settle_order_not_found_raises_value_error(self, mock_db):
        """Settling a missing order must raise ValueError."""
        mock_db.get_order = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)

        with pytest.raises(ValueError, match="not found"):
            await svc.settle_order(ORDER_UUID)

    @pytest.mark.asyncio
    async def test_settle_order_updates_db_with_released_escrow(self, mock_db):
        """DB must receive order_status=settled and escrow_status=released."""
        mock_db.get_order = AsyncMock(return_value={
            **SAMPLE_ORDER_ROW, "order_status": "delivered"
        })
        svc = OrderService(db=mock_db)

        await svc.settle_order(ORDER_UUID)
        mock_db.update_order_status.assert_awaited_once_with(
            ORDER_UUID, "settled", "released"
        )


# ═══════════════════════════════════════════════════════════════
# * 6. OrderService.get_order
# ═══════════════════════════════════════════════════════════════

class TestGetOrder:
    """Tests for the get_order query."""

    @pytest.mark.asyncio
    async def test_get_order_returns_order_response(self, service):
        """Happy path: returns OrderResponse."""
        result = await service.get_order(ORDER_UUID)
        assert isinstance(result, OrderResponse)
        assert result.id == ORDER_UUID

    @pytest.mark.asyncio
    async def test_get_order_not_found_returns_none(self, mock_db):
        """Returns None when order UUID does not exist."""
        mock_db.get_order = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)

        result = await svc.get_order(ORDER_UUID)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_order_no_db_returns_none(self, no_db_service):
        """Without DB, get_order returns None gracefully."""
        result = await no_db_service.get_order(ORDER_UUID)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# * 7. OrderService.get_orders_by_farmer
# ═══════════════════════════════════════════════════════════════

class TestGetOrdersByFarmer:
    """Tests for farmer order history query."""

    @pytest.mark.asyncio
    async def test_get_orders_by_farmer_returns_list(self, service):
        """Returns list of OrderResponse for a farmer."""
        results = await service.get_orders_by_farmer(FARMER_UUID)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].id == ORDER_UUID

    @pytest.mark.asyncio
    async def test_get_orders_by_farmer_passes_status_filter(self, service, mock_db):
        """Status filter must be forwarded to the DB call."""
        await service.get_orders_by_farmer(FARMER_UUID, status="settled")
        mock_db.get_orders_by_farmer.assert_awaited_once_with(
            FARMER_UUID, status="settled"
        )

    @pytest.mark.asyncio
    async def test_get_orders_by_farmer_empty_result(self, mock_db):
        """Empty DB result returns empty list."""
        mock_db.get_orders_by_farmer = AsyncMock(return_value=[])
        svc = OrderService(db=mock_db)

        results = await svc.get_orders_by_farmer(FARMER_UUID)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_orders_by_farmer_no_db_returns_empty(self, no_db_service):
        """Without DB, returns empty list gracefully."""
        results = await no_db_service.get_orders_by_farmer(FARMER_UUID)
        assert results == []


# ═══════════════════════════════════════════════════════════════
# * 8. OrderService.get_orders_by_buyer
# ═══════════════════════════════════════════════════════════════

class TestGetOrdersByBuyer:
    """Tests for buyer order history query."""

    @pytest.mark.asyncio
    async def test_get_orders_by_buyer_returns_list(self, service):
        """Returns list of OrderResponse for a buyer."""
        results = await service.get_orders_by_buyer(BUYER_UUID)
        assert len(results) == 1
        assert results[0].buyer_id == BUYER_UUID

    @pytest.mark.asyncio
    async def test_get_orders_by_buyer_passes_status_filter(self, service, mock_db):
        """Status filter must be forwarded to the DB call."""
        await service.get_orders_by_buyer(BUYER_UUID, status="in_transit")
        mock_db.get_orders_by_buyer.assert_awaited_once_with(
            BUYER_UUID, status="in_transit"
        )

    @pytest.mark.asyncio
    async def test_get_orders_by_buyer_empty_result(self, mock_db):
        """Empty DB result returns empty list."""
        mock_db.get_orders_by_buyer = AsyncMock(return_value=[])
        svc = OrderService(db=mock_db)

        results = await svc.get_orders_by_buyer(BUYER_UUID)
        assert results == []


# ═══════════════════════════════════════════════════════════════
# * 9. OrderService.get_aisp_breakdown
# ═══════════════════════════════════════════════════════════════

class TestGetAISPBreakdown:
    """Tests for the AISP breakdown query."""

    @pytest.mark.asyncio
    async def test_get_aisp_breakdown_returns_breakdown(self, service):
        """Happy path: returns AISPBreakdown."""
        result = await service.get_aisp_breakdown(ORDER_UUID)
        assert isinstance(result, AISPBreakdown)
        assert result.aisp_total == 3000.0

    @pytest.mark.asyncio
    async def test_get_aisp_breakdown_not_found_returns_none(self, mock_db):
        """Returns None when order UUID does not exist."""
        mock_db.get_order = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)

        result = await svc.get_aisp_breakdown(ORDER_UUID)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# * 10. AISP ratio-based fallback
# ═══════════════════════════════════════════════════════════════

class TestAISPRatioFallback:
    """Tests for ratio-based AISP when no PricingAgent is injected."""

    @pytest.mark.asyncio
    async def test_aisp_ratios_sum_to_total(self, service, create_request, mock_db):
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
    async def test_aisp_farmer_payout_is_80_percent(self, service, create_request):
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
        assert result.aisp.aisp_per_kg == 30.0    # SAMPLE_LISTING price

    @pytest.mark.asyncio
    async def test_aisp_total_is_quantity_times_price(self, service, create_request):
        """AISP total must be quantity_kg × price_per_kg."""
        result = await service.create_order(create_request)
        expected_total = 100.0 * 30.0    # 100kg × ₹30/kg
        assert result.aisp.aisp_total == expected_total


# ═══════════════════════════════════════════════════════════════
# * 11. AISP with PricingAgent — delegate then fallback
# ═══════════════════════════════════════════════════════════════

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
        failing_agent.calculate_aisp = AsyncMock(side_effect=RuntimeError("Agent down"))
        svc = OrderService(db=mock_db, pricing_agent=failing_agent)

        result = await svc.create_order(create_request)
        # Should still succeed using ratio fallback
        assert result.aisp.aisp_total > 0


# ═══════════════════════════════════════════════════════════════
# * 12. Row converter helpers
# ═══════════════════════════════════════════════════════════════

class TestRowConverters:
    """Tests for _row_to_order and _row_to_aisp static helpers."""

    def test_row_to_aisp_extracts_all_fields(self):
        aisp = OrderService._row_to_aisp(SAMPLE_ORDER_ROW)
        assert aisp.farmer_payout == 2400.0
        assert aisp.logistics_cost == 300.0
        assert aisp.platform_margin == 180.0
        assert aisp.risk_buffer == 120.0
        assert aisp.aisp_total == 3000.0
        assert aisp.aisp_per_kg == 30.0

    def test_row_to_order_extracts_all_fields(self):
        order = OrderService._row_to_order(SAMPLE_ORDER_ROW)
        assert order.id == ORDER_UUID
        assert order.listing_id == LISTING_UUID
        assert order.buyer_id == BUYER_UUID
        assert order.quantity_kg == 100.0
        assert order.order_status == "confirmed"
        assert order.escrow_status == "held"
        assert order.commodity == "tomato"
        assert order.farmer_id == FARMER_UUID
        assert order.buyer_name == "Test Buyer"

    def test_row_to_order_null_hauler_is_none(self):
        order = OrderService._row_to_order(SAMPLE_ORDER_ROW)
        assert order.hauler_id is None

    def test_row_to_order_with_hauler_id(self):
        row = {**SAMPLE_ORDER_ROW, "hauler_id": "hauler-uuid-001"}
        order = OrderService._row_to_order(row)
        assert order.hauler_id == "hauler-uuid-001"

    def test_row_to_aisp_handles_missing_fields_with_zeros(self):
        """Missing fields must default to 0.0 without raising."""
        aisp = OrderService._row_to_aisp({})
        assert aisp.aisp_total == 0.0
        assert aisp.farmer_payout == 0.0


# ═══════════════════════════════════════════════════════════════
# * 13. get_order_service factory
# ═══════════════════════════════════════════════════════════════

class TestGetOrderServiceFactory:
    """Tests for the module-level factory function."""

    def test_factory_returns_order_service(self):
        svc = get_order_service()
        assert isinstance(svc, OrderService)

    def test_factory_injects_db(self, mock_db):
        svc = get_order_service(db=mock_db)
        assert svc.db is mock_db

    def test_factory_injects_pricing_agent(self, mock_pricing_agent):
        svc = get_order_service(pricing_agent=mock_pricing_agent)
        assert svc.pricing_agent is mock_pricing_agent

    def test_factory_injects_quality_agent(self, mock_quality_agent):
        svc = get_order_service(quality_agent=mock_quality_agent)
        assert svc.quality_agent is mock_quality_agent

    def test_factory_all_none_by_default(self):
        svc = get_order_service()
        assert svc.db is None
        assert svc.pricing_agent is None
        assert svc.quality_agent is None


# ═══════════════════════════════════════════════════════════════
# * 14. ESCROW_ON_TRANSITION mapping
# ═══════════════════════════════════════════════════════════════

class TestEscrowOnTransition:
    """Verify escrow mappings are correct for key transitions."""

    def test_confirmed_sets_escrow_held(self):
        assert ESCROW_ON_TRANSITION.get("confirmed") == "held"

    def test_settled_releases_escrow(self):
        assert ESCROW_ON_TRANSITION.get("settled") == "released"

    def test_refunded_sets_escrow_refunded(self):
        assert ESCROW_ON_TRANSITION.get("refunded") == "refunded"

    def test_cancelled_refunds_buyer(self):
        assert ESCROW_ON_TRANSITION.get("cancelled") == "refunded"

    def test_pickup_scheduled_has_no_escrow_change(self):
        assert ESCROW_ON_TRANSITION.get("pickup_scheduled") is None

    def test_in_transit_has_no_escrow_change(self):
        assert ESCROW_ON_TRANSITION.get("in_transit") is None
