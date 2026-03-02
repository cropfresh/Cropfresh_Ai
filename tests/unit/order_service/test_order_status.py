"""
Unit tests — OrderService.update_status, settle_order, state machine.

* TEST MODULE: Status transitions, escrow flow, VALID_TRANSITIONS
"""

from unittest.mock import AsyncMock

import pytest

from src.api.services.order_service import (
    ESCROW_ON_TRANSITION,
    VALID_TRANSITIONS,
    OrderService,
)

from tests.unit.order_service.constants import ORDER_UUID, SAMPLE_ORDER_ROW


# * ═══════════════════════════════════════════════════════════════
# * UpdateStatus Tests
# * ═══════════════════════════════════════════════════════════════

class TestUpdateStatus:
    """Tests for the update_status state machine."""

    @pytest.mark.asyncio
    async def test_valid_transition_confirmed_to_pickup_scheduled(
        self, service, mock_db
    ):
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


# * ═══════════════════════════════════════════════════════════════
# * SettleOrder Tests
# * ═══════════════════════════════════════════════════════════════

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


# * ═══════════════════════════════════════════════════════════════
# * VALID_TRANSITIONS State Machine
# * ═══════════════════════════════════════════════════════════════

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

    @pytest.mark.parametrize("from_status,to_status,valid", [
        ("confirmed", "pickup_scheduled", True),
        ("confirmed", "delivered", False),
        ("in_transit", "disputed", True),
        ("in_transit", "delivered", True),
        ("settled", "confirmed", False),
        ("resolved", "settled", True),
        ("pickup_scheduled", "in_transit", True),
        ("delivered", "settled", True),
    ])
    def test_order_state_transitions(
        self, from_status: str, to_status: str, valid: bool
    ):
        """Parametrized: all legal and illegal state machine paths."""
        if valid:
            OrderService._assert_valid_transition(from_status, to_status)
        else:
            with pytest.raises(ValueError):
                OrderService._assert_valid_transition(from_status, to_status)


# * ═══════════════════════════════════════════════════════════════
# * ESCROW_ON_TRANSITION
# * ═══════════════════════════════════════════════════════════════

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
