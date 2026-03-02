"""
Unit tests — OrderService.raise_dispute.

* TEST MODULE: Dispute creation, twin diff trigger, status update
"""

from unittest.mock import AsyncMock

import pytest

from src.api.services.order_service import (
    DisputeResponse,
    OrderService,
    RaiseDisputeRequest,
)

from tests.unit.order_service.constants import (
    ORDER_UUID,
    SAMPLE_ORDER_ROW,
    TWIN_UUID,
)


@pytest.fixture
def in_transit_order_row():
    return {**SAMPLE_ORDER_ROW, "order_status": "in_transit"}


# * ═══════════════════════════════════════════════════════════════
# * RaiseDispute Tests
# * ═══════════════════════════════════════════════════════════════

class TestRaiseDispute:
    """Tests for the raise_dispute flow."""

    @pytest.mark.asyncio
    async def test_raise_dispute_returns_dispute_response(
        self, mock_db, in_transit_order_row
    ):
        """Happy path: returns DisputeResponse with all required fields."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db)
        req = RaiseDisputeRequest(raised_by="buyer", reason="Wrong grade delivered")
        result = await svc.raise_dispute(ORDER_UUID, req)
        assert isinstance(result, DisputeResponse)
        assert result.id == "eeeeeeee-0000-0000-0000-000000000001"
        assert result.order_id == ORDER_UUID
        assert result.raised_by == "buyer"
        assert result.status == "open"

    @pytest.mark.asyncio
    async def test_raise_dispute_advances_order_to_disputed(
        self, mock_db, in_transit_order_row
    ):
        """Order status must be updated to 'disputed' when dispute is raised."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db)
        req = RaiseDisputeRequest(raised_by="buyer", reason="Damage in transit")
        await svc.raise_dispute(ORDER_UUID, req)
        mock_db.update_order_status.assert_awaited_once_with(
            ORDER_UUID, "disputed", None
        )

    @pytest.mark.asyncio
    async def test_raise_dispute_from_invalid_status_raises_value_error(
        self, mock_db
    ):
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
        req = RaiseDisputeRequest(
            raised_by="farmer",
            reason="Buyer rejected unnecessarily",
        )
        result = await svc.raise_dispute(ORDER_UUID, req)
        mock_quality_agent.compare_twin.assert_not_awaited()
        assert result.diff_report is None

    @pytest.mark.asyncio
    async def test_raise_dispute_creates_db_record(
        self, mock_db, in_transit_order_row
    ):
        """Dispute must be persisted to the DB."""
        mock_db.get_order = AsyncMock(return_value=in_transit_order_row)
        svc = OrderService(db=mock_db)
        req = RaiseDisputeRequest(raised_by="buyer", reason="Substandard produce")
        await svc.raise_dispute(ORDER_UUID, req)
        mock_db.create_dispute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raise_dispute_order_not_found_raises_value_error(
        self, mock_db
    ):
        """Dispute on missing order must raise ValueError."""
        mock_db.get_order = AsyncMock(return_value=None)
        svc = OrderService(db=mock_db)
        req = RaiseDisputeRequest(raised_by="buyer", reason="test")
        with pytest.raises(ValueError, match="not found"):
            await svc.raise_dispute(ORDER_UUID, req)
