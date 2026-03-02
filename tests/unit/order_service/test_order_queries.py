"""
Unit tests — OrderService get_order, get_orders_by_farmer/buyer, get_aisp_breakdown.

* TEST MODULE: Query methods, filters, empty results
"""

from unittest.mock import AsyncMock

import pytest

from src.api.services.order_service import (
    AISPBreakdown,
    OrderResponse,
    OrderService,
)

from tests.unit.order_service.constants import (
    BUYER_UUID,
    FARMER_UUID,
    ORDER_UUID,
)


# * ═══════════════════════════════════════════════════════════════
# * GetOrder Tests
# * ═══════════════════════════════════════════════════════════════

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


# * ═══════════════════════════════════════════════════════════════
# * GetOrdersByFarmer Tests
# * ═══════════════════════════════════════════════════════════════

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
    async def test_get_orders_by_farmer_passes_status_filter(
        self, service, mock_db
    ):
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
    async def test_get_orders_by_farmer_no_db_returns_empty(
        self, no_db_service
    ):
        """Without DB, returns empty list gracefully."""
        results = await no_db_service.get_orders_by_farmer(FARMER_UUID)
        assert results == []


# * ═══════════════════════════════════════════════════════════════
# * GetOrdersByBuyer Tests
# * ═══════════════════════════════════════════════════════════════

class TestGetOrdersByBuyer:
    """Tests for buyer order history query."""

    @pytest.mark.asyncio
    async def test_get_orders_by_buyer_returns_list(self, service):
        """Returns list of OrderResponse for a buyer."""
        results = await service.get_orders_by_buyer(BUYER_UUID)
        assert len(results) == 1
        assert results[0].buyer_id == BUYER_UUID

    @pytest.mark.asyncio
    async def test_get_orders_by_buyer_passes_status_filter(
        self, service, mock_db
    ):
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


# * ═══════════════════════════════════════════════════════════════
# * GetAISPBreakdown Tests
# * ═══════════════════════════════════════════════════════════════

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
