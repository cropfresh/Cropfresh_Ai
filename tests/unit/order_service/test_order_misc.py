"""
Unit tests — OrderService row converters, factory, constants.

* TEST MODULE: _row_to_order, _row_to_aisp, get_order_service
"""

from unittest.mock import MagicMock

import pytest

from src.api.services.order_service import (
    OrderService,
    get_order_service,
)

from tests.unit.order_service.constants import (
    BUYER_UUID,
    FARMER_UUID,
    LISTING_UUID,
    ORDER_UUID,
    SAMPLE_ORDER_ROW,
)


# * ═══════════════════════════════════════════════════════════════
# * Row Converter Tests
# * ═══════════════════════════════════════════════════════════════

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


# * ═══════════════════════════════════════════════════════════════
# * get_order_service Factory
# * ═══════════════════════════════════════════════════════════════

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
