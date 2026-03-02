"""
Shared fixtures for OrderService unit tests.

* TEST MODULE: OrderService fixtures
NOTE: All external dependencies (DB, pricing agent, quality agent) are mocked.
"""

from unittest.mock import AsyncMock

import pytest

from src.api.services.order_service import CreateOrderRequest, OrderService

from tests.unit.order_service.constants import (
    BUYER_UUID,
    DISPUTE_UUID,
    FARMER_UUID,
    LISTING_UUID,
    ORDER_UUID,
    SAMPLE_LISTING,
    SAMPLE_ORDER_ROW,
    TWIN_UUID,
)


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
