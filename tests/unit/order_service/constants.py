"""
Shared constants and sample data for OrderService unit tests.
"""

from typing import Any

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
