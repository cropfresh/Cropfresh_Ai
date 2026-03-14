"""
Order service data models, constants, and state machine configuration.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# * Order State Machine
# ═══════════════════════════════════════════════════════════════

# * Legal forward-transitions for every order_status value.
# ! ALERT: Any transition not listed here will be rejected with a 422.
VALID_TRANSITIONS: dict[str, list[str]] = {
    "confirmed": ["pickup_scheduled", "cancelled"],
    "pickup_scheduled": ["in_transit", "cancelled"],
    "in_transit": ["delivered", "disputed"],
    "delivered": ["settled", "disputed"],
    "disputed": ["ai_analysed"],
    "ai_analysed": ["resolved", "escalated"],
    "resolved": ["settled"],
    "escalated": ["settled", "refunded"],
    "settled": [],       # Terminal — no further transitions
    "refunded": [],      # Terminal — escrow fully returned
    "cancelled": [],     # Terminal — order abandoned
}

# * Escrow status applied for each terminal/key transition
ESCROW_ON_TRANSITION: dict[str, str] = {
    "confirmed": "held",         # Escrow locked at order creation
    "settled": "released",       # Farmer paid on delivery confirmation
    "refunded": "refunded",      # Buyer refunded on dispute resolution
    "cancelled": "refunded",     # Cancelled before pickup — refund buyer
}

# * AISP cost breakdown ratios (of total transaction value)
AISP_FARMER_RATIO: float = 0.80
AISP_LOGISTICS_RATIO: float = 0.10
AISP_PLATFORM_RATIO: float = 0.06
AISP_RISK_RATIO: float = 0.04


# ═══════════════════════════════════════════════════════════════
# * Pydantic Models
# ═══════════════════════════════════════════════════════════════

class AISPBreakdown(BaseModel):
    """AI-Set Price breakdown for transaction value distribution."""
    farmer_payout: float
    logistics_cost: float
    platform_margin: float
    risk_buffer: float
    aisp_total: float
    aisp_per_kg: float


class CreateOrderRequest(BaseModel):
    """Request body for creating a new order from a matched listing."""
    listing_id: str
    buyer_id: str
    quantity_kg: float = Field(gt=0, description="Must be positive")
    hauler_id: Optional[str] = None
    override_price_per_kg: Optional[float] = Field(
        default=None, ge=0,
        description="Override AISP price (e.g., negotiated price)"
    )


class UpdateStatusRequest(BaseModel):
    """Request body for advancing an order through the state machine."""
    status: str
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Context for the transition (e.g., GPS coordinates, notes)"
    )


class RaiseDisputeRequest(BaseModel):
    """Request body for raising a dispute with supporting evidence."""
    raised_by: str = Field(description="'buyer' or 'farmer'")
    reason: str
    arrival_photos: Optional[list[str]] = None
    departure_twin_id: Optional[str] = None


class OrderResponse(BaseModel):
    """Complete order representation returned to callers."""
    id: str
    listing_id: str
    buyer_id: str
    hauler_id: Optional[str] = None
    quantity_kg: float
    order_status: str
    escrow_status: str
    aisp: AISPBreakdown
    commodity: Optional[str] = None
    farmer_id: Optional[str] = None
    buyer_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DisputeResponse(BaseModel):
    """Dispute record returned to callers."""
    id: str
    order_id: str
    raised_by: str
    reason: str
    status: str
    arrival_photos: Optional[list[str]] = None
    departure_twin_id: Optional[str] = None
    diff_report: Optional[dict[str, Any]] = None
    liability: Optional[str] = None
    claim_percent: Optional[float] = None
    created_at: Optional[datetime] = None
