"""
Order Management Service — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.api.services.order_pkg` package.
! Import from `src.api.services.order_pkg` directly in new code.
"""

from src.api.services.order_pkg.models import (
    AISP_FARMER_RATIO,
    AISP_LOGISTICS_RATIO,
    AISP_PLATFORM_RATIO,
    AISP_RISK_RATIO,
    AISPBreakdown,
    CreateOrderRequest,
    DisputeResponse,
    ESCROW_ON_TRANSITION,
    OrderResponse,
    RaiseDisputeRequest,
    UpdateStatusRequest,
    VALID_TRANSITIONS,
)
from src.api.services.order_pkg.service import OrderService, get_order_service

__all__ = [
    "AISP_FARMER_RATIO",
    "AISP_LOGISTICS_RATIO",
    "AISP_PLATFORM_RATIO",
    "AISP_RISK_RATIO",
    "AISPBreakdown",
    "CreateOrderRequest",
    "DisputeResponse",
    "ESCROW_ON_TRANSITION",
    "OrderResponse",
    "OrderService",
    "RaiseDisputeRequest",
    "UpdateStatusRequest",
    "VALID_TRANSITIONS",
    "get_order_service",
]
