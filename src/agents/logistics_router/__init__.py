"""
DPLE Logistics Routing Engine
==============================
Multi-pickup clustering and route optimization for <₹2.5/kg target.

Primary entry point:
    router.plan_route(pickups, delivery) → RouteResult
"""

from src.agents.logistics_router.engine import LogisticsRouter, get_logistics_router
from src.agents.logistics_router.models import (
    DeliveryPoint,
    PickupPoint,
    RouteResult,
)

__all__ = [
    "LogisticsRouter",
    "get_logistics_router",
    "PickupPoint",
    "DeliveryPoint",
    "RouteResult",
]
