"""
Logistics Router — Data Models
==============================
Data structures for pickup points, delivery points, and route results
used by the DPLE Logistics Routing Engine.
"""

# * LOGISTICS ROUTER MODELS MODULE
# NOTE: All models are dataclasses for lightweight serialisation.
# NOTE: PickupPoint/DeliveryPoint use (lat, lon) for GPS coordinates.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# * ═══════════════════════════════════════════════════════════════
# * Pickup and Delivery Points
# * ═══════════════════════════════════════════════════════════════

@dataclass
class PickupPoint:
    """
    Farm pickup location with weight for route planning.

    Used as input to LogisticsRouter.plan_route().
    """

    farm_id: str
    lat: float
    lon: float
    weight_kg: float
    listing_id: str = ""
    commodity_type: str = ""


@dataclass
class DeliveryPoint:
    """
    Buyer delivery location (destination).

    Used as input to LogisticsRouter.plan_route().
    """

    buyer_id: str
    lat: float
    lon: float
    address: str = ""


# * ═══════════════════════════════════════════════════════════════
# * Route Result
# * ═══════════════════════════════════════════════════════════════

@dataclass
class RouteResult:
    """
    Result of a planned multi-pickup route.

    Produced by LogisticsRouter.plan_route().
    Contains pickup sequence, vehicle assignment, cost breakdown,
    and utilization metrics for the <₹2.5/kg target.
    """

    route_id: str
    pickup_sequence: list[dict[str, Any]]  # Ordered farm stops
    total_distance_km: float
    total_weight_kg: float
    vehicle_type: str
    estimated_cost: float
    cost_per_kg: float
    utilization_pct: float
    estimated_duration_hours: float
    deadhead_km: float  # Empty return distance
    cluster_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for API responses."""
        return {
            "route_id": self.route_id,
            "pickup_sequence": self.pickup_sequence,
            "total_distance_km": round(self.total_distance_km, 2),
            "total_weight_kg": round(self.total_weight_kg, 2),
            "vehicle_type": self.vehicle_type,
            "estimated_cost": round(self.estimated_cost, 2),
            "cost_per_kg": round(self.cost_per_kg, 2),
            "utilization_pct": round(self.utilization_pct, 1),
            "estimated_duration_hours": round(self.estimated_duration_hours, 2),
            "deadhead_km": round(self.deadhead_km, 2),
            "cluster_size": self.cluster_size,
        }
