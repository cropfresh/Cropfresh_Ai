"""
Logistics Router — Cost Calculation
====================================
Cost model with deadhead factor for <₹2.5/kg target.
"""

# * COST MODULE
# NOTE: Cost = base_rate + per_km * total_distance_km.
# NOTE: Deadhead = return distance from last stop to depot (delivery).

from __future__ import annotations

from src.agents.logistics_router.geo import haversine_km
from src.agents.logistics_router.models import DeliveryPoint, PickupPoint
from src.agents.logistics_router.vehicle import VehicleConfig


def calculate_cost(
    pickup_sequence: list[PickupPoint],
    delivery: DeliveryPoint,
    vehicle: VehicleConfig,
    total_distance_km: float,
) -> tuple[float, float]:
    """
    Calculate estimated cost and deadhead distance.

    Args:
        pickup_sequence: Ordered pickup stops.
        delivery: Delivery destination.
        vehicle: Selected vehicle config.
        total_distance_km: Total route distance (includes return).

    Returns:
        (estimated_cost_inr, deadhead_km)
    """
    if not pickup_sequence:
        return vehicle.base_rate_inr, 0.0

    last = pickup_sequence[-1]
    deadhead_km = haversine_km(last.lat, last.lon, delivery.lat, delivery.lon)

    cost = vehicle.base_rate_inr + vehicle.per_km_rate_inr * total_distance_km
    return cost, deadhead_km
