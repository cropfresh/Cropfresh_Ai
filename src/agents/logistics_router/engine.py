"""
DPLE Logistics Routing Engine
==============================
Multi-pickup clustering and route optimization for <₹2.5/kg target.

Algorithm:
1. Cluster nearby farms (HDBSCAN, min_cluster_size=2)
2. For each cluster, solve TSP (OR-Tools or greedy)
3. Assign optimal vehicle type by total weight
4. Calculate cost with deadhead factor
5. Return route ranked by cost_per_kg
"""

# * LOGISTICS ROUTING ENGINE MODULE
# NOTE: Cluster-first, route-second approach per task11 research.
# NOTE: 5-farm cluster at 30km delivery should achieve <₹2.5/kg.

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from loguru import logger

from src.agents.logistics_router.clustering import cluster_pickups
from src.agents.logistics_router.cost import calculate_cost
from src.agents.logistics_router.models import (
    DeliveryPoint,
    PickupPoint,
    RouteResult,
)
from src.agents.logistics_router.routing import solve_tsp
from src.agents.logistics_router.vehicle import select_vehicle


class LogisticsRouter:
    """
    DPLE Logistics Routing Engine.

    Achieves <₹2.5/kg via multi-pickup clustering and vehicle utilization.
    """

    def __init__(self) -> None:
        pass

    async def plan_route(
        self,
        pickups: list[PickupPoint],
        delivery: DeliveryPoint,
        max_stops: int = 8,
        cold_chain_required: bool = False,
    ) -> Optional[RouteResult]:
        """
        Plan optimal multi-pickup route.

        Args:
            pickups: Farm locations with weights.
            delivery: Buyer delivery point.
            max_stops: Maximum pickup stops per route.
            cold_chain_required: True for perishables.

        Returns:
            RouteResult with cost_per_kg, utilization, etc.
            None if no pickups.
        """
        if not pickups:
            return None

        clusters = cluster_pickups(pickups, min_cluster_size=2)

        # * Always include the full pickup set as a candidate (depot-and-all).
        # HDBSCAN may split nearby farms into multiple small clusters when the
        # full set still fits in a single vehicle and achieves better cost/kg.
        all_as_one = list(pickups)
        candidate_clusters = [all_as_one] + [
            c for c in clusters if c != all_as_one and sorted(c, key=lambda p: p.farm_id) != sorted(all_as_one, key=lambda p: p.farm_id)
        ]

        best: Optional[RouteResult] = None

        for cluster in candidate_clusters:
            if len(cluster) > max_stops:
                continue
            total_weight = sum(p.weight_kg for p in cluster)
            if total_weight <= 0:
                continue

            ordered, total_km = solve_tsp(cluster, delivery)
            vehicle = select_vehicle(total_weight, cold_chain_required)
            cost, deadhead = calculate_cost(
                ordered, delivery, vehicle, total_km
            )
            utilization = (total_weight / vehicle.capacity_kg) * 100.0
            cost_per_kg = cost / total_weight if total_weight > 0 else 0.0
            duration_hours = total_km / 30.0

            route_id = f"rt-{uuid4().hex[:12]}"
            pickup_seq = [
                {
                    "farm_id": p.farm_id,
                    "lat": p.lat,
                    "lon": p.lon,
                    "weight_kg": p.weight_kg,
                }
                for p in ordered
            ]

            result = RouteResult(
                route_id=route_id,
                pickup_sequence=pickup_seq,
                total_distance_km=total_km,
                total_weight_kg=total_weight,
                vehicle_type=vehicle.vehicle_type,
                estimated_cost=cost,
                cost_per_kg=cost_per_kg,
                utilization_pct=min(100.0, utilization),
                estimated_duration_hours=duration_hours,
                deadhead_km=deadhead,
                cluster_size=len(cluster),
            )

            if best is None or result.cost_per_kg < best.cost_per_kg:
                best = result

        if best:
            logger.info(
                "Route {}: {} stops, {:.1f}kg, {:.2f}₹/kg, {}",
                best.route_id,
                best.cluster_size,
                best.total_weight_kg,
                best.cost_per_kg,
                best.vehicle_type,
            )
        return best


def get_logistics_router() -> LogisticsRouter:
    """Factory for LogisticsRouter."""
    return LogisticsRouter()
