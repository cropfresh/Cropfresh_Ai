"""
Logistics Router — Route Optimization
======================================
OR-Tools TSP/CVRP for optimal pickup sequence.
Falls back to greedy nearest-neighbor when OR-Tools unavailable.
"""

# * ROUTING MODULE
# NOTE: OR-Tools CVRP for optimal sequence; Haversine for distance matrix.
# NOTE: Depot = delivery (index 0); route is delivery -> pickups -> return to delivery.

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from src.agents.logistics_router.geo import build_distance_matrix, haversine_km
from src.agents.logistics_router.models import DeliveryPoint, PickupPoint

if TYPE_CHECKING:
    pass


def solve_tsp(
    pickups: list[PickupPoint],
    delivery: DeliveryPoint,
) -> tuple[list[PickupPoint], float]:
    """
    Solve TSP for optimal pickup sequence (depot = delivery).

    Uses OR-Tools when available; falls back to greedy nearest-neighbor.

    Args:
        pickups: Pickup points to visit.
        delivery: Delivery destination (depot for round-trip).

    Returns:
        (ordered_pickups, total_distance_km)
    """
    if not pickups:
        return [], 0.0

    if len(pickups) == 1:
        d_km = haversine_km(delivery.lat, delivery.lon, pickups[0].lat, pickups[0].lon)
        return pickups, d_km * 2

    try:
        return _solve_ortools(pickups, delivery)
    except ImportError:
        return _solve_greedy(pickups, delivery)


def _solve_ortools(
    pickups: list[PickupPoint],
    delivery: DeliveryPoint,
) -> tuple[list[PickupPoint], float]:
    """Solve using OR-Tools RoutingModel."""
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2

    points: list[Union[PickupPoint, DeliveryPoint]] = [delivery] + pickups
    matrix = build_distance_matrix(points)
    n = len(points)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx: int, to_idx: int) -> int:
        return matrix[from_idx][to_idx]

    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(distance_callback)
    )
    routing.AddDimension(
        routing.RegisterTransitCallback(distance_callback),
        0,
        int(1e9),
        True,
        "Distance",
    )
    search_params = routing.DefaultSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return _solve_greedy(pickups, delivery)

    route: list[PickupPoint] = []
    index = routing.Start(0)
    total_m = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node > 0:
            route.append(pickups[node - 1])
        next_index = solution.Value(routing.NextVar(index))
        total_m += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index

    total_km = total_m / 1000.0
    return route, total_km


def _solve_greedy(
    pickups: list[PickupPoint],
    delivery: DeliveryPoint,
) -> tuple[list[PickupPoint], float]:
    """Greedy nearest-neighbor TSP fallback."""
    if not pickups:
        return [], 0.0

    remaining = list(pickups)
    current_lat, current_lon = delivery.lat, delivery.lon
    route: list[PickupPoint] = []
    total_km = 0.0

    while remaining:
        best_idx = 0
        best_dist = 1e9
        for i, p in enumerate(remaining):
            d = haversine_km(current_lat, current_lon, p.lat, p.lon)
            if d < best_dist:
                best_dist = d
                best_idx = i
        next_p = remaining.pop(best_idx)
        route.append(next_p)
        total_km += best_dist
        current_lat, current_lon = next_p.lat, next_p.lon

    total_km += haversine_km(current_lat, current_lon, delivery.lat, delivery.lon)
    return route, total_km
