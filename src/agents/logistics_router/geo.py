"""
Logistics Router — Geospatial Utilities
========================================
Haversine distance and distance matrix for route optimization.
"""

# * GEO MODULE
# NOTE: Earth radius 6371 km for Haversine.
# NOTE: Distance matrix used by OR-Tools TSP/CVRP.

from __future__ import annotations

import math
from typing import Union

from src.agents.logistics_router.models import DeliveryPoint, PickupPoint

_EARTH_RADIUS_KM: float = 6371.0


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute Haversine distance between two GPS points in km.

    Args:
        lat1, lon1: First point (degrees).
        lat2, lon2: Second point (degrees).

    Returns:
        Distance in kilometres.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_KM * c


def build_distance_matrix(
    points: list[Union[PickupPoint, DeliveryPoint]],
) -> list[list[int]]:
    """
    Build integer distance matrix (metres) for OR-Tools.

    OR-Tools expects integer distances; we use metres to preserve precision.

    Args:
        points: Ordered list of pickup/delivery points.

    Returns:
        NxN matrix of distances in metres (integers).
    """
    n = len(points)
    matrix: list[list[int]] = [[0] * n for _ in range(n)]

    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            km = haversine_km(pi.lat, pi.lon, pj.lat, pj.lon)
            metres = int(round(km * 1000))
            matrix[i][j] = metres
            matrix[j][i] = metres

    return matrix
