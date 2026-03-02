"""
Logistics Router — Pickup Clustering
====================================
HDBSCAN clustering of farm locations for multi-pickup route optimization.
Groups nearby farms to reduce deadhead and achieve <₹2.5/kg cost.
"""

# * CLUSTERING MODULE
# NOTE: HDBSCAN is density-based and handles noise (single-farm outliers).
# NOTE: Haversine metric for GPS coordinates (coords must be in radians).
# NOTE: cluster_selection_epsilon is intentionally omitted — it has a known
#       Cython type bug in scikit-learn 1.8 with the haversine metric.
# NOTE: Noise points (label -1) are grouped into a single cluster by default,
#       allowing the engine to still route them as one multi-pickup run.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import HDBSCAN

from src.agents.logistics_router.models import PickupPoint

if TYPE_CHECKING:
    pass


def cluster_pickups(
    pickups: list[PickupPoint],
    min_cluster_size: int = 2,
    cluster_epsilon_km: float = 5.0,  # kept for API compatibility; unused internally
) -> list[list[PickupPoint]]:
    """
    Cluster nearby pickup points using HDBSCAN with Haversine metric.

    Args:
        pickups: List of farm pickup points.
        min_cluster_size: Minimum points per cluster (default 2).
        cluster_epsilon_km: Reserved for future use; currently unused to avoid
            a scikit-learn 1.8.x Cython compatibility issue with ``haversine``.

    Returns:
        List of clusters; each cluster is a list of PickupPoints.
        If HDBSCAN labels all points as noise (label -1), they are returned
        as one cluster so the engine can still plan a valid route.
    """
    if len(pickups) <= 1:
        return [pickups] if pickups else []

    if min_cluster_size < 2:
        min_cluster_size = 2

    coords = np.array([[p.lat, p.lon] for p in pickups])
    coords_rad = np.radians(coords)

    # NOTE: cluster_selection_epsilon omitted — triggers Cython scalar TypeError
    #       in sklearn 1.8.x when metric='haversine'.  Default behaviour (no
    #       epsilon) still produces correct density-based clusters.
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="haversine",
    )
    labels = clusterer.fit_predict(coords_rad)

    clusters: dict[int, list[PickupPoint]] = {}
    for i, label in enumerate(labels):
        # cast numpy int64 → Python int for dict-key compatibility
        clusters.setdefault(int(label), []).append(pickups[i])

    # * If HDBSCAN labels every point as noise (label -1), treat all as one cluster.
    # This occurs when farms are spread just beyond the density threshold but
    # close enough to be served by a single vehicle (correct business logic).
    if list(clusters.keys()) == [-1]:
        return [pickups]

    result: list[list[PickupPoint]] = []
    # Emit named clusters first (labels 0, 1, …) then noise cluster last.
    for label in sorted(clusters.keys(), reverse=True):
        result.append(clusters[label])

    return result
