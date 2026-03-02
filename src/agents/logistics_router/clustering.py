"""
Logistics Router — Pickup Clustering
====================================
HDBSCAN clustering of farm locations for multi-pickup route optimization.
Groups nearby farms to reduce deadhead and achieve <₹2.5/kg cost.
"""

# * CLUSTERING MODULE
# NOTE: HDBSCAN is density-based and handles noise (single-farm outliers).
# NOTE: Haversine metric for GPS coordinates; epsilon ~500m for cluster formation.

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
    cluster_epsilon_km: float = 0.5,
) -> list[list[PickupPoint]]:
    """
    Cluster nearby pickup points using HDBSCAN with Haversine metric.

    Args:
        pickups: List of farm pickup points.
        min_cluster_size: Minimum points per cluster (default 2).
        cluster_epsilon_km: Epsilon in km for cluster selection (~500m).

    Returns:
        List of clusters; each cluster is a list of PickupPoints.
        Noise points (label -1) are returned as single-point clusters.
    """
    if len(pickups) <= 1:
        return [pickups] if pickups else []

    if min_cluster_size < 2:
        min_cluster_size = 2

    coords = np.array([[p.lat, p.lon] for p in pickups])
    coords_rad = np.radians(coords)

    # * cluster_selection_epsilon: ~0.005 rad ≈ 500m
    epsilon_rad = cluster_epsilon_km / 111.0
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="haversine",
        cluster_selection_epsilon=epsilon_rad,
    )
    labels = clusterer.fit_predict(coords_rad)

    clusters: dict[int, list[PickupPoint]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(pickups[i])

    result: list[list[PickupPoint]] = []
    for label in sorted(clusters.keys(), reverse=True):
        result.append(clusters[label])

    return result
