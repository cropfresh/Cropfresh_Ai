"""
Buyer Matching Engine
=====================
Implements the multi-factor scoring logic for matching.
"""

import math
from datetime import datetime
from typing import Any, Optional

from .constants import GRADE_ORDER


class MatchingEngine:
    """
    Multi-factor matching engine for CropFresh marketplace.
    """

    WEIGHTS = {
        "proximity": 0.30,
        "quality": 0.25,
        "price_fit": 0.20,
        "demand_signal": 0.15,
        "reliability": 0.10,
    }

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km between two GPS points."""
        if lat1 == 0 and lon1 == 0:
            return 0.0
        if lat2 == 0 and lon2 == 0:
            return 0.0

        R = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def calculate_proximity_score(
        self,
        farmer_lat: float,
        farmer_lon: float,
        buyer_lat: float,
        buyer_lon: float,
        max_distance_km: float = 100.0,
    ) -> float:
        """
        Haversine proximity with exponential decay.
        """
        distance = self.haversine(farmer_lat, farmer_lon, buyer_lat, buyer_lon)
        if distance >= max_distance_km:
            return 0.0
        return math.exp(-distance / (max_distance_km * 0.3))

    def calculate_quality_match(self, listing_grade: str, buyer_min_grade: str) -> float:
        """
        Grade alignment scoring.
        """
        listing_val = GRADE_ORDER.get(listing_grade, 1)
        buyer_min_val = GRADE_ORDER.get(buyer_min_grade, 1)
        if listing_val < buyer_min_val:
            return 0.0
        if listing_val == buyer_min_val:
            return 1.0
        return 0.9

    def calculate_price_fit(self, asking_price: float, buyer_budget: float) -> float:
        """
        Price alignment scoring with overshoot penalty.
        """
        if buyer_budget <= 0:
            return 0.5
        if asking_price <= buyer_budget:
            return 1.0
        overshoot = (asking_price - buyer_budget) / buyer_budget
        if overshoot > 0.15:
            return 0.0
        return max(0.0, 1.0 - (overshoot * 5))

    def calculate_demand_signal(self, commodity: str, buyer_order_history: list[dict[str, Any]]) -> float:
        """
        Frequency + recency demand signal score.
        """
        relevant_orders = [
            order for order in buyer_order_history
            if str(order.get("commodity", "")).lower() == commodity.lower()
        ]
        if not relevant_orders:
            return 0.1
        frequency_score = min(1.0, len(relevant_orders) / 10.0)
        latest_date = self._extract_latest_order_date(relevant_orders)
        if latest_date is None:
            return frequency_score
        recency_days = (datetime.now() - latest_date).days
        recency_score = max(0.0, 1.0 - recency_days / 90.0)
        return 0.6 * frequency_score + 0.4 * recency_score

    def calculate_reliability(self, reliability_score: float) -> float:
        return min(max(reliability_score, 0.0), 1.0)

    def _extract_latest_order_date(self, orders: list[dict[str, Any]]) -> Optional[datetime]:
        extracted: list[datetime] = []
        for order in orders:
            date_value = order.get("date")
            if isinstance(date_value, datetime):
                extracted.append(date_value)
            elif isinstance(date_value, str):
                try:
                    extracted.append(datetime.fromisoformat(date_value))
                except ValueError:
                    continue
        if not extracted:
            return None
        return max(extracted)
