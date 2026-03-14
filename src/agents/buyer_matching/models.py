"""
Buyer Matching Models
=====================
Data structures for matching farmers and buyers.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ListingProfile(BaseModel):
    """Farmer listing available for matching."""
    listing_id: str
    farmer_id: str
    commodity: str
    variety: str = ""
    quantity_kg: float
    asking_price_per_kg: float
    grade: str = "Unverified"
    pickup_lat: float = 0.0
    pickup_lon: float = 0.0
    district: str = ""
    reliability_score: float = 0.7


class BuyerProfile(BaseModel):
    """Buyer with demand preferences."""
    buyer_id: str
    name: str = ""
    type: str = "retailer"
    district: str = ""
    delivery_lat: float = 0.0
    delivery_lon: float = 0.0
    preferred_grades: list[str] = Field(default_factory=lambda: ["A", "B"])
    min_grade: Optional[str] = None
    max_price_per_kg: float = 0.0
    demand_commodities: list[str] = Field(default_factory=list)
    demand_quantity_kg: float = 0.0
    order_history: list[dict[str, Any]] = Field(default_factory=list)


class MatchCandidate(BaseModel):
    """A scored buyer match for a listing."""
    listing_id: str
    farmer_id: str
    buyer_id: str
    buyer_name: str
    buyer_type: str
    match_score: float
    score: float
    proximity_km: float
    distance_km: float
    quality_match: float
    grade_compatible: bool
    price_fit: float
    price_compatible: bool
    demand_signal: float
    reliability: float
    quantity_fillable: float
    estimated_delivery_hours: float
    estimated_logistics_cost: float
    reasoning: str


class MatchResult(BaseModel):
    """Complete matching result for a listing."""
    listing_id: str
    commodity: str
    matches: list[MatchCandidate] = Field(default_factory=list)
    total_candidates_evaluated: int = 0
    cache_hit: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)
