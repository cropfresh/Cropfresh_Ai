"""
Buyer Matching Agent
====================
Matches active farmer listings to suitable buyers using
grade compatibility, location proximity, and demand signals.

Business Logic (from ARCHITECTURE.md):
  - Cluster farmers by GPS pickup location
  - Match buyers by grade preference, demand volume, route feasibility
  - Score and rank matches; return top-N candidates per listing

Author: CropFresh AI Team
Version: 2.0.0
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState
from src.orchestrator.llm_provider import BaseLLMProvider


# * ═══════════════════════════════════════════════════════════════
# * DATA MODELS
# * ═══════════════════════════════════════════════════════════════

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

class BuyerProfile(BaseModel):
    """Buyer with demand preferences."""
    buyer_id: str
    name: str = ""
    type: str = "retailer"
    district: str = ""
    delivery_lat: float = 0.0
    delivery_lon: float = 0.0
    preferred_grades: list[str] = Field(default_factory=lambda: ["A", "B"])
    max_price_per_kg: float = 0.0
    demand_commodities: list[str] = Field(default_factory=list)
    demand_quantity_kg: float = 0.0

class MatchCandidate(BaseModel):
    """A scored buyer match for a listing."""
    buyer_id: str
    buyer_name: str
    buyer_type: str
    score: float
    distance_km: float
    grade_compatible: bool
    price_compatible: bool
    quantity_fillable: float
    reasoning: str

class MatchResult(BaseModel):
    """Complete matching result for a listing."""
    listing_id: str
    commodity: str
    matches: list[MatchCandidate] = Field(default_factory=list)
    total_candidates_evaluated: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


# * ═══════════════════════════════════════════════════════════════
# * SCORING WEIGHTS
# * ═══════════════════════════════════════════════════════════════

WEIGHT_DISTANCE = 0.30
WEIGHT_GRADE = 0.25
WEIGHT_PRICE = 0.25
WEIGHT_QUANTITY = 0.20

MAX_MATCH_DISTANCE_KM = 150.0


class BuyerMatchingAgent(BaseAgent):
    """
    Matches farmer listings to buyers via multi-factor scoring.

    Factors:
      1. Distance — haversine between pickup and delivery GPS
      2. Grade compatibility — buyer accepts listing grade
      3. Price — buyer's max price >= asking price
      4. Quantity — buyer demand can absorb listing quantity

    Usage:
        agent = BuyerMatchingAgent(llm=provider)
        await agent.initialize()
        result = await agent.match(listing, buyers)
    """

    def __init__(self, llm: Optional[BaseLLMProvider] = None, **kwargs: Any):
        config = AgentConfig(
            name="buyer_matching",
            description="Matches farmer listings with suitable buyers by proximity, grade, and price",
            max_retries=1,
            temperature=0.3,
            max_tokens=500,
            kb_categories=["commerce", "market"],
        )
        super().__init__(config=config, llm=llm, **kwargs)

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return (
            "You are CropFresh's Buyer Matching Agent. "
            "Given a farmer listing and buyer profiles, rank the best matches "
            "and explain why each buyer is a good fit. Be concise."
        )

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """Process a natural-language matching request via the supervisor."""
        if not self.llm:
            return AgentResponse(
                content="Buyer matching requires an LLM. Please configure one.",
                agent_name=self.name,
                confidence=0.0,
            )

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query},
        ]
        response_text = await self.generate_with_llm(messages)

        return AgentResponse(
            content=response_text,
            agent_name=self.name,
            confidence=0.75,
            steps=["llm_match_analysis"],
        )

    async def match(
        self,
        listing: ListingProfile,
        buyers: list[BuyerProfile],
        top_n: int = 5,
    ) -> MatchResult:
        """
        Score and rank buyers against a single listing.

        Returns the top-N matches sorted by descending score.
        """
        candidates: list[MatchCandidate] = []

        for buyer in buyers:
            score, reasons = self._score_buyer(listing, buyer)
            if score <= 0:
                continue

            distance = self._haversine(
                listing.pickup_lat, listing.pickup_lon,
                buyer.delivery_lat, buyer.delivery_lon,
            )

            grade_ok = (
                listing.grade in buyer.preferred_grades
                or listing.grade == "Unverified"
            )
            price_ok = (
                buyer.max_price_per_kg <= 0
                or buyer.max_price_per_kg >= listing.asking_price_per_kg
            )
            fillable = min(listing.quantity_kg, buyer.demand_quantity_kg)

            candidates.append(MatchCandidate(
                buyer_id=buyer.buyer_id,
                buyer_name=buyer.name or buyer.buyer_id[:8],
                buyer_type=buyer.type,
                score=round(score, 3),
                distance_km=round(distance, 1),
                grade_compatible=grade_ok,
                price_compatible=price_ok,
                quantity_fillable=fillable,
                reasoning="; ".join(reasons),
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)

        return MatchResult(
            listing_id=listing.listing_id,
            commodity=listing.commodity,
            matches=candidates[:top_n],
            total_candidates_evaluated=len(buyers),
        )

    def _score_buyer(
        self,
        listing: ListingProfile,
        buyer: BuyerProfile,
    ) -> tuple[float, list[str]]:
        """Multi-factor scoring for a single buyer against a listing."""
        score = 0.0
        reasons: list[str] = []

        # 1. Distance score (closer = better)
        distance = self._haversine(
            listing.pickup_lat, listing.pickup_lon,
            buyer.delivery_lat, buyer.delivery_lon,
        )
        if distance > MAX_MATCH_DISTANCE_KM:
            return 0.0, ["too far"]

        dist_score = max(0, 1.0 - (distance / MAX_MATCH_DISTANCE_KM))
        score += dist_score * WEIGHT_DISTANCE
        reasons.append(f"distance {distance:.0f}km ({dist_score:.2f})")

        # 2. Grade compatibility
        grade_ok = (
            listing.grade in buyer.preferred_grades
            or listing.grade == "Unverified"
        )
        grade_score = 1.0 if grade_ok else 0.3
        score += grade_score * WEIGHT_GRADE
        reasons.append(f"grade {'match' if grade_ok else 'mismatch'}")

        # 3. Price compatibility
        if buyer.max_price_per_kg <= 0:
            price_score = 0.5
        elif buyer.max_price_per_kg >= listing.asking_price_per_kg:
            price_score = 1.0
        else:
            ratio = buyer.max_price_per_kg / listing.asking_price_per_kg
            price_score = max(0, ratio)
        score += price_score * WEIGHT_PRICE
        reasons.append(f"price score {price_score:.2f}")

        # 4. Quantity match
        if buyer.demand_quantity_kg <= 0:
            qty_score = 0.5
        else:
            fill_ratio = min(listing.quantity_kg / buyer.demand_quantity_kg, 1.0)
            qty_score = fill_ratio
        score += qty_score * WEIGHT_QUANTITY
        reasons.append(f"qty fill {qty_score:.2f}")

        return score, reasons

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
