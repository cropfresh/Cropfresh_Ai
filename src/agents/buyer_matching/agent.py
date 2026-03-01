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

import json
import math
from datetime import datetime, timedelta
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


MAX_MATCH_DISTANCE_KM = 150.0
CACHE_TTL_SECONDS = 300
GRADE_ORDER = {"A+": 4, "A": 3, "B": 2, "C": 1}


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
        distance = BuyerMatchingAgent._haversine(farmer_lat, farmer_lon, buyer_lat, buyer_lon)
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
        relevant_orders = [order for order in buyer_order_history if str(order.get("commodity", "")).lower() == commodity.lower()]
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

    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        redis_url: Optional[str] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
        **kwargs: Any,
    ):
        config = AgentConfig(
            name="buyer_matching",
            description="Matches farmer listings with suitable buyers by proximity, grade, and price",
            max_retries=1,
            temperature=0.3,
            max_tokens=500,
            kb_categories=["commerce", "market"],
        )
        super().__init__(config=config, llm=llm, **kwargs)
        self.engine = MatchingEngine()
        self.redis_url = redis_url
        self.cache_ttl_seconds = cache_ttl_seconds
        self._redis_client = None
        self._local_cache: dict[str, tuple[datetime, MatchResult]] = {}

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
        min_score: float = 0.0,
    ) -> MatchResult:
        """
        Score and rank buyers against a single listing.

        Returns the top-N matches sorted by descending score.
        """
        cache_key = self._build_cache_key(
            listing_id=listing.listing_id,
            buyer_ids=[buyer.buyer_id for buyer in buyers],
            suffix=f"top:{top_n}:min:{min_score:.3f}",
        )
        cached_result = await self._cache_get(cache_key)
        if cached_result:
            cached_result.cache_hit = True
            return cached_result

        candidates: list[MatchCandidate] = []

        for buyer in buyers:
            buyer_min_grade = self._resolve_buyer_min_grade(buyer)
            proximity = self.engine.calculate_proximity_score(
                listing.pickup_lat,
                listing.pickup_lon,
                buyer.delivery_lat,
                buyer.delivery_lon,
                max_distance_km=MAX_MATCH_DISTANCE_KM,
            )
            quality_match = self.engine.calculate_quality_match(listing.grade, buyer_min_grade)
            price_fit = self.engine.calculate_price_fit(listing.asking_price_per_kg, buyer.max_price_per_kg)
            demand_signal = self.engine.calculate_demand_signal(listing.commodity, buyer.order_history)
            if listing.commodity.lower() in [commodity.lower() for commodity in buyer.demand_commodities]:
                demand_signal = min(1.0, demand_signal + 0.2)
            reliability = self.engine.calculate_reliability(listing.reliability_score)

            match_score = (
                self.engine.WEIGHTS["proximity"] * proximity
                + self.engine.WEIGHTS["quality"] * quality_match
                + self.engine.WEIGHTS["price_fit"] * price_fit
                + self.engine.WEIGHTS["demand_signal"] * demand_signal
                + self.engine.WEIGHTS["reliability"] * reliability
            )
            if match_score < min_score:
                continue

            distance = self._haversine(
                listing.pickup_lat, listing.pickup_lon,
                buyer.delivery_lat, buyer.delivery_lon,
            )
            grade_ok = quality_match > 0.0
            price_ok = price_fit > 0.0
            fillable = min(listing.quantity_kg, buyer.demand_quantity_kg)
            if buyer.demand_quantity_kg <= 0:
                fillable = listing.quantity_kg
            estimated_delivery_hours = max(1.0, distance / 35.0)
            estimated_logistics_cost = round(distance * 0.8, 2)
            reasons = [
                f"proximity={proximity:.2f}",
                f"quality={quality_match:.2f}",
                f"price_fit={price_fit:.2f}",
                f"demand={demand_signal:.2f}",
                f"reliability={reliability:.2f}",
            ]

            candidates.append(MatchCandidate(
                listing_id=listing.listing_id,
                farmer_id=listing.farmer_id,
                buyer_id=buyer.buyer_id,
                buyer_name=buyer.name or buyer.buyer_id[:8],
                buyer_type=buyer.type,
                match_score=round(match_score, 3),
                score=round(match_score, 3),
                proximity_km=round(distance, 2),
                distance_km=round(distance, 1),
                quality_match=round(quality_match, 3),
                grade_compatible=grade_ok,
                price_fit=round(price_fit, 3),
                price_compatible=price_ok,
                demand_signal=round(demand_signal, 3),
                reliability=round(reliability, 3),
                quantity_fillable=fillable,
                estimated_delivery_hours=round(estimated_delivery_hours, 2),
                estimated_logistics_cost=estimated_logistics_cost,
                reasoning="; ".join(reasons),
            ))

        candidates.sort(key=lambda candidate: candidate.match_score, reverse=True)

        result = MatchResult(
            listing_id=listing.listing_id,
            commodity=listing.commodity,
            matches=candidates[:top_n],
            total_candidates_evaluated=len(buyers),
        )
        await self._cache_set(cache_key, result)
        return result

    async def find_matches(
        self,
        listing_id: str,
        max_results: int = 10,
        min_score: float = 0.3,
    ) -> list[MatchCandidate]:
        """
        Find top buyers for a listing.
        """
        listing, buyers = self._get_mock_listing_and_buyers(listing_id)
        result = await self.match(listing=listing, buyers=buyers, top_n=max_results, min_score=min_score)
        return result.matches

    async def find_farmers_for_buyer(
        self,
        buyer_id: str,
        commodity: str,
        quantity_needed_kg: float,
        max_price_per_kg: float,
        max_results: int = 10,
    ) -> list[MatchCandidate]:
        """
        Reverse matching: buyer specifies needs, find matching listings.
        """
        buyer = BuyerProfile(
            buyer_id=buyer_id,
            name="Requested Buyer",
            type="restaurant",
            district="Bangalore",
            delivery_lat=12.9716,
            delivery_lon=77.5946,
            preferred_grades=["A", "B"],
            max_price_per_kg=max_price_per_kg,
            demand_commodities=[commodity],
            demand_quantity_kg=quantity_needed_kg,
            order_history=[
                {"commodity": commodity, "date": datetime.now().isoformat()},
                {"commodity": commodity, "date": (datetime.now() - timedelta(days=7)).isoformat()},
            ],
        )
        listings = self._get_mock_listings_for_commodity(commodity)
        candidates: list[MatchCandidate] = []
        for listing in listings:
            matched = await self.match(
                listing=listing,
                buyers=[buyer],
                top_n=1,
                min_score=0.3,
            )
            if matched.matches:
                candidates.extend(matched.matches)
        candidates.sort(key=lambda candidate: candidate.match_score, reverse=True)
        return candidates[:max_results]

    def _resolve_buyer_min_grade(self, buyer: BuyerProfile) -> str:
        if buyer.min_grade:
            return buyer.min_grade
        if not buyer.preferred_grades:
            return "C"
        preferred_values = sorted(
            [GRADE_ORDER.get(grade, 1) for grade in buyer.preferred_grades]
        )
        preferred_min = preferred_values[0]
        for grade, value in GRADE_ORDER.items():
            if value == preferred_min:
                return grade
        return "C"

    def _build_cache_key(self, listing_id: str, buyer_ids: list[str], suffix: str = "") -> str:
        buyers = ",".join(sorted(buyer_ids))
        return f"match:{listing_id}:{buyers}:{suffix}"

    async def _get_redis(self):
        if self._redis_client is None and self.redis_url:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis_client.ping()
            except Exception as err:
                logger.warning(f"Redis cache unavailable for buyer matching: {err}")
                self._redis_client = None
        return self._redis_client

    async def _cache_get(self, key: str) -> Optional[MatchResult]:
        redis = await self._get_redis()
        if redis:
            try:
                raw = await redis.get(key)
                if raw:
                    return MatchResult.model_validate_json(raw)
            except Exception as err:
                logger.debug(f"Redis cache read failed ({key}): {err}")

        cached = self._local_cache.get(key)
        if not cached:
            return None
        expiry, value = cached
        if expiry <= datetime.now():
            self._local_cache.pop(key, None)
            return None
        return value.model_copy(deep=True)

    async def _cache_set(self, key: str, value: MatchResult) -> None:
        redis = await self._get_redis()
        if redis:
            try:
                await redis.setex(key, self.cache_ttl_seconds, value.model_dump_json())
            except Exception as err:
                logger.debug(f"Redis cache write failed ({key}): {err}")
        expiry = datetime.now() + timedelta(seconds=self.cache_ttl_seconds)
        self._local_cache[key] = (expiry, value.model_copy(deep=True))

    def _get_mock_listing_and_buyers(self, listing_id: str) -> tuple[ListingProfile, list[BuyerProfile]]:
        listing = ListingProfile(
            listing_id=listing_id,
            farmer_id="farmer-test-001",
            commodity="Tomato",
            variety="Hybrid",
            quantity_kg=200,
            asking_price_per_kg=24.0,
            grade="A",
            pickup_lat=13.13,
            pickup_lon=78.15,
            district="Kolar",
            reliability_score=0.86,
        )
        buyers = [
            BuyerProfile(
                buyer_id="buyer-near",
                name="Kolar Retail Hub",
                type="retailer",
                district="Kolar",
                delivery_lat=13.14,
                delivery_lon=78.16,
                preferred_grades=["A", "B"],
                max_price_per_kg=30.0,
                demand_commodities=["Tomato", "Onion"],
                demand_quantity_kg=300,
                order_history=[
                    {"commodity": "Tomato", "date": datetime.now().isoformat()},
                    {"commodity": "Tomato", "date": (datetime.now() - timedelta(days=14)).isoformat()},
                ],
            ),
            BuyerProfile(
                buyer_id="buyer-mid",
                name="Bangalore Fresh Stores",
                type="wholesaler",
                district="Bangalore",
                delivery_lat=12.98,
                delivery_lon=77.60,
                preferred_grades=["A"],
                max_price_per_kg=25.0,
                demand_commodities=["Tomato"],
                demand_quantity_kg=120,
                order_history=[{"commodity": "Tomato", "date": (datetime.now() - timedelta(days=21)).isoformat()}],
            ),
        ]
        return listing, buyers

    def _get_mock_listings_for_commodity(self, commodity: str) -> list[ListingProfile]:
        return [
            ListingProfile(
                listing_id="listing-a",
                farmer_id="farmer-a",
                commodity=commodity,
                variety="Hybrid",
                quantity_kg=180,
                asking_price_per_kg=23.0,
                grade="A",
                pickup_lat=13.13,
                pickup_lon=78.15,
                district="Kolar",
                reliability_score=0.9,
            ),
            ListingProfile(
                listing_id="listing-b",
                farmer_id="farmer-b",
                commodity=commodity,
                variety="Local",
                quantity_kg=240,
                asking_price_per_kg=27.0,
                grade="B",
                pickup_lat=13.05,
                pickup_lon=77.95,
                district="Bangalore Rural",
                reliability_score=0.75,
            ),
        ]

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
