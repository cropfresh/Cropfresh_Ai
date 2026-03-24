"""
Buyer Matching Agent
====================
Matches active farmer listings to suitable buyers using
grade compatibility, location proximity, and demand signals.
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState
from src.orchestrator.llm_provider import BaseLLMProvider

from .cache import BuyerMatchingCacheMixin
from .constants import CACHE_TTL_SECONDS, GRADE_ORDER, MAX_MATCH_DISTANCE_KM
from .engine import MatchingEngine
from .mock_data import BuyerMatchingMockDataMixin
from .models import BuyerProfile, ListingProfile, MatchCandidate, MatchResult


class BuyerMatchingAgent(BaseAgent, BuyerMatchingCacheMixin, BuyerMatchingMockDataMixin):
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

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Backward-compatible alias preserved for older tests and callers."""
        return MatchingEngine.haversine(lat1, lon1, lat2, lon2)

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
        response_text = await self.generate_with_llm(messages, context=context)

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
            price_fit = self.engine.calculate_price_fit(
                listing.asking_price_per_kg, buyer.max_price_per_kg
            )
            demand_signal = self.engine.calculate_demand_signal(
                listing.commodity, buyer.order_history
            )

            if listing.commodity.lower() in [c.lower() for c in buyer.demand_commodities]:
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

            distance = self.engine.haversine(
                listing.pickup_lat,
                listing.pickup_lon,
                buyer.delivery_lat,
                buyer.delivery_lon,
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

            candidates.append(
                MatchCandidate(
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
                )
            )

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
        """Find top buyers for a listing."""
        listing, buyers = self._get_mock_listing_and_buyers(listing_id)
        result = await self.match(
            listing=listing, buyers=buyers, top_n=max_results, min_score=min_score
        )
        return result.matches

    async def find_farmers_for_buyer(
        self,
        buyer_id: str,
        commodity: str,
        quantity_needed_kg: float,
        max_price_per_kg: float,
        max_results: int = 10,
    ) -> list[MatchCandidate]:
        """Reverse matching: buyer specifies needs, find matching listings."""
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
        candidates.sort(key=lambda c: c.match_score, reverse=True)
        return candidates[:max_results]

    def _resolve_buyer_min_grade(self, buyer: BuyerProfile) -> str:
        if buyer.min_grade:
            return buyer.min_grade
        if not buyer.preferred_grades:
            return "C"
        preferred_values = sorted([GRADE_ORDER.get(grade, 1) for grade in buyer.preferred_grades])
        preferred_min = preferred_values[0]
        for grade, value in GRADE_ORDER.items():
            if value == preferred_min:
                return grade
        return "C"
