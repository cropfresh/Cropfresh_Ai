"""
Unit tests for Buyer Matching Agent — multi-factor scoring,
haversine distance, and match ranking.
"""

import pytest

from src.agents.buyer_matching.agent import (
    BuyerMatchingAgent,
    BuyerProfile,
    ListingProfile,
    MatchingEngine,
)


@pytest.fixture
def agent() -> BuyerMatchingAgent:
    return BuyerMatchingAgent(llm=None)


@pytest.fixture
def listing() -> ListingProfile:
    return ListingProfile(
        listing_id="lst-001",
        farmer_id="farmer-001",
        commodity="Tomato",
        variety="Hybrid",
        quantity_kg=200,
        asking_price_per_kg=25.0,
        grade="A",
        pickup_lat=13.1300,
        pickup_lon=78.1500,
        district="Kolar",
        reliability_score=0.85,
    )


@pytest.fixture
def buyers() -> list[BuyerProfile]:
    return [
        BuyerProfile(
            buyer_id="buyer-close",
            name="Close Retailer",
            type="retailer",
            district="Kolar",
            delivery_lat=13.1400,
            delivery_lon=78.1600,
            preferred_grades=["A", "B"],
            min_grade="A",
            max_price_per_kg=30.0,
            demand_commodities=["Tomato"],
            demand_quantity_kg=300,
            order_history=[
                {"commodity": "Tomato", "date": "2026-03-01T10:00:00"},
                {"commodity": "Tomato", "date": "2026-02-20T10:00:00"},
            ],
        ),
        BuyerProfile(
            buyer_id="buyer-far",
            name="Far Restaurant",
            type="restaurant",
            district="Bangalore",
            delivery_lat=12.9700,
            delivery_lon=77.5900,
            preferred_grades=["A"],
            min_grade="A",
            max_price_per_kg=28.0,
            demand_commodities=["Tomato"],
            demand_quantity_kg=100,
            order_history=[{"commodity": "Tomato", "date": "2026-02-01T10:00:00"}],
        ),
        BuyerProfile(
            buyer_id="buyer-cheap",
            name="Budget Processor",
            type="processor",
            district="Kolar",
            delivery_lat=13.1200,
            delivery_lon=78.1400,
            preferred_grades=["B", "C"],
            min_grade="A+",
            max_price_per_kg=15.0,
            demand_commodities=["Tomato"],
            demand_quantity_kg=500,
            order_history=[{"commodity": "Cabbage", "date": "2026-02-10T10:00:00"}],
        ),
    ]


class TestBuyerMatching:

    @pytest.mark.asyncio
    async def test_match_returns_ranked_results(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """Match should return candidates sorted by score descending."""
        result = await agent.match(listing, buyers)

        assert result.listing_id == "lst-001"
        assert result.total_candidates_evaluated == 3
        assert len(result.matches) > 0

        scores = [m.match_score for m in result.matches]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_closer_buyer_scores_higher(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """Buyer closer to pickup should rank higher (all else equal-ish)."""
        result = await agent.match(listing, buyers)

        buyer_ids = [m.buyer_id for m in result.matches]
        assert buyer_ids[0] == "buyer-close"

    @pytest.mark.asyncio
    async def test_grade_mismatch_reduces_score(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """Buyer whose preferred grades don't include listing grade gets lower score."""
        result = await agent.match(listing, buyers)

        cheap = next(match for match in result.matches if match.buyer_id == "buyer-cheap")
        assert cheap.grade_compatible is False

    @pytest.mark.asyncio
    async def test_price_incompatibility_flagged(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """Buyer max_price < asking_price should be flagged."""
        result = await agent.match(listing, buyers)

        cheap = next(match for match in result.matches if match.buyer_id == "buyer-cheap")
        assert cheap.price_compatible is False

    @pytest.mark.asyncio
    async def test_empty_buyers_returns_no_matches(
        self, agent: BuyerMatchingAgent, listing: ListingProfile,
    ):
        """No buyers → empty matches list."""
        result = await agent.match(listing, [])

        assert len(result.matches) == 0
        assert result.total_candidates_evaluated == 0

    @pytest.mark.asyncio
    async def test_top_n_limits_results(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """top_n should limit number of returned matches."""
        result = await agent.match(listing, buyers, top_n=1)

        assert len(result.matches) <= 1

    @pytest.mark.asyncio
    async def test_find_matches_returns_ranked_with_mock_data(self, agent: BuyerMatchingAgent):
        """find_matches should return ranked matches from built-in mock data."""
        results = await agent.find_matches(listing_id="test", max_results=5, min_score=0.3)
        assert len(results) >= 1
        scores = [candidate.match_score for candidate in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_reverse_matching_returns_farmers(self, agent: BuyerMatchingAgent):
        """find_farmers_for_buyer should return matching listing candidates."""
        results = await agent.find_farmers_for_buyer(
            buyer_id="buyer-need-tomato",
            commodity="Tomato",
            quantity_needed_kg=200,
            max_price_per_kg=28.0,
            max_results=5,
        )
        assert len(results) >= 1
        assert all(candidate.listing_id for candidate in results)
        assert all(candidate.farmer_id for candidate in results)

    @pytest.mark.asyncio
    async def test_local_cache_hit_on_repeated_match(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """Second identical match should hit cache."""
        first = await agent.match(listing, buyers, top_n=3, min_score=0.0)
        second = await agent.match(listing, buyers, top_n=3, min_score=0.0)
        assert first.cache_hit is False
        assert second.cache_hit is True


class TestHaversine:

    def test_same_point_zero_distance(self):
        dist = BuyerMatchingAgent._haversine(13.0, 78.0, 13.0, 78.0)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        """Kolar to Bangalore ≈ 60-70 km."""
        dist = BuyerMatchingAgent._haversine(13.13, 78.15, 12.97, 77.59)
        assert 55 < dist < 75

    def test_zero_coords_returns_zero(self):
        dist = BuyerMatchingAgent._haversine(0, 0, 13.0, 78.0)
        assert dist == 0.0


class TestMatchingEngine:
    def test_proximity_nearby_scores_high(self):
        engine = MatchingEngine()
        score = engine.calculate_proximity_score(12.97, 77.59, 12.98, 77.60)
        assert score > 0.85

    def test_below_grade_returns_zero(self):
        engine = MatchingEngine()
        score = engine.calculate_quality_match("C", "A")
        assert score == 0.0
