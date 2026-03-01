"""
Unit tests for Buyer Matching Agent — multi-factor scoring,
haversine distance, and match ranking.
"""

import pytest

from src.agents.buyer_matching.agent import (
    BuyerMatchingAgent,
    BuyerProfile,
    ListingProfile,
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
            max_price_per_kg=30.0,
            demand_commodities=["Tomato"],
            demand_quantity_kg=300,
        ),
        BuyerProfile(
            buyer_id="buyer-far",
            name="Far Restaurant",
            type="restaurant",
            district="Bangalore",
            delivery_lat=12.9700,
            delivery_lon=77.5900,
            preferred_grades=["A"],
            max_price_per_kg=28.0,
            demand_commodities=["Tomato"],
            demand_quantity_kg=100,
        ),
        BuyerProfile(
            buyer_id="buyer-cheap",
            name="Budget Processor",
            type="processor",
            district="Kolar",
            delivery_lat=13.1200,
            delivery_lon=78.1400,
            preferred_grades=["B", "C"],
            max_price_per_kg=15.0,
            demand_commodities=["Tomato"],
            demand_quantity_kg=500,
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

        scores = [m.score for m in result.matches]
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

        cheap = next(m for m in result.matches if m.buyer_id == "buyer-cheap")
        assert cheap.grade_compatible is False

    @pytest.mark.asyncio
    async def test_price_incompatibility_flagged(
        self, agent: BuyerMatchingAgent, listing: ListingProfile, buyers: list[BuyerProfile],
    ):
        """Buyer max_price < asking_price should be flagged."""
        result = await agent.match(listing, buyers)

        cheap = next(m for m in result.matches if m.buyer_id == "buyer-cheap")
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
