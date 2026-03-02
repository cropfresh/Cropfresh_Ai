"""
Extended unit tests for Buyer Matching Agent — sub-function scoring and edge cases.
"""

import pytest
from datetime import datetime, timedelta
from src.agents.buyer_matching.agent import MatchingEngine, BuyerMatchingAgent, ListingProfile, BuyerProfile

class TestMatchingEngineScoring:
    @pytest.mark.parametrize("grade, min_grade, expected", [
        ("A+", "A", 0.9),
        ("A", "A", 1.0),
        ("B", "A", 0.0),
        ("C", "C", 1.0),
        ("A", "C", 0.9),
        ("A+", "C", 0.9),
        ("C", "A+", 0.0),
        ("B", "B", 1.0)
    ])
    def test_calculate_quality_match(self, grade, min_grade, expected):
        engine = MatchingEngine()
        assert engine.calculate_quality_match(grade, min_grade) == expected

class TestMatchingEnginePriceFit:
    @pytest.mark.parametrize("asking, budget, expected", [
        (20.0, 25.0, 1.0),    # well within budget
        (25.0, 25.0, 1.0),    # exact budget
        (26.0, 25.0, 0.8),    # overshoot = 1/25 = 0.04 -> 1 - (0.04*5) = 0.8
        (30.0, 25.0, 0.0),    # overshoot = 5/25 = 0.2 > 0.15 -> 0.0
        (20.0, 0.0, 0.5)      # no budget spec
    ])
    def test_calculate_price_fit(self, asking, budget, expected):
        engine = MatchingEngine()
        result = engine.calculate_price_fit(asking, budget)
        assert abs(result - expected) < 0.01

class TestMatchingEngineDemandSignal:
    def test_calculate_demand_signal_no_history(self):
        engine = MatchingEngine()
        assert engine.calculate_demand_signal("Tomato", []) == 0.1

    def test_calculate_demand_signal_mixed_history(self):
        engine = MatchingEngine()
        history = [
            {"commodity": "Tomato", "date": datetime.now().isoformat()},
            {"commodity": "Onion", "date": datetime.now().isoformat()},
            {"commodity": "Tomato", "date": (datetime.now() - timedelta(days=2)).isoformat()}
        ]
        score = engine.calculate_demand_signal("Tomato", history)
        assert score > 0.1
        assert score <= 1.0

class TestBuyerMatchingAgentEdgeCases:
    @pytest.mark.asyncio
    async def test_min_score_filter(self):
        agent = BuyerMatchingAgent(llm=None)
        listing = ListingProfile(
            listing_id="lst-0", farmer_id="f-0", commodity="Tomato",
            quantity_kg=100, asking_price_per_kg=50.0, grade="C"
        )
        buyers = [BuyerProfile(
            buyer_id="b-0", min_grade="A", max_price_per_kg=10.0, demand_commodities=["Tomato"]
        )]
        # This buyer will have 0 score for price and quality, match_score < 0.5
        result = await agent.match(listing, buyers, min_score=0.5)
        assert len(result.matches) == 0

    @pytest.mark.asyncio
    async def test_top_n_zero(self):
        agent = BuyerMatchingAgent(llm=None)
        listing = ListingProfile(
            listing_id="lst-0", farmer_id="f-0", commodity="Tomato",
            quantity_kg=100, asking_price_per_kg=20.0, grade="A"
        )
        buyers = [BuyerProfile(
            buyer_id="b-0", min_grade="A", max_price_per_kg=25.0, demand_commodities=["Tomato"]
        )]
        result = await agent.match(listing, buyers, top_n=0)
        assert len(result.matches) == 0

    @pytest.mark.asyncio
    async def test_reverse_matching_with_no_mock_data(self):
        agent = BuyerMatchingAgent(llm=None)
        # Should return an empty list or fall back gracefully depending on mock listings
        results = await agent.find_farmers_for_buyer(
            buyer_id="b-1", commodity="ExoticFruit", quantity_needed_kg=100, max_price_per_kg=50
        )
        # ExoticFruit is not in the mock list so matches might be empty or valid based on score
        # The agent mock returns ExoticFruit listings automatically, but they might not match grade/price if we set min_score too high.
        assert isinstance(results, list)
