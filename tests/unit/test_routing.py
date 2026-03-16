from __future__ import annotations

import pytest

from ai.rag.query_analyzer import AdaptiveQueryRouter, QueryAnalyzer, RetrievalRoute
from ai.rag.routing.prefilter import rule_based_query_analysis


@pytest.mark.asyncio
async def test_kannada_price_prefilter_routes_to_live_price(monkeypatch):
    monkeypatch.setenv("USE_ADAPTIVE_ROUTER", "true")
    router = AdaptiveQueryRouter(llm=None)

    decision = await router.route("Tomato bele yaavaga Kolar mandi alli?")

    assert decision.strategy is RetrievalRoute.LIVE_PRICE_API
    assert decision.pre_filter_matched is True


@pytest.mark.asyncio
async def test_compatibility_query_analyzer_keeps_kannada_scheme_category():
    analyzer = QueryAnalyzer(llm=None)

    result = await analyzer.analyze("PM-KISAN yojane ge hege arji maaduvudu?")

    assert result.category.value == "regulatory"
    assert result.query_type.value == "vector"


def test_rule_based_analysis_extracts_market_entities():
    result = rule_based_query_analysis("What is the tomato price in Hubli mandi today?")

    assert result.category.value == "market"
    assert "tomato" in result.crops
    assert "Hubli" in result.locations
