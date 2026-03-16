from __future__ import annotations

from typing import Optional

from src.rag.routing.models import (
    ROUTE_COST_MAP,
    QueryAnalysis,
    QueryCategory,
    QueryType,
    RetrievalRoute,
    RoutingDecision,
)

GREETING_TOKENS = {
    "hello", "hi", "hey", "namaste", "namaskar", "vanakkam", "thanks",
    "thank you", "bye", "goodbye", "help", "what is cropfresh",
}
IMAGE_TRIGGERS = {
    "[image", "photo of", "image of", "picture of", "pic of",
    "look at this", "attached image", "crop photo",
}
PRICE_KEYWORDS = {
    "price", "rate", "mandi", "market", "bhaav", "bhav", "ಬೆಲೆ", "ದರ",
    "bele", "quintal", "apmc",
}
PRICE_TIME_TOKENS = {
    "today", "current", "now", "live", "real-time", "aaj", "abhi",
    "ivattu", "indu", "eega", "ee dina",
}
WEATHER_KEYWORDS = {
    "weather", "rain", "rainfall", "forecast", "temperature", "monsoon",
    "barish", "mausam", "ಮಳೆ", "ಹವಾಮಾನ",
}
GRAPH_PATTERNS = {
    "which farmers", "who grows", "find farmers", "list buyers",
    "show me farmers", "supply chain", "how many farmers",
}
COMPLEX_PATTERNS = {
    "should i sell", "sell or store", "hold or sell", "compare",
    "vs ", "versus", "profitability", "better crop",
}
BROWSER_PATTERNS = {
    "latest scheme", "new scheme", "new policy", "news", "recently launched",
    "banned pesticide", "new government", "2026 scheme",
}
SCHEME_KEYWORDS = {
    "scheme", "subsidy", "yojana", "pm-kisan", "pmfby", "kcc", "msp",
    "yojane", "arji", "ಅರ್ಜಿ", "ಯೋಜನೆ",
}
AGRONOMY_KEYWORDS = {
    "grow", "plant", "cultivat", "harvest", "soil", "irrigation",
    "fertilizer", "seed", "crop", "pest", "disease", "hege", "niyantrisuvudu",
}
PLATFORM_KEYWORDS = {"cropfresh", "register", "login", "account", "app", "feature"}

CROPS = {
    "tomato", "onion", "potato", "paddy", "rice", "ragi", "maize", "cotton",
    "sugarcane", "coconut", "mulberry", "brinjal", "chilli",
}
LOCATIONS = {
    "karnataka", "kolar", "hubli", "dharwad", "belgaum", "belagavi",
    "mysore", "mysuru", "tumkur", "tiptur", "mandya", "bangalore",
    "bengaluru",
}


def extract_entities(query: str) -> dict[str, list[str]]:
    """Extract simple crop/location entities for observability."""
    q = query.lower()
    crops = sorted({crop for crop in CROPS if crop in q})
    locations = sorted({loc.title() for loc in LOCATIONS if loc in q})
    return {"crops": crops, "locations": locations, "schemes": []}


def prefilter_route(query: str, has_image: bool) -> Optional[RoutingDecision]:
    """Fast, no-LLM routing for the most obvious cases."""
    q = query.lower().strip()
    q_words = {word.strip("?!.,;:'\"") for word in q.split()}

    if has_image or any(token in q for token in IMAGE_TRIGGERS):
        return RoutingDecision(
            strategy=RetrievalRoute.MULTIMODAL,
            confidence=0.97,
            reason="Image attachment detected",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.MULTIMODAL],
            requires_image=True,
            pre_filter_matched=True,
            entities=extract_entities(query),
        )
    if (q_words & {"hi", "hey", "hello", "thanks", "bye", "help"}) or any(
        token in q for token in GREETING_TOKENS
    ):
        return RoutingDecision(
            strategy=RetrievalRoute.DIRECT_LLM,
            confidence=0.98,
            reason="Greeting or FAQ detected by rule",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.DIRECT_LLM],
            pre_filter_matched=True,
            entities=extract_entities(query),
        )

    has_price = any(token in q for token in PRICE_KEYWORDS)
    has_time = any(token in q for token in PRICE_TIME_TOKENS)
    has_market_context = "mandi" in q or "market" in q or bool(extract_entities(query)["locations"])
    if has_price and (has_time or has_market_context):
        return RoutingDecision(
            strategy=RetrievalRoute.LIVE_PRICE_API,
            confidence=0.92,
            reason="Price query detected with market or freshness hints",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.LIVE_PRICE_API],
            requires_live_data=True,
            pre_filter_matched=True,
            entities=extract_entities(query),
        )
    if any(token in q for token in WEATHER_KEYWORDS):
        return RoutingDecision(
            strategy=RetrievalRoute.WEATHER_API,
            confidence=0.9,
            reason="Weather or forecast query detected",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.WEATHER_API],
            requires_live_data=True,
            pre_filter_matched=True,
            entities=extract_entities(query),
        )
    return None


def fallback_route(query: str, has_image: bool = False) -> RoutingDecision:
    """Rule-based routing when no LLM decision is available."""
    q = query.lower()
    entities = extract_entities(query)

    if has_image:
        return prefilter_route(query, has_image=True)  # pragma: no cover
    if any(pattern in q for pattern in GRAPH_PATTERNS):
        return RoutingDecision(
            strategy=RetrievalRoute.GRAPH_TRAVERSAL,
            confidence=0.72,
            reason="Graph traversal pattern detected",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.GRAPH_TRAVERSAL],
            entities=entities,
        )
    if any(pattern in q for pattern in COMPLEX_PATTERNS):
        return RoutingDecision(
            strategy=RetrievalRoute.FULL_AGENTIC,
            confidence=0.7,
            reason="Complex decision query detected",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.FULL_AGENTIC],
            entities=entities,
        )
    if any(pattern in q for pattern in BROWSER_PATTERNS):
        return RoutingDecision(
            strategy=RetrievalRoute.BROWSER_SCRAPE,
            confidence=0.68,
            reason="Latest live scheme or policy query detected",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.BROWSER_SCRAPE],
            requires_live_data=True,
            entities=entities,
        )
    if any(token in q for token in PRICE_KEYWORDS):
        return RoutingDecision(
            strategy=RetrievalRoute.LIVE_PRICE_API,
            confidence=0.66,
            reason="Price vocabulary detected in fallback routing",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.LIVE_PRICE_API],
            requires_live_data=True,
            entities=entities,
        )
    return RoutingDecision(
        strategy=RetrievalRoute.VECTOR_ONLY,
        confidence=0.6,
        reason="Default vector search fallback",
        estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.VECTOR_ONLY],
        entities=entities,
    )


def rule_based_query_analysis(query: str) -> QueryAnalysis:
    """Legacy query analysis compatible with the original QueryAnalyzer."""
    q = query.lower()
    route = prefilter_route(query, has_image=False) or fallback_route(query)
    category = QueryCategory.GENERAL
    if any(token in q for token in PRICE_KEYWORDS | {"sell", "buy"}):
        category = QueryCategory.MARKET
    elif any(token in q for token in SCHEME_KEYWORDS):
        category = QueryCategory.REGULATORY
    elif any(token in q for token in AGRONOMY_KEYWORDS):
        category = QueryCategory.AGRONOMY
    elif any(token in q for token in PLATFORM_KEYWORDS):
        category = QueryCategory.PLATFORM

    query_type = QueryType.VECTOR_SEARCH
    if route.strategy in {
        RetrievalRoute.LIVE_PRICE_API,
        RetrievalRoute.WEATHER_API,
        RetrievalRoute.BROWSER_SCRAPE,
    }:
        query_type = QueryType.WEB_SEARCH
    elif route.strategy is RetrievalRoute.DIRECT_LLM:
        query_type = QueryType.DIRECT
    elif route.strategy is RetrievalRoute.FULL_AGENTIC:
        query_type = QueryType.DECOMPOSE

    entities = extract_entities(query)
    return QueryAnalysis(
        original_query=query,
        query_type=query_type,
        category=category,
        reasoning=route.reason,
        confidence=route.confidence,
        crops=entities["crops"],
        locations=entities["locations"],
        time_sensitive=route.requires_live_data,
    )
