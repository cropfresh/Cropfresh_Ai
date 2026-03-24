from __future__ import annotations

import json

from loguru import logger

from src.rag.routing.models import (
    QueryAnalysis,
    QueryCategory,
    QueryType,
    RetrievalRoute,
    RoutingDecision,
)

QUERY_ANALYZER_PROMPT = """You are a query analyzer for CropFresh AI.
Classify the query and respond as JSON with:
query_type, category, sub_queries, reasoning, crops, locations, time_sensitive."""

ADAPTIVE_ROUTER_PROMPT = """You are a routing classifier for CropFresh.
Respond ONLY as JSON with: route, confidence, reason, entities, requires_live_data, requires_image."""


async def llm_query_analysis(query: str, llm) -> QueryAnalysis:
    """LLM query analysis with graceful fallback-friendly parsing."""
    from src.orchestrator.llm_provider import LLMMessage

    messages = [
        LLMMessage(role="system", content=QUERY_ANALYZER_PROMPT),
        LLMMessage(role="user", content=query),
    ]
    response = await llm.generate(messages, temperature=0.0, max_tokens=300)
    result = json.loads(response.content)
    return QueryAnalysis(
        original_query=query,
        query_type=QueryType(result.get("query_type", "vector")),
        category=QueryCategory(result.get("category", "general")),
        sub_queries=result.get("sub_queries", []),
        reasoning=result.get("reasoning", ""),
        crops=result.get("crops", []),
        locations=result.get("locations", []),
        time_sensitive=bool(result.get("time_sensitive", False)),
    )


async def llm_route(query: str, has_image: bool, llm) -> RoutingDecision:
    """LLM routing for ambiguous cases."""
    from src.orchestrator.llm_provider import LLMMessage
    from src.rag.routing.models import ROUTE_COST_MAP

    messages = [
        LLMMessage(role="system", content=ADAPTIVE_ROUTER_PROMPT),
        LLMMessage(role="user", content=f"Query: {query}\nHas image: {has_image}"),
    ]
    response = await llm.generate(messages, temperature=0.0, max_tokens=200)
    result = json.loads(response.content)

    route = RetrievalRoute(result.get("route", "vector_only"))
    decision = RoutingDecision(
        strategy=route,
        confidence=float(result.get("confidence", 0.8)),
        reason=result.get("reason", "LLM classification"),
        estimated_cost_inr=ROUTE_COST_MAP[route],
        entities=result.get("entities", {}),
        requires_live_data=bool(result.get("requires_live_data", False)),
        requires_image=has_image or bool(result.get("requires_image", False)),
        pre_filter_matched=False,
    )
    logger.debug(
        "AdaptiveRouter LLM classified | route={} | confidence={:.2f} | query={}...",
        decision.strategy.value,
        decision.confidence,
        query[:60],
    )
    return decision
