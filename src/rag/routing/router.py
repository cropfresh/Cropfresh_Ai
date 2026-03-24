from __future__ import annotations

import os

from loguru import logger

from src.rag.routing.classifier import llm_query_analysis, llm_route
from src.rag.routing.models import QueryAnalysis, RoutingDecision
from src.rag.routing.prefilter import fallback_route, prefilter_route, rule_based_query_analysis


class QueryAnalyzer:
    """Compatibility query analyzer used by the legacy graph flow."""

    def __init__(self, llm=None):
        self.llm = llm

    async def analyze(self, query: str) -> QueryAnalysis:
        if self.llm is None:
            return rule_based_query_analysis(query)
        try:
            return await llm_query_analysis(query, self.llm)
        except Exception as exc:
            logger.warning("LLM query analysis failed: {}", exc)
            return rule_based_query_analysis(query)


class AdaptiveQueryRouter:
    """Adaptive router split out of the legacy oversized module."""

    def __init__(self, llm=None):
        self.llm = llm
        self._enabled = os.getenv("USE_ADAPTIVE_ROUTER", "false").lower() == "true"
        if self._enabled:
            logger.info(
                "AdaptiveQueryRouter: ENABLED | llm={}",
                "LLM" if llm else "rule-based only",
            )
        else:
            logger.info(
                "AdaptiveQueryRouter: USE_ADAPTIVE_ROUTER=false - defaulting to vector search",
            )

    async def route(self, query: str, has_image: bool = False) -> RoutingDecision:
        if not self._enabled:
            return fallback_route(query, has_image)

        prefiltered = prefilter_route(query, has_image)
        if prefiltered is not None:
            logger.debug(
                "AdaptiveRouter pre-filter match | strategy={} | query={}...",
                prefiltered.strategy.value,
                query[:60],
            )
            return prefiltered

        if self.llm is None:
            return fallback_route(query, has_image)

        try:
            return await llm_route(query, has_image, self.llm)
        except Exception as exc:
            logger.warning("AdaptiveRouter LLM classify failed: {}", exc)
            return fallback_route(query, has_image)
