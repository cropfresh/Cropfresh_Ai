"""
Query Rewriter — HyDE, Step-Back & Multi-Query Expansion
=========================================================
Rewrites user queries to improve retrieval recall.

Strategies:
  - step_back: Broaden specific queries before retrieval
  - hyde: Generate hypothetical doc → embed it for retrieval
  - multi_query: Generate 3 diverse reformulations → merge results
  - auto: Pick the best strategy based on query characteristics

Reference: ADR-010 — Advanced Agentic RAG System
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from ai.rag.query_rewriter_prompts import (
    HYDE_PROMPT,
    MULTI_QUERY_PROMPT,
    STEP_BACK_PROMPT,
    STRATEGY_CLASSIFIER_PROMPT,
)


class RewriteStrategy(str, Enum):
    """Available rewrite strategies."""
    STEP_BACK = "step_back"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    AUTO = "auto"
    NONE = "none"


class RewriteResult(BaseModel):
    """Result of query rewriting."""
    original_query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    strategy_used: str = "none"
    hyde_document: Optional[str] = None
    cost_inr: float = 0.0


class QueryRewriter:
    """
    Rewrites queries for improved retrieval recall.

    Uses Groq Llama-3.1-8B-Instant for fast, cheap rewriting (~₹0.001/call).
    Falls back to original query on any failure.
    """

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def rewrite(
        self,
        query: str,
        strategy: str = "auto",
    ) -> RewriteResult:
        """
        Rewrite a query using the specified strategy.

        Args:
            query: Original user query
            strategy: One of auto, step_back, hyde, multi_query, none

        Returns:
            RewriteResult with rewritten queries
        """
        if not query or not query.strip():
            return RewriteResult(original_query=query)

        result = RewriteResult(original_query=query)

        if strategy == "auto":
            strategy = await self._classify_strategy(query)

        result.strategy_used = strategy

        if strategy == "none" or self.llm is None:
            result.rewritten_queries = [query]
            return result

        try:
            if strategy == "step_back":
                result.rewritten_queries = await self._step_back(query)
            elif strategy == "hyde":
                hyde_doc = await self._generate_hyde(query)
                result.hyde_document = hyde_doc
                result.rewritten_queries = [query]
            elif strategy == "multi_query":
                result.rewritten_queries = await self._multi_query(query)
            else:
                result.rewritten_queries = [query]
        except Exception as e:
            logger.warning(f"Query rewrite failed ({strategy}): {e}")
            result.rewritten_queries = [query]
            result.strategy_used = "fallback"

        return result

    async def _classify_strategy(self, query: str) -> str:
        """Auto-select the best rewrite strategy for a query."""
        if self.llm is None:
            return self._classify_heuristic(query)

        try:
            from src.orchestrator.llm_provider import LLMMessage

            messages = [
                LLMMessage(role="user", content=STRATEGY_CLASSIFIER_PROMPT.format(query=query)),
            ]
            response = await self.llm.generate(messages, temperature=0.0, max_tokens=30)
            strategy = response.content.strip().lower().strip('"')

            if strategy in ("step_back", "hyde", "multi_query", "none"):
                return strategy
        except Exception as e:
            logger.debug(f"Strategy classification failed: {e}")

        return self._classify_heuristic(query)

    def _classify_heuristic(self, query: str) -> str:
        """Rule-based fallback for strategy selection."""
        q = query.lower()
        words = q.split()

        #! Safety: short/vague queries → HyDE for better recall
        if len(words) <= 3:
            return "hyde"

        # ? Multi-part questions benefit from decomposition
        if any(kw in q for kw in ("and", "versus", "compare", "difference")):
            return "multi_query"

        # Specific location/region queries → step-back for broader context
        # Only match "in/at/near" followed by place-like words
        location_indicators = (
            "district", "taluk", "mandi", "village", "region",
            "karnataka", "maharashtra", "andhra", "telangana",
            "tamil", "kerala", "punjab", "rajasthan", "gujarat",
            "kolar", "hubli", "mandya", "belgaum", "mysore",
        )
        if any(loc in q for loc in location_indicators):
            return "step_back"

        return "none"

    async def _step_back(self, query: str) -> list[str]:
        """Generate a broader step-back query + original."""
        from src.orchestrator.llm_provider import LLMMessage

        messages = [LLMMessage(role="user", content=STEP_BACK_PROMPT.format(query=query))]
        response = await self.llm.generate(messages, temperature=0.3, max_tokens=100)

        broader = response.content.strip()
        logger.info(f"Step-back: '{query}' → '{broader}'")
        return [query, broader]

    async def _generate_hyde(self, query: str) -> str:
        """Generate a hypothetical document for HyDE embedding."""
        from src.orchestrator.llm_provider import LLMMessage

        messages = [LLMMessage(role="user", content=HYDE_PROMPT.format(query=query))]
        response = await self.llm.generate(messages, temperature=0.5, max_tokens=200)

        hyde_doc = response.content.strip()
        logger.info(f"HyDE doc generated: {len(hyde_doc)} chars")
        return hyde_doc

    async def _multi_query(self, query: str) -> list[str]:
        """Generate 3 diverse reformulations of the query."""
        from src.orchestrator.llm_provider import LLMMessage

        messages = [LLMMessage(role="user", content=MULTI_QUERY_PROMPT.format(query=query))]
        response = await self.llm.generate(messages, temperature=0.4, max_tokens=200)

        try:
            queries = json.loads(response.content)
            if isinstance(queries, list) and len(queries) >= 2:
                logger.info(f"Multi-query: {len(queries)} reformulations")
                return [query] + queries[:3]
        except (json.JSONDecodeError, TypeError):
            # Parse line-by-line fallback
            lines = [ln.strip().lstrip("0123456789.-) ") for ln in response.content.splitlines() if ln.strip()]
            if lines:
                return [query] + lines[:3]

        return [query]
