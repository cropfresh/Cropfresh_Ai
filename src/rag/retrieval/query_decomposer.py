"""
Query Decomposer — Multi-part question splitting (ADR-010 Phase 4).

Breaks complex agricultural queries into atomic sub-questions,
each retrievable independently. Results are merged with deduplication.

Example:
  Input:  "What is the price of tomato in Kolar and how to control leaf curl?"
  Output: ["tomato price Kolar mandi", "tomato leaf curl disease control"]
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class DecomposedQuery(BaseModel):
    """Result of query decomposition."""

    original_query: str
    sub_queries: list[str] = Field(default_factory=list)
    is_multi_part: bool = False
    decomposition_method: str = "heuristic"  # "heuristic" | "llm"


class QueryDecomposer:
    """Decomposes complex queries into atomic sub-questions.

    Multi-part agricultural queries are common in voice-first contexts:
    "What is tomato price and which fertilizer for ragi and when to sow?"

    This splitter breaks them into independent retrieval targets.
    """

    # ? Conjunctions that indicate multi-part queries
    SPLIT_PATTERNS: list[str] = [
        r"\band\b",
        r"\balso\b",
        r"\bmattu\b",      # Kannada: "and"
        r"\bhagu\b",       # Kannada: "also"
        r"\bplus\b",
        r"\balong with\b",
    ]

    # ? Question starters indicating a new sub-question
    QUESTION_MARKERS: list[str] = [
        r"\bhow\b", r"\bwhat\b", r"\bwhen\b", r"\bwhere\b",
        r"\bwhich\b", r"\bwhy\b", r"\bhow much\b",
        r"\byaavaga\b",   # Kannada: "when"
        r"\byenu\b",      # Kannada: "what"
        r"\bhege\b",      # Kannada: "how"
    ]

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def decompose(self, query: str) -> DecomposedQuery:
        """Decompose a query into sub-queries.

        Args:
            query: Raw user query (may be multi-part).

        Returns:
            DecomposedQuery with atomic sub-questions.
        """
        if not query or len(query) < 15:
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query] if query else [],
            )

        if self.llm is not None:
            return await self._llm_decompose(query)

        return self._heuristic_decompose(query)

    def _heuristic_decompose(self, query: str) -> DecomposedQuery:
        """Split query using conjunction patterns."""
        # First check if it even looks multi-part
        has_conjunction = any(
            re.search(p, query, re.IGNORECASE)
            for p in self.SPLIT_PATTERNS
        )
        has_multiple_questions = sum(
            1 for p in self.QUESTION_MARKERS
            if re.search(p, query, re.IGNORECASE)
        ) >= 2

        if not has_conjunction and not has_multiple_questions:
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                is_multi_part=False,
            )

        # Split on conjunctions
        pattern = "|".join(self.SPLIT_PATTERNS)
        parts = re.split(f"({pattern})", query, flags=re.IGNORECASE)

        # Clean and filter parts
        sub_queries: list[str] = []
        for part in parts:
            cleaned = part.strip(" ,?.")
            if len(cleaned) < 8:
                continue
            # Skip if it's just a conjunction
            if re.fullmatch(pattern, cleaned, re.IGNORECASE):
                continue
            sub_queries.append(cleaned)

        if len(sub_queries) <= 1:
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                is_multi_part=False,
            )

        logger.info(
            f"QueryDecomposer: split into {len(sub_queries)} sub-queries | "
            f"original='{query[:60]}...'"
        )

        return DecomposedQuery(
            original_query=query,
            sub_queries=sub_queries,
            is_multi_part=True,
            decomposition_method="heuristic",
        )

    async def _llm_decompose(self, query: str) -> DecomposedQuery:
        """Decompose using LLM for complex queries."""
        try:
            import json

            from src.orchestrator.llm_provider import LLMMessage

            prompt = (
                "Break this farmer's question into independent sub-questions. "
                "Return JSON: {\"sub_queries\": [\"q1\", \"q2\"]}\n\n"
                f"Question: {query}"
            )
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=200,
            )
            result = json.loads(response.content)
            subs = result.get("sub_queries", [query])

            return DecomposedQuery(
                original_query=query,
                sub_queries=subs if len(subs) > 1 else [query],
                is_multi_part=len(subs) > 1,
                decomposition_method="llm",
            )
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
            return self._heuristic_decompose(query)
