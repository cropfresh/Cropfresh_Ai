"""
Unit tests for QueryRewriter (Phase 1 — ADR-010).

Tests cover:
  - HyDE strategy generates hypothetical document
  - Step-back strategy broadens specific queries
  - Multi-query strategy decomposes into 3+ queries
  - Auto strategy classification (heuristic fallback)
  - Graceful fallback on LLM failure
  - Empty/invalid query handling
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.rag.query_rewriter import QueryRewriter, RewriteResult, RewriteStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def rewriter(mock_llm):
    return QueryRewriter(llm=mock_llm)


@pytest.fixture
def rewriter_no_llm():
    return QueryRewriter(llm=None)


# ---------------------------------------------------------------------------
# Strategy classification tests
# ---------------------------------------------------------------------------


class TestStrategyClassification:
    def test_short_query_classified_as_hyde(self, rewriter: QueryRewriter):
        """Short/vague queries should default to HyDE."""
        strategy = rewriter._classify_heuristic("crop problem")
        assert strategy == "hyde"

    def test_location_query_classified_as_step_back(self, rewriter: QueryRewriter):
        """Queries with location hints should use step-back."""
        strategy = rewriter._classify_heuristic(
            "How to grow tomatoes in Kolar district?"
        )
        assert strategy == "step_back"

    def test_comparison_query_classified_as_multi_query(self, rewriter: QueryRewriter):
        """Comparative queries should use multi-query decomposition."""
        strategy = rewriter._classify_heuristic(
            "Difference between organic and chemical fertilizer"
        )
        assert strategy == "multi_query"

    def test_clear_query_classified_as_none(self, rewriter: QueryRewriter):
        """Well-formed clear queries need no rewriting."""
        strategy = rewriter._classify_heuristic(
            "What is the recommended NPK ratio for paddy?"
        )
        assert strategy == "none"


# ---------------------------------------------------------------------------
# Rewrite strategy execution tests
# ---------------------------------------------------------------------------


class TestStepBackStrategy:
    @pytest.mark.asyncio
    async def test_step_back_returns_two_queries(self, rewriter, mock_llm):
        """Step-back should return original + broader query."""
        mock_llm.generate.return_value = MagicMock(
            content="What are common tomato diseases and management?"
        )
        result = await rewriter.rewrite(
            "How to control leaf curl on tomato in Kolar?",
            strategy="step_back",
        )
        assert len(result.rewritten_queries) == 2
        assert result.strategy_used == "step_back"
        assert result.original_query in result.rewritten_queries


class TestHyDEStrategy:
    @pytest.mark.asyncio
    async def test_hyde_generates_document(self, rewriter, mock_llm):
        """HyDE should generate a hypothetical document."""
        mock_llm.generate.return_value = MagicMock(
            content="Tomato leaf curl virus is common in Karnataka..."
        )
        result = await rewriter.rewrite("crop problem", strategy="hyde")
        assert result.hyde_document is not None
        assert len(result.hyde_document) > 10
        assert result.strategy_used == "hyde"


class TestMultiQueryStrategy:
    @pytest.mark.asyncio
    async def test_multi_query_returns_multiple(self, rewriter, mock_llm):
        """Multi-query should return 3+ reformulations."""
        mock_llm.generate.return_value = MagicMock(
            content=json.dumps([
                "organic fertilizer benefits",
                "chemical fertilizer advantages",
                "organic vs chemical crop yield",
            ])
        )
        result = await rewriter.rewrite(
            "organic vs chemical fertilizer",
            strategy="multi_query",
        )
        assert len(result.rewritten_queries) >= 3
        assert result.strategy_used == "multi_query"

    @pytest.mark.asyncio
    async def test_multi_query_fallback_on_bad_json(self, rewriter, mock_llm):
        """Should parse line-by-line if JSON fails."""
        mock_llm.generate.return_value = MagicMock(
            content="1. query one\n2. query two\n3. query three"
        )
        result = await rewriter.rewrite(
            "tomato and onion prices",
            strategy="multi_query",
        )
        # Original + 3 parsed lines
        assert len(result.rewritten_queries) >= 2


# ---------------------------------------------------------------------------
# Edge cases and fallback
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, rewriter):
        """Empty query should return empty result."""
        result = await rewriter.rewrite("")
        assert result.rewritten_queries == []

    @pytest.mark.asyncio
    async def test_no_llm_returns_original(self, rewriter_no_llm):
        """Without LLM, should return original query unchanged."""
        query = "What is the recommended NPK ratio for paddy?"
        result = await rewriter_no_llm.rewrite(query)
        assert result.rewritten_queries == [query]
        assert result.strategy_used == "none"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_gracefully(self, rewriter, mock_llm):
        """LLM failure should return original query."""
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        result = await rewriter.rewrite("test query", strategy="step_back")
        assert result.rewritten_queries == ["test query"]
        assert result.strategy_used == "fallback"
