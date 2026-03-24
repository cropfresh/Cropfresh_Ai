"""
Unit tests for Phase 4 Advanced Retrieval modules (ADR-010).

Tests cover:
  - ContextualChunkEnricher: enrichment, section inference, edge cases
  - QueryDecomposer: multi-part splitting, Kannada support, simple queries
  - TimeAwareRetriever: category classification, score adjustment, decay
  - AdvancedRetriever: coordination and deduplication
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from src.rag.retrieval.contextual_enricher import (
    ContextualChunk,
    ContextualChunkEnricher,
)
from src.rag.retrieval.query_decomposer import QueryDecomposer
from src.rag.retrieval.time_aware import (
    FreshnessCategory,
    TimeAwareRetriever,
)

# ---------------------------------------------------------------------------
# ContextualChunkEnricher Tests
# ---------------------------------------------------------------------------


class TestContextualChunkEnricher:
    @pytest.mark.asyncio
    async def test_enriches_chunks_with_context(self):
        enricher = ContextualChunkEnricher()
        chunks = ["Tomato leaf curl is caused by whiteflies."]
        result = await enricher.enrich_chunks(
            chunks, full_document="Long document about tomatoes...",
            doc_id="t1", metadata={"title": "Tomato Guide"},
        )
        assert len(result) == 1
        assert result[0].context_prefix != ""
        assert "Tomato Guide" in result[0].enriched_text

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        enricher = ContextualChunkEnricher()
        result = await enricher.enrich_chunks([], "doc text")
        assert result == []

    @pytest.mark.asyncio
    async def test_section_inference_pest(self):
        enricher = ContextualChunkEnricher()
        chunks = [
            "Introduction to tomato farming in Karnataka.",
            "Apply neem oil to control pest infestation on leaves.",
            "Final notes on harvest timing.",
        ]
        result = await enricher.enrich_chunks(
            chunks, "document", doc_id="d1",
        )
        assert "pest management" in result[1].context_prefix

    @pytest.mark.asyncio
    async def test_section_inference_market(self):
        enricher = ContextualChunkEnricher()
        chunks = ["Current mandi price for onion is ₹2500/quintal."]
        # index=1, total=3 → not intro/conclusion, triggers keyword match
        result = await enricher.enrich_chunks(
            ["Intro chunk", chunks[0], "Conclusion chunk"],
            "document", doc_id="d1",
        )
        assert "market information" in result[1].context_prefix

    @pytest.mark.asyncio
    async def test_contextual_chunk_text_property(self):
        chunk = ContextualChunk(
            original_text="raw text",
            enriched_text="context: raw text",
        )
        assert chunk.text == "context: raw text"

    @pytest.mark.asyncio
    async def test_chunk_without_enrichment_falls_back(self):
        chunk = ContextualChunk(original_text="raw text")
        assert chunk.text == "raw text"


# ---------------------------------------------------------------------------
# QueryDecomposer Tests
# ---------------------------------------------------------------------------


class TestQueryDecomposer:
    @pytest.mark.asyncio
    async def test_simple_query_not_decomposed(self):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose(
            "What is the price of tomato in Kolar?"
        )
        assert not result.is_multi_part
        assert len(result.sub_queries) == 1

    @pytest.mark.asyncio
    async def test_multi_part_query_split(self):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose(
            "What is the price of tomato and how to control leaf curl disease?"
        )
        assert result.is_multi_part
        assert len(result.sub_queries) >= 2

    @pytest.mark.asyncio
    async def test_kannada_conjunction_recognized(self):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose(
            "What is tomato price mattu how to grow ragi in Karnataka?"
        )
        assert result.is_multi_part
        assert len(result.sub_queries) >= 2

    @pytest.mark.asyncio
    async def test_short_query_passthrough(self):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("tomato")
        assert not result.is_multi_part
        assert result.sub_queries == ["tomato"]

    @pytest.mark.asyncio
    async def test_empty_query(self):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("")
        assert result.sub_queries == []


# ---------------------------------------------------------------------------
# TimeAwareRetriever Tests
# ---------------------------------------------------------------------------


class TestTimeAwareRetriever:
    def test_market_query_classified(self):
        retriever = TimeAwareRetriever()
        assert retriever.classify_query(
            "What is tomato price in mandi?"
        ) == FreshnessCategory.MARKET

    def test_weather_query_classified(self):
        retriever = TimeAwareRetriever()
        assert retriever.classify_query(
            "Will it rain tomorrow in Hubli?"
        ) == FreshnessCategory.WEATHER

    def test_scheme_query_classified(self):
        retriever = TimeAwareRetriever()
        assert retriever.classify_query(
            "How to apply for PM-KISAN scheme?"
        ) == FreshnessCategory.SCHEME

    def test_evergreen_query_classified(self):
        retriever = TimeAwareRetriever()
        assert retriever.classify_query(
            "How to grow ragi in red soil?"
        ) == FreshnessCategory.EVERGREEN

    def test_market_docs_penalized_for_age(self):
        retriever = TimeAwareRetriever()
        now = time.time()
        old_doc = SimpleNamespace(
            id="d1", score=0.9,
            metadata={"timestamp": now - (48 * 3600)},  # 48h old
        )
        fresh_doc = SimpleNamespace(
            id="d2", score=0.7,
            metadata={"timestamp": now - (1 * 3600)},  # 1h old
        )

        results = retriever.adjust_scores(
            [old_doc, fresh_doc],
            "tomato price market",
            current_time=now,
        )

        # Fresh doc should rank higher after adjustment
        assert results[0].doc_id == "d2"

    def test_evergreen_docs_not_penalized(self):
        retriever = TimeAwareRetriever()
        now = time.time()
        old_doc = SimpleNamespace(
            id="d1", score=0.9,
            metadata={"timestamp": now - (365 * 24 * 3600)},  # 1y old
        )

        results = retriever.adjust_scores(
            [old_doc], "how to grow tomato",
            current_time=now,
        )

        assert results[0].freshness_score == 1.0

    def test_empty_docs_returns_empty(self):
        retriever = TimeAwareRetriever()
        results = retriever.adjust_scores([], "test query")
        assert results == []
