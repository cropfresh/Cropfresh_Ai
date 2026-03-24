"""
Unit tests for CitationEngine (Phase 1 — ADR-010).

Tests cover:
  - Heuristic citation placement (keyword overlap)
  - LLM-based citation with mock responses
  - Source list extraction from documents
  - Empty answer / empty docs edge cases
  - Sources section formatting
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.citation_engine import CitationEngine, CitedAnswer, Source

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def engine(mock_llm):
    return CitationEngine(llm=mock_llm)


@pytest.fixture
def engine_no_llm():
    return CitationEngine(llm=None)


@pytest.fixture
def sample_docs():
    """Minimal doc objects with .text, .id, and .metadata."""
    doc1 = MagicMock()
    doc1.text = "Tomato leaf curl virus (ToLCV) is transmitted by whiteflies."
    doc1.id = "doc-001"
    doc1.metadata = {"title": "IIHR Tomato Disease Guide"}

    doc2 = MagicMock()
    doc2.text = "Apply neem oil spray every 7 days as preventive measure."
    doc2.id = "doc-002"
    doc2.metadata = {"title": "KVK Kolar Pest Advisory"}

    return [doc1, doc2]


# ---------------------------------------------------------------------------
# Heuristic citation tests
# ---------------------------------------------------------------------------


class TestHeuristicCitations:
    @pytest.mark.asyncio
    async def test_heuristic_adds_markers(self, engine_no_llm, sample_docs):
        """Heuristic engine should add [1] or [2] markers."""
        answer = (
            "Tomato leaf curl is caused by whitefly-transmitted viruses. "
            "Apply neem oil spray every 7 days as preventive measure."
        )
        result = await engine_no_llm.add_citations(answer, sample_docs)

        assert isinstance(result, CitedAnswer)
        assert result.citation_count > 0
        assert re.search(r"\[\d+\]", result.answer)

    @pytest.mark.asyncio
    async def test_empty_answer_returns_unchanged(self, engine_no_llm):
        """Empty answer should return empty CitedAnswer."""
        result = await engine_no_llm.add_citations("", [])
        assert result.answer == ""
        assert result.citation_count == 0


# ---------------------------------------------------------------------------
# LLM-based citation tests
# ---------------------------------------------------------------------------


class TestLLMCitations:
    @pytest.mark.asyncio
    async def test_llm_citation_places_markers(self, engine, mock_llm, sample_docs):
        """LLM should return answer with inline citations."""
        mock_llm.generate.return_value = MagicMock(
            content=(
                "Tomato leaf curl is caused by whitefly-transmitted viruses [1]. "
                "Apply neem oil spray every 7 days [2]."
            )
        )
        result = await engine.add_citations(
            "Tomato leaf curl is caused by whitefly-transmitted viruses.",
            sample_docs,
        )
        assert "[1]" in result.answer
        assert result.citation_count >= 1

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristic(
        self, engine, mock_llm, sample_docs,
    ):
        """LLM failure should fall back to heuristic citations."""
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        result = await engine.add_citations(
            "Tomato leaf curl is caused by whiteflies.",
            sample_docs,
        )
        # Should still produce a result, not crash
        assert isinstance(result, CitedAnswer)


# ---------------------------------------------------------------------------
# Source extraction tests
# ---------------------------------------------------------------------------


class TestSourceExtraction:
    def test_sources_from_docs(self, engine, sample_docs):
        """Should extract title, snippet, and doc_id from documents."""
        sources = engine._build_sources(sample_docs)
        assert len(sources) == 2
        assert sources[0].title == "IIHR Tomato Disease Guide"
        assert sources[1].doc_id == "doc-002"

    def test_format_sources_section(self, engine):
        """Sources section should list [1], [2] with titles."""
        sources = [
            Source(index=1, title="IIHR Guide"),
            Source(index=2, title="KVK Advisory"),
        ]
        section = engine.format_sources_section(sources)
        assert "[1] IIHR Guide" in section
        assert "[2] KVK Advisory" in section

    def test_empty_sources_returns_empty_string(self, engine):
        assert engine.format_sources_section([]) == ""
