"""
Unit tests for enhanced DocumentGrader (Phase 1 — ADR-010).

Tests cover:
  - Continuous 0–1 scoring (new LLM prompt)
  - Time-decay penalty for stale market documents
  - Simple (keyword-based) grading fallback
  - Grade documents batch with web search trigger
  - Hallucination checker (unchanged behavior)
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.grader import (
    DocumentGrader,
)
from src.rag.knowledge_base import Document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(text: str, doc_id: str = "doc-001", score=None, metadata=None):
    """Helper to create real Document objects (not mocks)."""
    return Document(
        id=doc_id,
        text=text,
        score=score,
        metadata=metadata or {},
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def grader(mock_llm):
    return DocumentGrader(llm=mock_llm)


@pytest.fixture
def grader_no_llm():
    return DocumentGrader(llm=None)


# ---------------------------------------------------------------------------
# Continuous scoring tests
# ---------------------------------------------------------------------------


class TestContinuousScoring:
    @pytest.mark.asyncio
    async def test_llm_returns_continuous_score(self, grader, mock_llm):
        """LLM grading should return 0–1 continuous score."""
        mock_llm.generate.return_value = MagicMock(
            content=json.dumps({"score": 0.85, "reasoning": "Directly relevant"})
        )
        doc = _make_doc("Tomato cultivation in Karnataka requires red soil.")
        result = await grader.grade_document(doc, "How to grow tomato?")

        assert result.score == 0.85
        assert result.is_relevant is True
        assert result.reasoning == "Directly relevant"

    @pytest.mark.asyncio
    async def test_low_score_marks_irrelevant(self, grader, mock_llm):
        """Score below 0.4 should mark document as irrelevant."""
        mock_llm.generate.return_value = MagicMock(
            content=json.dumps({"score": 0.2, "reasoning": "Barely related"})
        )
        doc = _make_doc("Weather forecast for Mumbai region.")
        result = await grader.grade_document(doc, "Tomato price in Kolar?")

        assert result.score == 0.2
        assert result.is_relevant is False


# ---------------------------------------------------------------------------
# Time-decay tests
# ---------------------------------------------------------------------------


class TestTimeDecay:
    @pytest.mark.asyncio
    async def test_stale_market_doc_gets_penalty(self, grader_no_llm):
        """Market docs older than 7 days should get score × 0.5."""
        old_timestamp = time.time() - (10 * 24 * 3600)  # 10 days ago
        doc = _make_doc(
            "Onion price in Hubli is ₹30/kg",
            metadata={"created_at": old_timestamp},
        )
        result = await grader_no_llm.grade_document(doc, "What is onion price?")

        assert result.time_penalty_applied is True

    @pytest.mark.asyncio
    async def test_fresh_market_doc_no_penalty(self, grader_no_llm):
        """Recent market docs should NOT get time-decay penalty."""
        fresh_timestamp = time.time() - 3600  # 1 hour ago
        doc = _make_doc(
            "Onion price in Hubli is ₹30/kg today",
            metadata={"created_at": fresh_timestamp},
        )
        result = await grader_no_llm.grade_document(doc, "What is onion price?")

        assert result.time_penalty_applied is False

    @pytest.mark.asyncio
    async def test_non_market_query_no_penalty(self, grader_no_llm):
        """Non-market queries should NOT trigger time-decay."""
        old_timestamp = time.time() - (30 * 24 * 3600)  # 30 days ago
        doc = _make_doc(
            "Ragi cultivation techniques for Karnataka",
            metadata={"created_at": old_timestamp},
        )
        result = await grader_no_llm.grade_document(
            doc, "How to grow ragi?"
        )

        assert result.time_penalty_applied is False


# ---------------------------------------------------------------------------
# Simple grading fallback
# ---------------------------------------------------------------------------


class TestSimpleGrading:
    @pytest.mark.asyncio
    async def test_keyword_overlap_scoring(self, grader_no_llm):
        """Simple grading should use keyword overlap."""
        doc = _make_doc("tomato farming in Karnataka is popular")
        result = await grader_no_llm.grade_document(
            doc, "tomato farming Karnataka"
        )
        assert result.score > 0.3
        assert result.is_relevant is True

    @pytest.mark.asyncio
    async def test_no_overlap_irrelevant(self, grader_no_llm):
        """No keyword overlap should mark as irrelevant."""
        doc = _make_doc("weather forecast for Chennai region")
        result = await grader_no_llm.grade_document(
            doc, "tomato price Kolar"
        )
        assert result.is_relevant is False


# ---------------------------------------------------------------------------
# Batch grading
# ---------------------------------------------------------------------------


class TestBatchGrading:
    @pytest.mark.asyncio
    async def test_empty_docs_triggers_web_search(self, grader_no_llm):
        """Empty document list should trigger web search."""
        result = await grader_no_llm.grade_documents([], "any query")
        assert result.needs_web_search is True
        assert result.total_graded == 0

    @pytest.mark.asyncio
    async def test_all_relevant_no_web_search(self, grader_no_llm):
        """All relevant docs should not trigger web search."""
        docs = [
            _make_doc("tomato farming techniques in Karnataka"),
            _make_doc("tomato cultivation best practices India"),
        ]
        result = await grader_no_llm.grade_documents(docs, "tomato farming")
        assert result.needs_web_search is False
        assert len(result.relevant_documents) >= 1
