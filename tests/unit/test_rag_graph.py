"""
Unit tests for LangGraph RAG State Machine (Phase 3 — ADR-010).

Tests cover:
  - Graph compilation and structure
  - Individual node functions (rewrite, retrieve, grade, generate, cite)
  - Safety nodes (gate, evaluate, web_search)
  - Conditional edge routing
  - End-to-end graph execution
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.rag.graph.edges import after_evaluate, after_gate, after_grade
from ai.rag.graph.state import GraphRunResult, RAGGraphState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_state() -> RAGGraphState:
    """Minimal state for testing."""
    return {
        "query": "What is tomato price in Kolar?",
        "context": "",
        "has_image": False,
        "retry_count": 0,
        "max_retries": 2,
        "current_node": "start",
    }


@pytest.fixture
def graded_state() -> RAGGraphState:
    """State after grading with relevant docs."""
    return {
        "query": "tomato cultivation Karnataka",
        "relevant_documents": [MagicMock(text="Tomato doc", id="d1")],
        "needs_web_search": False,
        "retry_count": 0,
        "max_retries": 2,
    }


# ---------------------------------------------------------------------------
# Edge routing tests
# ---------------------------------------------------------------------------


class TestAfterGrade:
    def test_routes_to_generate_when_docs_available(self):
        state: RAGGraphState = {
            "needs_web_search": False,
            "relevant_documents": [MagicMock()],
        }
        assert after_grade(state) == "generate"

    def test_routes_to_web_search_when_no_docs(self):
        state: RAGGraphState = {
            "needs_web_search": True,
            "relevant_documents": [],
        }
        assert after_grade(state) == "web_search"

    def test_routes_generate_even_with_web_search_flag(self):
        """If web_search is needed but docs exist, still generate."""
        state: RAGGraphState = {
            "needs_web_search": True,
            "relevant_documents": [MagicMock()],
        }
        assert after_grade(state) == "generate"


class TestAfterEvaluate:
    def test_routes_to_gate_when_confident(self):
        state: RAGGraphState = {
            "should_retry": False,
            "retry_count": 0,
            "max_retries": 2,
            "confidence": 0.92,
        }
        assert after_evaluate(state) == "gate"

    def test_routes_to_rewrite_on_retry(self):
        state: RAGGraphState = {
            "should_retry": True,
            "retry_count": 0,
            "max_retries": 2,
            "confidence": 0.50,
        }
        assert after_evaluate(state) == "rewrite"

    def test_routes_to_gate_at_max_retries(self):
        """Should not retry beyond max_retries."""
        state: RAGGraphState = {
            "should_retry": True,
            "retry_count": 2,
            "max_retries": 2,
            "confidence": 0.50,
        }
        assert after_evaluate(state) == "gate"


class TestAfterGate:
    def test_always_routes_to_end(self):
        state: RAGGraphState = {"is_approved": True}
        assert after_gate(state) == "end"

    def test_declined_also_routes_to_end(self):
        state: RAGGraphState = {
            "is_approved": False,
            "safety_level": "safety_critical",
            "decline_reason": "Low grounding",
        }
        assert after_gate(state) == "end"


# ---------------------------------------------------------------------------
# Node function tests
# ---------------------------------------------------------------------------


class TestRewriteNode:
    @pytest.mark.asyncio
    async def test_rewrite_returns_queries(self, empty_state):
        from ai.rag.graph.nodes import rewrite_node

        result = await rewrite_node(empty_state)
        assert "rewritten_queries" in result
        assert len(result["rewritten_queries"]) >= 1

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from ai.rag.graph.nodes import rewrite_node

        result = await rewrite_node({"query": ""})
        assert result["rewritten_queries"] == []


class TestGradeNode:
    @pytest.mark.asyncio
    async def test_empty_docs_triggers_web_search(self):
        from ai.rag.graph.nodes import grade_node

        state: RAGGraphState = {"documents": [], "query": "test"}
        result = await grade_node(state)
        assert result["needs_web_search"] is True

    @pytest.mark.asyncio
    async def test_grades_with_docs(self):
        from ai.rag.graph.nodes import grade_node
        from src.rag.knowledge_base import Document

        docs = [
            Document(text="Tomato farming in Karnataka", id="d1"),
            Document(text="Onion price in Hubli today", id="d2"),
        ]
        state: RAGGraphState = {"documents": docs, "query": "tomato farming"}
        result = await grade_node(state)
        assert "relevant_documents" in result


class TestGateNode:
    @pytest.mark.asyncio
    async def test_gate_with_answer(self):
        from ai.rag.graph.nodes_safety import gate_node

        state: RAGGraphState = {
            "cited_answer": "Tomato leaf curl is caused by whiteflies.",
            "query": "What causes tomato leaf curl?",
            "relevant_documents": [
                MagicMock(text="Tomato leaf curl virus is transmitted by whiteflies."),
            ],
            "faithfulness": 0.9,
            "relevance": 0.9,
        }
        result = await gate_node(state)
        assert "is_approved" in result
        assert "safety_level" in result


class TestEvaluateNode:
    @pytest.mark.asyncio
    async def test_evaluate_returns_scores(self):
        from ai.rag.graph.nodes_safety import evaluate_node

        state: RAGGraphState = {
            "answer": "Some answer text.",
            "query": "test query",
            "relevant_documents": [],
            "retry_count": 0,
            "max_retries": 2,
        }
        result = await evaluate_node(state)
        assert "faithfulness" in result
        assert "relevance" in result
        assert "should_retry" in result

    @pytest.mark.asyncio
    async def test_evaluate_skips_at_max_retries(self):
        from ai.rag.graph.nodes_safety import evaluate_node

        state: RAGGraphState = {
            "answer": "Some answer.",
            "query": "test",
            "retry_count": 2,
            "max_retries": 2,
        }
        result = await evaluate_node(state)
        assert result["should_retry"] is False


# ---------------------------------------------------------------------------
# Graph compilation test
# ---------------------------------------------------------------------------


class TestGraphCompilation:
    def test_graph_compiles_successfully(self):
        """Graph should compile without errors."""
        from ai.rag.graph.builder import build_rag_graph

        compiled = build_rag_graph()
        assert compiled is not None

    def test_graph_run_result_model(self):
        """GraphRunResult should initialize with defaults."""
        result = GraphRunResult()
        assert result.answer == ""
        assert result.is_approved is False
        assert result.confidence == 0.0
