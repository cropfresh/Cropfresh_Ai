"""
Unit Tests: RAG Route — Redis Cache
=====================================
Tests the caching logic in /rag/query endpoint:
  - Cache miss triggers agent.answer() call
  - Cache hit returns cached result (agent.answer NOT called)
  - Cache key is deterministic (same q+context → same key)
  - Cache is skipped silently when Redis is unavailable
"""

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes.rag import _cache_key


# ─────────────────────────────────────────────────
# Cache key tests (pure function, no I/O)
# ─────────────────────────────────────────────────

class TestCacheKey:
    def test_same_inputs_same_key(self):
        k1 = _cache_key("How to grow tomatoes?", "Karnataka crop guide")
        k2 = _cache_key("How to grow tomatoes?", "Karnataka crop guide")
        assert k1 == k2

    def test_different_question_different_key(self):
        k1 = _cache_key("tomato", "")
        k2 = _cache_key("potato", "")
        assert k1 != k2

    def test_case_insensitive(self):
        """Leading/trailing whitespace and case folded."""
        k1 = _cache_key("  Tomato  ", "")
        k2 = _cache_key("tomato", "")
        assert k1 == k2

    def test_key_has_rag_prefix(self):
        key = _cache_key("q", "ctx")
        assert key.startswith("rag:")

    def test_key_is_sha256_length(self):
        key = _cache_key("q", "ctx")
        # "rag:" (4) + sha256 hex (64)
        assert len(key) == 68


# ─────────────────────────────────────────────────
# Integration-style tests for the /rag/query endpoint
# ─────────────────────────────────────────────────

def _make_test_app():
    """Build minimal FastAPI app mounting only the RAG router."""
    from fastapi import FastAPI
    from src.api.routes.rag import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Inject mock knowledge agent
    mock_agent = AsyncMock()
    mock_agent.answer = AsyncMock(return_value=MagicMock(
        answer="Tomato grows well in loamy soil.",
        sources=["ICAR Guideline v3"],
        confidence=0.87,
        query_type="agronomy",
        steps=["retrieve", "grade", "generate"],
    ))
    app.state.knowledge_agent = mock_agent
    app.state.redis = None  # Redis disabled for unit tests
    return app, mock_agent


class TestRAGQueryCacheMiss:
    """When Redis is None or key not found, agent.answer() must be called."""

    def test_cache_miss_calls_agent(self):
        app, mock_agent = _make_test_app()
        client = TestClient(app)

        r = client.post(
            "/api/v1/rag/query",
            json={"question": "How to grow tomatoes?", "context": ""},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Tomato grows well in loamy soil."
        assert data["cached"] is False
        mock_agent.answer.assert_called_once()

    def test_response_schema(self):
        app, _ = _make_test_app()
        client = TestClient(app)

        r = client.post("/api/v1/rag/query", json={"question": "soil health", "context": ""})
        data = r.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert "cached" in data


class TestRAGQueryCacheHit:
    """When Redis has a matching key, return cached result without calling agent."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_agent(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        from src.api.routes.rag import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.answer = AsyncMock(return_value=MagicMock(
            answer="From LLM", sources=[], confidence=0.9,
            query_type="agronomy", steps=[],
        ))
        app.state.knowledge_agent = mock_agent

        # Mock Redis that always returns a cached value
        cached_payload = json.dumps({
            "answer": "Cached answer",
            "sources": ["cache"],
            "confidence": 0.95,
            "query_type": "agronomy",
            "steps": ["cache_hit"],
        }).encode()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=cached_payload)
        mock_redis.setex = AsyncMock()
        app.state.redis = mock_redis

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post(
                "/api/v1/rag/query",
                json={"question": "soil pH", "context": ""},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Cached answer"
        assert data["cached"] is True
        # Agent should NOT be called
        mock_agent.answer.assert_not_called()


class TestRAGQueryNoAgent:
    """When knowledge_agent is not in app.state, return 503."""

    def test_503_when_agent_missing(self):
        from fastapi import FastAPI
        from src.api.routes.rag import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.state.knowledge_agent = None  # not initialized
        app.state.redis = None

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/api/v1/rag/query", json={"question": "test", "context": ""})
        assert r.status_code == 503
