"""
Unit Tests: API Key Authentication Middleware
=============================================
Tests the APIKeyMiddleware by instantiating it directly and calling
dispatch() with Mock request objects — bypasses Starlette's middleware
stack so kwargs are always correctly passed.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.middleware.auth import APIKeyMiddleware, _is_public


# ─────────────────────────────────────────────────
# Helper: mock request
# ─────────────────────────────────────────────────

def _mock_request(path: str, api_key: str = "") -> MagicMock:
    req = MagicMock()
    req.url.path = path
    req.method = "GET"
    header_dict = {}
    if api_key:
        header_dict["X-API-Key"] = api_key
    req.headers.get = lambda k, default="": header_dict.get(k, default)
    return req


def _mock_next(status: int = 200) -> AsyncMock:
    """Mock call_next that returns a simple 200 response."""
    resp = MagicMock()
    resp.status_code = status
    return AsyncMock(return_value=resp)


async def _dispatch(mw: APIKeyMiddleware, path: str, api_key: str = ""):
    req = _mock_request(path, api_key)
    return await mw.dispatch(req, _mock_next())


def _middleware(api_key: str = "", environment: str = "development") -> APIKeyMiddleware:
    """Instantiate middleware bypassing Starlette's __init__ super() chain."""
    mw = object.__new__(APIKeyMiddleware)
    mw._api_key = api_key or None
    mw._env_override = environment
    return mw


# ─────────────────────────────────────────────────
# _is_public() helper tests
# ─────────────────────────────────────────────────

class TestIsPublic:
    def test_root_is_public(self):
        assert _is_public("/") is True

    def test_health_is_public(self):
        assert _is_public("/health") is True
        assert _is_public("/health/ready") is True

    def test_docs_is_public(self):
        assert _is_public("/docs") is True
        assert _is_public("/openapi.json") is True

    def test_metrics_is_public(self):
        assert _is_public("/metrics") is True

    def test_static_is_public(self):
        assert _is_public("/static/voice_test_ui.html") is True

    def test_api_is_not_public(self):
        assert _is_public("/api/v1/rag/query") is False
        assert _is_public("/api/v1/chat") is False


# ─────────────────────────────────────────────────
# Public routes always pass
# ─────────────────────────────────────────────────

class TestPublicRoutes:
    @pytest.mark.asyncio
    async def test_health_passes_no_key_production(self):
        mw = _middleware(api_key="secret", environment="production")
        resp = await _dispatch(mw, "/health")
        # Must call call_next (pass-through) — status 200
        assert resp.status_code == 200


# ─────────────────────────────────────────────────
# Development environment — skip auth
# ─────────────────────────────────────────────────

class TestDevelopmentEnv:
    @pytest.mark.asyncio
    async def test_protected_no_key_dev_passes(self):
        mw = _middleware(api_key="secret", environment="development")
        resp = await _dispatch(mw, "/api/v1/rag/stats")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_protected_with_key_dev_passes(self):
        mw = _middleware(api_key="secret", environment="development")
        resp = await _dispatch(mw, "/api/v1/rag/stats", api_key="secret")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────
# Production / Staging — enforce auth
# ─────────────────────────────────────────────────

class TestProductionEnv:
    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self):
        from fastapi.responses import JSONResponse
        mw = _middleware(api_key="secret123", environment="production")
        result = await _dispatch(mw, "/api/v1/rag/stats", api_key="")
        # Should be a JSONResponse with status 401
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_returns_401(self):
        mw = _middleware(api_key="secret123", environment="production")
        result = await _dispatch(mw, "/api/v1/rag/stats", api_key="badkey")
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_key_passes(self):
        mw = _middleware(api_key="secret123", environment="production")
        result = await _dispatch(mw, "/api/v1/rag/stats", api_key="secret123")
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_staging_enforces_auth(self):
        mw = _middleware(api_key="stagingkey", environment="staging")
        result = await _dispatch(mw, "/api/v1/rag/stats", api_key="")
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_401_body_has_required_fields(self):
        mw = _middleware(api_key="secret", environment="production")
        result = await _dispatch(mw, "/api/v1/rag/stats")
        body = json.loads(result.body)
        assert body["error"] == "unauthorized"
        assert "message" in body


# ─────────────────────────────────────────────────
# No key configured — graceful bypass
# ─────────────────────────────────────────────────

class TestNoKeyConfigured:
    @pytest.mark.asyncio
    async def test_no_api_key_production_passes(self):
        """No API_KEY means auth never blocks traffic."""
        mw = _middleware(api_key="", environment="production")
        result = await _dispatch(mw, "/api/v1/rag/stats")
        assert result.status_code == 200
