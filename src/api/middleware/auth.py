"""
API Key Authentication Middleware
==================================
Validates X-API-Key header for all protected /api/v1/* routes.

Public routes (no auth required):
  - /               (root info)
  - /health         (liveness probe)
  - /health/ready   (readiness probe)
  - /docs           (Swagger UI — disabled in production)
  - /openapi.json   (OpenAPI schema)
  - /metrics        (Prometheus scrape — secured at network level)
  - /static/*       (test HTML/JS assets)
  - /ws/*           (WebSocket endpoints)

In development (ENVIRONMENT=development or API_KEY unset), auth is skipped.
"""

from __future__ import annotations

import os
from typing import Final

from fastapi.responses import JSONResponse
from loguru import logger

# Routes that NEVER require an API key
PUBLIC_PREFIXES: Final[tuple[str, ...]] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/static",
    "/api/v1/voice/ws",
    # NOTE: "/" is intentionally NOT here — startswith("/") matches all paths.
    # The root "/" is handled by the explicit `path == "/"` check in _is_public().
)


def _is_public(path: str) -> bool:
    """Return True if the path is exempt from auth checks."""
    if path == "/":
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


class APIKeyMiddleware:
    """
    Pure ASGI middleware for API-key guarding.
    Does not use BaseHTTPMiddleware to avoid breaking WebSocket Upgrades.
    """

    def __init__(self, app, api_key: str | None = None, environment: str | None = None):
        self.app = app
        # Resolve at startup — avoids per-request env lookups
        self._api_key: str | None = api_key or os.environ.get("API_KEY", "").strip() or None
        # environment can be injected directly (useful in tests) or read from os.environ
        self._env_override: str | None = environment

        if self._api_key:
            logger.info(
                "🔐 API key auth ENABLED. Key configured: {}...",
                self._api_key[:6],
            )
        else:
            logger.warning("⚠️  API key auth DISABLED — set API_KEY env var to enable.")

    def _current_environment(self) -> str:
        """Return the effective environment name used for auth checks."""
        return (self._env_override or os.environ.get("ENVIRONMENT", "development")).lower()

    def _is_auth_bypassed(self, path: str) -> bool:
        """Return whether auth should be skipped for this path and environment."""
        return _is_public(path) or self._current_environment() == "development" or not self._api_key

    def _is_authorized(self, provided_key: str) -> bool:
        """Return whether the provided key matches the configured API key."""
        return bool(provided_key and self._api_key and provided_key.strip() == self._api_key)

    async def dispatch(self, request, call_next):
        """
        Backward-compatible HTTP dispatch hook used by legacy unit tests.

        The runtime app uses the pure ASGI ``__call__`` path below so WebSocket
        upgrades continue to work correctly.
        """
        path = getattr(getattr(request, "url", None), "path", "")
        if self._is_auth_bypassed(path):
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "") or request.headers.get("x-api-key", "")
        if not self._is_authorized(provided_key):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Missing or invalid X-API-Key header",
                },
            )
        return await call_next(request)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Always pass through public routes
        if self._is_auth_bypassed(path):
            await self.app(scope, receive, send)
            return

        # Validate key
        # Headers in ASGI scope are list of (bytes, bytes)
        headers = dict((k.decode("latin-1").lower(), v.decode("latin-1")) for k, v in scope.get("headers", []))
        provided_key = headers.get("x-api-key", "").strip()

        if not self._is_authorized(provided_key):
            if scope["type"] == "websocket":
                # Reject WebSocket connection
                await send({"type": "websocket.close", "code": 1008})
                return
            else:
                # Reject HTTP request
                response = JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Missing or invalid X-API-Key header",
                    },
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
