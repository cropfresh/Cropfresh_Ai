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

In development (ENVIRONMENT=development or API_KEY unset), auth is skipped.
"""

from __future__ import annotations

import os
from typing import Final

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Routes that NEVER require an API key
PUBLIC_PREFIXES: Final[tuple[str, ...]] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/static",
    # NOTE: "/" is intentionally NOT here — startswith("/") matches all paths.
    # The root "/" is handled by the explicit `path == "/"` check in _is_public().
)


def _is_public(path: str) -> bool:
    """Return True if the path is exempt from auth checks."""
    if path == "/":
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Simple API-key guard for all /api/v1/* endpoints.

    The key is read from the X-API-Key request header.
    When API_KEY env var is empty or ENVIRONMENT=development, auth is skipped
    so local development works out-of-the-box without configuration.
    """

    def __init__(self, app, api_key: str | None = None, environment: str | None = None):
        super().__init__(app)
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

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Always pass through public routes
        if _is_public(path):
            return await call_next(request)

        # Skip auth in development or when no key is configured
        # _env_override is used in tests; production reads from os.environ
        current_env = (
            self._env_override
            or os.environ.get("ENVIRONMENT", "development")
        ).lower()
        if current_env == "development" or not self._api_key:
            return await call_next(request)

        # Validate key
        provided_key = request.headers.get("X-API-Key", "").strip()
        if not provided_key:
            logger.warning("Missing API key for {} {}", request.method, path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Missing X-API-Key header",
                    "docs": "/docs",
                },
            )

        if provided_key != self._api_key:
            logger.warning("Invalid API key for {} {}", request.method, path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Invalid API key",
                },
            )

        return await call_next(request)
