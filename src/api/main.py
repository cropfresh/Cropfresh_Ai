"""CropFresh AI FastAPI application entry point."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.config import get_settings
from src.api.middleware.auth import APIKeyMiddleware
from src.api.runtime.lifespan import lifespan
from src.api.runtime.router_setup import include_api_routers

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.SelectorEventLoop = asyncio.ProactorEventLoop

_settings = get_settings()

app = FastAPI(
    title="CropFresh AI Service",
    description=(
        "Agentic RAG + Voice + Market Intelligence backend for India's agricultural marketplace.\n\n"
        "**Auth:** Include `X-API-Key: <your_key>` for API routes outside development."
    ),
    version="0.2.0",
    docs_url="/docs" if not _settings.is_production else None,
    redoc_url="/redoc" if not _settings.is_production else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware, api_key=_settings.api_key or None)


@app.get("/", tags=["meta"], include_in_schema=False)
async def root():
    """Root endpoint."""
    if _settings.is_production:
        return {
            "service": "cropfresh-ai",
            "version": "0.2.0",
            "environment": _settings.environment,
        }
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/index.html")


@app.get("/health", tags=["health"])
async def health():
    """Simple liveness probe."""
    return {"status": "alive"}


@app.websocket("/ws/test")
async def test_websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello from bare websocket!")
    await websocket.close()


@app.get("/health/ready", tags=["health"])
async def readiness(request: Request):
    """Readiness probe for critical app services."""
    checks: dict[str, bool | str] = {
        "llm_provider": _settings.llm_provider,
        "llm_configured": _settings.has_llm_configured,
        "llm_initialized": getattr(request.app.state, "llm", None) is not None,
        "knowledge_agent": getattr(request.app.state, "knowledge_agent", None) is not None,
        "supervisor": getattr(request.app.state, "supervisor", None) is not None,
        "redis": getattr(request.app.state, "redis", None) is not None,
        "db": getattr(request.app.state, "db", None) is not None,
        "adcl_service": getattr(request.app.state, "adcl_service", None) is not None,
        "listing_service": getattr(request.app.state, "listing_service", None) is not None,
        "voice_agent": getattr(request.app.state, "voice_agent", None) is not None,
    }
    all_ready = bool(checks["supervisor"] and checks["adcl_service"] and checks["voice_agent"])
    return Response(
        content=__import__("orjson").dumps({"ready": all_ready, "checks": checks}),
        media_type="application/json",
        status_code=200 if all_ready else 503,
    )


@app.get("/metrics", tags=["observability"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        from src.production.observability import get_all_metrics

        metrics = get_all_metrics()
        lines = [
            "# HELP cropfresh_agent_requests_total Total requests per agent",
            "# TYPE cropfresh_agent_requests_total counter",
        ]
        for agent, values in metrics.get("by_agent", {}).items():
            lines.append(
                f'cropfresh_agent_requests_total{{agent="{agent}"}} {values["total_requests"]}'
            )
        return Response(content="\n".join(lines), media_type="text/plain")


include_api_routers(app)

static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("Static files mounted from {}", static_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=_settings.api_host,
        port=_settings.api_port,
        reload=_settings.debug,
        log_level=_settings.log_level.lower(),
    )
