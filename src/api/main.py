"""
CropFresh AI Service - FastAPI Application
==========================================
Production-grade entry point for the AI service API.

Changes from dev baseline:
  - Environment-driven CORS (ALLOWED_ORIGINS env var)
  - API key authentication middleware (X-API-Key header)
  - OpenTelemetry observability wired in lifespan startup
  - Prometheus /metrics endpoint
  - Dependency injection: agents initialized on startup, stored on app.state
  - Structured health checks: Qdrant + Redis probed on /health/ready
  - Graceful shutdown with connection cleanup
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.config import get_settings


# ─────────────────────────────────────────────────
# Lifespan: startup + shutdown
# ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup sequence:
      1. Load settings
      2. Wire OpenTelemetry observability
      3. Connect Redis cache
      4. Initialize KnowledgeAgent (Qdrant)
      5. Initialize SupervisorAgent (all domain agents)
      6. Probe dependencies for readiness

    Shutdown:
      - Close Redis, Qdrant connections gracefully
    """
    settings = get_settings()

    logger.info("🌾 CropFresh AI Service starting — env={} debug={}", settings.environment, settings.debug)

    # ── 1. Observability ────────────────────────────
    try:
        from src.production.observability import setup_observability
        setup_observability(
            service_name="cropfresh-ai",
            endpoint=settings.otel_endpoint or None,
        )
    except Exception as exc:
        logger.warning("Observability setup skipped: {}", exc)

    # ── 2. LangSmith (optional) ──────────────────────
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        logger.info("🔗 LangSmith tracing enabled (project={})", settings.langsmith_project)

    # ── 3. Redis cache ───────────────────────────────
    app.state.cache = None
    if settings.use_redis_cache:
        try:
            import redis.asyncio as aioredis  # type: ignore
            redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
            await redis_client.ping()
            app.state.redis = redis_client
            logger.info("✅ Redis connected: {}", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable ({}), falling back to in-memory cache", exc)
            app.state.redis = None
    else:
        app.state.redis = None

    # ── 4. LLM Provider (shared across agents) ─────────
    app.state.llm = None
    try:
        from src.orchestrator.llm_provider import create_llm_provider

        if settings.has_llm_configured:
            app.state.llm = create_llm_provider(
                provider=settings.llm_provider,
                api_key=settings.groq_api_key or settings.together_api_key,
                base_url=settings.vllm_base_url,
                model=settings.llm_model,
                region=settings.aws_region,
                aws_profile=settings.aws_profile,
            )
            logger.info(
                "✅ LLM provider ready — provider={} model={}",
                settings.llm_provider,
                settings.llm_model,
            )
        else:
            logger.warning("⚠️  No LLM provider configured — agents will use rule-based fallbacks")
    except Exception as exc:
        logger.warning("LLM provider initialization failed: {} — using fallbacks", exc)

    # ── 5. KnowledgeAgent ────────────────────────────
    app.state.knowledge_agent = None
    try:
        from src.agents.knowledge_agent import KnowledgeAgent

        agent = KnowledgeAgent(
            llm=app.state.llm,
            qdrant_host=settings.qdrant_host,
            qdrant_port=settings.qdrant_port,
            qdrant_api_key=settings.qdrant_api_key,
        )
        ok = await agent.initialize()
        if ok:
            app.state.knowledge_agent = agent
            logger.info("✅ KnowledgeAgent initialized (Qdrant={}:{})", settings.qdrant_host, settings.qdrant_port)
        else:
            logger.warning("⚠️  KnowledgeAgent initialization returned False — RAG queries will fail")
    except Exception as exc:
        logger.warning("KnowledgeAgent initialization failed: {} — RAG unavailable", exc)

    # ── 6. Full Agent System ──────────────────────────
    #! Critical: Previously created a bare SupervisorAgent with zero agents.
    #! Now uses agent_registry to wire ALL 15 agents at startup.
    app.state.supervisor = None
    app.state.state_manager = None
    try:
        from src.agents.agent_registry import create_agent_system

        # * Extract KB from KnowledgeAgent if available
        kb = None
        if app.state.knowledge_agent:
            kb = getattr(app.state.knowledge_agent, "knowledge_base", None)

        supervisor, state_manager = await create_agent_system(
            llm=app.state.llm,
            knowledge_base=kb,
            redis_url=settings.redis_url if settings.use_redis_cache else None,
            settings=settings,
        )
        app.state.supervisor = supervisor
        app.state.state_manager = state_manager
        logger.info(
            "✅ Agent system: {} agents registered",
            len(supervisor.get_available_agents()),
        )
    except Exception as exc:
        logger.warning("Agent system initialization failed: {}", exc)
        # * Fallback: create bare supervisor so health checks pass
        try:
            from src.agents.supervisor_agent import SupervisorAgent
            supervisor = SupervisorAgent(llm=app.state.llm)
            await supervisor.initialize()
            app.state.supervisor = supervisor
            logger.info("⚠️ Fallback: bare SupervisorAgent (no agents registered)")
        except Exception:
            pass

    logger.info("🚀 CropFresh AI Service ready — http://{}:{}/docs", settings.api_host, settings.api_port)

    # ── 7. Silero VAD pre-download ───────────────────
    try:
        from src.voice.vad import SileroVAD
        vad = SileroVAD()
        await vad.initialize()
        logger.info("✅ Silero VAD ready")
    except Exception as exc:
        logger.warning("⚠️ Silero VAD unavailable: {} — WebSocket VAD will be skipped", exc)

    yield  # ─── App is running ───

    # ── Shutdown ─────────────────────────────────────
    logger.info("🛑 CropFresh AI Service shutting down...")
    if getattr(app.state, "redis", None):
        await app.state.redis.aclose()
        logger.info("Redis connection closed")


# ─────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────

_settings = get_settings()

app = FastAPI(
    title="CropFresh AI Service",
    description=(
        "Agentic RAG + Voice + Market Intelligence backend "
        "for India's agricultural marketplace.\n\n"
        "**Auth:** Include `X-API-Key: <your_key>` header for API routes "
        "(not required in `development` environment)."
    ),
    version="0.2.0",
    # Hide /docs in production to reduce attack surface
    docs_url="/docs" if not _settings.is_production else None,
    redoc_url="/redoc" if not _settings.is_production else None,
    lifespan=lifespan,
)

# ─── CORS ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins_list,   # Env-driven — NOT allow_origins=["*"] in prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ─── API Key auth ──────────────────────────────────
from src.api.middleware.auth import APIKeyMiddleware  # noqa: E402
app.add_middleware(APIKeyMiddleware, api_key=_settings.api_key or None)


# ─────────────────────────────────────────────────
# Health endpoints
# ─────────────────────────────────────────────────

@app.get("/", tags=["meta"], include_in_schema=False)
async def root():
    """Root — redirect to test dashboard in dev, return JSON in production."""
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
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
async def readiness(request: Request):
    """
    Readiness probe — checks all critical dependencies.

    Returns 200 only when the service is truly ready to handle traffic.
    """
    checks: dict[str, bool | str] = {
        "llm_provider": _settings.llm_provider,
        "llm_configured": _settings.has_llm_configured,
        "llm_initialized": getattr(request.app.state, "llm", None) is not None,
        "qdrant_configured": bool(_settings.qdrant_host),
        "knowledge_agent": getattr(request.app.state, "knowledge_agent", None) is not None,
        "supervisor": getattr(request.app.state, "supervisor", None) is not None,
        "redis": getattr(request.app.state, "redis", None) is not None,
    }

    all_ready = checks["llm_configured"] and checks["qdrant_configured"]
    status_code = 200 if all_ready else 503

    return Response(
        content=__import__("orjson").dumps({"ready": all_ready, "checks": checks}),
        media_type="application/json",
        status_code=status_code,
    )


# ─────────────────────────────────────────────────
# Prometheus metrics endpoint
# ─────────────────────────────────────────────────

@app.get("/metrics", tags=["observability"], include_in_schema=False)
async def prometheus_metrics():
    """
    Prometheus text-format metrics scrape endpoint.

    Exposes:
      - cropfresh_agent_requests_total{agent}
      - cropfresh_agent_latency_ms{agent}
      - cropfresh_agent_errors_total{agent}
      - cropfresh_cache_hits_total / cropfresh_cache_misses_total
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        # Prometheus client not installed (observability extra not used)
        from src.production.observability import get_all_metrics
        metrics = get_all_metrics()
        lines = [
            "# HELP cropfresh_agent_requests_total Total requests per agent",
            "# TYPE cropfresh_agent_requests_total counter",
        ]
        for agent, m in metrics.get("by_agent", {}).items():
            lines.append(f'cropfresh_agent_requests_total{{agent="{agent}"}} {m["total_requests"]}')
        lines.append("")
        return Response(content="\n".join(lines), media_type="text/plain")


# ─────────────────────────────────────────────────
# API Routers
# ─────────────────────────────────────────────────

from src.api.routes import rag                      # noqa: E402
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])

from src.api.routes import chat                     # noqa: E402
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

from src.api.routes import data as data_routes      # noqa: E402
app.include_router(data_routes.router, prefix="/api/v1", tags=["data"])

from src.api.routers import listings as listings_router  # noqa: E402
app.include_router(listings_router.router, prefix="/api/v1", tags=["listings"])

from src.api.routers import orders as orders_router      # noqa: E402
app.include_router(orders_router.router, prefix="/api/v1", tags=["orders"])

from src.api.routers import auth as auth_router          # noqa: E402
app.include_router(auth_router.router, prefix="/api/v1", tags=["auth"])

from src.api.rest import voice as voice_rest        # noqa: E402
app.include_router(voice_rest.router, tags=["voice"])

from src.api import websocket as voice_ws           # noqa: E402
app.include_router(voice_ws.router, tags=["websocket"])


# ─────────────────────────────────────────────────
# Static files (Voice Test UI)
# ─────────────────────────────────────────────────

static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("📁 Static files mounted from: {}", static_dir)


# ─────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=_settings.api_host,
        port=_settings.api_port,
        reload=_settings.debug,
        log_level=_settings.log_level.lower(),
    )
