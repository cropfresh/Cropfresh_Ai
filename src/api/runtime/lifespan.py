"""FastAPI lifespan wiring for CropFresh AI."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger

from src.api.config import get_settings
from src.api.runtime.services import initialize_runtime_services, shutdown_runtime_services


@asynccontextmanager
async def lifespan(app: Any):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info(
        "CropFresh AI Service starting - env={} debug={}",
        settings.environment,
        settings.debug,
    )
    _configure_langsmith(settings)
    _setup_observability(settings)
    await _init_redis(app, settings)
    await _init_llm(app, settings)
    await _init_knowledge(app, settings)
    await _init_supervisor(app, settings)
    await initialize_runtime_services(app, settings)
    await _warm_vad()
    logger.info(
        "CropFresh AI Service ready - http://{}:{}/docs",
        settings.api_host,
        settings.api_port,
    )
    yield
    logger.info("CropFresh AI Service shutting down...")
    await shutdown_runtime_services(app)
    if getattr(app.state, "redis", None):
        await app.state.redis.aclose()


def _configure_langsmith(settings: Any) -> None:
    logger.info("Active event loop: {}", type(asyncio.get_running_loop()))
    if not settings.langsmith_api_key:
        return
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langsmith_tracing.lower())
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)


def _setup_observability(settings: Any) -> None:
    try:
        from src.production.observability import setup_observability

        setup_observability("cropfresh-ai", settings.otel_endpoint or None)
    except Exception as exc:
        logger.warning("Observability setup skipped: {}", exc)


async def _init_redis(app: Any, settings: Any) -> None:
    app.state.cache = None
    app.state.redis = None
    if not settings.use_redis_cache:
        return
    try:
        import redis.asyncio as aioredis  # type: ignore

        redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
        await redis_client.ping()
        app.state.redis = redis_client
    except Exception as exc:
        logger.warning("Redis unavailable ({}), falling back to in-memory cache", exc)


async def _init_llm(app: Any, settings: Any) -> None:
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
    except Exception as exc:
        logger.warning("LLM provider initialization failed: {}", exc)


async def _init_knowledge(app: Any, settings: Any) -> None:
    app.state.knowledge_agent = None
    try:
        from src.agents.knowledge_agent import KnowledgeAgent

        agent = KnowledgeAgent(
            llm=app.state.llm,
            qdrant_host=settings.qdrant_host,
            qdrant_port=settings.qdrant_port,
            qdrant_api_key=settings.qdrant_api_key,
        )
        if await agent.initialize():
            app.state.knowledge_agent = agent
    except Exception as exc:
        logger.warning("KnowledgeAgent initialization failed: {}", exc)


async def _init_supervisor(app: Any, settings: Any) -> None:
    app.state.supervisor = None
    app.state.state_manager = None
    try:
        from src.agents.agent_registry import create_agent_system

        kb = getattr(getattr(app.state, "knowledge_agent", None), "knowledge_base", None)
        supervisor, state_manager = await create_agent_system(
            llm=app.state.llm,
            knowledge_base=kb,
            redis_url=settings.redis_url if settings.use_redis_cache else None,
            settings=settings,
        )
        app.state.supervisor = supervisor
        app.state.state_manager = state_manager
    except Exception as exc:
        logger.warning("Agent system initialization failed: {}", exc)
        try:
            from src.agents.supervisor import SupervisorAgent

            supervisor = SupervisorAgent(llm=app.state.llm)
            await supervisor.initialize()
            app.state.supervisor = supervisor
        except Exception as fallback_exc:
            logger.warning("Fallback supervisor initialization failed: {}", fallback_exc)


async def _warm_vad() -> None:
    try:
        from src.voice.vad import SileroVAD

        vad = SileroVAD()
        await vad.initialize()
    except Exception as exc:
        logger.warning("Silero VAD unavailable: {}", exc)
