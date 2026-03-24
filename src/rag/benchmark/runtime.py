from __future__ import annotations

import os
from types import SimpleNamespace

from loguru import logger
from pydantic import ValidationError

from src.agents.knowledge_agent import KnowledgeAgent
from src.config.settings import get_settings
from src.orchestrator.llm_provider import create_llm_provider


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _llm_api_key(settings) -> str:
    if settings.llm_provider == "groq":
        return settings.groq_api_key
    if settings.llm_provider == "together":
        return settings.together_api_key
    return ""


def _load_benchmark_settings():
    try:
        return get_settings()
    except ValidationError as exc:
        logger.warning("Benchmark settings fallback activated: {}", exc)
        provider = os.getenv("LLM_PROVIDER", "bedrock")
        return SimpleNamespace(
            has_llm_configured=bool(
                provider == "bedrock"
                or (provider == "groq" and os.getenv("GROQ_API_KEY"))
                or (provider == "together" and os.getenv("TOGETHER_API_KEY"))
                or (provider == "vllm" and os.getenv("VLLM_BASE_URL"))
            ),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            aws_region=os.getenv("AWS_REGION", "ap-south-1"),
            aws_profile=os.getenv("AWS_PROFILE", ""),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        )


def _benchmark_llm_enabled(settings) -> bool:
    if _env_flag("RAG_BENCHMARK_DISABLE_LLM"):
        logger.info("Benchmark runtime: LLM disabled via RAG_BENCHMARK_DISABLE_LLM")
        return False
    if not settings.has_llm_configured:
        return False
    if settings.llm_provider == "bedrock" and not _env_flag("RAG_BENCHMARK_ALLOW_BEDROCK"):
        logger.warning(
            "Benchmark runtime: disabling Bedrock LLM by default. "
            "Set RAG_BENCHMARK_ALLOW_BEDROCK=true to override.",
        )
        return False
    return True


def build_benchmark_agent() -> KnowledgeAgent:
    """Create the canonical runtime agent used by live benchmarks."""
    settings = _load_benchmark_settings()
    llm = None
    if _benchmark_llm_enabled(settings):
        llm = create_llm_provider(
            provider=settings.llm_provider,
            api_key=_llm_api_key(settings),
            base_url=settings.vllm_base_url,
            model=settings.llm_model,
            region=settings.aws_region,
            aws_profile=settings.aws_profile,
        )
    return KnowledgeAgent(
        llm=llm,
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        qdrant_api_key=settings.qdrant_api_key,
    )
