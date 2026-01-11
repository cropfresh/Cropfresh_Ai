"""
Production Module
=================
Production-ready components for deployment.

Components:
- Observability: OpenTelemetry tracing/metrics
- Rate Limiter: Token bucket rate limiting
- Cache: Response caching layer
- Config: Production configuration
"""

from src.production.observability import setup_observability, trace_agent, AgentMetrics
from src.production.rate_limiter import RateLimiter, RateLimitExceeded
from src.production.cache import ResponseCache
from src.production.config import ProductionConfig, load_config

__all__ = [
    "setup_observability",
    "trace_agent",
    "AgentMetrics",
    "RateLimiter",
    "RateLimitExceeded",
    "ResponseCache",
    "ProductionConfig",
    "load_config",
]
