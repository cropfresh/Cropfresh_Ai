"""
Observability
=============
OpenTelemetry-based tracing and metrics.

Features:
- Distributed tracing
- Latency histograms
- Error rate metrics
- Token usage tracking
"""

import os
import functools
from datetime import datetime
from typing import Optional, Any, Callable
from contextlib import contextmanager

from loguru import logger
from pydantic import BaseModel, Field


# Check for Langsmith availability
try:
    from langsmith import traceable
    from langsmith.run_helpers import trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


class SpanContext(BaseModel):
    """Context for a trace span."""
    trace_id: str = ""
    span_id: str = ""
    name: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    attributes: dict = Field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None


class AgentMetrics(BaseModel):
    """Metrics for agent operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


# Global metrics storage
_agent_metrics: dict[str, AgentMetrics] = {}
_tracer = None
_meter = None


def setup_observability(
    service_name: str = "cropfresh-ai",
    endpoint: Optional[str] = None,
) -> bool:
    """
    Set up LangSmith tracing and metrics.
    
    Args:
        service_name: Name of this service
        endpoint: Langsmith endpoint (optional)
        
    Returns:
        True if setup successful
    """
    if "LANGCHAIN_API_KEY" not in os.environ:
        logger.warning("LANGCHAIN_API_KEY not set, LangSmith tracing may be disabled")
        return False
        
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = service_name
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        
    logger.info("LangSmith observability enabled for {}", service_name)
    return True


def trace_agent(agent_name: str):
    """
    Decorator to trace agent operations using LangSmith.
    
    Usage:
        @trace_agent("agronomy_agent")
        async def process_query(query: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        traced_func = traceable(name=f"{agent_name}.process", tags=["agent", agent_name])(func) if LANGSMITH_AVAILABLE else func

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            error = None
            result = None
            
            # Get or create metrics
            if agent_name not in _agent_metrics:
                _agent_metrics[agent_name] = AgentMetrics()
            metrics = _agent_metrics[agent_name]
            
            try:
                result = await traced_func(*args, **kwargs)
            except Exception as e:
                error = str(e)
                raise
            finally:
                # Update metrics
                latency = (datetime.now() - start_time).total_seconds() * 1000
                metrics.total_requests += 1
                metrics.total_latency_ms += latency
                
                if error:
                    metrics.failed_requests += 1
                else:
                    metrics.successful_requests += 1
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def trace_span(name: str, attributes: dict = None):
    """
    Context manager for creating LangSmith trace spans.
    
    Usage:
        with trace_span("database.query", {"table": "crops"}):
            result = await db.query(...)
    """
    if LANGSMITH_AVAILABLE:
        with trace(name=name, metadata=attributes or {}) as run:
            yield run
    else:
        yield SpanContext(name=name, attributes=attributes or {})


def record_tokens(agent_name: str, tokens_in: int, tokens_out: int):
    """Record token usage for an agent."""
    if agent_name not in _agent_metrics:
        _agent_metrics[agent_name] = AgentMetrics()
    
    _agent_metrics[agent_name].total_tokens_in += tokens_in
    _agent_metrics[agent_name].total_tokens_out += tokens_out


def get_metrics(agent_name: Optional[str] = None) -> dict:
    """Get metrics, optionally filtered by agent."""
    if agent_name:
        metrics = _agent_metrics.get(agent_name)
        return metrics.model_dump() if metrics else {}
    
    return {name: m.model_dump() for name, m in _agent_metrics.items()}


def get_all_metrics() -> dict:
    """Get aggregated metrics for all agents."""
    total = AgentMetrics()
    
    for metrics in _agent_metrics.values():
        total.total_requests += metrics.total_requests
        total.successful_requests += metrics.successful_requests
        total.failed_requests += metrics.failed_requests
        total.total_latency_ms += metrics.total_latency_ms
        total.total_tokens_in += metrics.total_tokens_in
        total.total_tokens_out += metrics.total_tokens_out
    
    return {
        "total": total.model_dump(),
        "by_agent": get_metrics(),
    }
