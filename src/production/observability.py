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


# Check for OpenTelemetry availability
OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    pass


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
    Set up OpenTelemetry tracing and metrics.
    
    Args:
        service_name: Name of this service
        endpoint: OTLP endpoint (optional)
        
    Returns:
        True if setup successful
    """
    global _tracer, _meter
    
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not installed, using fallback metrics")
        return False
    
    try:
        resource = Resource.create({"service.name": service_name})
        
        # Set up tracer
        provider = TracerProvider(resource=resource)
        
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        
        # Set up meter
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(__name__)
        
        logger.info("OpenTelemetry observability enabled for {}", service_name)
        return True
        
    except Exception as e:
        logger.error("Failed to set up OpenTelemetry: {}", str(e))
        return False


def trace_agent(agent_name: str):
    """
    Decorator to trace agent operations.
    
    Usage:
        @trace_agent("agronomy_agent")
        async def process_query(query: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            error = None
            result = None
            
            # Get or create metrics
            if agent_name not in _agent_metrics:
                _agent_metrics[agent_name] = AgentMetrics()
            metrics = _agent_metrics[agent_name]
            
            # Create span
            span_context = None
            if _tracer:
                with _tracer.start_as_current_span(f"{agent_name}.process") as span:
                    span.set_attribute("agent.name", agent_name)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                    except Exception as e:
                        error = str(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, error))
                        raise
            else:
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    error = str(e)
                    raise
            
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
    Context manager for creating trace spans.
    
    Usage:
        with trace_span("database.query", {"table": "crops"}):
            result = await db.query(...)
    """
    start_time = datetime.now()
    
    if _tracer:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
                span.set_status(trace.Status(trace.StatusCode.OK))
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
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
