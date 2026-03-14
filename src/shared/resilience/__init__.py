"""
Resilience Module
=================
Multi-Agent Self-Healing patterns for robust operation.

Components:
- Reflection: Agent self-correction via output analysis
- Recovery: Exponential backoff and retry strategies
- Circuit Breaker: Fail-fast for degraded services
- Health Monitor: Agent metrics and alerting
- Task Decomposition: Sub-task graphs for complex queries
"""

from src.resilience.circuit_breaker import CircuitBreaker, CircuitState
from src.resilience.feedback import FeedbackLoop, get_feedback_loop
from src.resilience.health_monitor import AgentHealth, HealthMonitor
from src.resilience.recovery import ErrorRecovery, RetryPolicy
from src.resilience.reflection import ReflectionEngine, ReflectionResult
from src.resilience.task_decomposer import TaskDecomposer, TaskGraph

__all__ = [
    "ReflectionEngine",
    "ReflectionResult",
    "RetryPolicy",
    "ErrorRecovery",
    "CircuitBreaker",
    "CircuitState",
    "HealthMonitor",
    "AgentHealth",
    "TaskDecomposer",
    "TaskGraph",
    "FeedbackLoop",
    "get_feedback_loop",
]
