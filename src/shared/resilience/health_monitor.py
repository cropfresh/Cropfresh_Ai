"""
Health Monitor
==============
Agent health monitoring and metrics collection.

Features:
- Request/response latency tracking
- Error rate monitoring
- Success rate calculation
- Alerting thresholds
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any

from loguru import logger
from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AgentHealth(BaseModel):
    """Health status for an agent."""
    agent_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: datetime = Field(default_factory=datetime.now)
    
    # Metrics
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0
    
    # Recent history
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    uptime_percent: float = 100.0


class HealthConfig(BaseModel):
    """Health monitoring configuration."""
    window_size: int = 100  # Number of requests to track
    error_rate_threshold: float = 0.1  # 10% errors = degraded
    critical_error_rate: float = 0.3  # 30% errors = unhealthy
    latency_threshold_ms: float = 5000  # 5s = degraded
    check_interval_sec: float = 30.0


class HealthMonitor:
    """
    Monitors agent health and collects metrics.
    
    Usage:
        monitor = HealthMonitor()
        
        # Record requests
        monitor.record_request("agronomy_agent", latency_ms=150)
        monitor.record_error("commerce_agent", "API timeout")
        
        # Get health
        health = monitor.get_health("agronomy_agent")
        all_health = monitor.get_all_health()
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        """
        Initialize health monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or HealthConfig()
        
        # Per-agent tracking
        self._agents: dict[str, dict] = {}
        self._alerts: list[dict] = []
    
    def _get_agent_data(self, agent_name: str) -> dict:
        """Get or create agent tracking data."""
        if agent_name not in self._agents:
            self._agents[agent_name] = {
                "latencies": deque(maxlen=self.config.window_size),
                "successes": deque(maxlen=self.config.window_size),
                "errors": [],
                "request_count": 0,
                "success_count": 0,
                "error_count": 0,
                "last_success": None,
                "last_error": None,
                "created_at": datetime.now(),
            }
        return self._agents[agent_name]
    
    def record_request(
        self,
        agent_name: str,
        latency_ms: float,
        success: bool = True,
    ):
        """
        Record a successful request.
        
        Args:
            agent_name: Agent identifier
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
        """
        data = self._get_agent_data(agent_name)
        
        data["request_count"] += 1
        data["latencies"].append(latency_ms)
        data["successes"].append(success)
        
        if success:
            data["success_count"] += 1
            data["last_success"] = datetime.now()
        
        # Check for latency alerts
        if latency_ms > self.config.latency_threshold_ms:
            self._add_alert(agent_name, "high_latency", f"Latency: {latency_ms:.0f}ms")
    
    def record_error(
        self,
        agent_name: str,
        error: str,
        latency_ms: Optional[float] = None,
    ):
        """
        Record an error.
        
        Args:
            agent_name: Agent identifier
            error: Error message
            latency_ms: Optional request latency
        """
        data = self._get_agent_data(agent_name)
        
        data["request_count"] += 1
        data["error_count"] += 1
        data["successes"].append(False)
        data["errors"].append({
            "error": error,
            "timestamp": datetime.now(),
        })
        data["last_error"] = error
        
        if latency_ms:
            data["latencies"].append(latency_ms)
        
        # Check for error rate alerts
        error_rate = self._calculate_error_rate(data)
        if error_rate > self.config.critical_error_rate:
            self._add_alert(agent_name, "critical_errors", f"Error rate: {error_rate:.1%}")
        elif error_rate > self.config.error_rate_threshold:
            self._add_alert(agent_name, "high_errors", f"Error rate: {error_rate:.1%}")
    
    def get_health(self, agent_name: str) -> AgentHealth:
        """
        Get health status for an agent.
        
        Args:
            agent_name: Agent identifier
            
        Returns:
            AgentHealth with metrics
        """
        if agent_name not in self._agents:
            return AgentHealth(agent_name=agent_name)
        
        data = self._agents[agent_name]
        
        # Calculate metrics
        latencies = list(data["latencies"])
        error_rate = self._calculate_error_rate(data)
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = self._calculate_percentile(latencies, 95) if latencies else 0
        
        # Determine status
        status = self._determine_status(error_rate, avg_latency)
        
        # Calculate uptime
        total = data["request_count"]
        uptime = (data["success_count"] / total * 100) if total > 0 else 100
        
        return AgentHealth(
            agent_name=agent_name,
            status=status,
            request_count=data["request_count"],
            success_count=data["success_count"],
            error_count=data["error_count"],
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            error_rate=error_rate,
            last_error=data["last_error"],
            last_success=data["last_success"],
            uptime_percent=uptime,
        )
    
    def get_all_health(self) -> dict[str, AgentHealth]:
        """Get health for all agents."""
        return {name: self.get_health(name) for name in self._agents}
    
    def get_alerts(self, since: Optional[datetime] = None) -> list[dict]:
        """Get recent alerts."""
        if since:
            return [a for a in self._alerts if a["timestamp"] > since]
        return self._alerts[-50:]  # Last 50 alerts
    
    def _calculate_error_rate(self, data: dict) -> float:
        """Calculate error rate from recent requests."""
        successes = list(data["successes"])
        if not successes:
            return 0.0
        failures = sum(1 for s in successes if not s)
        return failures / len(successes)
    
    def _calculate_percentile(self, values: list, percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _determine_status(self, error_rate: float, avg_latency: float) -> HealthStatus:
        """Determine health status from metrics."""
        if error_rate > self.config.critical_error_rate:
            return HealthStatus.UNHEALTHY
        
        if error_rate > self.config.error_rate_threshold:
            return HealthStatus.DEGRADED
        
        if avg_latency > self.config.latency_threshold_ms:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def _add_alert(self, agent_name: str, alert_type: str, message: str):
        """Add an alert."""
        self._alerts.append({
            "agent": agent_name,
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
        })
        
        # Keep only recent alerts
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]
        
        logger.warning("Health alert [{}]: {} - {}", agent_name, alert_type, message)
    
    def reset(self, agent_name: Optional[str] = None):
        """Reset metrics."""
        if agent_name:
            self._agents.pop(agent_name, None)
        else:
            self._agents.clear()
            self._alerts.clear()
