"""
Circuit Breaker
===============
Fail-fast pattern for degraded services.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing state, requests fail immediately
- HALF_OPEN: Testing if service recovered
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any

from loguru import logger
from pydantic import BaseModel, Field


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_sec: float = 30.0   # Time before testing recovery
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


class CircuitStats(BaseModel):
    """Circuit breaker statistics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_at: datetime = Field(default_factory=datetime.now)
    total_requests: int = 0
    total_failures: int = 0
    total_rejections: int = 0


class CircuitOpenError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreaker:
    """
    Implements circuit breaker pattern.
    
    Usage:
        breaker = CircuitBreaker("external_api")
        
        try:
            result = await breaker.call(my_api_function, arg1, arg2)
        except CircuitOpenError:
            # Use fallback
            result = fallback_value
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit
            config: Circuit configuration
            on_state_change: Callback when state changes
        """
        self.name = name
        self.config = config or CircuitConfig()
        self.on_state_change = on_state_change
        
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._half_open_calls = 0
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._stats.state
    
    @property
    def stats(self) -> CircuitStats:
        """Get current statistics."""
        return self._stats
    
    async def call(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
        """
        self._stats.total_requests += 1
        
        # Check circuit state
        async with self._lock:
            self._check_state_transition()
            
            if self._stats.state == CircuitState.OPEN:
                self._stats.total_rejections += 1
                raise CircuitOpenError(f"Circuit '{self.name}' is open")
            
            if self._stats.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit '{self.name}' is half-open (max calls reached)")
                self._half_open_calls += 1
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self._on_success()
            return result
            
        except Exception as e:
            # Record failure
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self._stats.success_count += 1
            self._stats.last_success_time = datetime.now()
            
            if self._stats.state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                
                if self._stats.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            
            elif self._stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._stats.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
        """Handle failed call."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.last_failure_time = datetime.now()
            
            if self._stats.state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                # Any failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)
            
            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _check_state_transition(self):
        """Check if state should transition based on timeout."""
        if self._stats.state == CircuitState.OPEN:
            time_since_open = datetime.now() - self._stats.state_changed_at
            if time_since_open.total_seconds() >= self.config.timeout_sec:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self._stats.state
        
        if old_state == new_state:
            return
        
        self._stats.state = new_state
        self._stats.state_changed_at = datetime.now()
        
        # Reset counters
        if new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._stats.success_count = 0
            self._half_open_calls = 0
        
        logger.info(
            "Circuit '{}' transitioned: {} -> {}",
            self.name, old_state.value, new_state.value
        )
        
        if self.on_state_change:
            self.on_state_change(self.name, old_state, new_state)
    
    def reset(self):
        """Reset circuit to closed state."""
        self._stats = CircuitStats()
        self._half_open_calls = 0
        logger.info("Circuit '{}' reset", self.name)
    
    def force_open(self):
        """Force circuit to open state."""
        self._transition_to(CircuitState.OPEN)
    
    def force_close(self):
        """Force circuit to closed state."""
        self._transition_to(CircuitState.CLOSED)


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Usage:
        registry = CircuitBreakerRegistry()
        
        # Get or create circuit for a service
        breaker = registry.get("enam_api")
        
        # Check health of all circuits
        health = registry.get_health()
    """
    
    def __init__(self, default_config: Optional[CircuitConfig] = None):
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitConfig()
    
    def get(self, name: str, config: Optional[CircuitConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
            )
        return self._breakers[name]
    
    def get_health(self) -> dict[str, dict]:
        """Get health status of all circuits."""
        return {
            name: {
                "state": breaker.state.value,
                "failure_count": breaker.stats.failure_count,
                "total_failures": breaker.stats.total_failures,
                "total_rejections": breaker.stats.total_rejections,
            }
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
