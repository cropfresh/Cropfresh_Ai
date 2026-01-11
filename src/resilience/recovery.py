"""
Error Recovery
==============
Retry strategies with exponential backoff for resilient operations.

Features:
- Configurable retry policies
- Exponential backoff with jitter
- Exception filtering
- Recovery callbacks
"""

import asyncio
import random
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any, Type

from loguru import logger
from pydantic import BaseModel, Field


class BackoffStrategy(str, Enum):
    """Backoff strategies."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_sec: float = 1.0
    max_delay_sec: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.3  # ±30%
    retryable_exceptions: list[str] = Field(default_factory=lambda: [
        "TimeoutError", "ConnectionError", "HTTPError", "RateLimitError"
    ])
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.initial_delay_sec
            
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay_sec * (attempt + 1)
            
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay_sec * (self.backoff_multiplier ** attempt)
            
        else:  # EXPONENTIAL_JITTER
            base_delay = self.initial_delay_sec * (self.backoff_multiplier ** attempt)
            jitter = random.uniform(-self.jitter_range, self.jitter_range)
            delay = base_delay * (1 + jitter)
        
        return min(delay, self.max_delay_sec)
    
    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        exception_name = type(exception).__name__
        return any(
            name.lower() in exception_name.lower()
            for name in self.retryable_exceptions
        )


class RecoveryResult(BaseModel):
    """Result of recovery attempt."""
    success: bool = False  # Default to False, updated on success
    result: Optional[Any] = None
    attempts: int = 0
    total_delay_sec: float = 0.0
    final_error: Optional[str] = None
    errors: list[str] = Field(default_factory=list)


class ErrorRecovery:
    """
    Handles error recovery with configurable retry strategies.
    
    Usage:
        recovery = ErrorRecovery()
        
        result = await recovery.execute_with_retry(
            async_func=my_api_call,
            args=("param1",),
            policy=RetryPolicy(max_retries=5)
        )
        
        if result.success:
            print(result.result)
        else:
            print(f"Failed after {result.attempts} attempts: {result.final_error}")
    """
    
    def __init__(
        self,
        default_policy: Optional[RetryPolicy] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        on_success: Optional[Callable[[Any, int], None]] = None,
        on_failure: Optional[Callable[[Exception, int], None]] = None,
    ):
        """
        Initialize error recovery.
        
        Args:
            default_policy: Default retry policy
            on_retry: Callback when retrying (attempt_num, exception)
            on_success: Callback on success (result, attempts)
            on_failure: Callback on final failure (exception, attempts)
        """
        self.default_policy = default_policy or RetryPolicy()
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure
    
    async def execute_with_retry(
        self,
        async_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        policy: Optional[RetryPolicy] = None,
    ) -> RecoveryResult:
        """
        Execute a function with retry on failure.
        
        Args:
            async_func: Async function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            policy: Retry policy (uses default if not provided)
            
        Returns:
            RecoveryResult with success status and result
        """
        kwargs = kwargs or {}
        policy = policy or self.default_policy
        
        result = RecoveryResult()
        last_exception = None
        
        for attempt in range(policy.max_retries + 1):
            result.attempts = attempt + 1
            
            try:
                # Execute function
                output = await async_func(*args, **kwargs)
                
                # Success
                result.success = True
                result.result = output
                
                if self.on_success:
                    self.on_success(output, result.attempts)
                
                logger.debug("Recovery succeeded on attempt {}", result.attempts)
                return result
                
            except Exception as e:
                last_exception = e
                error_msg = f"{type(e).__name__}: {str(e)}"
                result.errors.append(error_msg)
                
                # Check if we should retry
                if attempt >= policy.max_retries:
                    break
                
                if not policy.is_retryable(e):
                    logger.warning("Non-retryable exception: {}", error_msg)
                    break
                
                # Calculate delay
                delay = policy.get_delay(attempt)
                result.total_delay_sec += delay
                
                logger.warning(
                    "Attempt {} failed: {}. Retrying in {:.1f}s...",
                    attempt + 1, error_msg, delay
                )
                
                if self.on_retry:
                    self.on_retry(attempt + 1, e)
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        result.success = False
        result.final_error = str(last_exception) if last_exception else "Unknown error"
        
        if self.on_failure and last_exception:
            self.on_failure(last_exception, result.attempts)
        
        logger.error(
            "Recovery failed after {} attempts: {}",
            result.attempts, result.final_error
        )
        
        return result
    
    def with_fallback(
        self,
        fallback_value: Any,
    ) -> Callable:
        """
        Create a decorator that returns fallback on failure.
        
        Args:
            fallback_value: Value to return if all retries fail
            
        Returns:
            Decorator function
        """
        def decorator(async_func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                result = await self.execute_with_retry(
                    async_func, args, kwargs
                )
                return result.result if result.success else fallback_value
            return wrapper
        return decorator


# Convenience function
async def retry_async(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    *args,
    **kwargs,
) -> Any:
    """
    Simple retry wrapper for async functions.
    
    Usage:
        result = await retry_async(my_api_call, max_retries=5, arg1, arg2)
    """
    recovery = ErrorRecovery(
        default_policy=RetryPolicy(
            max_retries=max_retries,
            initial_delay_sec=initial_delay,
        )
    )
    
    result = await recovery.execute_with_retry(func, args, kwargs)
    
    if not result.success:
        raise Exception(result.final_error)
    
    return result.result
