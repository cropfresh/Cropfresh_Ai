"""
Rate Limiter
============
Token bucket rate limiting for API protection.

Features:
- Per-user rate limits
- Token bucket algorithm
- Burst allowance
- Quota management
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # Allow bursts
    token_refill_rate: float = 1.0  # Tokens per second


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: float = 60,
        limit_type: str = "global",
    ):
        super().__init__(message)
        self.retry_after = retry_after_seconds
        self.limit_type = limit_type


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(
        self,
        capacity: float,
        refill_rate: float,
    ):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.now()
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens consumed, False if insufficient
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_refill = now
    
    @property
    def available(self) -> float:
        """Available tokens."""
        self._refill()
        return self.tokens


class RateLimiter:
    """
    Rate limiter with per-user limits.
    
    Usage:
        limiter = RateLimiter()
        
        # Check limit
        try:
            await limiter.check("user_123")
            # Process request
        except RateLimitExceeded as e:
            # Return 429 with retry-after
            pass
    """
    
    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        
        # Per-user buckets
        self._buckets: dict[str, TokenBucket] = {}
        
        # Global counters
        self._minute_counts: dict[str, list] = {}
        self._hour_counts: dict[str, list] = {}
    
    async def check(
        self,
        user_id: str,
        tokens: float = 1.0,
    ) -> bool:
        """
        Check and consume rate limit.
        
        Args:
            user_id: User identifier
            tokens: Tokens to consume
            
        Returns:
            True if allowed
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        # Get or create bucket
        bucket = self._get_bucket(user_id)
        
        # Try token bucket first (burst protection)
        if not bucket.consume(tokens):
            raise RateLimitExceeded(
                f"Rate limit exceeded for {user_id}",
                retry_after_seconds=1.0 / self.config.token_refill_rate,
                limit_type="burst",
            )
        
        # Check minute limit
        now = datetime.now()
        await self._check_window(
            user_id,
            self._minute_counts,
            timedelta(minutes=1),
            self.config.requests_per_minute,
            "minute",
        )
        
        # Check hour limit
        await self._check_window(
            user_id,
            self._hour_counts,
            timedelta(hours=1),
            self.config.requests_per_hour,
            "hour",
        )
        
        return True
    
    def _get_bucket(self, user_id: str) -> TokenBucket:
        """Get or create token bucket for user."""
        if user_id not in self._buckets:
            self._buckets[user_id] = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=self.config.token_refill_rate,
            )
        return self._buckets[user_id]
    
    async def _check_window(
        self,
        user_id: str,
        counts: dict,
        window: timedelta,
        limit: int,
        limit_type: str,
    ):
        """Check sliding window rate limit."""
        now = datetime.now()
        cutoff = now - window
        
        # Get timestamps for this user
        if user_id not in counts:
            counts[user_id] = []
        
        # Remove old timestamps
        counts[user_id] = [t for t in counts[user_id] if t > cutoff]
        
        # Check limit
        if len(counts[user_id]) >= limit:
            oldest = min(counts[user_id]) if counts[user_id] else now
            retry_after = (oldest + window - now).total_seconds()
            
            raise RateLimitExceeded(
                f"{limit_type.capitalize()} limit ({limit}) exceeded",
                retry_after_seconds=max(1, retry_after),
                limit_type=limit_type,
            )
        
        # Add current request
        counts[user_id].append(now)
    
    def get_remaining(self, user_id: str) -> dict:
        """Get remaining limits for user."""
        now = datetime.now()
        
        # Minute remaining
        minute_used = len([
            t for t in self._minute_counts.get(user_id, [])
            if t > now - timedelta(minutes=1)
        ])
        
        # Hour remaining
        hour_used = len([
            t for t in self._hour_counts.get(user_id, [])
            if t > now - timedelta(hours=1)
        ])
        
        # Burst remaining
        bucket = self._buckets.get(user_id)
        burst_available = bucket.available if bucket else self.config.burst_size
        
        return {
            "minute": {
                "limit": self.config.requests_per_minute,
                "remaining": self.config.requests_per_minute - minute_used,
            },
            "hour": {
                "limit": self.config.requests_per_hour,
                "remaining": self.config.requests_per_hour - hour_used,
            },
            "burst": {
                "limit": self.config.burst_size,
                "available": burst_available,
            },
        }
    
    def reset(self, user_id: Optional[str] = None):
        """Reset limits for a user or all users."""
        if user_id:
            self._buckets.pop(user_id, None)
            self._minute_counts.pop(user_id, None)
            self._hour_counts.pop(user_id, None)
        else:
            self._buckets.clear()
            self._minute_counts.clear()
            self._hour_counts.clear()
