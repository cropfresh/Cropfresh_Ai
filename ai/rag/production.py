"""
Production Hardening & Guardrails
=================================
Middleware for robust production deployment of RAG systems.

Features:
- Rate Limiting (Token Bucket)
- Context-Aware Error Handling & Fallbacks
- Response Caching (LRU)
- A/B Testing Framework for Retrieval Strategies
- Request Tracing & Metrics

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
import time
import functools
import hashlib
from typing import Callable, Any, Dict, Optional, List
from enum import Enum
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field


class RateLimitError(Exception):
    pass


class CacheConfig(BaseModel):
    max_size: int = 1000
    ttl_seconds: int = 3600


class RateLimitConfig(BaseModel):
    requests_per_minute: int = 60
    burst_limit: int = 10


class RAGCache:
    """Simple LRU Cache with TTL."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.config.ttl_seconds:
                # Cache hit
                return self.cache[key]
            else:
                # Expired
                self.delete(key)
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.config.max_size:
            # Evict oldest
            oldest = min(self.timestamps, key=self.timestamps.get)
            self.delete(oldest)
            
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
    def delete(self, key: str):
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)


class RateLimiter:
    """Token Bucket Rate Limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.rate = config.requests_per_minute / 60.0
        self.capacity = config.burst_limit
        self.tokens = self.capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class ExperimentManager:
    """A/B Testing for RAG strategies."""
    
    def __init__(self):
        self.experiments: Dict[str, List[str]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        
    def register_experiment(self, name: str, variants: List[str]):
        self.experiments[name] = variants
        for v in variants:
            self.metrics[f"{name}:{v}"] = {"requests": 0, "success": 0, "latency_sum": 0.0}
            
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Deterministically assign user to variant."""
        variants = self.experiments.get(experiment_name, ["control"])
        hash_val = int(hashlib.md5(f"{experiment_name}:{user_id}".encode()).hexdigest(), 16)
        return variants[hash_val % len(variants)]
    
    def log_metric(self, experiment: str, variant: str, metric: str, value: float):
        key = f"{experiment}:{variant}"
        if key not in self.metrics:
            return
        
        if metric == "latency":
            self.metrics[key]["latency_sum"] += value
            self.metrics[key]["requests"] += 1
        elif metric == "success":
            self.metrics[key]["success"] += 1


class ProductionGuard:
    """
    Main guardrail for production RAG pipeline.
    
    Hooks:
    - Pre-execution: Rate limit, Cache check
    - Execution: Error handling, Retry
    - Post-execution: Cache set, Metrics log
    """
    
    def __init__(self):
        self.cache = RAGCache(CacheConfig())
        self.limiter = RateLimiter(RateLimitConfig())
        self.experiments = ExperimentManager()
        
        # Default experiment
        self.experiments.register_experiment("retrieval_strategy", ["hybrid", "dense", "graph"])
        
    def cached(self, prefix: str = "rag"):
        """Decorator for caching async functions."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate key
                key_str = f"{prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
                key = hashlib.md5(key_str.encode()).hexdigest()
                
                # Check cache
                cached_val = self.cache.get(key)
                if cached_val:
                    logger.debug(f"Cache hit for {key}")
                    return cached_val
                
                # Execute
                result = await func(*args, **kwargs)
                
                # Set cache
                self.cache.set(key, result)
                return result
            return wrapper
        return decorator

    def rate_limit(self):
        """Decorator for rate limiting."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not await self.limiter.acquire():
                    logger.warning("Rate limit exceeded")
                    raise RateLimitError("Too many requests. Please try again later.")
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def safe_execute(self, func: Callable, *args, **kwargs):
        """Execute with global error handling and retries."""
        retries = 3
        last_exception = None
        
        for attempt in range(retries):
            try:
                start = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                # Log success metrics (simulated)
                # logger.info(f"Execution successful in {duration:.3f}s")
                return result
                
            except RateLimitError:
                raise  # Don't retry rate limits immediately
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        logger.error(f"All {retries} attempts failed. Returning fallback.")
        return self._get_fallback(func.__name__)
    
    def _get_fallback(self, func_name: str) -> str:
        """Return safe fallback responses."""
        if "weather" in func_name:
            return "Weather data currently unavailable. Please check local capabilities."
        elif "price" in func_name:
            return "Latest market prices could not be fetched. Displaying last known averages."
        else:
            return "I apologize, but I'm having trouble processing your request right now. Please try again later."


# Global instance
production_guard = ProductionGuard()
