"""
Response Cache
==============
Caching layer for API responses.

Features:
- In-memory cache with TTL
- Cache key generation
- Cache invalidation
- Hit/miss metrics
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Cache configuration."""
    default_ttl_seconds: int = 300  # 5 minutes
    max_entries: int = 1000
    enable_metrics: bool = True


class CacheEntry(BaseModel):
    """A single cache entry."""
    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime
    hits: int = 0


class CacheStats(BaseModel):
    """Cache statistics."""
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0


class ResponseCache:
    """
    Response caching for agent outputs.
    
    Usage:
        cache = ResponseCache()
        
        # Check cache
        cached = await cache.get("tomato_prices_karnataka")
        if cached:
            return cached
        
        # Compute and cache
        result = await compute_expensive_result()
        await cache.set("tomato_prices_karnataka", result, ttl=300)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        
        if not entry:
            self._misses += 1
            return None
        
        # Check expiration
        if datetime.now() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            return None
        
        # Hit
        entry.hits += 1
        self._hits += 1
        
        return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        ttl = ttl or self.config.default_ttl_seconds
        
        # Evict if at capacity
        if len(self._cache) >= self.config.max_entries:
            self._evict_oldest()
        
        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=datetime.now() + timedelta(seconds=ttl),
        )
        
        self._cache[key] = entry
    
    async def delete(self, key: str) -> bool:
        """
        Delete from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching pattern.
        
        Args:
            pattern: Pattern to match (simple contains)
            
        Returns:
            Number of keys invalidated
        """
        to_delete = [k for k in self._cache if pattern in k]
        for key in to_delete:
            del self._cache[key]
        return len(to_delete)
    
    def _evict_oldest(self):
        """Evict oldest entries."""
        if not self._cache:
            return
        
        # Sort by creation time
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )
        
        # Remove oldest 10%
        evict_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:evict_count]:
            del self._cache[key]
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return CacheStats(
            total_entries=len(self._cache),
            total_hits=self._hits,
            total_misses=self._misses,
            hit_rate=hit_rate,
        )
    
    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(a) for a in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)
        
        return hashlib.md5(key_string.encode()).hexdigest()
