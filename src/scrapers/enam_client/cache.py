"""
eNAM Cache Manager
"""

from datetime import datetime
from typing import Any, Optional
from loguru import logger


class ENAMCacheManager:
    """Simple in-memory cache for eNAM data."""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._cache: dict[str, tuple[datetime, Any]] = {}
        
    def get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments."""
        return f"{prefix}:" + ":".join(str(a).lower() for a in args)

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if still valid."""
        if key in self._cache:
            cached_time, data = self._cache[key]
            if (datetime.now() - cached_time).seconds < self.ttl:
                logger.debug(f"Cache hit for {key}")
                return data
        return None

    def set(self, key: str, data: Any):
        """Store data in cache."""
        self._cache[key] = (datetime.now(), data)
        
    def get_freshness_stats(self) -> dict[str, Any]:
        """Return cache status and age for all entries."""
        now = datetime.now()
        cache_status = []
        
        for key, (cached_time, _) in self._cache.items():
            age_seconds = (now - cached_time).seconds
            cache_status.append({
                "key": key,
                "age_seconds": age_seconds,
                "fresh": age_seconds < self.ttl,
            })
            
        return {
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self.ttl,
            "entries": cache_status,
        }
