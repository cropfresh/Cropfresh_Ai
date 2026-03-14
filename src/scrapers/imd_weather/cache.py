"""
IMD Weather Cache Manager
"""

from datetime import datetime
from typing import Any, Optional
from loguru import logger


class IMDCacheManager:
    """Simple in-memory cache for IMD weather data."""
    
    def __init__(self, ttl: int = 1800):
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
