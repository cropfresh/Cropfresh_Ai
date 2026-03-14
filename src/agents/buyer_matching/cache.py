"""
Buyer Matching Cache
====================
Redis-backed and local in-memory caching for matching results.
"""

from datetime import datetime, timedelta
from typing import Optional, Any

from loguru import logger

from .models import MatchResult


class BuyerMatchingCacheMixin:
    """Mixin providing caching capabilities for the matching agent."""

    redis_url: Optional[str]
    cache_ttl_seconds: int
    _redis_client: Optional[Any] = None
    _local_cache: dict[str, tuple[datetime, MatchResult]]

    def _build_cache_key(self, listing_id: str, buyer_ids: list[str], suffix: str = "") -> str:
        buyers = ",".join(sorted(buyer_ids))
        return f"match:{listing_id}:{buyers}:{suffix}"

    async def _get_redis(self):
        if self._redis_client is None and self.redis_url:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis_client.ping()
            except Exception as err:
                logger.warning(f"Redis cache unavailable for buyer matching: {err}")
                self._redis_client = None
        return self._redis_client

    async def _cache_get(self, key: str) -> Optional[MatchResult]:
        redis = await self._get_redis()
        if redis:
            try:
                raw = await redis.get(key)
                if raw:
                    return MatchResult.model_validate_json(raw)
            except Exception as err:
                logger.debug(f"Redis cache read failed ({key}): {err}")

        cached = self._local_cache.get(key)
        if not cached:
            return None
        expiry, value = cached
        if expiry <= datetime.now():
            self._local_cache.pop(key, None)
            return None
        return value.model_copy(deep=True)

    async def _cache_set(self, key: str, value: MatchResult) -> None:
        redis = await self._get_redis()
        if redis:
            try:
                await redis.setex(key, self.cache_ttl_seconds, value.model_dump_json())
            except Exception as err:
                logger.debug(f"Redis cache write failed ({key}): {err}")
        expiry = datetime.now() + timedelta(seconds=self.cache_ttl_seconds)
        self._local_cache[key] = (expiry, value.model_copy(deep=True))
