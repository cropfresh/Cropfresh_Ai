"""Cache helpers for the rate hub."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from src.rates.models import MultiSourceRateResult


class RateCache:
    """Simple Redis-backed or in-memory cache for rate responses."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._memory: dict[str, tuple[datetime, str]] = {}

    async def get(self, key: str) -> Optional[MultiSourceRateResult]:
        """Return a cached result if present and not expired."""
        if self.redis is not None:
            payload = await self.redis.get(key)
            if payload:
                return MultiSourceRateResult.model_validate_json(payload)

        cached = self._memory.get(key)
        if not cached:
            return None

        expires_at, payload = cached
        if expires_at < datetime.utcnow():
            self._memory.pop(key, None)
            return None
        return MultiSourceRateResult.model_validate_json(payload)

    async def set(self, key: str, value: MultiSourceRateResult, ttl_minutes: int) -> None:
        """Store a result in cache."""
        payload = value.model_dump_json()
        if self.redis is not None:
            await self.redis.set(key, payload, ex=ttl_minutes * 60)
            return
        self._memory[key] = (datetime.utcnow() + timedelta(minutes=ttl_minutes), payload)
