"""Factory for the shared multi-source rate service."""

from __future__ import annotations

from typing import Optional

from src.rates.repository import RateRepository
from src.rates.service import RateService

_rate_service: Optional[RateService] = None


async def get_rate_service(db_client=None, redis_client=None, llm_provider=None, agmarknet_api_key: str = "") -> RateService:
    """Get or create the shared RateService instance."""
    global _rate_service
    if _rate_service is None:
        repository = RateRepository(db_client) if db_client is not None else None
        if repository is not None:
            await repository.initialize_schema()
        _rate_service = RateService(
            repository=repository,
            redis_client=redis_client,
            llm_provider=llm_provider,
            agmarknet_api_key=agmarknet_api_key,
        )
        return _rate_service

    if _rate_service.repository is None and db_client is not None:
        repository = RateRepository(db_client)
        await repository.initialize_schema()
        _rate_service.repository = repository
    return _rate_service
