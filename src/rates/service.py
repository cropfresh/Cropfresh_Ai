"""Shared multi-source rate service."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from src.db.models.price_records import RawPriceRecord
from src.rates.cache import RateCache
from src.rates.comparison import compare_records
from src.rates.connectors import PENDING_SOURCES, build_connectors
from src.rates.enums import AuthorityTier, ComparisonDepth, RateKind
from src.rates.models import MultiSourceRateResult, PendingSource, RateQuery, SourceHealthSnapshot
from src.rates.precedence import DEFAULT_TTLS_MINUTES
from src.rates.query_builder import build_rate_cache_key


class RateService:
    """Fan out to multiple connectors and aggregate responses."""

    def __init__(self, repository=None, redis_client=None, llm_provider=None, agmarknet_api_key: str = ""):
        self.repository = repository
        self.cache = RateCache(redis_client=redis_client)
        self.connectors = build_connectors(llm_provider=llm_provider, agmarknet_api_key=agmarknet_api_key)
        self.semaphore = asyncio.Semaphore(6)
        self._circuit_state: dict[str, tuple[int, datetime | None]] = {}
        self._health = {
            connector.source_id: SourceHealthSnapshot(
                source=connector.source_id,
                supports_live=connector.supports_live,
                fetch_mode=connector.fetch_mode,
            )
            for connector in self.connectors
        }

    async def query(self, query: RateQuery) -> MultiSourceRateResult:
        """Fetch and aggregate multi-source data for a query."""
        cache_key = build_rate_cache_key(query)
        if not query.force_live:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached
            if self.repository is not None:
                stored_records = await self.repository.get_rates(query)
                if stored_records:
                    return self._build_result(query, stored_records, warnings=["Using stored rates."])

        connectors = self._select_connectors(query)
        tasks = [self._run_connector(connector, query) for connector in connectors]
        results = await asyncio.gather(*tasks)
        records = [record for connector_records in results for record in connector_records]
        result = self._build_result(query, records)
        ttl_minutes = min((DEFAULT_TTLS_MINUTES.get(c.source_id, c.ttl_minutes) for c in connectors), default=60)
        await self.cache.set(cache_key, result, ttl_minutes)
        return result

    def get_source_health(self) -> list[SourceHealthSnapshot]:
        """Return source health ordered by source name."""
        return [self._health[name] for name in sorted(self._health)]

    def _select_connectors(self, query: RateQuery) -> list[Any]:
        """Select eligible connectors based on rate kind and comparison depth."""
        def allowed_tier(rate_kind: RateKind, tier: AuthorityTier) -> bool:
            if rate_kind in {RateKind.FUEL, RateKind.GOLD, RateKind.RETAIL_PRODUCE}:
                return True
            if query.comparison_depth is ComparisonDepth.OFFICIAL_ONLY:
                return tier in {AuthorityTier.OFFICIAL, AuthorityTier.REFERENCE_OFFICIAL}
            if query.comparison_depth is ComparisonDepth.OFFICIAL_PLUS_VALIDATORS:
                return tier in {
                    AuthorityTier.OFFICIAL,
                    AuthorityTier.REFERENCE_OFFICIAL,
                    AuthorityTier.VALIDATOR,
                }
            return True

        return [
            connector
            for connector in self.connectors
            if connector.rate_kind in query.rate_kinds and allowed_tier(connector.rate_kind, connector.authority_tier)
        ]

    async def _run_connector(self, connector, query: RateQuery):
        """Execute a single connector with timeout, retry, and circuit protection."""
        if not self._can_execute(connector.source_id):
            return []

        timeout_seconds = 25 if connector.uses_browser else 15
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with self.semaphore:
                    records = await asyncio.wait_for(connector.fetch(query), timeout=timeout_seconds)
                await self._persist_records(connector, records)
                self._record_success(connector.source_id)
                return records
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    continue
        self._record_failure(connector.source_id, str(last_error))
        return []

    async def _persist_records(self, connector, records) -> None:
        """Persist raw and normalized records when a repository exists."""
        if self.repository is None:
            return
        for record in records:
            raw_id = await self.repository.save_raw_record(
                RawPriceRecord(
                    source=connector.source_id,
                    raw_data=record.model_dump(mode="json"),
                    url=record.source_url,
                    scraped_at=record.fetched_at,
                )
            )
            record.raw_record_id = raw_id
            await self.repository.save_normalized_rate(record)

    def _can_execute(self, source: str) -> bool:
        """Return False when the circuit is currently open."""
        failures, open_until = self._circuit_state.get(source, (0, None))
        if open_until and open_until > datetime.utcnow():
            snapshot = self._health[source]
            snapshot.status = "unavailable"
            snapshot.circuit_open_until = open_until
            return False
        if open_until and open_until <= datetime.utcnow():
            self._circuit_state[source] = (failures, None)
        return True

    def _record_success(self, source: str) -> None:
        """Reset failure state after a successful fetch."""
        self._circuit_state[source] = (0, None)
        snapshot = self._health[source]
        snapshot.status = "healthy"
        snapshot.last_successful_fetch = datetime.utcnow()
        snapshot.last_error = None
        snapshot.consecutive_failures = 0
        snapshot.circuit_open_until = None

    def _record_failure(self, source: str, error: str) -> None:
        """Track connector failures and open the circuit after repeated issues."""
        failures, _ = self._circuit_state.get(source, (0, None))
        failures += 1
        open_until = datetime.utcnow() + timedelta(minutes=30) if failures >= 3 else None
        self._circuit_state[source] = (failures, open_until)
        snapshot = self._health[source]
        snapshot.status = "degraded" if failures < 3 else "unavailable"
        snapshot.last_error = error
        snapshot.consecutive_failures = failures
        snapshot.circuit_open_until = open_until

    def _pending_sources(self, query: RateQuery) -> list[PendingSource]:
        """Return relevant pending sources for the active rate kinds."""
        return [source for source in PENDING_SOURCES if source.rate_kind in query.rate_kinds]

    def _build_result(self, query: RateQuery, records, warnings: list[str] | None = None) -> MultiSourceRateResult:
        """Assemble the response contract from normalized records."""
        canonical_rates, comparison_quotes, compare_warnings = compare_records(records, query)
        return MultiSourceRateResult(
            query_target=query.model_dump(mode="json"),
            canonical_rates=canonical_rates,
            comparison_quotes=comparison_quotes,
            source_health=self.get_source_health(),
            warnings=list(dict.fromkeys((warnings or []) + compare_warnings)),
            pending_sources=self._pending_sources(query),
            fetched_at=datetime.utcnow(),
        )
