"""
CropFresh AI — Production-Grade Base Scraper
=============================================
Powered by Scrapling for adaptive parsing, anti-bot bypass,
and resilient web scraping with built-in retry/circuit-breaker.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

# Scrapling imports
from scrapling.fetchers import Fetcher, StealthyFetcher
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ============================================================================
# Enums & Models
# ============================================================================


class FetcherType(str, Enum):
    """Available Scrapling fetcher types."""
    BASIC = "basic"           # Fetcher — fast, no browser
    STEALTHY = "stealthy"     # StealthyFetcher — anti-bot bypass
    # PlayWrightFetcher available separately via scrapling[all]


class ScrapeResult(BaseModel):
    """Result from a scraping operation."""
    source: str
    url: str
    data: list[dict[str, Any]] = Field(default_factory=list)
    record_count: int = 0
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    from_cache: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.record_count > 0


class ScraperHealth(BaseModel):
    """Health status of a scraper."""
    name: str
    status: str = "unknown"  # healthy, degraded, unavailable
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    total_requests: int = 0
    successful_requests: int = 0
    avg_response_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


# ============================================================================
# Circuit Breaker (Simple built-in)
# ============================================================================


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, block requests
    HALF_OPEN = "half_open" # Testing recovery


class SimpleCircuitBreaker:
    """Lightweight circuit breaker for scrapers."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,  # seconds
        name: str = "scraper",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (
                time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                logger.info(f"⚡ Circuit breaker [{self.name}] → HALF_OPEN (testing recovery)")
                return True
            return False
        # HALF_OPEN — allow one attempt
        return True

    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"✅ Circuit breaker [{self.name}] → CLOSED (recovered)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"🔴 Circuit breaker [{self.name}] → OPEN "
                f"(after {self.failure_count} failures)"
            )


# ============================================================================
# Base Scraper (Production-Grade)
# ============================================================================


class ScraplingBaseScraper(ABC):
    """
    Production-grade base scraper powered by Scrapling.

    Features:
    - Adaptive parsing (survives website layout changes)
    - Anti-bot bypass via StealthyFetcher
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent hammering failed sources
    - Structured logging with scraper context
    - Response caching (in-memory with TTL)
    - Rate limiting (configurable per scraper)
    - Health tracking

    Usage:
        class MyScraper(ScraplingBaseScraper):
            name = "my_source"
            base_url = "https://example.com"
            fetcher_type = FetcherType.BASIC

            async def scrape(self, **kwargs) -> ScrapeResult:
                page = await self.fetch("https://example.com/data")
                items = page.css(".item::text").getall()
                return self.build_result(
                    url="https://example.com/data",
                    data=[{"item": i} for i in items],
                )
    """

    # Override in subclass
    name: str = "base"
    base_url: str = ""
    fetcher_type: FetcherType = FetcherType.BASIC

    # Configuration
    max_retries: int = 3
    cache_ttl_seconds: int = 300  # 5 minutes
    rate_limit_delay: float = 1.0  # seconds between requests
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 60.0

    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._circuit = SimpleCircuitBreaker(
            failure_threshold=self.circuit_breaker_threshold,
            recovery_timeout=self.circuit_breaker_recovery,
            name=self.name,
        )
        self._health = ScraperHealth(name=self.name)
        self._last_request_time: float = 0.0
        logger.info(f"🔧 Scraper [{self.name}] initialized (fetcher={self.fetcher_type.value})")

    # ── Fetching ──────────────────────────────────────────────────────────

    async def fetch(
        self,
        url: str,
        use_cache: bool = True,
        **kwargs,
    ) -> Any:
        """
        Fetch a URL using the configured Scrapling fetcher.

        Args:
            url: URL to fetch
            use_cache: Whether to use cached response
            **kwargs: Extra args passed to the fetcher

        Returns:
            Scrapling page/response object
        """
        # Check cache
        if use_cache:
            cached = self._get_cached(url)
            if cached is not None:
                logger.debug(f"📦 Cache HIT [{self.name}]: {url}")
                return cached

        # Check circuit breaker
        if not self._circuit.can_execute():
            raise ConnectionError(
                f"Circuit breaker OPEN for [{self.name}] — "
                f"source temporarily unavailable"
            )

        # Rate limiting
        await self._rate_limit()

        # Fetch with retry
        start_time = time.time()
        self._health.total_requests += 1

        try:
            page = await self._fetch_with_retry(url, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Track success
            self._circuit.record_success()
            self._health.successful_requests += 1
            self._health.last_success = datetime.utcnow()
            self._health.status = "healthy"
            self._update_avg_response(duration_ms)

            # Cache the response
            if use_cache:
                self._set_cached(url, page)

            logger.debug(
                f"✅ [{self.name}] fetched {url} "
                f"({duration_ms:.0f}ms, {self._health.success_rate:.0f}% success)"
            )
            return page

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._circuit.record_failure()
            self._health.last_error = str(e)
            self._health.status = "degraded"
            logger.error(f"❌ [{self.name}] failed to fetch {url}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def _fetch_with_retry(self, url: str, **kwargs) -> Any:
        """Fetch with automatic retry and exponential backoff."""
        if self.fetcher_type == FetcherType.STEALTHY:
            return await StealthyFetcher.async_fetch(url, **kwargs)
        else:
            # Basic Fetcher does not have async_fetch, we use fetch inside an executor or just fetch
            # Since Scrapling's Fetcher is built on httpx, we can use it directly
            return Fetcher(auto_match=False).get(url, **kwargs)

    # ── Rate Limiting ─────────────────────────────────────────────────────

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - elapsed
            logger.debug(f"⏳ [{self.name}] rate limiting: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        self._last_request_time = time.time()

    # ── Caching ───────────────────────────────────────────────────────────

    def _get_cached(self, key: str) -> Any:
        """Get a cached response if still valid."""
        if key in self._cache:
            value, cached_at = self._cache[key]
            if time.time() - cached_at < self.cache_ttl_seconds:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Cache a response with TTL."""
        self._cache[key] = (value, time.time())

    def clear_cache(self):
        """Clear all cached responses."""
        self._cache.clear()
        logger.info(f"🧹 [{self.name}] cache cleared")

    # ── Health Tracking ───────────────────────────────────────────────────

    def _update_avg_response(self, duration_ms: float):
        """Update rolling average response time."""
        n = self._health.successful_requests
        if n <= 1:
            self._health.avg_response_ms = duration_ms
        else:
            # Rolling average
            self._health.avg_response_ms = (
                self._health.avg_response_ms * (n - 1) + duration_ms
            ) / n

    def get_health(self) -> ScraperHealth:
        """Get current health status."""
        # Update status based on circuit breaker
        if self._circuit.state == CircuitState.OPEN:
            self._health.status = "unavailable"
        elif self._health.success_rate < 80:
            self._health.status = "degraded"
        return self._health

    # ── Result Builder ────────────────────────────────────────────────────

    def build_result(
        self,
        url: str,
        data: list[dict[str, Any]],
        from_cache: bool = False,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> ScrapeResult:
        """Build a standardized ScrapeResult."""
        return ScrapeResult(
            source=self.name,
            url=url,
            data=data,
            record_count=len(data),
            duration_ms=duration_ms,
            from_cache=from_cache,
            error=error,
        )

    # ── Abstract Methods ──────────────────────────────────────────────────

    @abstractmethod
    async def scrape(self, **kwargs) -> ScrapeResult:
        """Execute the scraping operation. Must be implemented by subclass."""
        ...

    async def close(self):
        """Clean up resources. Override if needed."""
        self.clear_cache()
        logger.info(f"🔧 Scraper [{self.name}] closed")
