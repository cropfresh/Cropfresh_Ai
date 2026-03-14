"""
Browser RAG Integration Engine.
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

from loguru import logger

from ai.rag.browser_rag_pkg.extractor import ContentExtractor, QualityFilter
from ai.rag.browser_rag_pkg.models import Citation, CitedAnswer, ScrapeIntent, TargetSource
from ai.rag.browser_rag_pkg.sources import AgriSourceSelector


class BrowserRAGIntegration:
    """
    Live web retrieval for time-sensitive agricultural information (ADR-010).

    Extends the vector knowledge base with real-time scraped content for:
    - Latest government scheme updates
    - Crop disease/pest alerts
    - Pesticide bans and restrictions
    - Export/import policy changes
    - Live market news

    Integrates with Scrapling's Playwright-based fetching via the existing
    ScraplingBaseScraper circuit breaker for reliability.
    """

    def __init__(self, cache_ttl_hours: float = 6.0):
        self.source_selector = AgriSourceSelector()
        self.extractor = ContentExtractor()
        self.quality_filter = QualityFilter()
        self.default_ttl = cache_ttl_hours
        self._scraper = None  # Lazy loaded

        logger.info("BrowserRAGIntegration initialized")

    @property
    def scraper(self):
        """Lazy load the ScraplingBaseScraper."""
        if self._scraper is None:
            try:
                from src.scrapers.base_scraper import ScraplingBaseScraper
                self._scraper = ScraplingBaseScraper()
                logger.info("BrowserRAGIntegration: Scrapling scraper loaded")
            except ImportError:
                logger.warning(
                    "BrowserRAGIntegration: ScraplingBaseScraper not available — "
                    "falling back to httpx basic fetcher"
                )
        return self._scraper

    async def retrieve_live(
        self,
        query: str,
        max_sources: int = 3,
        intent: Optional[ScrapeIntent] = None,
    ) -> list[Any]:
        start = time.perf_counter()

        if intent is None:
            intent = self.source_selector.classify_intent(query)

        sources = self.source_selector.get_sources(intent, max_sources=max_sources)

        logger.info(
            f"BrowserRAGIntegration.retrieve_live | "
            f"intent={intent} | sources={len(sources)} | query={query[:60]}..."
        )

        results = await asyncio.gather(
            *[self._scrape_source(source, query) for source in sources],
            return_exceptions=True,
        )

        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"BrowserRAGIntegration: source {i} failed: {result}")
                continue
            if result is not None:
                documents.append(result)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"BrowserRAGIntegration.retrieve_live: "
            f"{len(documents)}/{len(sources)} sources succeeded | {elapsed_ms:.0f}ms"
        )

        return documents

    async def _scrape_source(
        self,
        source: TargetSource,
        query: str,
    ) -> Optional[Any]:
        try:
            html = await self._fetch_url(source)
            if not html:
                return None

            domain = self._extract_domain(source.url)

            text = self.extractor.extract_text(
                html=html,
                domain=domain,
                css_override=source.css_selector,
            )

            if not self.quality_filter.is_valid(text):
                logger.debug(
                    f"BrowserRAGIntegration: {source.source_name} failed quality filter "
                    f"(words={len(text.split())})"
                )
                return None

            scraped_at = datetime.now(timezone.utc).isoformat()
            doc = SimpleNamespace(
                text=text,
                id=f"browser_{source.source_name.lower().replace(' ', '_')}",
                score=0.85,
                metadata={
                    "source": "browser_scrape",
                    "source_name": source.source_name,
                    "source_url": source.url,
                    "intent": source.intent.value,
                    "scraped_at": scraped_at,
                    "ttl_hours": source.ttl_hours,
                    "is_live": True,
                },
            )

            logger.debug(
                f"BrowserRAGIntegration: scraped {source.source_name} | "
                f"words={len(text.split())}"
            )
            return doc

        except Exception as e:
            logger.warning(
                f"BrowserRAGIntegration._scrape_source: {source.source_name} failed: {e}"
            )
            return None

    async def _fetch_url(self, source: TargetSource) -> Optional[str]:
        try:
            if self.scraper is not None:
                if source.fetcher_type == "stealth":
                    return await self.scraper.fetch_stealth(source.url)
                else:
                    return await self.scraper.fetch(source.url)

            return await self._httpx_fetch(source.url)

        except Exception as e:
            logger.warning(f"BrowserRAGIntegration._fetch_url: {source.url} error: {e}")
            return None

    async def _httpx_fetch(self, url: str) -> Optional[str]:
        try:
            import httpx

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
            }

            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text

        except Exception as e:
            logger.warning(f"BrowserRAGIntegration._httpx_fetch: {url} failed: {e}")
            return None

    def _extract_domain(self, url: str) -> str:
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else "unknown"

    def build_cited_answer(
        self,
        answer: str,
        used_docs: list[Any],
    ) -> CitedAnswer:
        citations = []
        scraped_at_times = []

        for doc in used_docs:
            metadata = getattr(doc, "metadata", {})
            if not metadata.get("is_live", False):
                continue

            scraped_at = metadata.get("scraped_at", "")
            source_url = metadata.get("source_url", "")
            source_name = metadata.get("source_name", "Web")
            text = getattr(doc, "text", "")
            excerpt = text[:200].strip() + "..." if len(text) > 200 else text

            freshness = self._compute_freshness(scraped_at)

            citation = Citation(
                source_url=source_url,
                source_name=source_name,
                scraped_at=scraped_at,
                excerpt=excerpt,
                freshness_label=freshness,
            )
            citations.append(citation)

            if scraped_at:
                scraped_at_times.append(scraped_at)

        overall_freshness = "Live" if citations else "Cached"

        return CitedAnswer(
            answer=answer,
            citations=citations,
            has_live_data=bool(citations),
            freshness_label=overall_freshness,
        )

    def _compute_freshness(self, scraped_at_iso: str) -> str:
        try:
            scraped_time = datetime.fromisoformat(scraped_at_iso.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta_hours = (now - scraped_time).total_seconds() / 3600

            if delta_hours < 0.5:
                return "Live"
            elif delta_hours < 3:
                return f"{int(delta_hours * 60)} minutes ago"
            elif delta_hours < 24:
                return f"{int(delta_hours)} hours ago"
            else:
                return f"{int(delta_hours / 24)} days ago"
        except Exception:
            return "Recent"
