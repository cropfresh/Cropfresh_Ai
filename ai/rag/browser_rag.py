"""
Browser-Augmented RAG Integration
===================================
Live web retrieval layer for time-sensitive agricultural information (ADR-010).

Augments the vector knowledge base with real-time web data for:
- Government scheme updates (agriculture.gov.in, farmer.gov.in)
- Disease and pest alerts (agrifarming.in, ICAR NBSS)
- Market news (commodityindia.com, agriwatch.com)
- Export/import policy changes
- Pesticide bans and weather advisories

Uses the existing Scrapling infrastructure (Playwright + Camoufox StealthyFetcher)
from src/scrapers/base_scraper.py for anti-bot evasion.

Architecture: docs/architecture/browser_scraping_rag.md
ADR: docs/decisions/ADR-010-browser-scraping-rag.md
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from types import SimpleNamespace

from loguru import logger
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Models
# ─────────────────────────────────────────────────────────────────────────────

class ScrapeIntent(str, Enum):
    """
    Intent category for browser scraping — determines which sources to target.
    """
    SCHEME_UPDATE    = "scheme_update"     # New government schemes
    DISEASE_ALERT    = "disease_alert"     # Crop disease/pest outbreaks
    PRICE_NEWS       = "price_news"        # Commodity price news
    PESTICIDE_BAN    = "pesticide_ban"     # Banned/restricted chemicals
    EXPORT_POLICY    = "export_policy"     # Export/import restrictions
    WEATHER_ADVISORY = "weather_advisory"  # Extreme weather advisories
    MARKET_NEWS      = "market_news"       # General agri-market news


class TargetSource(BaseModel):
    """A single target URL with scraping metadata."""
    url: str
    intent: ScrapeIntent
    source_name: str
    ttl_hours: float = Field(default=6.0, description="Cache TTL in hours")
    fetcher_type: str = Field(
        default="basic",
        description="'basic' for public pages, 'stealth' for bot-protected sites"
    )
    css_selector: Optional[str] = Field(
        default=None,
        description="Optional CSS selector to target specific content area"
    )


class Citation(BaseModel):
    """Source citation for a scraped answer."""
    source_url: str
    source_name: str
    scraped_at: str  # ISO 8601 datetime string
    excerpt: str = Field(description="Relevant 200-char excerpt from source")
    freshness_label: str = Field(
        default="Live",
        description="Human-readable freshness: 'Live', 'Today', '3 hours ago'"
    )


class CitedAnswer(BaseModel):
    """An answer with full source citations (used for browser-scraped answers)."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    has_live_data: bool = True
    freshness_label: str = "Live"


# ─────────────────────────────────────────────────────────────────────────────
# Agricultural Source Registry
# ─────────────────────────────────────────────────────────────────────────────

class AgriSourceSelector:
    """
    Maps query intents to priority-ranked agricultural web sources.

    Sources are chosen for:
    - Reliability (government-primary, then established agri news)
    - Scrape-friendliness (public HTML preferred over JS-heavy SPA)
    - Indian agricultural relevance
    """

    # Source registry: intent → list of TargetSource (priority order)
    SOURCES: dict[ScrapeIntent, list[TargetSource]] = {
        ScrapeIntent.SCHEME_UPDATE: [
            TargetSource(
                url="https://dbtbharat.gov.in/page/frontcontentview/?id=MjM=",
                intent=ScrapeIntent.SCHEME_UPDATE,
                source_name="DBT Bharat Agriculture",
                ttl_hours=24.0,
                fetcher_type="basic",
            ),
            TargetSource(
                url="https://farmer.gov.in/HelpDocument.aspx",
                intent=ScrapeIntent.SCHEME_UPDATE,
                source_name="Farmer.gov.in",
                ttl_hours=24.0,
                fetcher_type="basic",
                css_selector=".content-area",
            ),
            TargetSource(
                url="https://www.india.gov.in/topics/agriculture",
                intent=ScrapeIntent.SCHEME_UPDATE,
                source_name="India.gov.in Agriculture",
                ttl_hours=48.0,
                fetcher_type="basic",
            ),
        ],

        ScrapeIntent.DISEASE_ALERT: [
            TargetSource(
                url="https://www.agrifarming.in/crop-diseases",
                intent=ScrapeIntent.DISEASE_ALERT,
                source_name="AgrifarMing Crop Diseases",
                ttl_hours=12.0,
                fetcher_type="basic",
                css_selector="article, .post-content",
            ),
            TargetSource(
                url="https://icar.org.in/news",
                intent=ScrapeIntent.DISEASE_ALERT,
                source_name="ICAR News",
                ttl_hours=12.0,
                fetcher_type="basic",
                css_selector=".views-row",
            ),
            TargetSource(
                url="https://nhb.gov.in/horticultural-crops",
                intent=ScrapeIntent.DISEASE_ALERT,
                source_name="National Horticulture Board",
                ttl_hours=24.0,
                fetcher_type="basic",
            ),
        ],

        ScrapeIntent.PESTICIDE_BAN: [
            TargetSource(
                url="https://cibrc.nic.in/prt.php",
                intent=ScrapeIntent.PESTICIDE_BAN,
                source_name="CIB&RC Pesticide Registrations",
                ttl_hours=72.0,
                fetcher_type="basic",
                css_selector="table",
            ),
            TargetSource(
                url="https://ppqs.gov.in/divisions/insecticides-act-division",
                intent=ScrapeIntent.PESTICIDE_BAN,
                source_name="PPQS Insecticides Act",
                ttl_hours=72.0,
                fetcher_type="basic",
            ),
        ],

        ScrapeIntent.PRICE_NEWS: [
            TargetSource(
                url="https://www.commodityindia.com/agriculture-news",
                intent=ScrapeIntent.PRICE_NEWS,
                source_name="CommodityIndia Market News",
                ttl_hours=2.0,
                fetcher_type="basic",
                css_selector=".news-list, .article-body",
            ),
            TargetSource(
                url="https://agriwatch.com/news-2/",
                intent=ScrapeIntent.PRICE_NEWS,
                source_name="Agriwatch News",
                ttl_hours=3.0,
                fetcher_type="basic",
                css_selector=".entry-content",
            ),
        ],

        ScrapeIntent.EXPORT_POLICY: [
            TargetSource(
                url="https://apeda.gov.in/apedawebsite/news_letter/news_letter.htm",
                intent=ScrapeIntent.EXPORT_POLICY,
                source_name="APEDA News",
                ttl_hours=24.0,
                fetcher_type="basic",
                css_selector=".news",
            ),
            TargetSource(
                url="https://agriexchange.apeda.gov.in/",
                intent=ScrapeIntent.EXPORT_POLICY,
                source_name="APEDA AgriXchange",
                ttl_hours=6.0,
                fetcher_type="stealth",
            ),
        ],

        ScrapeIntent.WEATHER_ADVISORY: [
            TargetSource(
                url="https://mausam.imd.gov.in/responsive/agricultureweather.php",
                intent=ScrapeIntent.WEATHER_ADVISORY,
                source_name="IMD Agro-meteorological Advisory",
                ttl_hours=6.0,
                fetcher_type="basic",
                css_selector=".agromet",
            ),
        ],

        ScrapeIntent.MARKET_NEWS: [
            TargetSource(
                url="https://krishijagran.com/news/",
                intent=ScrapeIntent.MARKET_NEWS,
                source_name="Krishi Jagran",
                ttl_hours=4.0,
                fetcher_type="basic",
                css_selector="article",
            ),
            TargetSource(
                url="https://www.thehindubusinessline.com/markets/commodities/",
                intent=ScrapeIntent.MARKET_NEWS,
                source_name="Hindu BusinessLine Commodities",
                ttl_hours=4.0,
                fetcher_type="stealth",
                css_selector="article, .article-content",
            ),
        ],
    }

    # Intent classification rules (keyword → intent)
    INTENT_RULES: list[tuple[list[str], ScrapeIntent]] = [
        (["pest", "disease", "blight", "outbreak", "infection", "symptom"], ScrapeIntent.DISEASE_ALERT),
        (["pesticide", "chemical", "banned", "restricted", "herbicide"], ScrapeIntent.PESTICIDE_BAN),
        (["export", "import", "ban", "restriction", "mep", "minimum export"], ScrapeIntent.EXPORT_POLICY),
        (["weather", "monsoon", "rainfall", "forecast", "advisory"], ScrapeIntent.WEATHER_ADVISORY),
        (["price", "mandi", "market rate", "commodity", "bhaav"], ScrapeIntent.PRICE_NEWS),
        (["scheme", "subsidy", "government scheme", "yojana", "policy", "benefit"], ScrapeIntent.SCHEME_UPDATE),
    ]

    def classify_intent(self, query: str) -> ScrapeIntent:
        """
        Classify a query into a scraping intent.

        Args:
            query: User query text

        Returns:
            ScrapeIntent for the matching category (default: MARKET_NEWS)
        """
        q = query.lower()
        for keywords, intent in self.INTENT_RULES:
            if any(kw in q for kw in keywords):
                return intent
        return ScrapeIntent.MARKET_NEWS

    def get_sources(
        self,
        intent: ScrapeIntent,
        max_sources: int = 3,
    ) -> list[TargetSource]:
        """
        Get priority-ranked sources for a scraping intent.

        Args:
            intent: The scraping intent
            max_sources: Maximum number of sources to return

        Returns:
            List of TargetSource, most reliable first
        """
        sources = self.SOURCES.get(intent, self.SOURCES[ScrapeIntent.MARKET_NEWS])
        return sources[:max_sources]


# ─────────────────────────────────────────────────────────────────────────────
# Content Extractor (per-domain CSS selectors)
# ─────────────────────────────────────────────────────────────────────────────

class ContentExtractor:
    """
    Extracts clean text from scraped HTML using domain-specific CSS selectors.

    Falls back to a generic broad-content selector when domain-specific
    selectors fail or no match is found.
    """

    # Domain-specific CSS selectors
    DOMAIN_SELECTORS: dict[str, str] = {
        "farmer.gov.in":         ".content-area, #mainContent, .text",
        "icar.org.in":           ".views-row, .field-items, article",
        "agrifarming.in":        "article, .post-content, .entry-content",
        "commodityindia.com":    ".news-list, .article-body, .post",
        "agriwatch.com":         ".entry-content, .post-content",
        "krishijagran.com":      "article, .story-content, .content",
        "mausam.imd.gov.in":    ".agromet, .content, #main-content",
        "apeda.gov.in":          ".news, .content, table",
        "cibrc.nic.in":          "table, .content, #content",
        "_default":              "article, main, .content, .post, #content, p",
    }

    def extract_text(self, html: str, domain: str, css_override: Optional[str] = None) -> str:
        """
        Extract main text content from HTML.

        Args:
            html: Raw HTML content
            domain: Source domain (for selector lookup)
            css_override: Optional CSS selector from TargetSource

        Returns:
            Extracted clean text
        """
        try:
            from scrapling import Adaptor

            adaptor = Adaptor(html, auto_match=False)
            selector = css_override or self.DOMAIN_SELECTORS.get(
                domain, self.DOMAIN_SELECTORS["_default"]
            )

            # Try domain-specific selector
            elements = adaptor.css(selector)
            if elements:
                text = " ".join(el.text for el in elements if el.text)
                return self._clean_text(text)

            # Fallback to broad paragraph extraction
            paragraphs = adaptor.css("p")
            text = " ".join(p.text for p in paragraphs if p.text)
            return self._clean_text(text)

        except ImportError:
            logger.warning("Scrapling not installed — falling back to regex extraction")
            return self._regex_extract(html)

        except Exception as e:
            logger.warning(f"ContentExtractor: extraction failed for {domain}: {e}")
            return self._regex_extract(html)

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and normalize text."""
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common noise patterns
        text = re.sub(r'(Cookie Policy|Accept Cookies|Privacy Policy).*', '', text)
        return text[:5000]  # Cap at 5000 chars

    def _regex_extract(self, html: str) -> str:
        """Fallback: extract text between paragraph tags using regex."""
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
        text = ' '.join(re.sub(r'<[^>]+>', '', p) for p in paragraphs)
        return self._clean_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

class QualityFilter:
    """
    Filters out low-quality or error pages from scraped content.

    Applies minimum word count and error-page detection heuristics.
    """

    MIN_WORDS = 150
    ERROR_PATTERNS = [
        "404", "page not found", "access denied", "forbidden",
        "service unavailable", "captcha", "cloudflare",
        "we noticed unusual activity", "please enable javascript",
    ]

    def is_valid(self, text: str) -> bool:
        """
        Check if scraped text is valid content.

        Args:
            text: Extracted text

        Returns:
            True if content passes quality threshold
        """
        if not text or len(text.split()) < self.MIN_WORDS:
            return False

        text_lower = text.lower()
        if any(pattern in text_lower for pattern in self.ERROR_PATTERNS):
            return False

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Browser RAG Integration — Main Class
# ─────────────────────────────────────────────────────────────────────────────

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

    Usage:
        browser_rag = BrowserRAGIntegration()
        live_docs = await browser_rag.retrieve_live(
            query="Is Monocrotophos banned in 2026?",
            max_sources=3
        )
        cited = browser_rag.build_cited_answer(answer_text, live_docs)
    """

    def __init__(self, cache_ttl_hours: float = 6.0):
        """
        Initialize BrowserRAGIntegration.

        Args:
            cache_ttl_hours: Default cache TTL for scraped content (hours)
        """
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
        """
        Retrieve live documents for a query from web sources.

        Args:
            query: User query text
            max_sources: Maximum number of sources to scrape (default: 3)
            intent: Override intent classification (auto-detected if None)

        Returns:
            List of Document-like objects with metadata for RAG integration
        """
        start = time.perf_counter()

        # Classify intent
        if intent is None:
            intent = self.source_selector.classify_intent(query)

        sources = self.source_selector.get_sources(intent, max_sources=max_sources)

        logger.info(
            f"BrowserRAGIntegration.retrieve_live | "
            f"intent={intent} | sources={len(sources)} | query={query[:60]}..."
        )

        # Scrape all sources in parallel
        results = await asyncio.gather(
            *[self._scrape_source(source, query) for source in sources],
            return_exceptions=True,
        )

        # Collect valid documents
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
        """
        Scrape a single source URL and extract relevant text.

        Returns a Document-like SimpleNamespace or None if scraping fails.
        """
        try:
            html = await self._fetch_url(source)
            if not html:
                return None

            # Extract domain for selector lookup
            domain = self._extract_domain(source.url)

            # Extract text content
            text = self.extractor.extract_text(
                html=html,
                domain=domain,
                css_override=source.css_selector,
            )

            # Quality filter
            if not self.quality_filter.is_valid(text):
                logger.debug(
                    f"BrowserRAGIntegration: {source.source_name} failed quality filter "
                    f"(words={len(text.split())})"
                )
                return None

            # Create document with metadata
            scraped_at = datetime.now(timezone.utc).isoformat()
            doc = SimpleNamespace(
                text=text,
                id=f"browser_{source.source_name.lower().replace(' ', '_')}",
                score=0.85,  # Fixed confidence for browser-scraped content
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
        """
        Fetch HTML content from a URL using Scrapling or httpx fallback.

        Uses StealthyFetcher for bot-protected sites, basic fetcher otherwise.
        """
        try:
            if self.scraper is not None:
                # Use existing ScraplingBaseScraper
                if source.fetcher_type == "stealth":
                    return await self.scraper.fetch_stealth(source.url)
                else:
                    return await self.scraper.fetch(source.url)

            # Direct httpx fallback
            return await self._httpx_fetch(source.url)

        except Exception as e:
            logger.warning(f"BrowserRAGIntegration._fetch_url: {source.url} error: {e}")
            return None

    async def _httpx_fetch(self, url: str) -> Optional[str]:
        """Basic httpx fetcher as fallback when Scrapling is unavailable."""
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
        """Extract domain from a URL."""
        # Simple regex domain extraction
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else "unknown"

    def build_cited_answer(
        self,
        answer: str,
        used_docs: list[Any],
    ) -> CitedAnswer:
        """
        Build a CitedAnswer with source citations from browser-scraped documents.

        Args:
            answer: Generated answer text
            used_docs: Documents used for generation

        Returns:
            CitedAnswer with full citation list and freshness label
        """
        citations = []
        scraped_at_times = []

        for doc in used_docs:
            metadata = getattr(doc, "metadata", {})
            if not metadata.get("is_live", False):
                continue  # Skip non-browser-scraped docs

            scraped_at = metadata.get("scraped_at", "")
            source_url = metadata.get("source_url", "")
            source_name = metadata.get("source_name", "Web")
            text = getattr(doc, "text", "")
            excerpt = text[:200].strip() + "..." if len(text) > 200 else text

            # Compute freshness label
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

        # Overall freshness
        overall_freshness = "Live" if citations else "Cached"

        return CitedAnswer(
            answer=answer,
            citations=citations,
            has_live_data=bool(citations),
            freshness_label=overall_freshness,
        )

    def _compute_freshness(self, scraped_at_iso: str) -> str:
        """Compute a human-readable freshness label from ISO timestamp."""
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
