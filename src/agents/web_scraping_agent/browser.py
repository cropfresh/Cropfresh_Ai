"""
Browser Scraper Mixin
=====================
Handles Playwright-based navigation and content extraction.
"""

from datetime import datetime
from typing import Optional
from loguru import logger

from .models import ScrapingResult, ScrapingConfig
from .base import BaseWebScraper
from .parser import HTMLParserMixin
from .cache import ScraperCacheMixin
from src.tools.browser_stealth import apply_stealth


class BrowserScraperMixin(HTMLParserMixin, ScraperCacheMixin, BaseWebScraper):
    """Mixin for browser navigation and basic extraction without LLM."""

    async def scrape_to_markdown(
        self,
        url: str,
        config: Optional[ScrapingConfig] = None,
        use_cache: bool = True,
    ) -> ScrapingResult:
        """Scrape URL and return clean markdown content."""
        config = config or ScrapingConfig()
        start_time = datetime.now()

        if use_cache:
            cached = self._get_cached(url)
            if cached:
                logger.debug("Cache hit for {}", url)
                return cached

        try:
            if not self._browser:
                success = await self.initialize(config)
                if not success:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error="Playwright unavailable on this environment.",
                        scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    )

            page = await self._context.new_page()

            if config.stealth:
                await apply_stealth(page)

            await page.goto(url, timeout=config.timeout, wait_until=config.wait_for_load_state)

            if config.wait_for_selector:
                await page.wait_for_selector(config.wait_for_selector, timeout=config.timeout)

            html = await page.content()
            markdown = self._html_to_markdown(html)

            if config.screenshot:
                screenshot_path = self.cache_dir / f"{self._url_hash(url)}.png"
                await page.screenshot(path=str(screenshot_path), full_page=config.full_page_screenshot)

            await page.close()

            result = ScrapingResult(
                url=url,
                success=True,
                markdown=markdown,
                html=html,
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

            if use_cache:
                self._set_cached(url, result)

            logger.info("Scraped {} ({:.0f}ms)", url, result.scrape_time_ms)
            return result

        except Exception as e:
            logger.error("Failed to scrape {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                error=str(e),
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def scrape_with_css(
        self,
        url: str,
        selectors: dict[str, str],
        config: Optional[ScrapingConfig] = None,
    ) -> ScrapingResult:
        """Extract data using CSS selectors (no LLM cost)."""
        config = config or ScrapingConfig()
        start_time = datetime.now()

        try:
            if not self._browser:
                success = await self.initialize(config)
                if not success:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error="Playwright unavailable on this environment.",
                        scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    )

            page = await self._context.new_page()

            if config.stealth:
                await apply_stealth(page)

            await page.goto(url, timeout=config.timeout, wait_until=config.wait_for_load_state)

            extracted = {}
            for field_name, selector in selectors.items():
                try:
                    elements = await page.query_selector_all(selector)
                    if len(elements) == 1:
                        extracted[field_name] = await elements[0].inner_text()
                    else:
                        extracted[field_name] = [await el.inner_text() for el in elements]
                except Exception as e:
                    logger.warning("Failed to extract {}: {}", field_name, str(e))
                    extracted[field_name] = None

            await page.close()

            return ScrapingResult(
                url=url,
                success=True,
                extracted_data=extracted,
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            logger.error("CSS extraction failed for {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                error=str(e),
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
