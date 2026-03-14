"""
Base Web Scraper
================
Provides Playwright browser context initialization and teardown.
"""

import sys
from datetime import timedelta
from typing import Any, Optional
from pathlib import Path

from loguru import logger
from playwright.async_api import async_playwright, Browser, BrowserContext

from .models import ScrapingConfig
from src.tools.browser_stealth import get_random_user_agent


class BaseWebScraper:
    """Base class holding Playwright instance and configuration."""

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_minutes: int = 15,
    ):
        self.llm_provider = llm_provider
        self.cache_dir = cache_dir or Path("data/scraping_cache")
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._playwright_supported = True

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("BaseWebScraper initialized")

    async def initialize(self, config: Optional[ScrapingConfig] = None) -> bool:
        """Initialize browser instance."""
        config = config or ScrapingConfig()

        if not self._playwright_supported:
            return False

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=config.headless,
            )

            self._context = await self._browser.new_context(
                viewport={"width": config.viewport_width, "height": config.viewport_height},
                user_agent=get_random_user_agent() if config.stealth else None,
            )

            logger.info("Browser initialized (headless={})", config.headless)
            return True
        except NotImplementedError as e:
            if sys.platform == "win32":
                logger.error("Playwright not supported on current Windows asyncio loop. Use WSL or Custom Loop.")
            else:
                logger.error("Playwright initialization failed: {}", str(e))
            self._playwright_supported = False
            return False
        except Exception as e:
            logger.error("Failed to initialize Playwright browser: {}", str(e))
            self._playwright_supported = False
            return False

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")
