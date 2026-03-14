"""
ENAM Scraper Extract
====================
Scrapling-powered scraper for eNAM portal — Live trading data.
"""

import re
import time
from datetime import date
from typing import Any, Optional

from loguru import logger

from src.agents.web_scraping_agent import WebScrapingAgent
from src.scrapers.base_scraper import FetcherType, ScrapeResult, ScraplingBaseScraper

from .models import MandiPrice, MandiPriceList


class ENAMScraper(ScraplingBaseScraper):
    """
    Scraper for eNAM portal — Live trading data.

    URL: https://enam.gov.in
    Uses Scrapling's StealthyFetcher for anti-bot bypass.
    """
    name = "enam"
    base_url = "https://enam.gov.in"
    fetcher_type = FetcherType.STEALTHY
    cache_ttl_seconds = 180
    rate_limit_delay = 3.0

    DASHBOARD_URL = "https://enam.gov.in/web/dashboard/trade-data"

    def __init__(self, llm_provider: Optional[Any] = None):
        super().__init__()
        self.web_agent = WebScrapingAgent(llm_provider=llm_provider) if llm_provider else None

    async def scrape(
        self,
        commodity: Optional[str] = None,
        state: Optional[str] = None,
        **kwargs,
    ) -> ScrapeResult:
        """Get live prices from eNAM dashboard."""
        start_time = time.time()

        try:
            page = await self.fetch(self.DASHBOARD_URL)
            prices = self._parse_enam_data(page, commodity, state)

            if not prices and self.web_agent:
                logger.info("eNAM CSS extraction failed — falling back to LLM WebScrapingAgent")
                prices = await self._fallback_llm_extract(page, commodity, state)

            duration_ms = (time.time() - start_time) * 1000
            return self.build_result(
                url=self.DASHBOARD_URL,
                data=[p.model_dump() for p in prices],
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"eNAM scraping failed: {e}")
            return self.build_result(
                url=self.DASHBOARD_URL,
                data=[],
                error=str(e),
                duration_ms=duration_ms,
            )

    def _parse_enam_data(
        self,
        page: Any,
        commodity: Optional[str],
        state: Optional[str],
    ) -> list[MandiPrice]:
        prices: list[MandiPrice] = []

        try:
            rows = page.css("table.table tbody tr")
            for row in rows:
                cells = row.css("td::text").getall()
                if len(cells) >= 6:
                    try:
                        row_commodity = cells[0].strip()
                        row_state = cells[1].strip()

                        if commodity and commodity.lower() not in row_commodity.lower():
                            continue
                        if state and state.lower() not in row_state.lower():
                            continue

                        prices.append(
                            MandiPrice(
                                commodity=row_commodity,
                                mandi=cells[2].strip(),
                                state=row_state,
                                min_price=self._safe_float(cells[3]),
                                max_price=self._safe_float(cells[4]),
                                modal_price=self._safe_float(cells[5]) or 0.0,
                                date=date.today(),
                                source="enam",
                            )
                        )
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.warning(f"eNAM table parsing failed: {e}")

        if not prices:
            logger.warning("No live eNAM data parsed")

        return prices

    async def _fallback_llm_extract(
        self, page: Any, commodity: Optional[str], state: Optional[str]
    ) -> list[MandiPrice]:
        if not self.web_agent:
            return []

        html_content = ""
        if hasattr(page, "body"):
            html_content = page.body.decode("utf-8", "ignore")
        elif hasattr(page, "html"):
            html_content = page.html

        if not html_content:
            return []

        instruction = "Extract all live trade prices from the eNAM dashboard table."
        result = await self.web_agent.extract_with_schema(
            html_content=html_content,
            url=self.DASHBOARD_URL,
            schema=MandiPriceList,
            instruction=instruction
        )

        fallback_prices = []
        if result.success and isinstance(result.extracted_data, dict):
            for item in result.extracted_data.get("prices", []):
                try:
                    c = item.get("commodity", "")
                    s = item.get("state", "")

                    if commodity and commodity.lower() not in c.lower(): continue
                    if state and state.lower() not in s.lower(): continue

                    item["source"] = "enam_llm"
                    fallback_prices.append(MandiPrice(**item))
                except Exception as e:
                    logger.debug(f"LLM item parse error: {e}")

        return fallback_prices

    def _safe_float(self, value: str) -> Optional[float]:
        try:
            cleaned = re.sub(r"[^\d.]", "", value.strip())
            return float(cleaned) if cleaned else None
        except (ValueError, AttributeError):
            return None
