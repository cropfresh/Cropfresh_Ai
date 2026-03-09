"""
State Government Portals Scraper
================================
Handles data extraction from various State Government agricultural portals.
Provides standard configurations and LLM-based fallback for unknown pages.
"""

import time
import re
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from src.scrapers.base_scraper import FetcherType, ScrapeResult, ScraplingBaseScraper
from src.agents.web_scraping_agent import WebScrapingAgent


class StateAgriculturalAlert(BaseModel):
    """Schema for state government agricultural alerts/schemes/notices."""
    state: str
    title: str
    description: Optional[str] = None
    date_issued: Optional[str] = None
    url: Optional[str] = None
    source: str


class StateAlertList(BaseModel):
    """Wrapper for array of state alerts returned by LLM extraction."""
    alerts: list[StateAgriculturalAlert]


class StatePortalScraper(ScraplingBaseScraper):
    """
    Generic scraper for various State Government portals.
    Uses specific CSS patterns if known, otherwise relies on WebScrapingAgent.
    """
    name = "state_portals"
    fetcher_type = FetcherType.STEALTHY
    cache_ttl_seconds = 3600  # 1 hour
    
    PORTAL_URLS = {
        "UP": "https://upagripardarshi.gov.in/",
        "MH": "https://krishi.maharashtra.gov.in/",
    }

    def __init__(self, llm_provider: Optional[Any] = None):
        super().__init__()
        self.web_agent = WebScrapingAgent(llm_provider=llm_provider) if llm_provider else None

    async def scrape(self, state: str = "UP", **kwargs) -> ScrapeResult:
        """
        Scrape alerts/data for a specific state portal.
        
        Args:
            state: State code (e.g., 'UP', 'MH')
        """
        state_code = state.upper()
        if state_code not in self.PORTAL_URLS:
            if not self.web_agent:
                return self.build_result(url="", data=[], error=f"Unsupported state: {state}")
            logger.info(f"No predefined URL for state {state}, dynamic mapping not yet supported.")
            return self.build_result(url="", data=[], error=f"Unknown state code: {state}")

        url = self.PORTAL_URLS[state_code]
        start_time = time.time()
        
        try:
            page = await self.fetch(url)
            alerts = []
            
            # 1. Attempt known CSS selectors based on state
            if state_code == "UP":
                alerts = self._parse_up_agri(page)
            elif state_code == "MH":
                alerts = self._parse_maha_agri(page)
                
            # 2. LLM Fallback if CSS parsing failed or returned empty
            if not alerts and self.web_agent:
                logger.info(f"{state_code} Portal CSS extraction failed, falling back to LLM WebScrapingAgent")
                alerts = await self._fallback_llm_extract(page, url, state_code)
                
            if not alerts:
                logger.warning(f"No alerts found for '{state_code}' portal")
                
            duration_ms = (time.time() - start_time) * 1000
            return self.build_result(
                url=url,
                data=[a.model_dump() for a in alerts],
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"State portal scraping failed for {state_code}: {e}")
            return self.build_result(
                url=url,
                data=[],
                error=str(e),
                duration_ms=duration_ms
            )

    def _parse_up_agri(self, page: Any) -> list[StateAgriculturalAlert]:
        alerts = []
        try:
            # Example CSS extraction logic based on common markup for Marquees
            items = page.css(".marquee span, .news-list li, marquee a")
            for item in items:
                text = item.css("::text").get()
                if not text:
                    continue
                text = text.strip()
                link = item.css("::attr(href)").get()
                if text:
                    alerts.append(StateAgriculturalAlert(
                        state="UP",
                        title=text,
                        url=link,
                        source="up_agri_portal"
                    ))
        except Exception as e:
            logger.debug(f"UP Agri CSS parse error: {e}")
        return alerts

    def _parse_maha_agri(self, page: Any) -> list[StateAgriculturalAlert]:
        alerts = []
        try:
            items = page.css(".marqee_text a, .whatsnew a, marquee a")
            for item in items:
                text = item.css("::text").get()
                if not text:
                    continue
                text = text.strip()
                link = item.css("::attr(href)").get()
                if text:
                    alerts.append(StateAgriculturalAlert(
                        state="MH",
                        title=text,
                        url=link,
                        source="maha_agri_portal"
                    ))
        except Exception as e:
            logger.debug(f"MahaAgri CSS parse error: {e}")
        return alerts

    async def _fallback_llm_extract(self, page: Any, url: str, state_code: str) -> list[StateAgriculturalAlert]:
        if not self.web_agent:
            return []
            
        html_content = ""
        if hasattr(page, "body"):
            html_content = page.body.decode("utf-8", "ignore")
        elif hasattr(page, "html"):
            html_content = page.html
            
        if not html_content:
            return []
            
        instruction = "Extract the latest agricultural news, schemes, announcements, and alerts from the page."
        result = await self.web_agent.extract_with_schema(
            html_content=html_content,
            url=url,
            schema=StateAlertList,
            instruction=instruction
        )
        
        fallback_alerts = []
        if result.success and isinstance(result.extracted_data, dict):
            for item in result.extracted_data.get("alerts", []):
                try:
                    # Enforce the specific state code and tag as LLM-extracted
                    item["state"] = state_code
                    item["source"] = f"state_portal_{state_code.lower()}_llm"
                    fallback_alerts.append(StateAgriculturalAlert(**item))
                except Exception as e:
                    logger.debug(f"LLM item parse error for State Portal: {e}")
                    
        return fallback_alerts
