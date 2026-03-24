"""Base connector contract for multi-source rates."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import httpx

from src.rates.enums import AuthorityTier, FetchMode, RateKind
from src.rates.models import NormalizedRateRecord, RateQuery


class BaseRateConnector:
    """Base contract shared by all rate connectors."""

    source_id: str = "base"
    rate_kind: RateKind = RateKind.MANDI_WHOLESALE
    authority_tier: AuthorityTier = AuthorityTier.VALIDATOR
    fetch_mode: FetchMode = FetchMode.LIVE
    ttl_minutes: int = 360
    supports_live: bool = True
    source_url: str = ""
    uses_browser: bool = False
    allow_llm_fallback: bool = False

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    async def fetch(self, query: RateQuery) -> list[NormalizedRateRecord]:
        """Fetch normalized records for a query."""
        raise NotImplementedError

    def build_url(self, query: RateQuery) -> str:
        """Return the target URL for a query."""
        del query
        return self.source_url

    async def fetch_page_content(self, query: RateQuery) -> str:
        """Fetch page content using HTTP first, then optional browser fallbacks."""
        url = self.build_url(query)
        try:
            return await self.fetch_text(url)
        except Exception:
            if self.uses_browser:
                content = await self._fetch_with_browser(url)
                if content:
                    return content
            if self.allow_llm_fallback:
                content = await self._fetch_with_web_agent(url)
                if content:
                    return content
            raise

    async def fetch_text(self, url: str) -> str:
        """Fetch text from a public URL."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept-Language": "en-IN,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def _fetch_with_browser(self, url: str) -> Optional[str]:
        """Fetch dynamic content through BrowserAgent when required."""
        from src.agents.browser_agent import ActionType, BrowserAction, BrowserAgent

        agent = BrowserAgent(headless=True, stealth=True)
        try:
            await agent.start_session()
            await agent.execute_action(BrowserAction(action=ActionType.GOTO, value=url))
            return await agent.get_page_content()
        except Exception:
            return None
        finally:
            await agent.close_session()

    async def _fetch_with_web_agent(self, url: str) -> Optional[str]:
        """Fetch markdown via WebScrapingAgent as a final fallback."""
        if self.llm_provider is None:
            return None

        from src.agents.web_scraping_agent import WebScrapingAgent

        agent = WebScrapingAgent(llm_provider=self.llm_provider)
        try:
            await agent.initialize()
            result = await agent.scrape_to_markdown(url)
            return result.markdown if result.success else None
        except Exception:
            return None
        finally:
            await agent.close()

    def build_record(
        self,
        *,
        query: RateQuery,
        location_label: str,
        price_date: date,
        unit: str,
        source_url: Optional[str] = None,
        commodity: Optional[str] = None,
        variety: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        price_value: Optional[float] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        modal_price: Optional[float] = None,
        freshness: str = "live",
    ) -> NormalizedRateRecord:
        """Construct one normalized record."""
        return NormalizedRateRecord(
            rate_kind=self.rate_kind,
            commodity=commodity or query.commodity,
            variety=variety,
            state=query.state,
            district=district or query.district,
            market=market or query.market,
            location_label=location_label,
            price_date=price_date,
            unit=unit,
            currency="INR",
            price_value=price_value,
            min_price=min_price,
            max_price=max_price,
            modal_price=modal_price,
            source=self.source_id,
            authority_tier=self.authority_tier,
            source_url=source_url or self.build_url(query),
            freshness=freshness,
            fetched_at=datetime.utcnow(),
        )

    def matches_query(self, record: NormalizedRateRecord, query: RateQuery) -> bool:
        """Apply commodity and location filters after parsing."""
        if query.commodity and record.commodity and query.commodity.lower() not in record.commodity.lower():
            return False
        if query.market and record.location_label and query.market.lower() not in record.location_label.lower():
            return False
        if query.district and record.district and query.district.lower() not in record.district.lower():
            return False
        return True
