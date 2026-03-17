"""Connector for the direct AGMARKNET scraper."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery
from src.scrapers.agmarknet.client import AgmarknetScraper


class AgmarknetScrapeConnector(BaseRateConnector):
    """Fetch mandi wholesale data from the AGMARKNET report scraper."""

    source_id = "agmarknet_scrape"
    rate_kind = RateKind.MANDI_WHOLESALE
    authority_tier = AuthorityTier.OFFICIAL
    ttl_minutes = 120
    source_url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"

    def __init__(self, llm_provider=None):
        super().__init__(llm_provider=llm_provider)
        self.scraper = AgmarknetScraper()

    async def fetch(self, query: RateQuery):
        result = await self.scraper.scrape(
            state=query.state,
            commodity=query.commodity or "",
            district=query.district,
            market=query.market,
        )
        records = []
        for row in result.data:
            record = self.build_record(
                query=query,
                commodity=row.get("commodity") or query.commodity,
                variety=row.get("variety"),
                district=row.get("district"),
                market=row.get("market"),
                location_label=row.get("market") or row.get("district") or query.location_label,
                price_date=query.target_date,
                unit=row.get("unit") or "INR/quintal",
                source_url=result.url or self.source_url,
                min_price=row.get("min_price"),
                max_price=row.get("max_price"),
                modal_price=row.get("modal_price"),
                price_value=row.get("modal_price"),
            )
            if self.matches_query(record, query):
                records.append(record)
        return records
