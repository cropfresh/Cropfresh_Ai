"""Connector for the eNAM dashboard scraper."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery
from src.scrapers.agri_scrapers.enam import ENAMScraper


class ENAMDashboardConnector(BaseRateConnector):
    """Fetch mandi wholesale data from the public eNAM dashboard."""

    source_id = "enam_dashboard"
    rate_kind = RateKind.MANDI_WHOLESALE
    authority_tier = AuthorityTier.OFFICIAL
    ttl_minutes = 120
    source_url = ENAMScraper.DASHBOARD_URL
    uses_browser = True
    allow_llm_fallback = True

    def __init__(self, llm_provider=None):
        super().__init__(llm_provider=llm_provider)
        self.scraper = ENAMScraper(llm_provider=llm_provider)

    async def fetch(self, query: RateQuery):
        result = await self.scraper.scrape(
            commodity=query.commodity,
            state=query.state,
            district=query.district,
        )
        records = []
        for row in result.data:
            record = self.build_record(
                query=query,
                commodity=row.get("commodity") or query.commodity,
                district=row.get("district"),
                market=row.get("mandi") or row.get("market"),
                location_label=row.get("mandi") or row.get("market") or query.location_label,
                price_date=row.get("date") or query.target_date,
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
