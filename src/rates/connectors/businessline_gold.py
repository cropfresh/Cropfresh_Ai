"""Connector for BusinessLine Karnataka gold rates."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_rate_from_text
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class BusinessLineGoldConnector(BaseRateConnector):
    """Scrape Karnataka gold rates from BusinessLine."""

    source_id = "businessline_gold"
    rate_kind = RateKind.GOLD
    authority_tier = AuthorityTier.REFERENCE_OFFICIAL
    ttl_minutes = 60
    source_url = "https://www.thehindubusinessline.com/gold-rate-today/Karnataka/"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        rate_gram = extract_rate_from_text(html, ("per gram", "1 gram", "22 carat"))
        if rate_gram is None:
            return []
        return [
            self.build_record(
                query=query,
                commodity="gold",
                location_label=query.state,
                price_date=query.target_date,
                unit="INR/gram",
                price_value=rate_gram,
                modal_price=rate_gram,
            )
        ]
