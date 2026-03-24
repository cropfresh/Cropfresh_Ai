"""Connector for IIFL Karnataka gold rates."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_rate_from_text
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class IIFLGoldConnector(BaseRateConnector):
    """Scrape Karnataka gold rates from IIFL."""

    source_id = "iifl_gold"
    rate_kind = RateKind.GOLD
    authority_tier = AuthorityTier.RETAIL_REFERENCE
    ttl_minutes = 60
    source_url = "https://www.iifl.com/gold-rates-today/gold-rate-karnataka"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        rate_gram = extract_rate_from_text(html, ("per gram", "1 gram", "22k"))
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
