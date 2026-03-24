"""Connector for Park+ Karnataka fuel pages."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_rate_from_text
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class ParkPlusFuelConnector(BaseRateConnector):
    """Scrape Karnataka fuel rates from Park+."""

    source_id = "parkplus_fuel"
    rate_kind = RateKind.FUEL
    authority_tier = AuthorityTier.RETAIL_REFERENCE
    ttl_minutes = 60
    source_url = "https://parkplus.io/fuel-price/karnataka"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        petrol = extract_rate_from_text(html, ("petrol",))
        diesel = extract_rate_from_text(html, ("diesel",))
        records = []
        if petrol is not None:
            records.append(
                self.build_record(
                    query=query,
                    commodity="petrol",
                    location_label=query.state,
                    price_date=query.target_date,
                    unit="INR/litre",
                    price_value=petrol,
                    modal_price=petrol,
                )
            )
        if diesel is not None:
            records.append(
                self.build_record(
                    query=query,
                    commodity="diesel",
                    location_label=query.state,
                    price_date=query.target_date,
                    unit="INR/litre",
                    price_value=diesel,
                    modal_price=diesel,
                )
            )
        return records
