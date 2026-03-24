"""Connector for retail produce pages on VegetableMarketPrice."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_table_rows, match_field, parse_date, parse_price
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class VegetableMarketPriceConnector(BaseRateConnector):
    """Scrape retail produce prices from VegetableMarketPrice."""

    source_id = "vegetablemarketprice"
    rate_kind = RateKind.RETAIL_PRODUCE
    authority_tier = AuthorityTier.RETAIL_REFERENCE
    ttl_minutes = 360
    source_url = "https://vegetablemarketprice.com/market/karnataka/today"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        records = []
        for row in extract_table_rows(html):
            commodity = match_field(row, "commodity") or query.commodity or ""
            price_value = parse_price(match_field(row, "price") or match_field(row, "modal_price"))
            record = self.build_record(
                query=query,
                commodity=commodity,
                location_label=query.location_label,
                price_date=parse_date(match_field(row, "date"), fallback=query.target_date),
                unit="INR/kg",
                price_value=price_value,
                modal_price=price_value,
            )
            if self.matches_query(record, query):
                records.append(record)
        return records
