"""Connector for the Shyali APMC mirror."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_table_rows, match_field, parse_date, parse_price
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class ShyaliConnector(BaseRateConnector):
    """Scrape the Shyali mirror as a validator source."""

    source_id = "shyali"
    rate_kind = RateKind.MANDI_WHOLESALE
    authority_tier = AuthorityTier.VALIDATOR
    ttl_minutes = 360
    source_url = "https://www.shyaliproducts.com/apmc-prices"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        records = []
        for row in extract_table_rows(html):
            market = match_field(row, "market") or query.location_label
            commodity = match_field(row, "commodity") or query.commodity or ""
            modal_price = parse_price(match_field(row, "modal_price") or match_field(row, "price"))
            record = self.build_record(
                query=query,
                commodity=commodity,
                market=market,
                location_label=market,
                price_date=parse_date(match_field(row, "date"), fallback=query.target_date),
                unit="INR/quintal",
                modal_price=modal_price,
                price_value=modal_price,
                min_price=parse_price(match_field(row, "min_price")),
                max_price=parse_price(match_field(row, "max_price")),
            )
            if self.matches_query(record, query):
                records.append(record)
        return records
