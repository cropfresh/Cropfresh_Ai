"""Connector for KRAMA minimum support price data."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_table_rows, match_field, parse_price
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class KRAMAFloorPriceConnector(BaseRateConnector):
    """Scrape Karnataka KRAMA minimum floor price tables."""

    source_id = "krama_floor_price"
    rate_kind = RateKind.SUPPORT_PRICE
    authority_tier = AuthorityTier.REFERENCE_OFFICIAL
    ttl_minutes = 1440
    source_url = "https://krama.karnataka.gov.in/Markets/minimumsupportprice"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        records = []
        for row in extract_table_rows(html):
            commodity = match_field(row, "commodity") or query.commodity or ""
            price_value = parse_price(match_field(row, "price"))
            record = self.build_record(
                query=query,
                commodity=commodity,
                variety=match_field(row, "variety") or None,
                location_label=query.state,
                price_date=query.target_date,
                unit="INR/quintal",
                price_value=price_value,
                modal_price=price_value,
                source_url=self.source_url,
                freshness="reference",
            )
            if self.matches_query(record, query):
                records.append(record)
        return records
