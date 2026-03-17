"""Connector for CommodityMarketLive state pages."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_table_rows, match_field, parse_date, parse_price
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class CommodityMarketLiveConnector(BaseRateConnector):
    """Scrape CommodityMarketLive as a validator source."""

    source_id = "commoditymarketlive"
    rate_kind = RateKind.MANDI_WHOLESALE
    authority_tier = AuthorityTier.VALIDATOR
    ttl_minutes = 360
    source_url = "https://www.commoditymarketlive.com/mandi-price-state/karnataka"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        records = []
        for row in extract_table_rows(html):
            commodity = match_field(row, "commodity") or query.commodity or ""
            modal_price = parse_price(match_field(row, "modal_price") or match_field(row, "price"))
            record = self.build_record(
                query=query,
                commodity=commodity,
                location_label=query.location_label,
                price_date=parse_date(match_field(row, "date"), fallback=query.target_date),
                unit="INR/quintal",
                modal_price=modal_price,
                price_value=modal_price,
            )
            if self.matches_query(record, query):
                records.append(record)
        return records
