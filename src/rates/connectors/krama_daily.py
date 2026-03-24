"""Connector for KRAMA daily mandi reports."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_table_rows, match_field, parse_date, parse_price
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class KRAMADailyConnector(BaseRateConnector):
    """Scrape Karnataka's KRAMA daily report table."""

    source_id = "krama_daily"
    rate_kind = RateKind.MANDI_WHOLESALE
    authority_tier = AuthorityTier.OFFICIAL
    ttl_minutes = 120
    source_url = "https://krama.karnataka.gov.in/Reports/Main_rep"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        records = []
        for row in extract_table_rows(html):
            commodity = match_field(row, "commodity") or query.commodity or ""
            market = match_field(row, "market") or query.market or query.location_label
            district = match_field(row, "district") or query.district
            modal_price = parse_price(match_field(row, "modal_price") or match_field(row, "price"))
            record = self.build_record(
                query=query,
                commodity=commodity,
                district=district,
                market=market,
                location_label=market or district or query.location_label,
                price_date=parse_date(match_field(row, "date"), fallback=query.target_date),
                unit=match_field(row, "unit") or "INR/quintal",
                min_price=parse_price(match_field(row, "min_price")),
                max_price=parse_price(match_field(row, "max_price")),
                modal_price=modal_price,
                price_value=modal_price,
            )
            if self.matches_query(record, query):
                records.append(record)
        return records
