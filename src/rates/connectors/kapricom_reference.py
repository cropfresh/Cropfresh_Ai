"""Connector for KAPRICOM reference price pages."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.connectors.html_utils import extract_table_rows, match_field, parse_price
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery


class KAPRICOMReferenceConnector(BaseRateConnector):
    """Scrape KAPRICOM analytical reference prices when tabular data is exposed."""

    source_id = "kapricom_reference"
    rate_kind = RateKind.SUPPORT_PRICE
    authority_tier = AuthorityTier.REFERENCE_OFFICIAL
    ttl_minutes = 1440
    source_url = "https://kapricom.karnataka.gov.in/english"

    async def fetch(self, query: RateQuery):
        html = await self.fetch_page_content(query)
        records = []
        for row in extract_table_rows(html):
            commodity = match_field(row, "commodity") or query.commodity or ""
            if not commodity:
                continue
            price_value = parse_price(match_field(row, "price") or match_field(row, "modal_price"))
            if price_value is None:
                continue
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
