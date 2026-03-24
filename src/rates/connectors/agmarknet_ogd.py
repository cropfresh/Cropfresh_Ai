"""Connector for AGMARKNET OGD/API data."""

from __future__ import annotations

from src.rates.connectors.base import BaseRateConnector
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import RateQuery
from src.tools.agmarknet import AgmarknetTool


class AgmarknetOGDConnector(BaseRateConnector):
    """Fetch mandi wholesale data from the AGMARKNET OGD endpoint."""

    source_id = "agmarknet_ogd"
    rate_kind = RateKind.MANDI_WHOLESALE
    authority_tier = AuthorityTier.OFFICIAL
    ttl_minutes = 120
    source_url = AgmarknetTool.BASE_URL

    def __init__(self, api_key: str = "", llm_provider=None):
        super().__init__(llm_provider=llm_provider)
        self.tool = AgmarknetTool(api_key=api_key)

    async def fetch(self, query: RateQuery):
        prices = await self.tool.get_prices(
            commodity=(query.commodity or "").title(),
            state=query.state,
            district=query.district,
            market=query.market,
            limit=50,
        )
        records = [
            self.build_record(
                query=query,
                commodity=price.commodity,
                district=price.district,
                market=price.market,
                location_label=price.market or price.district or query.location_label,
                price_date=price.date.date(),
                unit="INR/quintal",
                source_url=self.source_url,
                min_price=price.min_price,
                max_price=price.max_price,
                modal_price=price.modal_price,
                price_value=price.modal_price,
            )
            for price in prices
        ]
        return [record for record in records if self.matches_query(record, query)]
