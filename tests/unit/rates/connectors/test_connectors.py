from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.rates.connectors.agmarknet_ogd import AgmarknetOGDConnector
from src.rates.connectors.agmarknet_scrape import AgmarknetScrapeConnector
from src.rates.connectors.agriplus import AgriplusConnector
from src.rates.connectors.businessline_gold import BusinessLineGoldConnector
from src.rates.connectors.commoditymarketlive import CommodityMarketLiveConnector
from src.rates.connectors.enam_dashboard import ENAMDashboardConnector
from src.rates.connectors.iifl_gold import IIFLGoldConnector
from src.rates.connectors.kapricom_reference import KAPRICOMReferenceConnector
from src.rates.connectors.krama_daily import KRAMADailyConnector
from src.rates.connectors.krama_floor_price import KRAMAFloorPriceConnector
from src.rates.connectors.napanta import NaPantaConnector
from src.rates.connectors.parkplus_fuel import ParkPlusFuelConnector
from src.rates.connectors.petroldieselprice import PetrolDieselPriceConnector
from src.rates.connectors.shyali import ShyaliConnector
from src.rates.connectors.todaypricerates import TodayPriceRatesConnector
from src.rates.connectors.vegetablemarketprice import VegetableMarketPriceConnector
from src.rates.query_builder import normalize_rate_query

MANDI_HTML = """
<table>
  <tr><th>Commodity</th><th>District</th><th>Market</th><th>Date</th><th>Min Price</th><th>Max Price</th><th>Modal Price</th><th>Unit</th></tr>
  <tr><td>Tomato</td><td>Kolar</td><td>Kolar</td><td>2026-03-17</td><td>1000</td><td>1400</td><td>1200</td><td>INR/quintal</td></tr>
</table>
"""
SUPPORT_HTML = """
<table>
  <tr><th>Commodity</th><th>Support Price</th><th>Variety</th></tr>
  <tr><td>Copra</td><td>12000</td><td>Milling</td></tr>
</table>
"""
RETAIL_HTML = """
<table>
  <tr><th>Commodity</th><th>Date</th><th>Price</th></tr>
  <tr><td>Tomato</td><td>2026-03-17</td><td>45</td></tr>
</table>
"""
FUEL_HTML = "<html><body>Petrol 102.55 Diesel 88.10</body></html>"
GOLD_HTML = "<html><body>22 carat gold rate 1 gram 6789 per gram</body></html>"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connector", "query", "html", "expected_unit"),
    [
        (KRAMADailyConnector(), normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar"), MANDI_HTML, "INR/quintal"),
        (KRAMAFloorPriceConnector(), normalize_rate_query(rate_kinds=["support_price"], commodity="copra"), SUPPORT_HTML, "INR/quintal"),
        (KAPRICOMReferenceConnector(), normalize_rate_query(rate_kinds=["support_price"], commodity="copra"), SUPPORT_HTML, "INR/quintal"),
        (NaPantaConnector(), normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar"), MANDI_HTML, "INR/quintal"),
        (AgriplusConnector(), normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar"), MANDI_HTML, "INR/quintal"),
        (CommodityMarketLiveConnector(), normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar"), MANDI_HTML, "INR/quintal"),
        (ShyaliConnector(), normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar"), MANDI_HTML, "INR/quintal"),
        (VegetableMarketPriceConnector(), normalize_rate_query(rate_kinds=["retail_produce"], commodity="tomato", market="Bengaluru"), RETAIL_HTML, "INR/kg"),
        (TodayPriceRatesConnector(), normalize_rate_query(rate_kinds=["retail_produce"], commodity="tomato", market="Bengaluru"), RETAIL_HTML, "INR/kg"),
        (PetrolDieselPriceConnector(), normalize_rate_query(rate_kinds=["fuel"]), FUEL_HTML, "INR/litre"),
        (ParkPlusFuelConnector(), normalize_rate_query(rate_kinds=["fuel"]), FUEL_HTML, "INR/litre"),
        (BusinessLineGoldConnector(), normalize_rate_query(rate_kinds=["gold"]), GOLD_HTML, "INR/gram"),
        (IIFLGoldConnector(), normalize_rate_query(rate_kinds=["gold"]), GOLD_HTML, "INR/gram"),
    ],
)
async def test_html_connectors_parse_records(connector, query, html, expected_unit) -> None:
    connector.fetch_page_content = AsyncMock(return_value=html)
    records = await connector.fetch(query)
    assert records
    assert all(record.source == connector.source_id for record in records)
    assert all(record.unit == expected_unit for record in records)


@pytest.mark.asyncio
async def test_agmarknet_ogd_connector_maps_tool_results() -> None:
    connector = AgmarknetOGDConnector()
    connector.tool.get_prices = AsyncMock(
        return_value=[
            SimpleNamespace(
                commodity="Tomato",
                district="Kolar",
                market="Kolar",
                date=datetime(2026, 3, 17, 6, 0, 0),
                min_price=1000.0,
                max_price=1400.0,
                modal_price=1200.0,
            )
        ]
    )
    query = normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar")
    records = await connector.fetch(query)
    assert records[0].source == "agmarknet_ogd"
    assert records[0].modal_price == 1200.0


@pytest.mark.asyncio
async def test_agmarknet_scrape_connector_maps_scraper_results() -> None:
    connector = AgmarknetScrapeConnector()
    connector.scraper.scrape = AsyncMock(
        return_value=SimpleNamespace(
            data=[{"commodity": "Tomato", "district": "Kolar", "market": "Kolar", "modal_price": 1250.0}],
            url="https://example.com/agmarknet",
        )
    )
    query = normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar")
    records = await connector.fetch(query)
    assert records[0].source == "agmarknet_scrape"
    assert records[0].price_value == 1250.0


@pytest.mark.asyncio
async def test_enam_dashboard_connector_maps_scraper_results() -> None:
    connector = ENAMDashboardConnector()
    connector.scraper.scrape = AsyncMock(
        return_value=SimpleNamespace(
            data=[{"commodity": "Tomato", "district": "Kolar", "mandi": "Kolar", "modal_price": 1300.0, "date": date(2026, 3, 17)}],
            url="https://example.com/enam",
        )
    )
    query = normalize_rate_query(rate_kinds=["mandi_wholesale"], commodity="tomato", market="Kolar")
    records = await connector.fetch(query)
    assert records[0].source == "enam_dashboard"
    assert records[0].price_value == 1300.0
