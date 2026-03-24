"""Connector registry for the multi-source rate hub."""

from __future__ import annotations

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


def build_connectors(llm_provider=None, agmarknet_api_key: str = "") -> list:
    """Instantiate all enabled connectors."""
    return [
        KRAMADailyConnector(llm_provider=llm_provider),
        AgmarknetOGDConnector(api_key=agmarknet_api_key, llm_provider=llm_provider),
        AgmarknetScrapeConnector(llm_provider=llm_provider),
        ENAMDashboardConnector(llm_provider=llm_provider),
        KRAMAFloorPriceConnector(llm_provider=llm_provider),
        KAPRICOMReferenceConnector(llm_provider=llm_provider),
        NaPantaConnector(llm_provider=llm_provider),
        AgriplusConnector(llm_provider=llm_provider),
        CommodityMarketLiveConnector(llm_provider=llm_provider),
        ShyaliConnector(llm_provider=llm_provider),
        VegetableMarketPriceConnector(llm_provider=llm_provider),
        TodayPriceRatesConnector(llm_provider=llm_provider),
        PetrolDieselPriceConnector(llm_provider=llm_provider),
        ParkPlusFuelConnector(llm_provider=llm_provider),
        BusinessLineGoldConnector(llm_provider=llm_provider),
        IIFLGoldConnector(llm_provider=llm_provider),
    ]
