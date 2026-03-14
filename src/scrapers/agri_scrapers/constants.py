"""
Source Constants 
================
Data source configurations for agricultural data.
"""

from enum import Enum


class DataSource(str, Enum):
    """Available data sources."""
    ENAM = "enam"
    AGMARKNET = "agmarknet"
    IMD = "imd"
    DATA_GOV = "data_gov"
    PM_KISAN = "pm_kisan"
    RURAL_VOICE = "rural_voice"
    AI_KOSHA = "ai_kosha"


SOURCE_URLS = {
    DataSource.ENAM: "https://enam.gov.in/web/dashboard/trade-data",
    DataSource.AGMARKNET: "https://agmarknet.gov.in/SearchCmmMkt.aspx",
    DataSource.IMD: "https://mausam.imd.gov.in/",
    DataSource.DATA_GOV: "https://data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi",
    DataSource.PM_KISAN: "https://pmkisan.gov.in/",
    DataSource.RURAL_VOICE: "https://ruralvoice.in/rss/latest-posts",
    DataSource.AI_KOSHA: "https://indiaai.gov.in/ai-kosha",
}
