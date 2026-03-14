"""
eNAM API Client Proxy
=====================
Proxy to avoid duplicating code between src/tools and src/scrapers.
Features are imported directly from src/scrapers/enam_client.
"""

from src.scrapers.enam_client import (
    PriceTrendDirection,
    MandiPrice,
    PriceTrend,
    MarketSummary,
    ENAMClient,
    get_enam_client,
)

__all__ = [
    "PriceTrendDirection",
    "MandiPrice",
    "PriceTrend",
    "MarketSummary",
    "ENAMClient",
    "get_enam_client",
]
