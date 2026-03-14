"""
eNAM Client Models
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PriceTrendDirection(str, Enum):
    """Price trend direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class MandiPrice(BaseModel):
    """Live mandi price data from eNAM."""

    commodity: str
    variety: str = ""
    state: str
    district: str
    market: str

    # Prices in ₹/quintal
    min_price: float
    max_price: float
    modal_price: float

    # Quantity
    arrival_qty: float = 0.0  # in quintals
    traded_qty: float = 0.0  # in quintals

    # Timestamp
    price_date: datetime
    last_updated: datetime = Field(default_factory=datetime.now)

    # Source
    source: str = "enam"

    @property
    def modal_price_per_kg(self) -> float:
        """Convert quintal price to per-kg."""
        return self.modal_price / 100

    @property
    def price_range_str(self) -> str:
        """Format price range as string."""
        return f"₹{self.min_price:,.0f} - ₹{self.max_price:,.0f}/quintal"


class PriceTrend(BaseModel):
    """Price trend analysis over time."""

    commodity: str
    state: str
    market: str

    # Current price
    current_price: float

    # Historical prices
    price_7d_ago: float = 0.0
    price_30d_ago: float = 0.0

    # Trend analysis
    trend_7d: PriceTrendDirection = PriceTrendDirection.STABLE
    trend_30d: PriceTrendDirection = PriceTrendDirection.STABLE

    change_7d_pct: float = 0.0
    change_30d_pct: float = 0.0

    # Forecast
    forecast_next_week: str = ""

    # Analysis date
    analysis_date: datetime = Field(default_factory=datetime.now)


class MarketSummary(BaseModel):
    """Summary of market activity."""

    commodity: str
    state: str

    # Aggregated data
    total_arrivals: float = 0.0
    total_traded: float = 0.0
    avg_modal_price: float = 0.0
    min_price_across_markets: float = 0.0
    max_price_across_markets: float = 0.0

    # Top markets
    top_markets: list[dict] = Field(default_factory=list)

    # Date
    date: datetime = Field(default_factory=datetime.now)
