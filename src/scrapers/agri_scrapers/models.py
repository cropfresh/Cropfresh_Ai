"""
Agricultural Data Models
========================
Pydantic schemas for structured agricultural data.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class MandiPrice(BaseModel):
    """Mandi (market) price data structure."""
    commodity: str
    variety: Optional[str] = None
    mandi: str
    district: Optional[str] = None
    state: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: float
    unit: str = "Rs/Quintal"
    date: date
    source: str = "agmarknet"

    @property
    def modal_price_per_kg(self) -> float:
        """Return the modal price normalized to INR/kg for legacy callers."""
        return self.modal_price / 100


class MandiPriceList(BaseModel):
    """Wrapper for array of Mandi prices returned by LLM extraction."""
    prices: list[MandiPrice]


class WeatherData(BaseModel):
    """Weather data structure."""
    location: str
    district: str
    state: str
    temperature_celsius: Optional[float] = None
    humidity_percent: Optional[float] = None
    rainfall_mm: Optional[float] = None
    weather_condition: Optional[str] = None
    forecast_date: date
    advisory: Optional[str] = None
    source: str = "IMD"


class GovScheme(BaseModel):
    """Government scheme information."""
    name: str
    department: str
    description: str
    eligibility: Optional[str] = None
    benefits: Optional[str] = None
    application_url: Optional[str] = None
    deadline: Optional[date] = None
    state: str = "All India"


class NewsArticle(BaseModel):
    """Agricultural news article."""
    title: str
    summary: Optional[str] = None
    url: str
    source: str
    published_date: Optional[datetime] = None
    category: Optional[str] = None
