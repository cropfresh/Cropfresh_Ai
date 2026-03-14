"""
Price Prediction Models
=======================
Data structures for price prediction agent.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class PricePrediction(BaseModel):
    """Complete price prediction result for a commodity."""
    commodity: str
    district: str
    current_price: float                # ₹/kg — latest observed
    predicted_price_7d: float           # ₹/kg — 7-day forward estimate
    predicted_price_30d: float          # ₹/kg — 30-day forward estimate (trend-adjusted)
    confidence: float                   # 0.0–1.0
    trend: str                          # 'rising' | 'falling' | 'stable'
    trend_strength: float               # 0.0–1.0
    seasonal_factor: str                # 'peak_harvest' | 'off_season' | 'normal'
    factors: list[str] = Field(default_factory=list)   # human-readable drivers
    recommendation: str                 # 'sell_now' | 'hold_3d' | 'hold_7d' | 'hold_30d'
    data_source: str                    # 'historical' | 'model' | 'llm_estimate'
    predicted_at: datetime = Field(default_factory=datetime.now)
