"""
Price record models for the price intelligence pipeline.
"""
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, Field


class RawPriceRecord(BaseModel):
    """Raw record scraped from a source connector."""
    source: str
    raw_data: dict
    url: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.now)


class NormalizedPriceRecord(BaseModel):
    """Canonical price record for aggregation and querying."""
    commodity: str
    market: str
    price_date: date
    source: str
    
    variety: Optional[str] = None
    state: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: Optional[float] = None
    unit: Optional[str] = None
    
    raw_record_id: Optional[str] = None
    id: Optional[str] = None
    created_at: Optional[datetime] = None
