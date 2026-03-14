"""
AI Kosha Models
===============
Data structures and enumerations for the AI Kosha client.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AIKoshaCategory(str, Enum):
    """AI Kosha dataset categories relevant to agriculture."""
    AGRICULTURE = "Agriculture, Forestry and Rural Development"
    AQUACULTURE = "Aquaculture, Livestock and Fisheries"
    ENVIRONMENT = "Environment and Climate"
    HEALTH = "Health and Nutrition"
    SATELLITE = "Satellite and Remote Sensing"
    METEOROLOGY = "Meteorology and Weather"


class AIKoshaDataset(BaseModel):
    """A dataset available on AI Kosha."""
    id: str
    title: str
    description: str = ""
    category: str = ""
    source_organization: str = ""
    format: str = ""  # CSV, JSON, Parquet, etc.
    record_count: Optional[int] = None
    last_updated: Optional[datetime] = None
    download_url: Optional[str] = None
    api_url: Optional[str] = None
    license: str = ""
    ai_readiness_score: Optional[float] = None
    tags: list[str] = Field(default_factory=list)


class AIKoshaSearchResult(BaseModel):
    """Search results from AI Kosha."""
    total_results: int = 0
    page: int = 1
    per_page: int = 20
    datasets: list[AIKoshaDataset] = Field(default_factory=list)
    query: str = ""
    filters: dict[str, str] = Field(default_factory=dict)
