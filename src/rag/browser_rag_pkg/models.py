"""
Browser RAG data models.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScrapeIntent(str, Enum):
    """
    Intent category for browser scraping — determines which sources to target.
    """
    SCHEME_UPDATE    = "scheme_update"     # New government schemes
    DISEASE_ALERT    = "disease_alert"     # Crop disease/pest outbreaks
    PRICE_NEWS       = "price_news"        # Commodity price news
    PESTICIDE_BAN    = "pesticide_ban"     # Banned/restricted chemicals
    EXPORT_POLICY    = "export_policy"     # Export/import restrictions
    WEATHER_ADVISORY = "weather_advisory"  # Extreme weather advisories
    MARKET_NEWS      = "market_news"       # General agri-market news


class TargetSource(BaseModel):
    """A single target URL with scraping metadata."""
    url: str
    intent: ScrapeIntent
    source_name: str
    ttl_hours: float = Field(default=6.0, description="Cache TTL in hours")
    fetcher_type: str = Field(
        default="basic",
        description="'basic' for public pages, 'stealth' for bot-protected sites"
    )
    css_selector: Optional[str] = Field(
        default=None,
        description="Optional CSS selector to target specific content area"
    )


class Citation(BaseModel):
    """Source citation for a scraped answer."""
    source_url: str
    source_name: str
    scraped_at: str  # ISO 8601 datetime string
    excerpt: str = Field(description="Relevant 200-char excerpt from source")
    freshness_label: str = Field(
        default="Live",
        description="Human-readable freshness: 'Live', 'Today', '3 hours ago'"
    )


class CitedAnswer(BaseModel):
    """An answer with full source citations (used for browser-scraped answers)."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    has_live_data: bool = True
    freshness_label: str = "Live"
