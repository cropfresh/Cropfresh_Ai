"""
Web Scraping Agent Models
=========================
Pydantic models for web scraping inputs and outputs.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ScrapingResult(BaseModel):
    """Result from web scraping operation."""
    url: str
    success: bool
    markdown: str = ""
    html: str = ""
    extracted_data: Any = Field(default_factory=dict)
    error: Optional[str] = None
    cached: bool = False
    scrape_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class ScrapingConfig(BaseModel):
    """Configuration for scraping operations."""
    timeout: int = 30000  # ms
    wait_for_selector: Optional[str] = None
    wait_for_load_state: str = "networkidle"
    screenshot: bool = False
    full_page_screenshot: bool = True
    stealth: bool = True
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
