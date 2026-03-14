"""
IMD Weather Scraper
===================
Scraper for IMD weather and agricultural advisories.
"""

import re
import time
from datetime import date
from typing import Any, Optional

from loguru import logger

from src.scrapers.base_scraper import FetcherType, ScrapeResult, ScraplingBaseScraper

from .models import WeatherData


class IMDWeatherScraper(ScraplingBaseScraper):
    """
    Scraper for IMD weather and agricultural advisories.

    URLs:
    - mausam.imd.gov.in - Weather forecasts
    - imdagrimet.gov.in - Agricultural advisories
    """
    name = "imd_weather"
    base_url = "https://mausam.imd.gov.in"
    fetcher_type = FetcherType.BASIC
    cache_ttl_seconds = 1800
    rate_limit_delay = 2.0

    WEATHER_URL = "https://mausam.imd.gov.in/"
    AGRIMET_URL = "https://imdagrimet.gov.in/"

    async def scrape(
        self,
        state: str = "Karnataka",
        district: Optional[str] = None,
        include_advisory: bool = False,
        **kwargs,
    ) -> ScrapeResult:
        """Get weather forecast for a location."""
        start_time = time.time()

        try:
            page = await self.fetch(self.WEATHER_URL)
            weather = self._parse_weather(page, state, district)

            advisory = None
            if include_advisory:
                try:
                    advisory_page = await self.fetch(self.AGRIMET_URL)
                    advisory = self._parse_advisory(advisory_page, state, district)
                    if advisory and weather:
                        weather[0].advisory = advisory
                except Exception as e:
                    logger.debug(f"Advisory fetch failed (non-critical): {e}")

            duration_ms = (time.time() - start_time) * 1000
            return self.build_result(
                url=self.WEATHER_URL,
                data=[w.model_dump() for w in weather],
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"IMD weather scraping failed: {e}")
            return self.build_result(
                url=self.WEATHER_URL,
                data=[],
                error=str(e),
                duration_ms=duration_ms,
            )

    def _parse_weather(
        self,
        page: Any,
        state: str,
        district: Optional[str],
    ) -> list[WeatherData]:
        weather: list[WeatherData] = []

        try:
            temp_el = page.css(".temperature::text").get()
            humidity_el = page.css(".humidity::text").get()
            condition_el = page.css(".weather-condition::text").get()

            if temp_el or humidity_el:
                weather.append(
                    WeatherData(
                        location=district or state,
                        district=district or "State-wide",
                        state=state,
                        temperature_celsius=self._extract_number(temp_el),
                        humidity_percent=self._extract_number(humidity_el),
                        weather_condition=condition_el,
                        forecast_date=date.today(),
                    )
                )
        except Exception as e:
            logger.debug(f"Weather element extraction failed: {e}")

        if not weather:
            logger.warning("No weather data parsed")

        return weather

    def _parse_advisory(
        self,
        page: Any,
        state: str,
        district: Optional[str],
    ) -> Optional[str]:
        try:
            advisory_el = page.css(".advisory-text::text").get()
            if advisory_el:
                return advisory_el.strip()

            advisory_el = page.find_by_text("advisory", first_match=True)
            if advisory_el:
                return advisory_el.get_all_text().strip()[:500]
        except Exception:
            pass
        return None

    def _extract_number(self, text: Optional[str]) -> Optional[float]:
        if not text:
            return None
        match = re.search(r"[\d.]+", text)
        return float(match.group()) if match else None
