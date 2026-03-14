"""
Agmarknet source connector built on the `ScraplingBaseScraper`.
"""
from typing import Optional, List, Any
import time
from loguru import logger
from datetime import datetime
import httpx
from src.scrapers.base_scraper import ScraplingBaseScraper, FetcherType, ScrapeResult

from src.scrapers.agmarknet.parser import AgmarknetParser


class AgmarknetScraper(ScraplingBaseScraper):
    """
    Scraper for official Agmarknet portal.
    Uses httpx to perform direct JSON API requests, bypassing the React SPA.
    """
    
    name: str = "agmarknet"
    base_url: str = "https://api.agmarknet.gov.in/v1"
    fetcher_type: FetcherType = FetcherType.BASIC
    # Polite settings for government portals
    rate_limit_delay: float = 2.0 
    
    _filters_cache: Optional[dict[str, Any]] = None
    
    async def _get_filters(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if self._filters_cache is None:
            logger.info("Fetching taxonomy filters from Agmarknet API...")
            res = await client.get(f"{self.base_url}/daily-price-arrival/filters", headers={"Referer": "https://agmarknet.gov.in/"})
            res.raise_for_status()
            self._filters_cache = res.json().get("data", {})
        return self._filters_cache or {}
    
    async def scrape(self, state: str, commodity: str, date_from: Optional[str] = None, date_to: Optional[str] = None, district: Optional[str] = None, market: Optional[str] = None, **kwargs) -> ScrapeResult:
        """Executes a scrape against Agmarknet using the direct JSON API."""
        search_url = f"{self.base_url}/daily-price-arrival/report"
        logger.info(f"Scraping Agmarknet API for {commodity} in {state}")
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                filters = await self._get_filters(client)
                
                # Resolve names to IDs
                cmdt_id, group_id = None, None
                for c in filters.get("cmdt_data", []):
                    if c.get("cmdt_name", "").lower() == commodity.lower():
                        cmdt_id = c.get("cmdt_id")
                        group_id = c.get("cmdt_group_id")
                        break
                        
                state_id = None
                for s in filters.get("state_data", []):
                    if s.get("state_name", "").lower() == state.lower():
                        state_id = s.get("state_id")
                        break
                        
                if not cmdt_id or not state_id:
                    error_msg = f"Could not find ID for commodity '{commodity}' or state '{state}'"
                    logger.error(error_msg)
                    return self.build_result(url=search_url, data=[], duration_ms=(time.time() - start_time) * 1000, error=error_msg)
                
                # Resolve district ID if provided
                district_param = "[100001]"
                if district:
                    for d in filters.get("district_data", []):
                        if d.get("district_name") and str(d.get("state_id")) == str(state_id) and str(d.get("district_name")).lower() == district.lower():
                            district_param = f"[{d.get('id')}]"
                            break
                            
                today_str = datetime.now().strftime("%Y-%m-%d")
                
                params = {
                    "from_date": date_from or today_str,
                    "to_date": date_to or today_str,
                    "data_type": "100004", # Price
                    "group": str(group_id),
                    "commodity": str(cmdt_id),
                    "state": f"[{state_id}]",
                    "district": district_param, # using resolved district
                    "market": "[100002]",   # default all
                    "grade": "[100003]",    # default all
                    "variety": "[100007]",  # default all
                    "page": "1",
                    "limit": "500"          # high limit to avoid pagination if possible
                }
                
                logger.info(f"Fetching data from API: {search_url}")
                res = await client.get(search_url, params=params, headers={"Referer": "https://agmarknet.gov.in/"})
                res.raise_for_status()
                json_data = res.json()
                
            # Parse the JSON
            parsed_data = AgmarknetParser.parse_json_response(json_data, search_url)
            
            # Decorate parsed data with context if missing 
            for rec in parsed_data:
                if rec.get("state") == "Unknown" and state:
                    rec["state"] = state
                if rec.get("commodity") == "Unknown" and commodity:
                    rec["commodity"] = commodity
                    
            duration_ms = (time.time() - start_time) * 1000
            
            return self.build_result(
                url=str(res.url),
                data=parsed_data,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to scrape Agmarknet API: {e}")
            return self.build_result(url=search_url, data=[], duration_ms=duration_ms, error=str(e))


def get_agmarknet_scraper() -> AgmarknetScraper:
    """Return a ready-to-use AgmarknetScraper (no-arg factory)."""
    return AgmarknetScraper()
