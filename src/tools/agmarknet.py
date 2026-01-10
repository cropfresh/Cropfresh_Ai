"""
Agmarknet Tool
==============
Fetches current wholesale prices from Indian agricultural markets (mandis).

API Sources:
- Primary: data.gov.in OGD Platform
- Backup: CEDA API (Ashoka University)
"""

from datetime import datetime
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel


class AgmarknetPrice(BaseModel):
    """Price data from Agmarknet."""
    
    commodity: str
    state: str
    district: str
    market: str
    variety: str = ""
    date: datetime
    min_price: float  # ₹/quintal
    max_price: float  # ₹/quintal
    modal_price: float  # ₹/quintal
    unit: str = "quintal"
    
    @property
    def modal_price_per_kg(self) -> float:
        """Convert quintal price to per-kg."""
        return self.modal_price / 100


class AgmarknetTool:
    """
    Agmarknet API Integration.
    
    Fetches real-time and historical prices from Indian mandis.
    
    Usage:
        tool = AgmarknetTool(api_key="your_key")
        prices = await tool.get_prices("Tomato", "Karnataka", "Kolar")
    """
    
    # OGD Platform API
    BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    
    # CEDA Backup API
    CEDA_URL = "https://ceda.ashoka.edu.in/api/agmarknet/prices"
    
    def __init__(self, api_key: str = "", cache_ttl: int = 900):
        """
        Initialize Agmarknet tool.
        
        Args:
            api_key: data.gov.in API key
            cache_ttl: Cache TTL in seconds (default 15 min)
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self._cache: dict = {}
    
    async def get_prices(
        self,
        commodity: str,
        state: str,
        district: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 20,
    ) -> list[AgmarknetPrice]:
        """
        Fetch current prices from Agmarknet.
        
        Args:
            commodity: Crop name (e.g., "Tomato", "Potato", "Onion")
            state: Indian state (e.g., "Karnataka", "Maharashtra")
            district: Optional district filter
            market: Optional specific mandi name
            limit: Max results to return
            
        Returns:
            List of AgmarknetPrice objects
        """
        cache_key = f"{commodity}:{state}:{district}:{market}"
        
        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"Returning cached prices for {cache_key}")
                return cached_data
        
        # Build query params
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "filters[commodity]": commodity,
            "filters[state]": state,
        }
        
        if district:
            params["filters[district]"] = district
        if market:
            params["filters[market]"] = market
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.warning(f"Primary API failed: {e}, trying CEDA backup")
            return await self._fallback_ceda(commodity, state, district, limit)
        
        # Parse results
        prices = []
        for record in data.get("records", []):
            try:
                prices.append(AgmarknetPrice(
                    commodity=record.get("commodity", commodity),
                    state=record.get("state", state),
                    district=record.get("district", district or ""),
                    market=record.get("market", ""),
                    variety=record.get("variety", ""),
                    date=datetime.strptime(
                        record.get("arrival_date", datetime.now().strftime("%d/%m/%Y")),
                        "%d/%m/%Y"
                    ),
                    min_price=float(record.get("min_price", 0)),
                    max_price=float(record.get("max_price", 0)),
                    modal_price=float(record.get("modal_price", 0)),
                ))
            except Exception as e:
                logger.debug(f"Error parsing record: {e}")
                continue
        
        # Update cache
        self._cache[cache_key] = (datetime.now(), prices)
        
        return prices
    
    async def _fallback_ceda(
        self,
        commodity: str,
        state: str,
        district: Optional[str],
        limit: int,
    ) -> list[AgmarknetPrice]:
        """Fallback to CEDA API if primary fails."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    self.CEDA_URL,
                    params={
                        "commodity": commodity,
                        "state": state,
                        "district": district or "",
                        "limit": limit,
                    }
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error(f"CEDA API also failed: {e}")
            return []
        
        # Parse CEDA format (may differ slightly)
        prices = []
        for record in data.get("data", []):
            try:
                prices.append(AgmarknetPrice(
                    commodity=record.get("commodity", commodity),
                    state=record.get("state", state),
                    district=record.get("district", ""),
                    market=record.get("market", ""),
                    date=datetime.now(),
                    min_price=float(record.get("min_price", 0)),
                    max_price=float(record.get("max_price", 0)),
                    modal_price=float(record.get("modal_price", 0)),
                ))
            except Exception:
                continue
        
        return prices
    
    def get_mock_prices(
        self,
        commodity: str,
        state: str = "Karnataka",
        district: str = "Kolar",
    ) -> list[AgmarknetPrice]:
        """
        Return mock prices for testing without API key.
        
        This provides realistic sample data for development.
        """
        mock_data = {
            "Tomato": {"min": 1500, "max": 3500, "modal": 2500},
            "Potato": {"min": 1200, "max": 2000, "modal": 1600},
            "Onion": {"min": 1800, "max": 3000, "modal": 2400},
            "Carrot": {"min": 2000, "max": 3500, "modal": 2800},
            "Capsicum": {"min": 3000, "max": 5000, "modal": 4000},
            "Beans": {"min": 3500, "max": 5500, "modal": 4500},
            "Cabbage": {"min": 800, "max": 1500, "modal": 1100},
            "Cauliflower": {"min": 1200, "max": 2500, "modal": 1800},
        }
        
        prices = mock_data.get(commodity, {"min": 2000, "max": 4000, "modal": 3000})
        
        return [
            AgmarknetPrice(
                commodity=commodity,
                state=state,
                district=district,
                market=f"{district} Main Market",
                date=datetime.now(),
                min_price=prices["min"],
                max_price=prices["max"],
                modal_price=prices["modal"],
            )
        ]
