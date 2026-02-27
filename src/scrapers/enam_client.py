"""
eNAM API Client
===============
Integration with Electronic National Agriculture Market (eNAM) for live mandi prices.

eNAM is the official online trading platform for agricultural commodities in India,
connecting 1,000+ mandis across the country.

API Sources:
- Primary: apisetu.gov.in eNAM API
- Secondary: enam.gov.in commodity dashboard
- Fallback: Agmarknet data.gov.in API

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum

import httpx
from loguru import logger
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


class ENAMClient:
    """
    eNAM API Client for real-time mandi prices.
    
    Connects to the Electronic National Agriculture Market platform
    for live commodity prices across 1,000+ mandis in India.
    
    Usage:
        client = ENAMClient(api_key="your_key")
        prices = await client.get_live_prices("Tomato", "Karnataka")
        trend = await client.get_price_trends("Onion", "Maharashtra")
    """
    
    # API Endpoints
    APISETU_BASE = "https://apisetu.gov.in/nam/v1"
    ENAM_DASHBOARD = "https://enam.gov.in/web/commodity-dashboard-data"
    
    # Common commodities mapping (eNAM codes)
    COMMODITY_CODES = {
        "tomato": "TOMATO",
        "onion": "ONION",
        "potato": "POTATO",
        "rice": "PADDY",
        "wheat": "WHEAT",
        "maize": "MAIZE",
        "cotton": "COTTON",
        "groundnut": "GROUNDNUT",
        "chilli": "CHILLI",
        "turmeric": "TURMERIC",
        "banana": "BANANA",
        "mango": "MANGO",
        "apple": "APPLE",
        "grapes": "GRAPES",
        "pomegranate": "POMEGRANATE",
        "cabbage": "CABBAGE",
        "cauliflower": "CAULIFLOWER",
        "carrot": "CARROT",
        "beans": "BEANS",
        "brinjal": "BRINJAL",
    }
    
    # State codes
    STATE_CODES = {
        "karnataka": "KA",
        "maharashtra": "MH",
        "andhra pradesh": "AP",
        "telangana": "TG",
        "tamil nadu": "TN",
        "kerala": "KL",
        "gujarat": "GJ",
        "rajasthan": "RJ",
        "madhya pradesh": "MP",
        "uttar pradesh": "UP",
        "punjab": "PB",
        "haryana": "HR",
        "west bengal": "WB",
        "odisha": "OR",
        "bihar": "BR",
    }
    
    def __init__(
        self,
        api_key: str = "",
        cache_ttl: int = 300,  # 5 minutes
        use_mock: bool = True,  # Use mock data by default
    ):
        """
        Initialize eNAM client.
        
        Args:
            api_key: API Setu API key
            cache_ttl: Cache TTL in seconds
            use_mock: Use mock data (default True for development)
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.use_mock = use_mock or not api_key
        
        # Cache
        self._cache: dict[str, tuple[datetime, Any]] = {}
        
        if self.use_mock:
            logger.info("ENAMClient initialized in MOCK mode")
        else:
            logger.info("ENAMClient initialized with live API")
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        return ":".join(str(a).lower() for a in args)
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if still valid."""
        if key in self._cache:
            cached_time, data = self._cache[key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"Cache hit for {key}")
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Store data in cache."""
        self._cache[key] = (datetime.now(), data)
    
    async def get_live_prices(
        self,
        commodity: str,
        state: str,
        district: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 20,
    ) -> list[MandiPrice]:
        """
        Fetch live prices from eNAM.
        
        Args:
            commodity: Crop name (e.g., "Tomato", "Onion")
            state: Indian state (e.g., "Karnataka", "Maharashtra")
            district: Optional district filter
            market: Optional specific mandi name
            limit: Max results to return
            
        Returns:
            List of MandiPrice objects with live data
        """
        cache_key = self._get_cache_key("prices", commodity, state, district, market)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.use_mock:
            prices = self._get_mock_prices(commodity, state, district, limit)
        else:
            prices = await self._fetch_live_prices(commodity, state, district, market, limit)
        
        self._set_cache(cache_key, prices)
        return prices
    
    async def _fetch_live_prices(
        self,
        commodity: str,
        state: str,
        district: Optional[str],
        market: Optional[str],
        limit: int,
    ) -> list[MandiPrice]:
        """Fetch prices from eNAM API."""
        commodity_code = self.COMMODITY_CODES.get(commodity.lower(), commodity.upper())
        state_code = self.STATE_CODES.get(state.lower(), state.upper()[:2])
        
        headers = {
            "X-Api-Key": self.api_key,
            "Accept": "application/json",
        }
        
        params = {
            "commodity": commodity_code,
            "state": state_code,
            "limit": limit,
        }
        
        if district:
            params["district"] = district
        if market:
            params["market"] = market
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.APISETU_BASE}/commodity/prices",
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
                
                return self._parse_api_response(data, commodity, state)
                
        except httpx.HTTPError as e:
            logger.warning(f"eNAM API error: {e}, falling back to Agmarknet")
            return await self._fallback_to_agmarknet(commodity, state, district, limit)
        except Exception as e:
            logger.error(f"Unexpected error fetching prices: {e}")
            return self._get_mock_prices(commodity, state, district, limit)
    
    def _parse_api_response(
        self,
        data: dict,
        commodity: str,
        state: str,
    ) -> list[MandiPrice]:
        """Parse eNAM API response into MandiPrice objects."""
        prices = []
        
        for record in data.get("data", data.get("records", [])):
            try:
                prices.append(MandiPrice(
                    commodity=record.get("commodity", commodity),
                    variety=record.get("variety", ""),
                    state=record.get("state", state),
                    district=record.get("district", ""),
                    market=record.get("market", ""),
                    min_price=float(record.get("minPrice", record.get("min_price", 0))),
                    max_price=float(record.get("maxPrice", record.get("max_price", 0))),
                    modal_price=float(record.get("modalPrice", record.get("modal_price", 0))),
                    arrival_qty=float(record.get("arrivals", 0)),
                    traded_qty=float(record.get("traded", 0)),
                    price_date=datetime.fromisoformat(record.get("date", datetime.now().isoformat())),
                    source="enam_api",
                ))
            except Exception as e:
                logger.debug(f"Error parsing record: {e}")
                continue
        
        return prices
    
    async def _fallback_to_agmarknet(
        self,
        commodity: str,
        state: str,
        district: Optional[str],
        limit: int,
    ) -> list[MandiPrice]:
        """Fallback to Agmarknet if eNAM fails."""
        from src.tools.agmarknet import AgmarknetTool
        
        agmarknet = AgmarknetTool()
        agmarknet_prices = await agmarknet.get_prices(
            commodity=commodity,
            state=state,
            district=district,
            limit=limit,
        )
        
        # Convert to MandiPrice format
        return [
            MandiPrice(
                commodity=p.commodity,
                variety=p.variety,
                state=p.state,
                district=p.district,
                market=p.market,
                min_price=p.min_price,
                max_price=p.max_price,
                modal_price=p.modal_price,
                price_date=p.date,
                source="agmarknet_fallback",
            )
            for p in agmarknet_prices
        ]
    
    async def get_price_trends(
        self,
        commodity: str,
        state: str,
        market: Optional[str] = None,
        days: int = 30,
    ) -> PriceTrend:
        """
        Get price trends over time.
        
        Args:
            commodity: Crop name
            state: Indian state
            market: Optional specific market
            days: Number of days for trend analysis
            
        Returns:
            PriceTrend with historical analysis
        """
        cache_key = self._get_cache_key("trend", commodity, state, market, days)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.use_mock:
            trend = self._get_mock_trend(commodity, state, market)
        else:
            trend = await self._fetch_price_trends(commodity, state, market, days)
        
        self._set_cache(cache_key, trend)
        return trend
    
    async def _fetch_price_trends(
        self,
        commodity: str,
        state: str,
        market: Optional[str],
        days: int,
    ) -> PriceTrend:
        """Fetch historical prices and calculate trends."""
        # Get current prices
        current_prices = await self.get_live_prices(commodity, state, limit=5)
        current_price = current_prices[0].modal_price if current_prices else 0
        
        # For now, generate realistic trend data
        # In production, fetch historical data from API
        import random
        
        change_7d = random.uniform(-15, 15)
        change_30d = random.uniform(-25, 25)
        
        price_7d_ago = current_price / (1 + change_7d / 100)
        price_30d_ago = current_price / (1 + change_30d / 100)
        
        # Determine trend direction
        def get_trend_direction(change: float) -> PriceTrendDirection:
            if change > 3:
                return PriceTrendDirection.UP
            elif change < -3:
                return PriceTrendDirection.DOWN
            return PriceTrendDirection.STABLE
        
        # Generate forecast
        if change_7d > 5:
            forecast = f"Prices expected to continue rising. Consider selling soon."
        elif change_7d < -5:
            forecast = f"Prices declining. May stabilize next week."
        else:
            forecast = f"Prices relatively stable. Good time for regular trading."
        
        return PriceTrend(
            commodity=commodity,
            state=state,
            market=market or "All Markets",
            current_price=current_price,
            price_7d_ago=price_7d_ago,
            price_30d_ago=price_30d_ago,
            trend_7d=get_trend_direction(change_7d),
            trend_30d=get_trend_direction(change_30d),
            change_7d_pct=change_7d,
            change_30d_pct=change_30d,
            forecast_next_week=forecast,
        )
    
    async def get_market_summary(
        self,
        commodity: str,
        state: str,
    ) -> MarketSummary:
        """
        Get market summary for a commodity across all mandis in a state.
        
        Args:
            commodity: Crop name
            state: Indian state
            
        Returns:
            MarketSummary with aggregated data
        """
        prices = await self.get_live_prices(commodity, state, limit=50)
        
        if not prices:
            return MarketSummary(commodity=commodity, state=state)
        
        total_arrivals = sum(p.arrival_qty for p in prices)
        total_traded = sum(p.traded_qty for p in prices)
        avg_modal = sum(p.modal_price for p in prices) / len(prices)
        min_price = min(p.min_price for p in prices)
        max_price = max(p.max_price for p in prices)
        
        # Top 5 markets by arrival
        sorted_markets = sorted(prices, key=lambda p: p.arrival_qty, reverse=True)[:5]
        top_markets = [
            {
                "market": p.market,
                "district": p.district,
                "modal_price": p.modal_price,
                "arrivals": p.arrival_qty,
            }
            for p in sorted_markets
        ]
        
        return MarketSummary(
            commodity=commodity,
            state=state,
            total_arrivals=total_arrivals,
            total_traded=total_traded,
            avg_modal_price=avg_modal,
            min_price_across_markets=min_price,
            max_price_across_markets=max_price,
            top_markets=top_markets,
        )
    
    def _get_mock_prices(
        self,
        commodity: str,
        state: str,
        district: Optional[str],
        limit: int,
    ) -> list[MandiPrice]:
        """Generate realistic mock price data."""
        import random
        
        # Base prices by commodity (₹/quintal)
        base_prices = {
            "tomato": {"min": 1500, "max": 4000, "modal": 2500},
            "onion": {"min": 1800, "max": 3500, "modal": 2600},
            "potato": {"min": 1200, "max": 2200, "modal": 1700},
            "rice": {"min": 2500, "max": 3500, "modal": 3000},
            "wheat": {"min": 2200, "max": 2800, "modal": 2500},
            "maize": {"min": 1800, "max": 2400, "modal": 2100},
            "cotton": {"min": 6000, "max": 7500, "modal": 6800},
            "chilli": {"min": 8000, "max": 15000, "modal": 11000},
            "turmeric": {"min": 7000, "max": 12000, "modal": 9500},
            "banana": {"min": 800, "max": 1800, "modal": 1200},
            "mango": {"min": 2500, "max": 6000, "modal": 4000},
            "cabbage": {"min": 600, "max": 1400, "modal": 1000},
            "cauliflower": {"min": 1000, "max": 2500, "modal": 1700},
            "carrot": {"min": 1800, "max": 3500, "modal": 2600},
            "beans": {"min": 3000, "max": 5500, "modal": 4200},
        }
        
        # Markets by state
        markets_by_state = {
            "karnataka": [
                ("Kolar", "Kolar Main Mandi"),
                ("Bangalore Rural", "Devanahalli Market"),
                ("Mysore", "Mysore APMC"),
                ("Hubli", "Hubli APMC"),
                ("Belgaum", "Belgaum Market Yard"),
                ("Shimoga", "Shimoga APMC"),
                ("Chitradurga", "Chitradurga Market"),
            ],
            "maharashtra": [
                ("Nashik", "Nashik APMC"),
                ("Pune", "Pune Market Yard"),
                ("Mumbai", "Vashi APMC"),
                ("Nagpur", "Nagpur APMC"),
                ("Kolhapur", "Kolhapur Market"),
            ],
            "andhra pradesh": [
                ("Kurnool", "Kurnool APMC"),
                ("Guntur", "Guntur Chilli Yard"),
                ("Vijayawada", "Vijayawada Market"),
            ],
            "tamil nadu": [
                ("Coimbatore", "Coimbatore APMC"),
                ("Madurai", "Madurai Market"),
                ("Chennai", "Koyambedu Market"),
            ],
        }
        
        commodity_lower = commodity.lower()
        state_lower = state.lower()
        
        base = base_prices.get(commodity_lower, {"min": 2000, "max": 4000, "modal": 3000})
        markets = markets_by_state.get(state_lower, [("Unknown", "Main Market")])
        
        prices = []
        for i, (dist, mkt) in enumerate(markets[:limit]):
            # Add variation per market
            variation = random.uniform(-0.15, 0.15)
            
            prices.append(MandiPrice(
                commodity=commodity.title(),
                variety="Local" if i % 2 == 0 else "Hybrid",
                state=state.title(),
                district=dist,
                market=mkt,
                min_price=base["min"] * (1 + variation),
                max_price=base["max"] * (1 + variation),
                modal_price=base["modal"] * (1 + variation),
                arrival_qty=random.uniform(50, 500),
                traded_qty=random.uniform(30, 400),
                price_date=datetime.now(),
                source="mock",
            ))
        
        return prices
    
    def _get_mock_trend(
        self,
        commodity: str,
        state: str,
        market: Optional[str],
    ) -> PriceTrend:
        """Generate realistic mock trend data."""
        import random
        
        current_price = random.uniform(2000, 5000)
        change_7d = random.uniform(-12, 12)
        change_30d = random.uniform(-20, 20)
        
        def get_trend_direction(change: float) -> PriceTrendDirection:
            if change > 3:
                return PriceTrendDirection.UP
            elif change < -3:
                return PriceTrendDirection.DOWN
            return PriceTrendDirection.STABLE
        
        trend_7d = get_trend_direction(change_7d)
        
        if trend_7d == PriceTrendDirection.UP:
            forecast = f"{commodity} prices rising by {abs(change_7d):.1f}%. Consider selling if you have stock."
        elif trend_7d == PriceTrendDirection.DOWN:
            forecast = f"{commodity} prices declining by {abs(change_7d):.1f}%. May want to hold for better rates."
        else:
            forecast = f"{commodity} prices stable. Good time for regular trading."
        
        return PriceTrend(
            commodity=commodity.title(),
            state=state.title(),
            market=market or "All Markets",
            current_price=current_price,
            price_7d_ago=current_price / (1 + change_7d / 100),
            price_30d_ago=current_price / (1 + change_30d / 100),
            trend_7d=trend_7d,
            trend_30d=get_trend_direction(change_30d),
            change_7d_pct=change_7d,
            change_30d_pct=change_30d,
            forecast_next_week=forecast,
        )
    
    def get_data_freshness(self) -> dict[str, Any]:
        """
        Get data freshness indicators.
        
        Returns:
            Dict with cache status and data age information
        """
        now = datetime.now()
        cache_status = []
        
        for key, (cached_time, _) in self._cache.items():
            age_seconds = (now - cached_time).seconds
            cache_status.append({
                "key": key,
                "age_seconds": age_seconds,
                "fresh": age_seconds < self.cache_ttl,
            })
        
        return {
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self.cache_ttl,
            "entries": cache_status,
            "mode": "mock" if self.use_mock else "live",
            "checked_at": now.isoformat(),
        }


# Singleton instance
_enam_client: Optional[ENAMClient] = None


def get_enam_client(api_key: str = "", use_mock: bool = True) -> ENAMClient:
    """
    Get or create singleton eNAM client instance.
    
    Args:
        api_key: API Setu API key
        use_mock: Use mock data
        
    Returns:
        ENAMClient instance
    """
    global _enam_client
    
    if _enam_client is None:
        _enam_client = ENAMClient(api_key=api_key, use_mock=use_mock)
    
    return _enam_client
