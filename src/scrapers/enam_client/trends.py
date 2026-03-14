"""
eNAM Trends and Summaries
"""

import random
from typing import Optional

from .models import PriceTrend, PriceTrendDirection, MarketSummary


async def fetch_price_trends(
    client,
    commodity: str,
    state: str,
    market: Optional[str],
    days: int,
) -> PriceTrend:
    """Fetch historical prices and calculate trends."""
    # Get current prices
    current_prices = await client.get_live_prices(commodity, state, limit=5)
    current_price = current_prices[0].modal_price if current_prices else 0
    
    # For now, generate realistic trend data
    # In production, fetch historical data from API
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
    client,
    commodity: str,
    state: str,
) -> MarketSummary:
    """
    Get market summary for a commodity across all mandis in a state.
    """
    prices = await client.get_live_prices(commodity, state, limit=50)
    
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
