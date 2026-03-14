"""
eNAM Mock Data Generation
"""

import random
from datetime import datetime
from typing import Optional

from .models import MandiPrice, PriceTrend, PriceTrendDirection


def get_mock_prices_data(
    commodity: str,
    state: str,
    district: Optional[str],
    limit: int,
) -> list[MandiPrice]:
    """Generate realistic mock price data."""
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


def get_mock_trend_data(
    commodity: str,
    state: str,
    market: Optional[str],
) -> PriceTrend:
    """Generate realistic mock trend data."""
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
