"""Legacy Agmarknet scraper compatibility helpers."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.scrapers.agri_scrapers.models import MandiPrice

KARNATAKA_MANDIS = [
    "Bangalore",
    "Mysore",
    "Hubli",
    "Kolar",
    "Belgaum",
    "Mangalore",
    "Davanagere",
    "Tumkur",
]

TARGET_COMMODITIES = [
    "Tomato",
    "Onion",
    "Potato",
    "Cabbage",
    "Cauliflower",
    "Beans",
    "Carrot",
    "Capsicum",
    "Chilli",
    "Brinjal",
]

_BASE_PRICES = {
    "tomato": 2400.0,
    "onion": 2100.0,
    "potato": 1800.0,
    "cabbage": 1200.0,
    "cauliflower": 1700.0,
    "beans": 3200.0,
    "carrot": 2600.0,
    "capsicum": 3600.0,
    "chilli": 4200.0,
    "brinjal": 1900.0,
}


def build_dev_prices(commodity: str, state: str = "Karnataka") -> list[MandiPrice]:
    """Return deterministic fallback mandi prices across major Karnataka mandis."""
    commodity_name = commodity or "Tomato"
    base_price = _BASE_PRICES.get(commodity_name.lower(), 2200.0)
    prices: list[MandiPrice] = []
    for index, mandi in enumerate(KARNATAKA_MANDIS):
        modal_price = base_price + (index * 75.0)
        prices.append(
            MandiPrice(
                commodity=commodity_name,
                mandi=mandi,
                district=mandi,
                state=state,
                min_price=round(modal_price * 0.8, 2),
                max_price=round(modal_price * 1.2, 2),
                modal_price=round(modal_price, 2),
                date=date.today(),
                source="agmarknet",
            )
        )
    return prices


def build_prices_from_rows(
    rows: list[dict[str, Any]],
    *,
    commodity: str,
    state: str,
) -> list[MandiPrice]:
    """Convert scraper rows into the legacy ``MandiPrice`` model."""
    prices: list[MandiPrice] = []
    for row in rows:
        mandi = row.get("market") or row.get("district") or "Unknown"
        district = row.get("district") or mandi
        modal_price = row.get("modal_price") or row.get("price_value")
        if modal_price is None:
            continue
        prices.append(
            MandiPrice(
                commodity=row.get("commodity") or commodity,
                variety=row.get("variety"),
                mandi=mandi,
                district=district,
                state=row.get("state") or state,
                min_price=row.get("min_price"),
                max_price=row.get("max_price"),
                modal_price=float(modal_price),
                unit=row.get("unit") or "Rs/Quintal",
                date=date.today(),
                source=row.get("source") or "agmarknet",
            )
        )
    return prices
