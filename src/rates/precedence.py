"""Source precedence, TTLs, and discrepancy thresholds."""

from __future__ import annotations

from src.rates.enums import AuthorityTier, RateKind

DISCREPANCY_WARNING_THRESHOLD = 0.15
OFFICIAL_STALE_HOURS = 24
REFERENCE_STALE_HOURS = 72

DEFAULT_TTLS_MINUTES: dict[str, int] = {
    "krama_daily": 120,
    "agmarknet_ogd": 120,
    "agmarknet_scrape": 120,
    "enam_dashboard": 120,
    "napanta": 360,
    "agriplus": 360,
    "commoditymarketlive": 360,
    "shyali": 360,
    "vegetablemarketprice": 360,
    "todaypricerates": 360,
    "petroldieselprice": 60,
    "parkplus_fuel": 60,
    "businessline_gold": 60,
    "iifl_gold": 60,
    "krama_floor_price": 1440,
    "kapricom_reference": 1440,
}

SOURCE_PRECEDENCE: dict[RateKind, list[str]] = {
    RateKind.MANDI_WHOLESALE: [
        "krama_daily",
        "agmarknet_ogd",
        "agmarknet_scrape",
        "enam_dashboard",
        "napanta",
        "agriplus",
        "commoditymarketlive",
        "shyali",
    ],
    RateKind.RETAIL_PRODUCE: [
        "vegetablemarketprice",
        "todaypricerates",
    ],
    RateKind.FUEL: [
        "petroldieselprice",
        "parkplus_fuel",
    ],
    RateKind.GOLD: [
        "businessline_gold",
        "iifl_gold",
    ],
    RateKind.SUPPORT_PRICE: [
        "krama_floor_price",
        "kapricom_reference",
    ],
}

AUTHORITY_RANK: dict[AuthorityTier, int] = {
    AuthorityTier.OFFICIAL: 0,
    AuthorityTier.REFERENCE_OFFICIAL: 1,
    AuthorityTier.VALIDATOR: 2,
    AuthorityTier.RETAIL_REFERENCE: 3,
    AuthorityTier.PENDING_ACCESS: 4,
}


def source_rank(rate_kind: RateKind, source: str, authority_tier: AuthorityTier) -> tuple[int, int]:
    """Return a stable official-first sort key for a source."""
    ordered_sources = SOURCE_PRECEDENCE.get(rate_kind, [])
    return (
        ordered_sources.index(source) if source in ordered_sources else 999,
        AUTHORITY_RANK.get(authority_tier, 999),
    )
