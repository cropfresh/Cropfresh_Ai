"""
Entity extraction utilities for AgentStateManager.

Provides regex-based entity extraction for agricultural terms
(commodities, quantities, districts, prices) without LLM calls.
"""

import re
from typing import Any

# Compiled once at module level for efficiency
_ENTITY_PATTERNS: dict[str, Any] = {}


def _get_entity_patterns() -> dict:
    """Lazy-compile entity regex patterns (module-level cache)."""
    global _ENTITY_PATTERNS
    if not _ENTITY_PATTERNS:
        _ENTITY_PATTERNS = {
            "commodity": re.compile(
                r"\b(tomato|tamatar|potato|aloo|alugedde|onion|pyaaz|eerulli|"
                r"carrot|gajjari|okra|bhindi|bendekai|cauliflower|gobhi|"
                r"beans|hurali|brinjal|cucumber|chilli|mirchi|capsicum|cabbage)\b",
                re.IGNORECASE,
            ),
            "quantity_kg": re.compile(
                r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilogram)", re.IGNORECASE
            ),
            "quantity_quintal": re.compile(
                r"(\d+(?:\.\d+)?)\s*(?:quintal|quintals|q\b)", re.IGNORECASE
            ),
            "district": re.compile(
                r"\b(kolar|tumkur|tumakuru|hassan|mysuru|mysore|belagavi|belgaum|"
                r"hubli|dharwad|gadag|bidar|raichur|bagalkot|mandya|shimoga|"
                r"shivamogga|davangere|chitradurga|chikkaballapur|bangalore|bengaluru|"
                r"udupi|mangalore|ballari|bellary)\b",
                re.IGNORECASE,
            ),
            "price_per_kg": re.compile(
                r"₹\s*(\d+(?:\.\d+)?)\s*/?\s*kg", re.IGNORECASE
            ),
        }
    return _ENTITY_PATTERNS


# Commodity alias → canonical name
_COMMODITY_MAP = {
    "tamatar": "tomato", "aloo": "potato", "alugedde": "potato",
    "pyaaz": "onion", "eerulli": "onion", "gajjari": "carrot",
    "bhindi": "okra", "bendekai": "okra", "gobhi": "cauliflower",
    "hurali": "beans", "mirchi": "chilli",
}


def extract_entities(text: str) -> dict[str, Any]:
    """
    Extract agricultural entities from text.

    Returns dict of found entity key/value pairs (commodity, quantity_kg,
    district, price_per_kg etc.)
    """
    patterns = _get_entity_patterns()
    found: dict[str, Any] = {}

    commodity_match = patterns["commodity"].search(text)
    if commodity_match:
        raw_term = commodity_match.group(0).lower()
        canonical = _COMMODITY_MAP.get(raw_term, raw_term).capitalize()
        found["commodity"] = canonical

    qty_kg = patterns["quantity_kg"].search(text)
    if qty_kg:
        found["quantity_kg"] = float(qty_kg.group(1))

    qty_q = patterns["quantity_quintal"].search(text)
    if qty_q:
        found["quantity_quintal"] = float(qty_q.group(1))
        found["quantity_kg"] = found["quantity_quintal"] * 100.0

    district_match = patterns["district"].search(text)
    if district_match:
        found["district"] = district_match.group(0).title()

    price_match = patterns["price_per_kg"].search(text)
    if price_match:
        found["price_per_kg"] = float(price_match.group(1))

    return found
