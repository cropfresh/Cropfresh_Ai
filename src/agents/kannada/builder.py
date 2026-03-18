"""Dynamic Kannada context assembly for shared agent prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.kannada.adcl_terms import ADCL_TERMS
from src.agents.kannada.administrative_terms import ADMIN_TERMS
from src.agents.kannada.agronomy_terms import AGRONOMY_TERMS
from src.agents.kannada.crop_varieties import CROP_VARIETIES_TERMS
from src.agents.kannada.dialect_context import build_dialect_context
from src.agents.kannada.equipment_terms import EQUIPMENT_TERMS
from src.agents.kannada.financial_terms import FINANCIAL_TERMS
from src.agents.kannada.guidelines import KANNADA_GUIDELINES
from src.agents.kannada.listing_terms import LISTING_TERMS
from src.agents.kannada.logistics_terms import LOGISTICS_TERMS
from src.agents.kannada.market_terms import MARKET_TERMS
from src.agents.kannada.matching_terms import MATCHING_TERMS
from src.agents.kannada.platform_terms import PLATFORM_TERMS
from src.agents.kannada.price_prediction_terms import PRICE_PREDICTION_TERMS
from src.agents.kannada.quality_terms import QUALITY_TERMS
from src.agents.kannada.runtime_context import build_runtime_context_blocks
from src.agents.kannada.soil_terms import SOIL_TERMS
from src.agents.kannada.weather_terms import WEATHER_TERMS


def get_kannada_context(
    domain_name: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Assemble Kannada behavior, dialect hints, and domain vocabulary."""
    domain = _resolve_domain(domain_name)
    domain_parts = {
        "agronomy": [
            AGRONOMY_TERMS,
            CROP_VARIETIES_TERMS,
            EQUIPMENT_TERMS,
            WEATHER_TERMS,
            SOIL_TERMS,
        ],
        "commerce": [MARKET_TERMS, FINANCIAL_TERMS, ADMIN_TERMS],
        "platform": [PLATFORM_TERMS, FINANCIAL_TERMS, ADMIN_TERMS],
        "crop_listing": [LISTING_TERMS, MARKET_TERMS, PLATFORM_TERMS],
        "buyer_matching": [MATCHING_TERMS, MARKET_TERMS, FINANCIAL_TERMS],
        "quality_assessment": [QUALITY_TERMS, PLATFORM_TERMS],
        "logistics": [LOGISTICS_TERMS, MARKET_TERMS],
        "adcl": [ADCL_TERMS, MARKET_TERMS, AGRONOMY_TERMS, WEATHER_TERMS],
        "price_prediction": [PRICE_PREDICTION_TERMS, MARKET_TERMS, WEATHER_TERMS],
        "general": [PLATFORM_TERMS, MARKET_TERMS, AGRONOMY_TERMS],
    }
    parts = [
        KANNADA_GUIDELINES,
        *domain_parts[domain],
        build_dialect_context(context),
        *build_runtime_context_blocks(context),
    ]
    return "\n\n".join(part for part in parts if part)


def _resolve_domain(domain_name: str | None) -> str:
    domain = domain_name.lower() if domain_name else "general"
    aliases = {
        "agronomy": ("agronomy",),
        "commerce": ("commerce", "pricing", "market"),
        "platform": ("platform", "support", "register"),
        "crop_listing": ("crop_listing", "listing"),
        "buyer_matching": ("buyer_matching", "matching", "buyer_match"),
        "quality_assessment": ("quality_assessment", "quality"),
        "logistics": ("logistics", "delivery", "transport"),
        "adcl": ("adcl", "recommend", "sow"),
        "price_prediction": ("price_prediction", "forecast", "trend"),
    }

    for resolved, patterns in aliases.items():
        if any(pattern in domain for pattern in patterns):
            return resolved
    return "general"
