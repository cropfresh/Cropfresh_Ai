"""Dynamic Kannada context assembly for shared agent prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.kannada.adcl_terms import ADCL_TERMS
from src.agents.kannada.administrative_terms import ADMIN_TERMS
from src.agents.kannada.agronomy_terms import AGRONOMY_TERMS
from src.agents.kannada.conversation_patterns import KANNADA_CONVERSATION_PATTERNS
from src.agents.kannada.crop_varieties import CROP_VARIETIES_TERMS
from src.agents.kannada.dialect_context import build_dialect_context
from src.agents.kannada.dialect_patterns import KANNADA_DIALECT_PATTERNS
from src.agents.kannada.domain_resolution import resolve_domain_name
from src.agents.kannada.equipment_terms import EQUIPMENT_TERMS
from src.agents.kannada.few_shot_examples import KANNADA_FEW_SHOT_EXAMPLES
from src.agents.kannada.financial_terms import FINANCIAL_TERMS
from src.agents.kannada.guidelines import KANNADA_GUIDELINES
from src.agents.kannada.listing_terms import LISTING_TERMS
from src.agents.kannada.logistics_terms import LOGISTICS_TERMS
from src.agents.kannada.market_terms import MARKET_TERMS
from src.agents.kannada.matching_terms import MATCHING_TERMS
from src.agents.kannada.platform_terms import PLATFORM_TERMS
from src.agents.kannada.price_prediction_terms import PRICE_PREDICTION_TERMS
from src.agents.kannada.quality_terms import QUALITY_TERMS
from src.agents.kannada.retriever import enrich_runtime_context
from src.agents.kannada.runtime_context import build_runtime_context_blocks
from src.agents.kannada.soil_terms import SOIL_TERMS
from src.agents.kannada.weather_terms import WEATHER_TERMS

KANNADA_RUNTIME_RETRIEVAL = """## Kannada Runtime Retrieval
- Use the following retrieved Kannada lexicon and local-context hints only when they match this user query.
- Prefer these retrieved hints over generic wording when they improve clarity.
"""


def get_kannada_context(
    domain_name: str | None = None,
    context: Mapping[str, Any] | None = None,
    query: str = "",
    include_static: bool = True,
) -> str:
    """Assemble Kannada behavior, dialect hints, and domain vocabulary."""
    domain = resolve_domain_name(domain_name)
    runtime_context = enrich_runtime_context(domain, context=context, query=query)
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
    parts: list[str] = []
    if include_static:
        parts.extend(
            [
                KANNADA_GUIDELINES,
                KANNADA_DIALECT_PATTERNS,
                KANNADA_CONVERSATION_PATTERNS,
                KANNADA_FEW_SHOT_EXAMPLES,
                *domain_parts[domain],
                build_dialect_context(runtime_context),
            ]
        )

    runtime_blocks = build_runtime_context_blocks(runtime_context)
    if runtime_blocks and not include_static:
        parts.append(KANNADA_RUNTIME_RETRIEVAL)
    parts.extend(runtime_blocks)
    return "\n\n".join(part for part in parts if part)
