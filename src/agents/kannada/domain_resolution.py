"""Shared Kannada domain alias resolution."""

from __future__ import annotations

ALIASES = {
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


def resolve_domain_name(domain_name: str | None) -> str:
    """Map agent or feature names to the shared Kannada domain buckets."""
    domain = domain_name.lower() if domain_name else "general"
    for resolved, patterns in ALIASES.items():
        if any(pattern in domain for pattern in patterns):
            return resolved
    return "general"
