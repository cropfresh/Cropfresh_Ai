"""Enums for the multi-source rate hub."""

from __future__ import annotations

from enum import Enum


class RateKind(str, Enum):
    """Supported daily rate categories."""

    MANDI_WHOLESALE = "mandi_wholesale"
    RETAIL_PRODUCE = "retail_produce"
    FUEL = "fuel"
    GOLD = "gold"
    SUPPORT_PRICE = "support_price"


class AuthorityTier(str, Enum):
    """Authority tier used for canonical selection."""

    OFFICIAL = "official"
    REFERENCE_OFFICIAL = "reference_official"
    VALIDATOR = "validator"
    RETAIL_REFERENCE = "retail_reference"
    PENDING_ACCESS = "pending_access"


class ComparisonDepth(str, Enum):
    """How many sources to include in comparison output."""

    OFFICIAL_ONLY = "official_only"
    OFFICIAL_PLUS_VALIDATORS = "official_plus_validators"
    ALL_SOURCES = "all_sources"


class FetchMode(str, Enum):
    """Execution mode for a source connector."""

    LIVE = "live"
    CACHED = "cached"
    PENDING = "pending"
