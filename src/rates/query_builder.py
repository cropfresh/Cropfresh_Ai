"""Helpers for query normalization and cache keys."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from typing import Iterable

from src.rates.enums import RateKind
from src.rates.models import RateQuery


def normalize_rate_kinds(rate_kinds: Iterable[str | RateKind]) -> list[RateKind]:
    """Normalize rate kind input while preserving order."""
    normalized: list[RateKind] = []
    seen: set[RateKind] = set()
    for item in rate_kinds:
        kind = item if isinstance(item, RateKind) else RateKind(str(item))
        if kind not in seen:
            normalized.append(kind)
            seen.add(kind)
    return normalized


def normalize_rate_query(**kwargs) -> RateQuery:
    """Build a validated RateQuery with normalized kind values."""
    kwargs["rate_kinds"] = normalize_rate_kinds(kwargs.get("rate_kinds", []))
    if kwargs.get("date") is None:
        kwargs["date"] = date.today()
    return RateQuery(**kwargs)


def build_location_label(state: str, district: str | None = None, market: str | None = None) -> str:
    """Return the most specific location label available."""
    return market or district or state


def build_rate_cache_key(query: RateQuery, source: str | None = None) -> str:
    """Stable cache key used by service-level caching."""
    parts = [
        ",".join(kind.value for kind in query.rate_kinds),
        query.commodity or "",
        query.state,
        query.district or "",
        query.market or "",
        query.target_date.isoformat(),
        str(query.include_reference),
        query.comparison_depth.value,
        str(query.force_live),
        source or "",
    ]
    return sha256("|".join(parts).encode("utf-8")).hexdigest()
