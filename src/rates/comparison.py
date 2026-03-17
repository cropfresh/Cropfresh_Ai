"""Comparison helpers for official-first rate selection."""

from __future__ import annotations

from collections import defaultdict

from src.rates.models import CanonicalRate, NormalizedRateRecord, RateQuery, SourceQuote
from src.rates.precedence import (
    DISCREPANCY_WARNING_THRESHOLD,
    OFFICIAL_STALE_HOURS,
    REFERENCE_STALE_HOURS,
    source_rank,
)


def _group_key(record: NormalizedRateRecord) -> tuple[str, str, str]:
    return (record.rate_kind.value, record.commodity or "", record.location_label.lower())


def _price_value(record: NormalizedRateRecord) -> float | None:
    return record.price_value or record.modal_price or record.max_price or record.min_price


def _freshness_warning(record: NormalizedRateRecord, query: RateQuery) -> str | None:
    delta_days = (query.target_date - record.price_date).days
    if delta_days <= 0:
        return None
    max_hours = OFFICIAL_STALE_HOURS if record.rate_kind.value != "support_price" else REFERENCE_STALE_HOURS
    if delta_days * 24 >= max_hours:
        return f"{record.source} is older than the freshness target for {record.rate_kind.value}."
    return None


def _quote(record: NormalizedRateRecord) -> SourceQuote:
    return SourceQuote(
        rate_kind=record.rate_kind,
        source=record.source,
        authority_tier=record.authority_tier,
        commodity=record.commodity,
        location_label=record.location_label,
        price_date=record.price_date,
        unit=record.unit,
        currency=record.currency,
        price_value=record.price_value,
        min_price=record.min_price,
        max_price=record.max_price,
        modal_price=record.modal_price,
        freshness=record.freshness,
        source_url=record.source_url,
        fetched_at=record.fetched_at,
    )


def compare_records(records: list[NormalizedRateRecord], query: RateQuery) -> tuple[list[CanonicalRate], list[SourceQuote], list[str]]:
    """Select canonical records and build side-by-side comparison quotes."""
    grouped: dict[tuple[str, str, str], list[NormalizedRateRecord]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)

    canonical_rates: list[CanonicalRate] = []
    comparison_quotes: list[SourceQuote] = []
    warnings: list[str] = []

    for group_records in grouped.values():
        ordered = sorted(
            group_records,
            key=lambda record: (
                source_rank(record.rate_kind, record.source, record.authority_tier),
                -record.fetched_at.timestamp(),
            ),
        )
        best = ordered[0]
        quotes = [_quote(record) for record in ordered]
        comparison_quotes.extend(quotes)
        canonical_rates.append(
            CanonicalRate(
                rate_kind=best.rate_kind,
                source=best.source,
                authority_tier=best.authority_tier,
                commodity=best.commodity,
                location_label=best.location_label,
                price_date=best.price_date,
                unit=best.unit,
                currency=best.currency,
                price_value=best.price_value,
                min_price=best.min_price,
                max_price=best.max_price,
                modal_price=best.modal_price,
                comparison_count=len(quotes),
                freshness=best.freshness,
            )
        )

        price_values = [_price_value(record) for record in ordered if _price_value(record) is not None]
        if len({record.unit for record in ordered}) > 1:
            warnings.append(f"Unit mismatch detected for {best.rate_kind.value} at {best.location_label}.")
        if len({record.price_date for record in ordered}) > 1:
            warnings.append(f"Date mismatch detected for {best.rate_kind.value} at {best.location_label}.")
        if price_values:
            low, high = min(price_values), max(price_values)
            if low > 0 and ((high - low) / low) >= DISCREPANCY_WARNING_THRESHOLD:
                warnings.append(f"Large source discrepancy detected for {best.rate_kind.value} at {best.location_label}.")
        if not any(record.authority_tier.value in {"official", "reference_official"} for record in ordered):
            warnings.append(f"No official source was available for {best.rate_kind.value} at {best.location_label}.")
        freshness_warning = _freshness_warning(best, query)
        if freshness_warning:
            warnings.append(freshness_warning)

    if not canonical_rates:
        warnings.append("No rates were available for the requested filters.")
    return canonical_rates, comparison_quotes, list(dict.fromkeys(warnings))
