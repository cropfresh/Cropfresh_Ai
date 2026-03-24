from datetime import date, datetime, timedelta

from src.rates.comparison import compare_records
from src.rates.enums import AuthorityTier, RateKind
from src.rates.models import NormalizedRateRecord
from src.rates.query_builder import normalize_rate_query

NOW = datetime(2026, 3, 17, 10, 0, 0)


def _record(
    source: str,
    authority_tier: AuthorityTier,
    price_value: float,
    *,
    unit: str = "INR/quintal",
    price_date: date = date(2026, 3, 17),
    fetched_at: datetime = NOW,
) -> NormalizedRateRecord:
    return NormalizedRateRecord(
        rate_kind=RateKind.MANDI_WHOLESALE,
        commodity="tomato",
        state="Karnataka",
        district="Kolar",
        market="Kolar",
        location_label="Kolar",
        price_date=price_date,
        unit=unit,
        price_value=price_value,
        modal_price=price_value,
        source=source,
        authority_tier=authority_tier,
        source_url=f"https://example.com/{source}",
        fetched_at=fetched_at,
    )


def test_compare_records_prefers_official_source_precedence() -> None:
    query = normalize_rate_query(
        rate_kinds=["mandi_wholesale"],
        commodity="tomato",
        market="Kolar",
        date=NOW.date(),
    )
    records = [
        _record("agmarknet_ogd", AuthorityTier.OFFICIAL, 2200.0),
        _record("krama_daily", AuthorityTier.OFFICIAL, 2100.0, fetched_at=NOW - timedelta(minutes=5)),
    ]

    canonical_rates, comparison_quotes, warnings = compare_records(records, query)

    assert canonical_rates[0].source == "krama_daily"
    assert canonical_rates[0].comparison_count == 2
    assert len(comparison_quotes) == 2
    assert warnings == []


def test_compare_records_emits_data_quality_warnings() -> None:
    query = normalize_rate_query(
        rate_kinds=["mandi_wholesale"],
        commodity="tomato",
        market="Kolar",
        date=date(2026, 3, 17),
    )
    records = [
        _record(
            "napanta",
            AuthorityTier.VALIDATOR,
            1000.0,
            unit="INR/kg",
            price_date=date(2026, 3, 10),
        ),
        _record(
            "agriplus",
            AuthorityTier.VALIDATOR,
            1500.0,
            unit="INR/quintal",
            price_date=date(2026, 3, 17),
            fetched_at=NOW + timedelta(minutes=1),
        ),
    ]

    _, _, warnings = compare_records(records, query)

    assert any("Unit mismatch" in warning for warning in warnings)
    assert any("Date mismatch" in warning for warning in warnings)
    assert any("Large source discrepancy" in warning for warning in warnings)
    assert any("No official source" in warning for warning in warnings)
    assert any("older than the freshness target" in warning for warning in warnings)
