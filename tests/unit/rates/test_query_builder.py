from datetime import date

from src.rates.enums import RateKind
from src.rates.query_builder import build_rate_cache_key, normalize_rate_kinds, normalize_rate_query


def test_normalize_rate_kinds_dedupes_and_preserves_order() -> None:
    kinds = normalize_rate_kinds(["fuel", RateKind.GOLD, "fuel", "gold"])
    assert kinds == [RateKind.FUEL, RateKind.GOLD]


def test_normalize_rate_query_sets_defaults() -> None:
    query = normalize_rate_query(rate_kinds=["fuel"])
    assert query.state == "Karnataka"
    assert query.rate_kinds == [RateKind.FUEL]
    assert query.date == date.today()


def test_build_rate_cache_key_changes_with_source() -> None:
    query = normalize_rate_query(rate_kinds=["fuel"], market="Bengaluru")
    key_without_source = build_rate_cache_key(query)
    key_with_source = build_rate_cache_key(query, source="petroldieselprice")
    assert key_without_source != key_with_source
