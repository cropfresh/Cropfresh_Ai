"""Tests for structured Kannada retrieval and runtime-context enrichment."""

from src.agents.kannada.retriever import (
    enrich_runtime_context,
    retrieve_dialect_lexicon,
    retrieve_domain_context,
)


def test_retrieve_dialect_lexicon_matches_query_and_location():
    entries = retrieve_dialect_lexicon(
        "ivattu market sakkath jamagitta",
        {"user_profile": {"district": "Dharwad"}},
    )
    slangs = {entry["slang"] for entry in entries}
    assert "sakkath" in slangs
    assert "jamagitta" in slangs


def test_retrieve_domain_context_matches_domain_crop_and_query():
    entries = retrieve_domain_context(
        "price_prediction",
        query="onion price forecast for next week, should I hold?",
        context={"entities": {"commodity": "onion"}, "user_profile": {"district": "Dharwad"}},
    )
    assert entries
    assert any(entry["type"] == "forecast_playbook" for entry in entries)
    assert any("storage cost" in entry["details"] for entry in entries)


def test_enrich_runtime_context_merges_without_duplicating_existing_entries():
    context = enrich_runtime_context(
        "buyer_matching",
        context={
            "user_profile": {"district": "Kalaburagi"},
            "dialect_lexicon": [
                {
                    "dialect_tag": "HYDERABAD_KARNATAKA",
                    "slang": "hisaab",
                    "normalized_kannada": "ಲೆಕ್ಕ / ಖಾತೆ ವಿವರ",
                }
            ],
        },
        query="payment hisaab clear madsi",
    )
    matches = [entry for entry in context["dialect_lexicon"] if entry["slang"] == "hisaab"]
    assert len(matches) == 1
    assert context["kannada_context_info"]
