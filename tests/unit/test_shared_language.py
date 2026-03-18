"""Unit tests for the shared language facade."""

from src.shared.language import (
    detect_response_language,
    ensure_language_context,
    normalize_language_code,
    resolve_language,
    split_session_context,
)


def test_normalize_language_code_accepts_names_and_locales():
    assert normalize_language_code("Kannada") == "kn"
    assert normalize_language_code("kn-IN") == "kn"
    assert normalize_language_code("english") == "en"


def test_detect_response_language_handles_transliterated_kannada():
    assert detect_response_language("Tomato bele yaavaga Kolar mandi alli?") == "kn"


def test_resolve_language_prefers_current_query_over_stored_profile():
    assert (
        resolve_language(
            query="ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?",
            context={"user_profile": {"language_pref": "en"}},
        )
        == "kn"
    )


def test_ensure_language_context_preserves_language_pref():
    context = ensure_language_context(
        {"user_profile": {"language_pref": "kannada"}},
        query="What is the tomato price?",
    )
    assert context["user_profile"]["language"] == "kn"
    assert context["user_profile"]["language_pref"] == "kn"


def test_split_session_context_routes_profile_fields_and_entities():
    user_profile, entities = split_session_context(
        {
            "language_pref": "kannada",
            "district": "Kolar",
            "farmer_id": "farmer-123",
            "commodity": "tomato",
        },
        query="Tomato bele yaavaga?",
    )
    assert user_profile["language"] == "kn"
    assert user_profile["language_pref"] == "kn"
    assert user_profile["district"] == "Kolar"
    assert entities["farmer_id"] == "farmer-123"
    assert entities["commodity"] == "tomato"
