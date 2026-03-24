"""Tests for centralized Kannada domain-context expansion."""

from src.agents.kannada import get_kannada_context


def test_crop_listing_domain_gets_listing_guidance():
    context = get_kannada_context("crop_listing")
    assert "Listing & Selling Flow" in context
    assert "Grama Kannada AI Sahayaka" in context
    assert "Useful Kannada Response Patterns" in context


def test_buyer_matching_domain_gets_matching_guidance():
    context = get_kannada_context("buyer_matching")
    assert "Buyer Matching Guidance" in context
    assert "code-mixed usage" in context
    assert "Kannada Dialect Bucket Guide" in context


def test_quality_domain_gets_quality_guidance():
    context = get_kannada_context("quality_assessment")
    assert "Quality & Grading Guidance" in context
    assert "Shelf Life" in context


def test_adcl_agent_name_resolves_to_crop_recommendation_guidance():
    context = get_kannada_context("adcl_agent")
    assert "Crop Recommendation & Weekly Demand Guidance" in context
    assert "step-by-step guidance" in context


def test_builder_adds_dialect_hint_from_profile_district():
    context = get_kannada_context(
        "general",
        {"user_profile": {"district": "Mysuru", "language": "kn"}},
    )
    assert "Likely dialect bucket: OLD_MYSURU_RURAL" in context
    assert "Never print the dialect bucket label to the user" in context


def test_builder_renders_dialect_lexicon_blocks():
    context = get_kannada_context(
        "buyer_matching",
        {
            "dialect_lexicon": [
                {
                    "dialect_tag": "NORTH_KA_RURAL",
                    "slang": "sakkath",
                    "normalized_kannada": "tumba chennagide",
                    "english_gloss": "really good",
                    "example_user_sentence": "ivattu market sakkath jamagitta",
                    "example_ai_reply": "indu market tumba jamagide anta arthha",
                }
            ]
        },
    )
    assert "[DIALECT_LEXICON: NORTH_KA_RURAL]" in context
    assert 'normalized_kannada = "tumba chennagide"' in context


def test_builder_renders_local_kannada_context_blocks():
    context = get_kannada_context(
        "agronomy",
        {
            "kannada_context_info": [
                {
                    "type": "crop_practice",
                    "crop": "ragi",
                    "region": "Old Mysuru",
                    "details": "Prefer lighter irrigation after initial establishment.",
                }
            ]
        },
    )
    assert "[CONTEXT_KANNADA_INFO]" in context
    assert 'crop = "ragi"' in context


def test_builder_includes_shared_kannada_few_shots():
    context = get_kannada_context("general")
    assert "Kannada Few-Shot Examples" in context
    assert "listing create madakke" in context


def test_builder_retrieves_structured_kannada_entries_from_query():
    context = get_kannada_context(
        "price_prediction",
        {"user_profile": {"district": "Dharwad", "language": "kn"}},
        query="onion price forecast next week, hold madla?",
    )
    assert "## Kannada Dialect Lexicon Hints" in context
    assert "## Kannada Local Context Hints" in context
    assert "storage cost" in context
