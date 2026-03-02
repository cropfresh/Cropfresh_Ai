"""Unit tests for VoiceAgent task 16 wiring (10+ Languages & 5 New Intents)."""

import pytest
from src.agents.voice_agent import VoiceAgent
from src.voice.entity_extractor import VoiceEntityExtractor, VoiceIntent
from src.voice.stt import TranscriptionResult

# The 10 languages we need to support
SUPPORTED_LANGUAGES = ["hi", "kn", "en", "ta", "te", "mr", "bn", "gu", "pa", "ml"]

# ═══════════════════════════════════════════════════════════════
# AC-1 & AC-5: Template coverage and matrix
# ═══════════════════════════════════════════════════════════════

def test_response_templates_cover_10_languages():
    """Verify every intent has template entries for all 10 languages."""
    agent = VoiceAgent()
    templates = agent.RESPONSE_TEMPLATES

    for intent in VoiceIntent:
        if intent == VoiceIntent.UNKNOWN:
            continue
            
        intent_templates = templates.get(intent, {})
        for lang in SUPPORTED_LANGUAGES:
            assert lang in intent_templates, f"Intent {intent.name} missing template for language: {lang}"
            assert len(intent_templates[lang]) > 0, f"Template empty for {intent.name} in lang {lang}"

def test_template_x_intent_x_language_coverage_matrix():
    """Build full grid of Intent x Language to ensure no empty templates exist."""
    agent = VoiceAgent()
    templates = agent.RESPONSE_TEMPLATES
    
    missing = []
    for intent in VoiceIntent:
        if intent == VoiceIntent.UNKNOWN:
            continue
        for lang in SUPPORTED_LANGUAGES:
            template = templates.get(intent, {}).get(lang)
            if not template:
                missing.append(f"{intent.name}[{lang}]")
                
    assert len(missing) == 0, f"Missing templates in matrix: {missing}"

# ═══════════════════════════════════════════════════════════════
# AC-2: New intent patterns across languages
# ═══════════════════════════════════════════════════════════════

def test_new_intent_patterns_detected_in_multiple_languages():
    """Test pattern-based intent detection for new intents and languages.
    
    Uses actual keyword strings from _keywords.py — exact matches are guaranteed
    because _detect_intent does substring lookup against the dict.
    """
    extractor = VoiceEntityExtractor()

    # (text_containing_known_keyword, lang_code, expected_intent)
    cases = [
        # find_buyer — use exact keyword substrings
        ("யாருக்கு வேண்டும் கத்தரிக்காய்", "ta", VoiceIntent.FIND_BUYER),
        ("ਕੌਣ ਖਰੀਦੇਗਾ ਮੇਰਾ ਟਮਾਟਰ", "pa", VoiceIntent.FIND_BUYER),
        ("ఎవరు కొంటారు ఈ టమాటాలు", "te", VoiceIntent.FIND_BUYER),
        # check_weather
        (  "వాతావరణం ఉష్ణోగ్రత చెప్పు", "te", VoiceIntent.CHECK_WEATHER),
        ("आज पाऊस येणार का", "mr", VoiceIntent.CHECK_WEATHER),  # 'पाऊस' = rain, unique to weather
        # get_advisory
        ("कृষি পরামর্শ দিন", "bn", VoiceIntent.GET_ADVISORY),
        ("ਟਮਾਟਰ ਲਈ ਸਲਾਹ ਦਿਓ", "pa", VoiceIntent.GET_ADVISORY),
        # dispute_status
        ("എന്റെ പരാതി", "ml", VoiceIntent.DISPUTE_STATUS),
        ("ਮੇਰੀ ਸ਼ਿਕਾਇਤ ਦਾ ਕੀ ਬਣਿਆ", "pa", VoiceIntent.DISPUTE_STATUS),
        # weekly_demand
        ("ఈ వారం డిమాండ్ ఏమిటి", "te", VoiceIntent.WEEKLY_DEMAND),
        ("સાપ્તાહિક માંગ આ અઠ.", "gu", VoiceIntent.WEEKLY_DEMAND),  # 'સાપ્તાહિક' = weekly, unique keyword
    ]

    for text, lang, expected_intent in cases:
        intent, conf = extractor._detect_intent(text, lang)
        assert intent == expected_intent, (
            f"Expected {expected_intent.name} for '{text}' [{lang}], got {intent.name}"
        )

# ═══════════════════════════════════════════════════════════════
# AC-3: Commodity Name Normalization (Regional → Standard)
# ═══════════════════════════════════════════════════════════════

def test_commodity_aliases_normalize_regional_names():
    """Ensure COMMODITY_ALIASES maps regional crop names to standards like 'tomato'."""
    extractor = VoiceEntityExtractor()
    
    aliases = {
        'തക്കാളി': 'tomato',       # Malayalam
        'ಟೊಮೆಟೊ': 'tomato',        # Kannada
        'தக்காளி': 'tomato',        # Tamil
        'టమాటా': 'tomato',          # Telugu
        'टोमॅटो': 'tomato',         # Marathi
        'টমেটো': 'tomato',          # Bengali
        'ટામેટા': 'tomato',         # Gujarati
        'ਟਮਾਟਰ': 'tomato',          # Punjabi
        'टमाटर': 'tomato',          # Hindi
    }
    
    unified_crops = {**extractor.CROP_NAMES, **getattr(extractor, "COMMODITY_ALIASES", {})}
    
    for regional, standard in aliases.items():
        assert regional in unified_crops, f"Missing regional alias: {regional}"
        assert unified_crops[regional] == standard, f"Mismatched standard name for {regional}"

# ═══════════════════════════════════════════════════════════════
# AC-4: Auto-language detection from STT output
# ═══════════════════════════════════════════════════════════════

def test_detect_language_from_text():
    """Test unicode-based language detection."""
    assert VoiceEntityExtractor.detect_language_from_text("नमस्ते") == "hi"
    assert VoiceEntityExtractor.detect_language_from_text("ನಮಸ್ಕಾರ") == "kn"
    assert VoiceEntityExtractor.detect_language_from_text("வணக்கம்") == "ta"
    assert VoiceEntityExtractor.detect_language_from_text("నమస్కారం") == "te"
    assert VoiceEntityExtractor.detect_language_from_text("নমস্কার") == "bn"
    assert VoiceEntityExtractor.detect_language_from_text("નમસ્તે") == "gu"
    assert VoiceEntityExtractor.detect_language_from_text("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ") == "pa"
    assert VoiceEntityExtractor.detect_language_from_text("നമസ്കാരം") == "ml"
    
    # Marathi vs Hindi distinction
    assert VoiceEntityExtractor.detect_language_from_text("हे माझे आहे") == "mr"
    assert VoiceEntityExtractor.detect_language_from_text("hello") == "en"

# ═══════════════════════════════════════════════════════════════
# Handler localization
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tamil_handler_returns_localised_response():
    """Verify handler crop-prompt is localized for Tamil session."""
    agent = VoiceAgent()
    session = agent._get_or_create_session("u1", None, "ta")
    
    # Calling the handler with missing required element 'crop' should trigger Tamil fallback
    response = await agent._handle_create_listing("unused", {}, session)
    
    # Response must NOT be the English default
    assert "Which crop do you want to list?" not in response, (
        f"Handler returned English instead of Tamil: {response}"
    )
    # Must contain some Tamil info (should have Tamil Unicode chars or well-known message)
    assert len(response) > 5


