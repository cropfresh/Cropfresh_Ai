"""
entity_extractor package
========================

Public surface (backward-compatible with the old flat module):

    from src.voice.entity_extractor import (
        VoiceIntent,
        ExtractionResult,
        VoiceEntityExtractor,
    )

Sub-modules (import directly for data):
    _intents   – VoiceIntent, ExtractionResult
    _keywords  – INTENT_KEYWORDS
    _crops     – CROP_NAMES, COMMODITY_ALIASES
    _patterns  – QUANTITY_PATTERNS, PRICE_PATTERNS, LOCATION_PATTERNS, UNIT_MAP
    _language  – detect_language_from_text
    _extractor – VoiceEntityExtractor
"""

from src.voice.entity_extractor._intents import VoiceIntent, ExtractionResult
from src.voice.entity_extractor._extractor import VoiceEntityExtractor
from src.voice.entity_extractor._language import detect_language_from_text

# Data constants also re-exported for convenience
from src.voice.entity_extractor._keywords import INTENT_KEYWORDS
from src.voice.entity_extractor._crops import CROP_NAMES, COMMODITY_ALIASES
from src.voice.entity_extractor._patterns import (
    QUANTITY_PATTERNS,
    PRICE_PATTERNS,
    LOCATION_PATTERNS,
    UNIT_MAP,
)

__all__ = [
    # Core types
    "VoiceIntent",
    "ExtractionResult",
    "VoiceEntityExtractor",
    # Language utility
    "detect_language_from_text",
    # Data dicts
    "INTENT_KEYWORDS",
    "CROP_NAMES",
    "COMMODITY_ALIASES",
    "QUANTITY_PATTERNS",
    "PRICE_PATTERNS",
    "LOCATION_PATTERNS",
    "UNIT_MAP",
]
