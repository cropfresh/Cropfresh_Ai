"""Shared language constants and keyword hints."""

from __future__ import annotations

SUPPORTED_LANGUAGE_CODES = {
    "en",
    "hi",
    "kn",
    "ta",
    "te",
    "mr",
    "bn",
    "gu",
    "pa",
    "ml",
}

LANGUAGE_FIELDS = (
    "response_language",
    "language",
    "language_pref",
    "preferred_language",
    "detected_language",
)

USER_PROFILE_FIELDS = {
    "type",
    "user_type",
    "name",
    "district",
    "village",
    "location",
    "language",
    "language_pref",
    "preferred_language",
    "farm_size_acres",
    "crops",
    "buyer_type",
    "subscription_tier",
    "credit_limit",
    "latitude",
    "longitude",
    "phone",
}

LANGUAGE_ALIASES = {
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "hindi": "hi",
    "hi-in": "hi",
    "kannada": "kn",
    "kn-in": "kn",
    "tamil": "ta",
    "ta-in": "ta",
    "telugu": "te",
    "te-in": "te",
    "marathi": "mr",
    "mr-in": "mr",
    "bengali": "bn",
    "bn-in": "bn",
    "gujarati": "gu",
    "gu-in": "gu",
    "punjabi": "pa",
    "pa-in": "pa",
    "gurmukhi": "pa",
    "malayalam": "ml",
    "ml-in": "ml",
}

KANNADA_LATIN_STRONG_HINTS = {
    "yaavaga",
    "yeshtu",
    "hege",
    "mattu",
    "yojane",
    "arji",
    "niyantrisuvudu",
    "maaduvudu",
    "maadodu",
    "maadide",
    "ivattu",
    "indu",
    "eega",
    "beku",
    "haakabeku",
    "raitare",
}

KANNADA_LATIN_SUPPORT_HINTS = {
    "bele",
    "dara",
    "alli",
    "gida",
    "sarkaar",
    "raita",
    "sahaya",
    "yojana",
    "maadi",
    "banni",
}
