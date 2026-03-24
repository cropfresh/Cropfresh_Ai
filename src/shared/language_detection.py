"""Language normalization and lightweight detection helpers."""

from __future__ import annotations

import re
from typing import Any

from src.shared.language_values import (
    KANNADA_LATIN_STRONG_HINTS,
    KANNADA_LATIN_SUPPORT_HINTS,
    LANGUAGE_ALIASES,
    SUPPORTED_LANGUAGE_CODES,
)
from src.shared.script_language import detect_language_from_text as detect_script_language


def normalize_language_code(
    value: Any,
    default: str = "en",
    allow_auto: bool = False,
) -> str:
    """Normalize language names, locale tags, and ISO codes."""
    fallback = normalize_default_language(default, allow_auto=allow_auto)
    if value is None:
        return fallback

    code = str(value).strip().lower().replace("_", "-")
    if not code:
        return fallback

    if code == "auto":
        return "auto" if allow_auto else fallback

    if code in SUPPORTED_LANGUAGE_CODES:
        return code

    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]

    primary = code.split("-", 1)[0]
    if primary in SUPPORTED_LANGUAGE_CODES:
        return primary
    if primary in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[primary]

    return fallback


def detect_response_language(text: str) -> str:
    """Detect the best response language, including transliterated Kannada."""
    detected = normalize_language_code(detect_script_language(text or ""), default="en")
    if detected != "en":
        return detected

    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    if tokens & KANNADA_LATIN_STRONG_HINTS:
        return "kn"
    if len(tokens & (KANNADA_LATIN_STRONG_HINTS | KANNADA_LATIN_SUPPORT_HINTS)) >= 2:
        return "kn"
    return "en"


def normalize_default_language(default: str, allow_auto: bool = False) -> str:
    """Normalize the configured default while preserving blank fallbacks."""
    if default == "auto" and allow_auto:
        return "auto"

    code = str(default).strip().lower().replace("_", "-")
    if not code:
        return ""
    if code == "auto":
        return "en"
    if code in SUPPORTED_LANGUAGE_CODES:
        return code
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]

    primary = code.split("-", 1)[0]
    if primary in SUPPORTED_LANGUAGE_CODES:
        return primary
    if primary in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[primary]
    return "en"
