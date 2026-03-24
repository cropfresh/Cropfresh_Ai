"""Raw unicode-script language detection shared across voice and chat."""

from __future__ import annotations


def detect_language_from_text(text: str) -> str:
    """Detect language from unicode script ranges for supported languages."""
    if not text:
        return "en"

    if any("\u0900" <= char <= "\u097f" for char in text):
        marathi_markers = {
            "माझे",
            "माझी",
            "आहे",
            "काय",
            "नाही",
            "होते",
            "तुमचे",
            "मराठी",
        }
        if any(marker in text for marker in marathi_markers):
            return "mr"
        return "hi"

    if any("\u0c80" <= char <= "\u0cff" for char in text):
        return "kn"
    if any("\u0c00" <= char <= "\u0c7f" for char in text):
        return "te"
    if any("\u0b80" <= char <= "\u0bff" for char in text):
        return "ta"
    if any("\u0d00" <= char <= "\u0d7f" for char in text):
        return "ml"
    if any("\u0a80" <= char <= "\u0aff" for char in text):
        return "gu"
    if any("\u0980" <= char <= "\u09ff" for char in text):
        return "bn"
    if any("\u0a00" <= char <= "\u0a7f" for char in text):
        return "pa"
    return "en"
