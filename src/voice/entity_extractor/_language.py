"""
Unicode-range based language detection for 10 Indian languages.
"""


def detect_language_from_text(text: str) -> str:
    """
    Detect spoken/typed language from text using Unicode script ranges.

    Supported codes:  hi, kn, ta, te, mr, bn, gu, pa, ml, en
    Falls back to 'en' for ASCII / unknown scripts.
    """
    if not text:
        return "en"

    # Devanagari  (Hindi AND Marathi share the same script)
    if any('\u0900' <= c <= '\u097F' for c in text):
        # Simple heuristics to distinguish Marathi from Hindi
        _mr_markers = {'माझे', 'माझी', 'आहे', 'काय', 'नाही', 'होते', 'तुमचे', 'मराठी'}
        if any(w in text for w in _mr_markers):
            return "mr"
        return "hi"

    # Kannada
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"

    # Telugu
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"

    # Tamil
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"

    # Malayalam
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return "ml"

    # Gujarati
    if any('\u0A80' <= c <= '\u0AFF' for c in text):
        return "gu"

    # Bengali / Assamese
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"

    # Gurmukhi (Punjabi)
    if any('\u0A00' <= c <= '\u0A7F' for c in text):
        return "pa"

    return "en"
