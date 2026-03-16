"""
Language helpers for multilingual RAG responses.
"""

from __future__ import annotations

import re

from src.voice.entity_extractor._language import detect_language_from_text

_KANNADA_LATIN_STRONG_HINTS = {
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

_KANNADA_LATIN_SUPPORT_HINTS = {
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

_MESSAGES = {
    "no_information": {
        "en": "I don't have enough information to answer.",
        "kn": "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ನನಗಿಲ್ಲ.",
    },
    "extractive_prefix": {
        "en": "Based on the retrieved documents: ",
        "kn": "ಪಡೆಯಲಾದ ದಾಖಲೆಗಳ ಆಧಾರದಲ್ಲಿ: ",
    },
    "generation_error": {
        "en": "Unable to generate an answer at this time.",
        "kn": "ಈ ಕ್ಷಣದಲ್ಲಿ ಉತ್ತರವನ್ನು ರಚಿಸಲು ಆಗುತ್ತಿಲ್ಲ.",
    },
    "decline_safe": {
        "en": (
            "I don't have enough information about this topic in my knowledge base. "
            "Please try rephrasing your question or consult a local expert."
        ),
        "kn": (
            "ಈ ವಿಷಯದ ಬಗ್ಗೆ ನನ್ನ ಜ್ಞಾನ ಭಂಡಾರದಲ್ಲಿ ಸಾಕಷ್ಟು ಪರಿಶೀಲಿತ ಮಾಹಿತಿ ಇಲ್ಲ. "
            "ದಯವಿಟ್ಟು ಪ್ರಶ್ನೆಯನ್ನು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಕೇಳಿ ಅಥವಾ ಸ್ಥಳೀಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
        ),
    },
    "decline_safety_critical": {
        "en": (
            "I don't have enough verified information to answer this safely. "
            "For pesticide dosages, financial advice, or health-related queries, "
            "please consult your local KVK (Krishi Vigyan Kendra) or agriculture officer."
        ),
        "kn": (
            "ಈ ಪ್ರಶ್ನೆಗೆ ಸುರಕ್ಷಿತವಾಗಿ ಉತ್ತರಿಸಲು ನನಗಿರುವ ಪರಿಶೀಲಿತ ಮಾಹಿತಿ ಸಾಕಾಗುವುದಿಲ್ಲ. "
            "ಕೀಟನಾಶಕ ಪ್ರಮಾಣ, ಹಣಕಾಸು ಸಲಹೆ, ಅಥವಾ ಆರೋಗ್ಯ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳಿಗಾಗಿ "
            "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
        ),
    },
    "decline_platform": {
        "en": (
            "I'm not sure about this. Please contact CropFresh support "
            "or check the Help section in the app."
        ),
        "kn": (
            "ಈ ವಿಷಯದಲ್ಲಿ ನನಗೆ ಖಚಿತ ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು CropFresh ಸಹಾಯವಾಣಿಯನ್ನು "
            "ಸಂಪರ್ಕಿಸಿ ಅಥವಾ ಆಪ್‌ನ Help ವಿಭಾಗವನ್ನು ನೋಡಿ."
        ),
    },
}


def detect_response_language(text: str) -> str:
    """Detect the best response language, with Kanglish support for Kannada."""
    detected = detect_language_from_text(text or "")
    if detected != "en":
        return detected

    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    if tokens & _KANNADA_LATIN_STRONG_HINTS:
        return "kn"
    if len(tokens & (_KANNADA_LATIN_STRONG_HINTS | _KANNADA_LATIN_SUPPORT_HINTS)) >= 2:
        return "kn"
    return "en"


def build_generation_language_instruction(
    query: str,
    route: str = "",
    language: str | None = None,
) -> str:
    """Return concise language instructions for answer generation prompts."""
    language = language or detect_response_language(query)
    if language == "kn":
        market_line = ""
        if route == "live_price_api":
            market_line = (
                "\n- Use Kannada market wording such as ಬೆಲೆ, ಮಾರುಕಟ್ಟೆ, ಕನಿಷ್ಠ ದರ, "
                "ಗರಿಷ್ಠ ದರ, ಮಾದರಿ ದರ, and ಕ್ವಿಂಟಾಲ್ when relevant."
            )
        return (
            "The farmer asked in Kannada. Respond fully in natural, respectful Kannada."
            "\n- Use Kannada script unless the user explicitly asks for English transliteration."
            "\n- Keep sentences short and voice-friendly."
            "\n- Prefer Kannada agricultural terms like ಬೆಳೆ, ಬೆಲೆ, ರೋಗ, ಕೀಟ, ಯೋಜನೆ, and ಅರ್ಜಿ."
            "\n- Keep English only for unavoidable technical terms."
            f"{market_line}"
        )
    if language == "en":
        return "Respond in clear, simple English suitable for farmers."
    return "Respond in the same language as the user's question. Keep the answer simple and practical."


def get_localized_message(message_key: str, language: str) -> str:
    """Fetch a localized canned response, defaulting to English."""
    options = _MESSAGES[message_key]
    return options.get(language, options["en"])
