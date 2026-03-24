"""Context helpers for language-aware chat and agent routing."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from src.shared.language_detection import (
    detect_response_language,
    normalize_default_language,
    normalize_language_code,
)
from src.shared.language_values import LANGUAGE_FIELDS, USER_PROFILE_FIELDS


def resolve_language(
    query: str = "",
    context: Mapping[str, Any] | None = None,
    default: str = "en",
    allow_auto: bool = False,
) -> str:
    """Resolve the effective language for the current turn."""
    fallback = normalize_default_language(default, allow_auto=allow_auto)
    payload = coerce_mapping(context)

    explicit = extract_language(payload, allow_auto=allow_auto)
    if explicit and (allow_auto or explicit != "auto"):
        return explicit

    profile = coerce_mapping(payload.get("user_profile"))
    profile_language = extract_language(profile, allow_auto=allow_auto)
    detected = detect_response_language(query) if query else ""
    if detected and detected != "en":
        return detected

    if profile_language and (allow_auto or profile_language != "auto"):
        return profile_language

    if detected:
        return detected

    return fallback


def ensure_language_context(
    context: Mapping[str, Any] | None,
    query: str = "",
    default_language: str = "en",
) -> dict[str, Any]:
    """Return a copied context with canonical language fields populated."""
    normalized = deepcopy(dict(context or {}))
    user_profile = coerce_mapping(normalized.get("user_profile"))
    entities = coerce_mapping(normalized.get("entities"))

    resolved_language = resolve_language(
        query=query,
        context={"user_profile": user_profile, **normalized},
        default=default_language,
    )

    profile_pref = normalize_language_code(user_profile.get("language_pref"), default="")
    if not profile_pref:
        profile_pref = normalize_language_code(
            normalized.get("language_pref"),
            default="",
        )

    user_profile["language"] = resolved_language
    user_profile["language_pref"] = profile_pref or resolved_language

    normalized["user_profile"] = user_profile
    normalized["entities"] = entities
    normalized["language"] = resolved_language
    normalized["response_language"] = resolved_language
    return normalized


def split_session_context(
    payload: Mapping[str, Any] | None,
    query: str = "",
    default_language: str = "en",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split incoming request context into session user_profile and entities."""
    incoming = deepcopy(dict(payload or {}))
    user_profile = coerce_mapping(incoming.pop("user_profile", {}))
    entities = coerce_mapping(incoming.pop("entities", {}))

    for key, value in incoming.items():
        if key in USER_PROFILE_FIELDS or key in LANGUAGE_FIELDS:
            user_profile[key] = value
        else:
            entities[key] = value

    normalized = ensure_language_context(
        {"user_profile": user_profile, "entities": entities},
        query=query,
        default_language=default_language,
    )
    return normalized["user_profile"], normalized["entities"]


def coerce_mapping(value: Any) -> dict[str, Any]:
    """Convert a mapping-like object into a mutable dict."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def extract_language(payload: Mapping[str, Any], allow_auto: bool = False) -> str:
    """Read the first canonical language field from a payload."""
    for key in LANGUAGE_FIELDS:
        normalized = normalize_language_code(
            payload.get(key),
            default="",
            allow_auto=allow_auto,
        )
        if normalized and (allow_auto or normalized != "auto"):
            return normalized
    return ""
