"""Safe settings helpers for rate-hub integrations."""

from __future__ import annotations

from src.config import get_settings


def get_agmarknet_api_key() -> str:
    """Return the Agmarknet API key without failing on unrelated settings issues."""
    try:
        return get_settings().agmarknet_api_key
    except Exception:
        return ""
