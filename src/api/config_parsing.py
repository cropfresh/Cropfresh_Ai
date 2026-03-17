"""Helpers for normalizing loose environment values into typed settings."""

from __future__ import annotations

from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on", "debug", "dev", "development", "local"}
_FALSE_VALUES = {"0", "false", "no", "off", "release", "prod", "production", "staging"}


def normalize_environment(value: str) -> str:
    """Validate and normalize the deployment environment value."""
    allowed = {"development", "staging", "production"}
    normalized = value.lower()
    if normalized not in allowed:
        raise ValueError(f"environment must be one of {allowed}")
    return normalized


def parse_env_bool(value: Any) -> bool:
    """Parse booleans while tolerating common deployment strings like `release`."""
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES or normalized == "":
        return False
    raise ValueError("value must be a boolean-like string")
