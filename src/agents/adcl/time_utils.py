"""Small UTC helpers shared by ADCL modules."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return a JSON-safe UTC timestamp string."""
    return utc_now().isoformat()
