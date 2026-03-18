"""District-aware Kannada dialect hints for the shared prompt builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_DISTRICT_BUCKETS = {
    "mandya": "OLD_MYSURU_RURAL",
    "mysore": "OLD_MYSURU_RURAL",
    "mysuru": "OLD_MYSURU_RURAL",
    "hassan": "OLD_MYSURU_RURAL",
    "bengaluru": "BENGALURU_URBAN_MIXED",
    "bangalore": "BENGALURU_URBAN_MIXED",
    "hubballi": "NORTH_KA_RURAL",
    "hubli": "NORTH_KA_RURAL",
    "dharwad": "NORTH_KA_RURAL",
    "haveri": "NORTH_KA_RURAL",
    "gadag": "NORTH_KA_RURAL",
    "belagavi": "NORTH_KA_RURAL",
    "belgaum": "NORTH_KA_RURAL",
    "bagalkot": "NORTH_KA_RURAL",
    "kalaburagi": "HYDERABAD_KARNATAKA",
    "gulbarga": "HYDERABAD_KARNATAKA",
    "yadgir": "HYDERABAD_KARNATAKA",
    "raichur": "HYDERABAD_KARNATAKA",
    "ballari": "HYDERABAD_KARNATAKA",
    "bellary": "HYDERABAD_KARNATAKA",
    "koppal": "HYDERABAD_KARNATAKA",
    "mangaluru": "COASTAL_KA",
    "mangalore": "COASTAL_KA",
    "udupi": "COASTAL_KA",
    "kundapura": "COASTAL_KA",
    "dakshina kannada": "COASTAL_KA",
    "tumakuru": "CENTRAL_KA",
    "tumkur": "CENTRAL_KA",
    "shivamogga": "CENTRAL_KA",
    "shimoga": "CENTRAL_KA",
    "chitradurga": "CENTRAL_KA",
}


def build_dialect_context(context: Mapping[str, Any] | None) -> str:
    """Build a district-aware dialect hint for Kannada prompting."""
    signal = _extract_signal(context)
    if not signal:
        return ""

    bucket = _resolve_bucket(signal)
    if not bucket:
        return ""

    return "\n".join(
        [
            "## Kannada Dialect Hint",
            f"- Likely dialect bucket: {bucket}",
            f"- Reason: location signal = {signal}",
            "- Use this bucket only for internal style matching and slang interpretation.",
            "- Never print the dialect bucket label to the user.",
            "- Match local particles and borrowed words only when they improve clarity.",
        ]
    )


def _extract_signal(context: Mapping[str, Any] | None) -> str:
    payload = _coerce_mapping(context)
    profile = _coerce_mapping(payload.get("user_profile"))
    entities = _coerce_mapping(payload.get("entities"))

    for source in (profile, entities, payload):
        for key in ("district", "location", "region", "taluk"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _resolve_bucket(signal: str) -> str:
    signal_lower = signal.lower()
    for district, bucket in _DISTRICT_BUCKETS.items():
        if district in signal_lower:
            return bucket
    return "OTHER / MIXED" if signal_lower else ""


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
