"""
TTS Utilities
=============
Helper functions for Text-to-Speech processing.
"""

from typing import Union


def normalize_edge_rate(rate: Union[str, float, int, None]) -> str:
    """Normalize speed/rate to Edge TTS SSML format (e.g., '+0%', '+20%', '-20%')."""
    if isinstance(rate, str):
        return rate
    if rate is None:
        return "+0%"
    try:
        rate_pct = int((float(rate) - 1.0) * 100)
        return f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    except (ValueError, TypeError):
        return "+0%"
