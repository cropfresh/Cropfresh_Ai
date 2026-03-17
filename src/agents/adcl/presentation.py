"""Presentation helpers for ADCL caller surfaces."""

from __future__ import annotations

from typing import Any


def format_weekly_report(report: Any, district: str) -> str:
    """Format a weekly report for agent/chat responses."""
    lines = [f"Weekly Crop Recommendations - {district}", ""]
    crops = getattr(report, "crops", [])
    if not crops:
        lines.append("No crop recommendations are available this week.")
        return "\n".join(lines)

    for index, crop in enumerate(crops[:5], start=1):
        badge = "GREEN" if getattr(crop, "green_label", False) else "WATCH"
        recommendation = getattr(crop, "recommendation", "")
        lines.append(
            f"{index}. {getattr(crop, 'commodity', 'Unknown')} "
            f"[{badge}] score={getattr(crop, 'demand_score', 0.0):.2f}"
        )
        if recommendation:
            lines.append(f"   {recommendation}")

    generated_at = getattr(report, "generated_at", None)
    if generated_at:
        lines.extend(["", f"Generated: {generated_at}"])
    return "\n".join(lines)
