"""
ADCL Agent — Data Models
========================
Data structures for the Adaptive Demand Crop List weekly report.

ADCLCrop  : per-commodity demand + price + green label.
WeeklyReport : full weekly output stored in adcl_reports table.
"""

# * ADCL MODELS MODULE
# NOTE: Dataclasses for lightweight serialisation, matching task12.md schema.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


# * ═══════════════════════════════════════════════════════════════
# * Crop-level record
# * ═══════════════════════════════════════════════════════════════

@dataclass
class ADCLCrop:
    """
    Single commodity entry in the weekly demand list.

    Produced by scoring.score_and_label() and consumed by
    summary.SummaryGenerator and ADCLAgent.generate_weekly_report().
    """

    commodity: str
    demand_score: float              # 0.0–1.0 (normalised by max demand)
    predicted_price_per_kg: float    # From PricePredictionAgent (₹/kg)
    price_trend: str                 # 'rising' | 'stable' | 'falling'
    seasonal_fit: str                # 'in_season' | 'off_season' | 'year_round'
    green_label: bool                # True = recommended to grow
    buyer_count: int                 # Unique buyers who ordered this
    total_demand_kg: float           # Estimated weekly demand (kg)
    recommendation: str = ""        # Farmer-friendly advice (English)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return {
            "commodity": self.commodity,
            "demand_score": round(self.demand_score, 3),
            "predicted_price_per_kg": round(self.predicted_price_per_kg, 2),
            "price_trend": self.price_trend,
            "seasonal_fit": self.seasonal_fit,
            "green_label": self.green_label,
            "buyer_count": self.buyer_count,
            "total_demand_kg": round(self.total_demand_kg, 1),
            "recommendation": self.recommendation,
        }


# * ═══════════════════════════════════════════════════════════════
# * Weekly report
# * ═══════════════════════════════════════════════════════════════

@dataclass
class WeeklyReport:
    """
    Full ADCL weekly report output.

    Stored in adcl_reports table; consumed by farmer-facing API.
    """

    week_start: date
    crops: list[ADCLCrop]
    generated_by: str = "adcl_agent"
    summary_en: str = ""            # English narrative
    summary_hi: str = ""            # Hindi narrative
    summary_kn: str = ""            # Kannada narrative

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return {
            "week_start": self.week_start.isoformat(),
            "generated_by": self.generated_by,
            "summary_en": self.summary_en,
            "summary_hi": self.summary_hi,
            "summary_kn": self.summary_kn,
            "crops": [c.to_dict() for c in self.crops],
            "green_count": sum(1 for c in self.crops if c.green_label),
        }
