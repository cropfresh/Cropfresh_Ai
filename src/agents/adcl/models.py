"""Dataclasses for the ADCL service contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.agents.adcl.time_utils import utc_now


@dataclass
class ADCLCrop:
    """Canonical crop-level ADCL payload reused across callers."""

    commodity: str
    demand_score: float
    predicted_price_per_kg: float
    price_trend: str
    seasonal_fit: str
    green_label: bool
    buyer_count: int
    total_demand_kg: float
    recommendation: str = ""
    demand_trend: str = "stable"
    sow_season_fit: str = "year_round"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    freshness: dict[str, Any] = field(default_factory=dict)
    source_health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize one crop recommendation into a JSON-safe payload."""
        return {
            "commodity": self.commodity,
            "demand_score": round(self.demand_score, 3),
            "predicted_price_per_kg": round(self.predicted_price_per_kg, 2),
            "price_trend": self.price_trend,
            "demand_trend": self.demand_trend,
            "seasonal_fit": self.seasonal_fit,
            "sow_season_fit": self.sow_season_fit,
            "green_label": self.green_label,
            "buyer_count": self.buyer_count,
            "total_demand_kg": round(self.total_demand_kg, 1),
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "freshness": self.freshness,
            "source_health": self.source_health,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ADCLCrop":
        """Build an ADCLCrop from persisted JSON data."""
        return cls(
            commodity=str(data.get("commodity", "")),
            demand_score=float(data.get("demand_score", 0.0)),
            predicted_price_per_kg=float(data.get("predicted_price_per_kg", 0.0)),
            price_trend=str(data.get("price_trend", "stable")),
            seasonal_fit=str(data.get("seasonal_fit", "year_round")),
            green_label=bool(data.get("green_label", False)),
            buyer_count=int(data.get("buyer_count", 0)),
            total_demand_kg=float(data.get("total_demand_kg", 0.0)),
            recommendation=str(data.get("recommendation", "")),
            demand_trend=str(data.get("demand_trend", "stable")),
            sow_season_fit=str(data.get("sow_season_fit", "year_round")),
            evidence=list(data.get("evidence", [])),
            freshness=dict(data.get("freshness", {})),
            source_health=dict(data.get("source_health", {})),
        )


@dataclass
class WeeklyReport:
    """Weekly district ADCL report persisted in `adcl_reports`."""

    week_start: date
    crops: list[ADCLCrop]
    district: str = "Bangalore"
    generated_by: str = "adcl_service"
    generated_at: datetime = field(default_factory=utc_now)
    summary_en: str = ""
    summary_hi: str = ""
    summary_kn: str = ""
    freshness: dict[str, Any] = field(default_factory=dict)
    source_health: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report into a JSON-safe payload."""
        return {
            "week_start": self.week_start.isoformat(),
            "district": self.district,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at.isoformat(),
            "summary_en": self.summary_en,
            "summary_hi": self.summary_hi,
            "summary_kn": self.summary_kn,
            "freshness": self.freshness,
            "source_health": self.source_health,
            "metadata": self.metadata,
            "crops": [crop.to_dict() for crop in self.crops],
            "green_count": sum(1 for crop in self.crops if crop.green_label),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeeklyReport":
        """Rebuild a report from the persisted DB/API payload."""
        crops = [ADCLCrop.from_dict(item) for item in data.get("crops", [])]
        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        if not isinstance(generated_at, datetime):
            generated_at = utc_now()
        week_start = data.get("week_start", date.today().isoformat())
        if isinstance(week_start, str):
            week_start = date.fromisoformat(week_start)
        return cls(
            week_start=week_start,
            district=str(data.get("district", "Bangalore")),
            crops=crops,
            generated_by=str(data.get("generated_by", "adcl_service")),
            generated_at=generated_at,
            summary_en=str(data.get("summary_en", "")),
            summary_hi=str(data.get("summary_hi", "")),
            summary_kn=str(data.get("summary_kn", "")),
            freshness=dict(data.get("freshness", {})),
            source_health=dict(data.get("source_health", {})),
            metadata=dict(data.get("metadata", {})),
        )
