"""
Digital Twin Logic Utilities
============================
Pure, stateless functions for grading, defect generation, and explanation logic.
"""

from datetime import datetime
from typing import Any

from src.agents.digital_twin.models import DigitalTwin

# Grade sequence for arrival estimation (best → worst)
GRADE_SEQUENCE: tuple[str, ...] = ("A+", "A", "B", "C")

# Realistic transit clamping range in hours
MAX_TRANSIT_HOURS: float = 72.0


def compute_transit_hours(departed_at: datetime, arrived_at: datetime) -> float:
    """Compute hours between departure twin creation and arrival, clamped."""
    delta = arrived_at - departed_at
    hours = delta.total_seconds() / 3600.0
    return max(0.0, min(MAX_TRANSIT_HOURS, hours))


def estimate_arrival_grade(
    departure_twin: DigitalTwin,
    arrival_photos: list[str],
    transit_hours: float,
) -> str:
    """Estimate arrival grade from departure snapshot and transit duration."""
    if not arrival_photos:
        return "C"

    dep_idx = GRADE_SEQUENCE.index(departure_twin.grade) if departure_twin.grade in GRADE_SEQUENCE else 1
    shelf_life_hours = float(departure_twin.shelf_life_days) * 24.0

    if shelf_life_hours <= 0:
        return GRADE_SEQUENCE[min(dep_idx + 1, len(GRADE_SEQUENCE) - 1)]

    degradation_ratio = transit_hours / shelf_life_hours

    if degradation_ratio < 0.10:
        grade_drops = 0
    elif degradation_ratio < 0.25:
        grade_drops = 1
    elif degradation_ratio < 0.50:
        grade_drops = 1 if dep_idx < 2 else 2
    else:
        grade_drops = 2

    arrival_idx = min(dep_idx + grade_drops, len(GRADE_SEQUENCE) - 1)
    return GRADE_SEQUENCE[arrival_idx]


def infer_arrival_defects(grade_arrival: str, departure_defects: list[str]) -> list[str]:
    """Infer likely arrival defects from departure defects."""
    grade_index = {"A+": 0, "A": 1, "B": 2, "C": 3}
    arrival_val = grade_index.get(grade_arrival, 2)

    defects = list(departure_defects)
    if arrival_val >= 2 and "bruise" not in defects:
        defects.append("bruise")
    if arrival_val >= 3 and "overripe" not in defects:
        defects.append("overripe")
    return defects


def compute_report_confidence(
    similarity_score: float,
    has_photos: bool,
    departure_confidence: float,
    analysis_method: str,
) -> float:
    """Compute overall DiffReport confidence."""
    if not has_photos:
        return 0.40

    photo_factor = 0.85
    method_bonus = {"ssim": 0.15, "perceptual_hash": 0.10, "rule_based": 0.0}.get(
        analysis_method, 0.0
    )
    base = (photo_factor + departure_confidence) / 2.0
    return round(min(1.0, base + method_bonus), 3)


def build_explanation(
    departure_twin: DigitalTwin,
    grade_arrival: str,
    new_defects: list[str],
    transit_hours: float,
    similarity_score: float,
    liability_result: Any,
) -> str:
    """Build a human-readable dispute resolution explanation."""
    parts: list[str] = [
        f"Departure: Grade {departure_twin.grade}, {departure_twin.defect_count} defect(s), "
        f"shelf life {departure_twin.shelf_life_days} days.",
    ]

    if grade_arrival == departure_twin.grade:
        parts.append(f"Arrival: Grade {grade_arrival} — no grade change detected.")
    else:
        parts.append(
            f"Arrival: Grade {grade_arrival} (dropped from {departure_twin.grade}) "
            f"after {transit_hours:.1f}h transit."
        )

    if new_defects:
        parts.append(f"New defects at arrival: {', '.join(new_defects)}.")

    parts.append(f"Image similarity: {similarity_score:.1%}.")
    parts.append(liability_result.reasoning)

    return " ".join(parts)
