"""
Digital Twin — Data Models
==========================
Immutable data structures for departure snapshot, arrival data,
and AI-powered diff reports used in dispute resolution.
"""

# * DIGITAL TWIN MODELS MODULE
# NOTE: All models are dataclasses for lightweight serialisation without Pydantic overhead.
# NOTE: to_dict() methods produce the JSONB payload stored in the disputes.diff_report column.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# * ═══════════════════════════════════════════════════════════════
# * Departure Snapshot
# * ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DigitalTwin:
    """
    Immutable departure snapshot of produce quality.

    Created at the farm gate when produce is handed to the hauler.
    Captures photos, AI grade, GPS coordinates, and defect annotations.
    This snapshot is the ground truth for any future dispute comparison.

    frozen=True enforces immutability in Python (FR9); the DB trigger
    `trg_digital_twins_immutable` enforces immutability at the storage layer.
    """

    twin_id: str
    listing_id: str
    farmer_photos: list[str]        # S3 URLs — farmer-submitted photos
    agent_photos: list[str]         # S3 URLs — field agent verification photos
    grade: str                      # Departure grade: 'A+' | 'A' | 'B' | 'C'
    confidence: float               # AI confidence [0.0, 1.0]
    defect_types: list[str]         # Known defect labels at departure
    defect_count: int               # len(defect_types) cached for fast comparison
    shelf_life_days: int            # Predicted shelf life in days
    gps_lat: float                  # GPS latitude at farm gate
    gps_lng: float                  # GPS longitude at farm gate
    ai_annotations: dict[str, Any]  # Bounding boxes from YOLO / annotation model
    # * FR9: DINOv2 ViT-S/14 softmax output [p_A+, p_A, p_B, p_C] — immutable audit trail
    dinov2_confidence_vector: tuple[float, ...]  = field(default_factory=tuple)  # frozen requires tuple not list
    created_at: datetime = field(default_factory=datetime.now)

    def all_photos(self) -> list[str]:
        """Combined list of farmer + agent photos for analysis."""
        return self.farmer_photos + self.agent_photos

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for DB storage."""
        return {
            "twin_id": self.twin_id,
            "listing_id": self.listing_id,
            "farmer_photos": list(self.farmer_photos),
            "agent_photos": list(self.agent_photos),
            "grade": self.grade,
            "confidence": self.confidence,
            "defect_types": list(self.defect_types),
            "defect_count": self.defect_count,
            "shelf_life_days": self.shelf_life_days,
            "gps_lat": self.gps_lat,
            "gps_lng": self.gps_lng,
            "ai_annotations": self.ai_annotations,
            # * FR9: DINOv2 confidence vector serialised as a plain list
            "dinov2_confidence_vector": list(self.dinov2_confidence_vector),
            "created_at": self.created_at.isoformat(),
        }


# * ═══════════════════════════════════════════════════════════════
# * Arrival State
# * ═══════════════════════════════════════════════════════════════

@dataclass
class ArrivalData:
    """
    Arrival state captured by the buyer at the delivery point.

    Buyer submits photos and GPS. These are compared against the
    departure twin to generate the diff report.
    """

    arrival_photos: list[str]   # S3 URLs — buyer-submitted arrival photos
    gps_lat: float              # GPS latitude at delivery point
    gps_lng: float              # GPS longitude at delivery point
    arrived_at: datetime = field(default_factory=datetime.now)


# * ═══════════════════════════════════════════════════════════════
# * Diff Report
# * ═══════════════════════════════════════════════════════════════

@dataclass
class DiffReport:
    """
    AI-powered quality diff between departure twin and arrival state.

    Produced by DigitalTwinEngine.generate_diff_report().
    Stored in disputes.diff_report (JSONB) for dispute resolution.
    """

    quality_delta: float        # -1.0 to 0.0  (0.0 = no change; -1.0 = max degradation)
    grade_departure: str        # Grade at departure  e.g. 'A'
    grade_arrival: str          # Grade at arrival    e.g. 'B'
    new_defects: list[str]      # Defects detected at arrival not present at departure
    similarity_score: float     # Image similarity [0.0, 1.0] (SSIM / hash / rule-based)
    transit_hours: float        # Hours between departure twin creation and arrival
    liability: str              # 'farmer' | 'hauler' | 'buyer' | 'shared' | 'none'
    claim_percent: float        # Compensation claim as % of order value [0, 100]
    confidence: float           # Report confidence [0.0, 1.0]
    explanation: str            # Human-readable dispute resolution explanation
    analysis_method: str        # 'ssim' | 'perceptual_hash' | 'rule_based'

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for DB storage in disputes.diff_report."""
        return {
            "quality_delta": round(self.quality_delta, 4),
            "grade_departure": self.grade_departure,
            "grade_arrival": self.grade_arrival,
            "new_defects": self.new_defects,
            "similarity_score": round(self.similarity_score, 4),
            "transit_hours": round(self.transit_hours, 2),
            "liability": self.liability,
            "claim_percent": round(self.claim_percent, 1),
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "analysis_method": self.analysis_method,
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict())

    @property
    def is_disputed(self) -> bool:
        """True when a meaningful quality drop was detected."""
        return self.quality_delta < -0.01 or bool(self.new_defects)

    @property
    def grade_dropped(self) -> bool:
        """True when arrival grade is lower than departure grade."""
        return self.grade_arrival != self.grade_departure


# * ═══════════════════════════════════════════════════════════════
# * Summary type-alias used by REST endpoints
# * ═══════════════════════════════════════════════════════════════

TwinDiffPayload = Optional[dict[str, Any]]
