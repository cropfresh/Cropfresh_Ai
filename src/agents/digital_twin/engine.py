"""
Digital Twin Engine
====================
Creates immutable departure snapshots of produce quality and compares
them against buyer arrival photos to generate AI-powered diff reports
for dispute resolution.

Primary entry points:
    engine.create_departure_twin(...)  → DigitalTwin
    engine.compare_arrival(...)        → DiffReport
    engine.generate_diff_report(...)   → DiffReport

Architecture:
    - DB is optional; engine works in-memory for tests and dev
    - All image analysis degrades gracefully to rule-based fallback
    - Liability assignment follows the CropFresh liability matrix
"""

# * DIGITAL TWIN ENGINE MODULE
# NOTE: DB dependency injected; degrade gracefully when absent (dev/test mode).
# NOTE: Image analysis is best-effort; rule-based fallback always available.
# ! SECURITY: twin_id values should be treated as sensitive — do not log full IDs in production.

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

from src.agents.digital_twin.diff_analysis import (
    compute_grade_delta,
    compute_new_defects,
    compute_similarity,
)
from src.agents.digital_twin.liability import determine_liability
from src.agents.digital_twin.models import ArrivalData, DiffReport, DigitalTwin
from src.agents.digital_twin.similarity import ResNetSimilarityEngine
from src.agents.quality_assessment.vision_models import QualityResult


# * Grade sequence for arrival estimation (best → worst)
_GRADE_SEQUENCE: tuple[str, ...] = ("A+", "A", "B", "C")

# * Realistic transit clamping range in hours
_MAX_TRANSIT_HOURS: float = 72.0


# * ═══════════════════════════════════════════════════════════════
# * Engine
# * ═══════════════════════════════════════════════════════════════

class DigitalTwinEngine:
    """
    Digital Twin Engine for produce quality tracking and dispute resolution.

    Creates an immutable departure snapshot (DigitalTwin) at the farm gate,
    then compares it against buyer-submitted arrival photos via AI diff
    analysis to produce a DiffReport with liability recommendation.

    Usage:
        engine = DigitalTwinEngine(db=db_client)
        twin = await engine.create_departure_twin(
            listing_id="lst-abc123",
            farmer_photos=["s3://bucket/farm_front.jpg"],
            agent_photos=["s3://bucket/agent_check.jpg"],
            quality_result=quality_result,
            gps=(12.9716, 77.5946),
        )
        diff = await engine.compare_arrival(
            twin_id=twin.twin_id,
            arrival_photos=["s3://bucket/arrival.jpg"],
            arrival_gps=(12.9700, 77.5900),
        )
    """

    def __init__(self, db: Optional[Any] = None) -> None:
        # NOTE: db is AuroraPostgresClient — optional for dev/test
        self.db = db
        self._twin_cache: dict[str, DigitalTwin] = {}
        # * ResNet50 similarity engine — degrades to phash/rule-based when model absent
        self.similarity_engine = ResNetSimilarityEngine()

    # ─────────────────────────────────────────────────────────
    # * Public — Departure Twin Creation
    # ─────────────────────────────────────────────────────────

    async def create_departure_twin(
        self,
        listing_id: str,
        farmer_photos: list[str],
        agent_photos: list[str],
        quality_result: QualityResult,
        gps: tuple[float, float],
        dinov2_confidence_vector: list[float] | None = None,
    ) -> DigitalTwin:
        """
        Create an immutable departure snapshot for a listing.

        Captures farmer + agent photos, AI grade, GPS coordinates,
        and defect annotations. Persists to DB when available and
        always caches in memory for fast lookups.

        Args:
            listing_id:     UUID of the crop listing.
            farmer_photos:  S3 URLs of farmer-submitted photos (3–5 angles recommended).
            agent_photos:   S3 URLs of field agent verification photos.
            quality_result: QualityResult from the CV-QG quality assessment agent.
            gps:            (latitude, longitude) GPS coordinates at departure.
            dinov2_confidence_vector: Optional explicit DINOv2 vector; if omitted,
                            taken from quality_result.dinov2_confidence_vector (FR9).

        Returns:
            DigitalTwin with a stable twin_id for future comparison.
        """
        twin_id = f"dt-{uuid4().hex[:12]}"
        gps_lat, gps_lng = gps

        # * FR9: prefer explicit arg; fall back to quality_result field
        dino_vec = tuple(
            dinov2_confidence_vector
            if dinov2_confidence_vector is not None
            else quality_result.dinov2_confidence_vector
        )

        twin = DigitalTwin(
            twin_id=twin_id,
            listing_id=listing_id,
            farmer_photos=farmer_photos,
            agent_photos=agent_photos,
            grade=quality_result.grade,
            confidence=quality_result.confidence,
            defect_types=quality_result.defects,
            defect_count=quality_result.defect_count,
            shelf_life_days=quality_result.shelf_life_days,
            gps_lat=gps_lat,
            gps_lng=gps_lng,
            ai_annotations={"bboxes": quality_result.annotations},
            dinov2_confidence_vector=dino_vec,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )

        self._twin_cache[twin_id] = twin
        await self._persist_twin(twin)

        logger.info(
            "Departure twin {} created: listing={} grade={} confidence={:.2f} "
            "defects={} dino_vector_len={} farmer_photos={} agent_photos={}",
            twin_id, listing_id, twin.grade, twin.confidence,
            twin.defect_count, len(dino_vec), len(farmer_photos), len(agent_photos),
        )
        return twin

    # ─────────────────────────────────────────────────────────
    # * Public — Arrival Comparison
    # ─────────────────────────────────────────────────────────

    async def compare_arrival(
        self,
        twin_id: str,
        arrival_photos: list[str],
        arrival_gps: tuple[float, float] = (0.0, 0.0),
        arrived_at: Optional[datetime] = None,
    ) -> DiffReport:
        """
        Compare a departure twin against buyer-submitted arrival photos.

        Fetches the departure twin from cache or DB, then calls
        generate_diff_report() to produce the full AI-powered analysis.

        Args:
            twin_id:        ID returned by create_departure_twin().
            arrival_photos: S3 URLs of buyer arrival photos.
            arrival_gps:    (lat, lng) at delivery point (default 0,0 if unknown).
            arrived_at:     Arrival timestamp (defaults to now).

        Returns:
            DiffReport with quality delta, similarity score, liability, and explanation.

        Raises:
            ValueError: If departure twin is not found in cache or DB.
        """
        departure_twin = await self._fetch_twin(twin_id)
        if departure_twin is None:
            raise ValueError(
                f"Digital twin '{twin_id}' not found. "
                "Ensure create_departure_twin() was called before compare_arrival()."
            )

        arrival = ArrivalData(
            arrival_photos=arrival_photos,
            gps_lat=arrival_gps[0],
            gps_lng=arrival_gps[1],
            arrived_at=arrived_at or datetime.now(UTC).replace(tzinfo=None),
        )
        return await self.generate_diff_report(departure_twin, arrival)

    # ─────────────────────────────────────────────────────────
    # * Public — Diff Report Generation
    # ─────────────────────────────────────────────────────────

    async def generate_diff_report(
        self,
        departure_twin: DigitalTwin,
        arrival_data: ArrivalData,
    ) -> DiffReport:
        """
        Generate an AI-powered diff report for a departure twin vs. arrival state.

        Pipeline:
          1. Compute transit hours from timestamps
          2. Estimate arrival grade from departure data + transit duration
          3. Infer arrival defects (departure defects + transit-induced defects)
          4. Compute image similarity (SSIM → perceptual hash → rule-based)
          5. Compute quality delta (grade_departure vs grade_arrival)
          6. Identify new defects introduced during transit
          7. Apply liability matrix to assign responsible party
          8. Build report confidence from all contributing factors
          9. Generate human-readable explanation

        Args:
            departure_twin: Immutable departure snapshot (DigitalTwin).
            arrival_data:   Buyer arrival state (ArrivalData).

        Returns:
            DiffReport with full analysis and liability recommendation.
        """
        # * Step 1: Transit duration
        transit_hours = _compute_transit_hours(
            departure_twin.created_at, arrival_data.arrived_at
        )

        # * Step 2: Estimate arrival grade
        grade_arrival = _estimate_arrival_grade(
            departure_twin=departure_twin,
            arrival_photos=arrival_data.arrival_photos,
            transit_hours=transit_hours,
        )

        # * Step 3: Infer arrival defects from grade + known departure defects
        arrival_defects = _infer_arrival_defects(grade_arrival, departure_twin.defect_types)

        # * Step 4: Image similarity — ResNet50 preferred; falls back to SSIM/phash/rule-based
        substitution_flag = False
        if self.similarity_engine.available:
            batch_result = self.similarity_engine.compare_url_batches(
                departure_twin.all_photos(),
                arrival_data.arrival_photos,
            )
            similarity_score  = batch_result["similarity_score"]
            substitution_flag = batch_result["substitution_flag"]
            analysis_method   = "resnet50"
            logger.debug(
                "ResNet50 similarity: score={:.4f} min={:.4f} substitution={}",
                similarity_score, batch_result["min_score"], substitution_flag,
            )
        else:
            # * Fallback: SSIM → perceptual hash → rule-based (existing diff_analysis pipeline)
            similarity_score, analysis_method = compute_similarity(
                departure_photos=departure_twin.all_photos(),
                arrival_photos=arrival_data.arrival_photos,
                grade_departure=departure_twin.grade,
                grade_arrival=grade_arrival,
                departure_defects=departure_twin.defect_types,
                arrival_defects=arrival_defects,
            )

        # * Step 5: Quality delta
        quality_delta = compute_grade_delta(departure_twin.grade, grade_arrival)

        # * Step 6: New defects introduced during transit
        new_defects = compute_new_defects(departure_twin.defect_types, arrival_defects)

        # * Step 7: Liability determination (substitution_flag escalates to hauler 100%)
        liability_result = determine_liability(
            grade_departure=departure_twin.grade,
            grade_arrival=grade_arrival,
            quality_delta=quality_delta,
            transit_hours=transit_hours,
            new_defects=new_defects,
            has_arrival_photos=bool(arrival_data.arrival_photos),
            substitution_flag=substitution_flag,
        )

        # * Step 8: Report confidence
        confidence = _compute_report_confidence(
            similarity_score=similarity_score,
            has_photos=bool(arrival_data.arrival_photos),
            departure_confidence=departure_twin.confidence,
            analysis_method=analysis_method,
        )

        # * Step 9: Explanation
        explanation = _build_explanation(
            departure_twin=departure_twin,
            grade_arrival=grade_arrival,
            new_defects=new_defects,
            transit_hours=transit_hours,
            similarity_score=similarity_score,
            liability_result=liability_result,
        )

        diff = DiffReport(
            quality_delta=quality_delta,
            grade_departure=departure_twin.grade,
            grade_arrival=grade_arrival,
            new_defects=new_defects,
            similarity_score=similarity_score,
            transit_hours=transit_hours,
            liability=liability_result.liable_party,
            claim_percent=liability_result.claim_percent,
            confidence=confidence,
            explanation=explanation,
            analysis_method=analysis_method,
        )

        logger.info(
            "DiffReport: {} → {} | delta={:.3f} | sim={:.3f} | "
            "liability={} | claim={}% | conf={:.3f} | method={}",
            diff.grade_departure, diff.grade_arrival, diff.quality_delta,
            diff.similarity_score, diff.liability, diff.claim_percent,
            diff.confidence, diff.analysis_method,
        )
        return diff

    # ─────────────────────────────────────────────────────────
    # * Private — DB helpers
    # ─────────────────────────────────────────────────────────

    async def _persist_twin(self, twin: DigitalTwin) -> None:
        """Persist digital twin to DB when available; skip silently otherwise."""
        if not (self.db and hasattr(self.db, "create_digital_twin")):
            return
        try:
            # * FR9 immutability: INSERT ... ON CONFLICT DO NOTHING ensures write-once
            await self.db.create_digital_twin({
                "listing_id": twin.listing_id,
                "farmer_photos": list(twin.farmer_photos),
                "agent_photos": list(twin.agent_photos),
                "ai_annotations": twin.ai_annotations,
                "grade": twin.grade,
                "confidence": twin.confidence,
                "defect_types": list(twin.defect_types),
                "shelf_life_days": twin.shelf_life_days,
                # * FR9: persist the DINOv2 softmax vector for immutable audit trail
                "dinov2_confidence_vector": list(twin.dinov2_confidence_vector),
            }, conflict="ignore")  # DB driver passes ON CONFLICT DO NOTHING
        except Exception as exc:
            logger.warning(f"Failed to persist digital twin {twin.twin_id}: {exc}")

    async def _fetch_twin(self, twin_id: str) -> Optional[DigitalTwin]:
        """Fetch digital twin — checks memory cache first, then DB."""
        if twin_id in self._twin_cache:
            return self._twin_cache[twin_id]

        if self.db and hasattr(self.db, "get_digital_twin"):
            try:
                row = await self.db.get_digital_twin(twin_id)
                if row:
                    twin = _row_to_twin(row)
                    self._twin_cache[twin_id] = twin
                    return twin
            except Exception as exc:
                logger.warning(f"Failed to fetch digital twin {twin_id}: {exc}")

        return None


# * ═══════════════════════════════════════════════════════════════
# * Module-level pure helpers (stateless, testable without engine)
# * ═══════════════════════════════════════════════════════════════

def _compute_transit_hours(departed_at: datetime, arrived_at: datetime) -> float:
    """
    Compute hours between departure twin creation and arrival.

    Returns:
        Float clamped to [0.0, 72.0] hours.
    """
    delta = arrived_at - departed_at
    hours = delta.total_seconds() / 3600.0
    return max(0.0, min(_MAX_TRANSIT_HOURS, hours))


def _estimate_arrival_grade(
    departure_twin: DigitalTwin,
    arrival_photos: list[str],
    transit_hours: float,
) -> str:
    """
    Estimate arrival grade from departure snapshot and transit duration.

    Without real vision inference on arrival photos, grade is degraded
    based on the fraction of shelf life consumed during transit.

    Args:
        departure_twin: Departure snapshot.
        arrival_photos: Buyer arrival photos (used only for presence check).
        transit_hours:  Transit duration in hours.

    Returns:
        Estimated arrival grade string.
    """
    # ! If no photos submitted, assume worst case for buyer claim purposes
    if not arrival_photos:
        return "C"

    dep_idx = _GRADE_SEQUENCE.index(departure_twin.grade) if departure_twin.grade in _GRADE_SEQUENCE else 1
    shelf_life_hours = float(departure_twin.shelf_life_days) * 24.0

    if shelf_life_hours <= 0:
        return _GRADE_SEQUENCE[min(dep_idx + 1, len(_GRADE_SEQUENCE) - 1)]

    degradation_ratio = transit_hours / shelf_life_hours

    # * Map consumed shelf-life fraction to grade drops
    if degradation_ratio < 0.10:
        grade_drops = 0
    elif degradation_ratio < 0.25:
        grade_drops = 1
    elif degradation_ratio < 0.50:
        grade_drops = 1 if dep_idx < 2 else 2
    else:
        grade_drops = 2

    arrival_idx = min(dep_idx + grade_drops, len(_GRADE_SEQUENCE) - 1)
    return _GRADE_SEQUENCE[arrival_idx]


def _infer_arrival_defects(grade_arrival: str, departure_defects: list[str]) -> list[str]:
    """
    Infer likely arrival defects from arrival grade and departure defects.

    Adds transit-induced defects (bruise, overripe) when grade has degraded.

    Args:
        grade_arrival:     Estimated or actual arrival grade.
        departure_defects: Known defects at departure.

    Returns:
        Extended defect list for arrival state.
    """
    grade_index = {"A+": 0, "A": 1, "B": 2, "C": 3}
    arrival_val = grade_index.get(grade_arrival, 2)

    defects = list(departure_defects)
    if arrival_val >= 2 and "bruise" not in defects:
        defects.append("bruise")
    if arrival_val >= 3 and "overripe" not in defects:
        defects.append("overripe")
    return defects


def _compute_report_confidence(
    similarity_score: float,
    has_photos: bool,
    departure_confidence: float,
    analysis_method: str,
) -> float:
    """
    Compute overall DiffReport confidence from contributing factors.

    Args:
        similarity_score:      Image similarity score [0, 1].
        has_photos:            Whether arrival photos were submitted.
        departure_confidence:  AI confidence from departure assessment.
        analysis_method:       Analysis method used.

    Returns:
        Confidence score [0.0, 1.0].
    """
    if not has_photos:
        return 0.40

    photo_factor = 0.85
    method_bonus = {"ssim": 0.15, "perceptual_hash": 0.10, "rule_based": 0.0}.get(
        analysis_method, 0.0
    )
    base = (photo_factor + departure_confidence) / 2.0
    return round(min(1.0, base + method_bonus), 3)


def _build_explanation(
    departure_twin: DigitalTwin,
    grade_arrival: str,
    new_defects: list[str],
    transit_hours: float,
    similarity_score: float,
    liability_result: Any,
) -> str:
    """
    Build a human-readable dispute resolution explanation.

    Args:
        departure_twin:   Departure snapshot.
        grade_arrival:    Estimated arrival grade.
        new_defects:      Defects introduced during transit.
        transit_hours:    Transit duration.
        similarity_score: Image similarity score.
        liability_result: LiabilityResult from determine_liability().

    Returns:
        Multi-sentence explanation string.
    """
    from src.agents.digital_twin.liability import LiabilityResult

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


def _row_to_twin(row: dict[str, Any]) -> DigitalTwin:
    """
    Convert a DB row dict to a DigitalTwin dataclass.

    Handles JSONB ai_annotations stored as string or dict.

    Args:
        row: Raw asyncpg row dict from digital_twins table.

    Returns:
        Hydrated DigitalTwin instance.
    """
    ai_annotations = row.get("ai_annotations") or {}
    if isinstance(ai_annotations, str):
        try:
            ai_annotations = json.loads(ai_annotations)
        except Exception:
            ai_annotations = {}

    gps = row.get("gps_location") or {}

    return DigitalTwin(
        twin_id=str(row.get("id", "")),
        listing_id=str(row.get("listing_id", "")),
        farmer_photos=list(row.get("farmer_photos") or []),
        agent_photos=list(row.get("agent_photos") or []),
        grade=str(row.get("grade", "B")),
        confidence=float(row.get("confidence") or 0.70),
        defect_types=list(row.get("defect_types") or []),
        defect_count=len(list(row.get("defect_types") or [])),
        shelf_life_days=int(row.get("shelf_life_days") or 3),
        gps_lat=float(gps.get("lat", 0.0)) if isinstance(gps, dict) else 0.0,
        gps_lng=float(gps.get("lng", 0.0)) if isinstance(gps, dict) else 0.0,
        ai_annotations=ai_annotations,
        # * FR9: rehydrate DINOv2 vector as a tuple (required by frozen dataclass)
        dinov2_confidence_vector=tuple(row.get("dinov2_confidence_vector") or []),
        created_at=row.get("created_at") or datetime.now(UTC).replace(tzinfo=None),
    )


# * ═══════════════════════════════════════════════════════════════
# * Factory
# * ═══════════════════════════════════════════════════════════════

def get_digital_twin_engine(db: Optional[Any] = None) -> DigitalTwinEngine:
    """
    Factory for creating a DigitalTwinEngine with optional DB dependency.

    Args:
        db: AuroraPostgresClient instance (optional).

    Returns:
        Configured DigitalTwinEngine.
    """
    return DigitalTwinEngine(db=db)
