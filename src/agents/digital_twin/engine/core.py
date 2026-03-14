"""
Digital Twin Core Engine
========================
Creates immutable departure snapshots of produce quality and compares
them against buyer arrival photos to generate AI-powered diff reports
for dispute resolution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

from src.agents.digital_twin.models import ArrivalData, DiffReport, DigitalTwin
from src.agents.digital_twin.similarity import ResNetSimilarityEngine
from src.agents.quality_assessment.vision_models import QualityResult

from .storage import StorageMixin
from .report import DiffReportMixin


class DigitalTwinEngine(StorageMixin, DiffReportMixin):
    """
    Digital Twin Engine for produce quality tracking and dispute resolution.

    Creates an immutable departure snapshot (DigitalTwin) at the farm gate,
    then compares it against buyer-submitted arrival photos via AI diff
    analysis to produce a DiffReport with liability recommendation.
    """

    def __init__(self, db: Optional[Any] = None) -> None:
        self.db = db
        self._twin_cache: dict[str, DigitalTwin] = {}
        self.similarity_engine = ResNetSimilarityEngine()

    async def create_departure_twin(
        self,
        listing_id: str,
        farmer_photos: list[str],
        agent_photos: list[str],
        quality_result: QualityResult,
        gps: tuple[float, float],
        dinov2_confidence_vector: list[float] | None = None,
    ) -> DigitalTwin:
        """Create an immutable departure snapshot for a listing."""
        twin_id = f"dt-{uuid4().hex[:12]}"
        gps_lat, gps_lng = gps

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

    async def compare_arrival(
        self,
        twin_id: str,
        arrival_photos: list[str],
        arrival_gps: tuple[float, float] = (0.0, 0.0),
        arrived_at: Optional[datetime] = None,
    ) -> DiffReport:
        """Compare a departure twin against buyer-submitted arrival photos."""
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


def get_digital_twin_engine(db: Optional[Any] = None) -> DigitalTwinEngine:
    """Factory for creating a DigitalTwinEngine with optional DB dependency."""
    return DigitalTwinEngine(db=db)
