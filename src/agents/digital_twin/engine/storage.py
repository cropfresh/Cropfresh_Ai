"""
Engine Storage Mixin
====================
Provides database persist and fetch capabilities for digital twins.
"""

import json
from datetime import UTC, datetime
from typing import Any, Optional

from loguru import logger

from src.agents.digital_twin.models import DigitalTwin


def row_to_twin(row: dict[str, Any]) -> DigitalTwin:
    """Convert a DB row dict to a DigitalTwin dataclass."""
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
        dinov2_confidence_vector=tuple(row.get("dinov2_confidence_vector") or []),
        created_at=row.get("created_at") or datetime.now(UTC).replace(tzinfo=None),
    )


class StorageMixin:
    """Mixin for DB lifecycle of a DigitalTwin."""

    # These instance vars must be present on the inheriting class
    db: Optional[Any]
    _twin_cache: dict[str, DigitalTwin]

    async def _persist_twin(self, twin: DigitalTwin) -> None:
        """Persist digital twin to DB when available; skip silently otherwise."""
        if not (self.db and hasattr(self.db, "create_digital_twin")):
            return
        try:
            await self.db.create_digital_twin({
                "listing_id": twin.listing_id,
                "farmer_photos": list(twin.farmer_photos),
                "agent_photos": list(twin.agent_photos),
                "ai_annotations": twin.ai_annotations,
                "grade": twin.grade,
                "confidence": twin.confidence,
                "defect_types": list(twin.defect_types),
                "shelf_life_days": twin.shelf_life_days,
                "dinov2_confidence_vector": list(twin.dinov2_confidence_vector),
            }, conflict="ignore")
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
                    twin = row_to_twin(row)
                    self._twin_cache[twin_id] = twin
                    return twin
            except Exception as exc:
                logger.warning(f"Failed to fetch digital twin {twin_id}: {exc}")

        return None
