"""Typed models for real-data quality-vision training exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VALID_GRADES = frozenset({"A+", "A", "B", "C"})


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """A single defect bounding box in xyxy pixel format."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(slots=True)
class VisionTrainingRecord:
    """Live labeled record sourced from a digital twin."""

    twin_id: str
    listing_id: str
    commodity: str
    grade: str
    confidence: float
    defect_types: list[str] = field(default_factory=list)
    ai_annotations: dict[str, Any] = field(default_factory=dict)
    farmer_photos: list[str] = field(default_factory=list)
    agent_photos: list[str] = field(default_factory=list)
    created_at: str = ""
    agent_verified: bool = False

    def photo_candidates(self, prefer_agent_photos: bool = True) -> list[tuple[str, str]]:
        """Return ordered (role, uri) photo candidates for export."""
        ordered: list[tuple[str, str]] = []
        groups = [("agent", self.agent_photos), ("farmer", self.farmer_photos)]
        if not prefer_agent_photos:
            groups.reverse()
        for role, values in groups:
            ordered.extend((role, uri) for uri in values if uri)
        return ordered


@dataclass(slots=True)
class VisionDatasetItem:
    """Materialized image item written into the training export."""

    twin_id: str
    listing_id: str
    commodity: str
    grade: str
    confidence: float
    split: str
    photo_role: str
    source_uri: str
    source_image_path: str
    classification_image_path: str
    detection_image_path: str | None
    detection_label_path: str | None
    defect_types: list[str] = field(default_factory=list)
    boxes: list[BoundingBox] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict for the training manifest."""
        payload = asdict(self)
        payload["boxes"] = [asdict(box) for box in self.boxes]
        return payload


@dataclass(slots=True)
class DatasetBuildSummary:
    """Counters returned after exporting a real-data training set."""

    records_seen: int = 0
    records_skipped: int = 0
    images_written: int = 0
    classification_images: int = 0
    detection_images: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
