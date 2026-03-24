"""Canonical manifest schema for CropFresh quality-vision datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.agents.quality_assessment.training.commodity_registry import (
    UNKNOWN_COMMODITY_ID,
    get_commodity_id,
    normalize_commodity,
)
from src.agents.quality_assessment.training.models import VALID_GRADES, BoundingBox
from src.agents.quality_assessment.yolo_detector import DEFECT_CLASS_NAMES

DatasetSplit = Literal["train", "val", "test"]


class DefectBoxSchema(BaseModel):
    """Manifest-safe representation of a single defect bounding box."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def validate_box(self) -> "DefectBoxSchema":
        self.label = str(self.label).strip().lower().replace(" ", "_")
        if self.label not in DEFECT_CLASS_NAMES:
            raise ValueError(f"Unknown defect label '{self.label}'")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive area")
        return self

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(label=self.label, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


class VisionManifestRow(BaseModel):
    """Canonical row stored in dataset JSON/JSONL manifests."""

    lot_id: str
    image_id: str
    commodity: str
    commodity_id: int | None = None
    grade: str
    split: DatasetSplit | None = None
    farm_id: str
    device_model: str = ""
    label_source: str
    rubric_version: str
    source_uri: str
    defect_boxes: list[DefectBoxSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fields(self) -> "VisionManifestRow":
        self.commodity = normalize_commodity(self.commodity)
        self.commodity_id = self.commodity_id or get_commodity_id(self.commodity)
        if self.commodity_id < UNKNOWN_COMMODITY_ID:
            raise ValueError("commodity_id must be non-negative")
        if self.grade not in VALID_GRADES:
            raise ValueError(f"Unsupported grade '{self.grade}'")
        return self

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_bounding_boxes(self) -> list[BoundingBox]:
        return [box.to_bounding_box() for box in self.defect_boxes]

    def with_split(self, split: DatasetSplit) -> "VisionManifestRow":
        return self.model_copy(update={"split": split})


def coerce_manifest_row(value: VisionManifestRow | dict) -> VisionManifestRow:
    """Normalize dict or model inputs to a validated manifest row."""
    if isinstance(value, VisionManifestRow):
        return value
    return VisionManifestRow.model_validate(value)


def load_manifest_rows(manifest_path: str | Path) -> list[VisionManifestRow]:
    """Read a JSON or JSONL manifest into validated row models."""
    path = Path(manifest_path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [VisionManifestRow.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Manifest JSON must contain a top-level array")
    return [VisionManifestRow.model_validate(item) for item in payload]
