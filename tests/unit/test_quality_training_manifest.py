"""Unit tests for the canonical quality-training manifest schema."""

from __future__ import annotations

import pytest

from src.agents.quality_assessment.training.manifest_schema import VisionManifestRow


def test_manifest_normalizes_commodity_and_assigns_id():
    row = VisionManifestRow(
        lot_id="lot-1",
        image_id="img-1",
        commodity="Tomato",
        grade="A",
        farm_id="farm-1",
        label_source="human",
        rubric_version="v1",
        source_uri="C:/tmp/tomato.jpg",
    )

    assert row.commodity == "tomato"
    assert row.commodity_id == 1


def test_manifest_serializes_defect_boxes():
    row = VisionManifestRow(
        lot_id="lot-2",
        image_id="img-2",
        commodity="Onion",
        grade="B",
        split="train",
        farm_id="farm-2",
        label_source="human",
        rubric_version="v1",
        source_uri="C:/tmp/onion.jpg",
        defect_boxes=[{"label": "bruise", "x1": 1, "y1": 2, "x2": 10, "y2": 12}],
    )

    payload = row.to_dict()
    assert payload["defect_boxes"][0]["label"] == "bruise"
    assert row.to_bounding_boxes()[0].x2 == 10


def test_manifest_rejects_invalid_grade():
    with pytest.raises(ValueError, match="Unsupported grade"):
        VisionManifestRow(
            lot_id="lot-3",
            image_id="img-3",
            commodity="Potato",
            grade="Premium",
            farm_id="farm-3",
            label_source="human",
            rubric_version="v1",
            source_uri="C:/tmp/potato.jpg",
        )
