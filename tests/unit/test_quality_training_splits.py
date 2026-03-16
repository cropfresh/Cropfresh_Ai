"""Unit tests for deterministic grouped split assignment."""

from __future__ import annotations

import pytest

from src.agents.quality_assessment.training.manifest_schema import VisionManifestRow
from src.agents.quality_assessment.training.splitter import assign_grouped_splits


def _row(lot_id: str, image_id: str, farm_id: str) -> VisionManifestRow:
    return VisionManifestRow(
        lot_id=lot_id,
        image_id=image_id,
        commodity="Tomato",
        grade="A",
        farm_id=farm_id,
        label_source="human",
        rubric_version="v1",
        source_uri=f"C:/tmp/{image_id}.jpg",
    )


def test_assign_grouped_splits_keeps_same_farm_together():
    rows = [
        _row("lot-a", "img-1", "farm-1"),
        _row("lot-b", "img-2", "farm-1"),
        _row("lot-c", "img-3", "farm-2"),
        _row("lot-d", "img-4", "farm-3"),
    ]

    assigned = assign_grouped_splits(rows, seed=7)
    farm_splits = {(row.farm_id, row.split) for row in assigned}
    assert len([pair for pair in farm_splits if pair[0] == "farm-1"]) == 1


def test_assign_grouped_splits_is_deterministic():
    rows = [_row(f"lot-{idx}", f"img-{idx}", f"farm-{idx}") for idx in range(6)]
    first = assign_grouped_splits(rows, seed=11)
    second = assign_grouped_splits(rows, seed=11)
    assert [row.split for row in first] == [row.split for row in second]


def test_assign_grouped_splits_rejects_bad_ratios():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_grouped_splits([_row("lot-1", "img-1", "farm-1")], train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
