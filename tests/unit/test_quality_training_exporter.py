"""Unit tests for exporting quality manifests into training layouts."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.agents.quality_assessment.training.dataset_exporter import (
    export_quality_dataset,
)


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (24, 24), color=color).save(path)


def test_export_quality_dataset_writes_layouts_and_yaml(tmp_path: Path):
    source_a = tmp_path / "a.jpg"
    source_b = tmp_path / "b.jpg"
    _make_image(source_a, (220, 10, 10))
    _make_image(source_b, (10, 220, 10))

    rows = [
        {
            "lot_id": "lot-1",
            "image_id": "img-a",
            "commodity": "Tomato",
            "grade": "A",
            "split": "train",
            "farm_id": "farm-1",
            "label_source": "human",
            "rubric_version": "v1",
            "source_uri": str(source_a),
            "defect_boxes": [{"label": "bruise", "x1": 1, "y1": 1, "x2": 8, "y2": 8}],
        },
        {
            "lot_id": "lot-2",
            "image_id": "img-b",
            "commodity": "Onion",
            "grade": "B",
            "split": "val",
            "farm_id": "farm-2",
            "label_source": "human",
            "rubric_version": "v1",
            "source_uri": str(source_b),
            "defect_boxes": [],
        },
    ]

    output_dir = tmp_path / "exported"
    summary = export_quality_dataset(rows, output_dir)

    assert summary.to_dict() == {
        "records_seen": 2,
        "records_skipped": 0,
        "images_written": 4,
        "classification_images": 2,
        "detection_images": 2,
    }
    assert (output_dir / "classification" / "train" / "A" / "img-a.jpg").exists()
    assert (output_dir / "detection" / "labels" / "train" / "img-a.txt").read_text(encoding="utf-8").startswith("0 ")
    manifest_lines = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(manifest_lines[0])["classification_image_path"].endswith("img-a.jpg")
    dataset_yaml = (output_dir / "detection" / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in dataset_yaml
    assert "0: bruise" in dataset_yaml
