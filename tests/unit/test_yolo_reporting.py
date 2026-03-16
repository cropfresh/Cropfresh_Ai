"""Unit tests for YOLO training metric extraction and gating."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.quality_assessment.training.yolo_reporting import (
    YoloValidationMetrics,
    assert_minimum_metrics,
    extract_validation_metrics,
    write_yolo_report,
)


def test_extract_validation_metrics_reads_ultralytics_box_contract():
    results = SimpleNamespace(box=SimpleNamespace(mp=0.81, mr=0.77, map50=0.88, map=0.62))

    metrics = extract_validation_metrics(results)

    assert metrics == YoloValidationMetrics(
        precision=0.81,
        recall=0.77,
        map50=0.88,
        map50_95=0.62,
    )


def test_assert_minimum_metrics_raises_for_underperforming_detector():
    metrics = YoloValidationMetrics(precision=0.69, recall=0.71, map50=0.68, map50_95=0.5)

    with pytest.raises(ValueError, match="map50 0.6800 < 0.7000, precision 0.6900 < 0.7000"):
        assert_minimum_metrics(metrics, min_map50=0.7, min_precision=0.7, min_recall=0.7)


def test_write_yolo_report_serializes_metrics_json(tmp_path: Path):
    report_path = tmp_path / "yolo.metrics.json"
    write_yolo_report(
        report_path,
        metrics=YoloValidationMetrics(precision=0.84, recall=0.79, map50=0.9, map50_95=0.66),
        artifact_path=Path("models/vision/yolov26n_agri_defects.onnx"),
        data_config=Path("exports/detection/dataset.yaml"),
        training_config={"epochs": 60, "imgsz": 640},
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["artifact_path"].endswith("yolov26n_agri_defects.onnx")
    assert payload["validation"]["precision"] == pytest.approx(0.84)
    assert payload["training_config"]["epochs"] == 60
