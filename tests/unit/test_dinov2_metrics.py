"""Unit tests for DINO training metrics and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.quality_assessment.training.dinov2_metrics import (
    EvaluationMetrics,
    TrainingReport,
    evaluate_predictions,
    write_training_report,
)


def test_evaluate_predictions_computes_exact_grade_and_commodity_metrics():
    metrics = evaluate_predictions(
        labels=[0, 0, 1, 2, 3, 3],
        predictions=[0, 1, 1, 2, 2, 3],
        commodity_ids=[1, 1, 1, 2, 2, 2],
    )

    assert metrics.accuracy == pytest.approx(0.666667, abs=1e-6)
    assert metrics.macro_f1 == pytest.approx(0.666667, abs=1e-6)
    assert metrics.confusion_matrix == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ]
    grade_metrics = {item.label: item for item in metrics.per_grade}
    assert grade_metrics["A+"].recall == pytest.approx(0.5)
    assert grade_metrics["C"].precision == pytest.approx(1.0)
    commodity_metrics = {item.commodity: item for item in metrics.per_commodity}
    assert commodity_metrics["tomato"].accuracy == pytest.approx(0.666667, abs=1e-6)
    assert commodity_metrics["onion"].support == 3


def test_write_training_report_persists_validation_payload(tmp_path: Path):
    validation = EvaluationMetrics(
        accuracy=0.82,
        macro_f1=0.79,
        per_grade=[],
        per_commodity=[],
        confusion_matrix=[[2, 0], [1, 3]],
    )
    report = TrainingReport(
        artifact_path="models/vision/dinov2_grade_classifier.onnx",
        checkpoint_path="models/vision/dinov2-best.pt",
        training_config={"epochs_requested": 12, "seed": 42},
        train_label_counts={"A+": 8, "A": 10, "B": 6, "C": 4},
        val_label_counts={"A+": 2, "A": 3, "B": 2, "C": 1},
        best_epoch=7,
        epochs_completed=9,
        history=[{"epoch": 1.0, "train_loss": 0.9, "val_accuracy": 0.7, "val_macro_f1": 0.68}],
        validation=validation,
    )

    report_path = tmp_path / "dinov2.metrics.json"
    write_training_report(report_path, report)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["artifact_path"].endswith("dinov2_grade_classifier.onnx")
    assert payload["best_epoch"] == 7
    assert payload["validation"]["macro_f1"] == pytest.approx(0.79)
    assert payload["train_label_counts"]["A"] == 10
