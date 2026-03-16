"""Evaluation helpers and report writers for DINO quality-grading training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.agents.quality_assessment.training.commodity_registry import get_commodity_slug
from src.agents.quality_assessment.training.dinov2_data import GRADE_LABELS


@dataclass(slots=True)
class GradeMetric:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(slots=True)
class CommodityMetric:
    commodity: str
    accuracy: float
    support: int


@dataclass(slots=True)
class EvaluationMetrics:
    accuracy: float
    macro_f1: float
    per_grade: list[GradeMetric]
    per_commodity: list[CommodityMetric]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingReport:
    artifact_path: str
    checkpoint_path: str
    training_config: dict[str, Any]
    train_label_counts: dict[str, int]
    val_label_counts: dict[str, int]
    best_epoch: int
    epochs_completed: int
    history: list[dict[str, float]]
    validation: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation"] = self.validation.to_dict()
        return payload


def evaluate_predictions(
    labels: list[int], predictions: list[int], commodity_ids: list[int]
) -> EvaluationMetrics:
    """Compute overall, per-grade, and per-commodity validation metrics."""
    if not labels:
        raise ValueError("At least one labeled prediction is required for evaluation")
    if len(labels) != len(predictions) or len(labels) != len(commodity_ids):
        raise ValueError("labels, predictions, and commodity_ids must have the same length")

    label_indexes = list(range(len(GRADE_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_indexes,
        zero_division=0,
    )
    per_grade = [
        GradeMetric(
            label=GRADE_LABELS[index],
            precision=round(float(precision[index]), 6),
            recall=round(float(recall[index]), 6),
            f1=round(float(f1[index]), 6),
            support=int(support[index]),
        )
        for index in label_indexes
    ]
    return EvaluationMetrics(
        accuracy=round(float(accuracy_score(labels, predictions)), 6),
        macro_f1=round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 6),
        per_grade=per_grade,
        per_commodity=_build_commodity_metrics(labels, predictions, commodity_ids),
        confusion_matrix=confusion_matrix(labels, predictions, labels=label_indexes).tolist(),
    )


def write_training_report(report_path: Path, report: TrainingReport) -> None:
    """Persist a stable JSON report for reproducible training reviews."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _build_commodity_metrics(
    labels: list[int],
    predictions: list[int],
    commodity_ids: list[int],
) -> list[CommodityMetric]:
    counts: dict[int, int] = {}
    correct: dict[int, int] = {}
    for label, prediction, commodity_id in zip(labels, predictions, commodity_ids):
        counts[commodity_id] = counts.get(commodity_id, 0) + 1
        correct[commodity_id] = correct.get(commodity_id, 0) + int(label == prediction)
    metrics = [
        CommodityMetric(
            commodity=get_commodity_slug(commodity_id),
            accuracy=round(correct[commodity_id] / counts[commodity_id], 6),
            support=counts[commodity_id],
        )
        for commodity_id in counts
    ]
    return sorted(metrics, key=lambda item: (-item.support, item.commodity))
