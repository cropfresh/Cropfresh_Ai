"""Validation summaries and quality gates for YOLO defect training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class YoloValidationMetrics:
    precision: float
    recall: float
    map50: float
    map50_95: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def extract_validation_metrics(results: Any) -> YoloValidationMetrics:
    """Normalize Ultralytics validation results into stable float metrics."""
    source = getattr(results, "box", results)
    return YoloValidationMetrics(
        precision=_extract_metric(source, "mp"),
        recall=_extract_metric(source, "mr"),
        map50=_extract_metric(source, "map50"),
        map50_95=_extract_metric(source, "map"),
    )


def assert_minimum_metrics(
    metrics: YoloValidationMetrics,
    *,
    min_map50: float,
    min_precision: float,
    min_recall: float,
) -> None:
    """Fail the training run when the detector misses launch-quality thresholds."""
    failures: list[str] = []
    if metrics.map50 < min_map50:
        failures.append(f"map50 {metrics.map50:.4f} < {min_map50:.4f}")
    if metrics.precision < min_precision:
        failures.append(f"precision {metrics.precision:.4f} < {min_precision:.4f}")
    if metrics.recall < min_recall:
        failures.append(f"recall {metrics.recall:.4f} < {min_recall:.4f}")
    if failures:
        raise ValueError("YOLO validation gate failed: " + ", ".join(failures))


def write_yolo_report(
    report_path: Path,
    *,
    metrics: YoloValidationMetrics,
    artifact_path: Path,
    data_config: Path,
    training_config: dict[str, Any],
) -> None:
    """Persist a stable JSON metrics report for the trained detector."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_path": str(artifact_path),
        "data_config": str(data_config),
        "training_config": training_config,
        "validation": metrics.to_dict(),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _extract_metric(source: Any, name: str) -> float:
    value = getattr(source, name, None)
    if value is None:
        raise ValueError(f"YOLO validation results missing metric '{name}'")
    return round(float(value), 6)
