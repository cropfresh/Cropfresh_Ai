"""Annotation parsing and YOLO export helpers for quality vision training."""

from __future__ import annotations

from typing import Any

from src.agents.quality_assessment.training.models import BoundingBox
from src.agents.quality_assessment.yolo_detector import DEFECT_CLASS_NAMES


def extract_bounding_boxes(ai_annotations: dict[str, Any]) -> list[BoundingBox]:
    """Normalize stored twin annotations into typed bounding boxes."""
    candidates = (
        ai_annotations.get("bboxes")
        or ai_annotations.get("boxes")
        or ai_annotations.get("annotations")
        or []
    )
    if not isinstance(candidates, list):
        return []

    boxes: list[BoundingBox] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        label = _read_label(candidate)
        coords = _read_coords(candidate)
        if label not in DEFECT_CLASS_NAMES or coords is None:
            continue
        boxes.append(BoundingBox(label=label, x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]))
    return boxes


def to_yolo_label_lines(
    boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
) -> list[str]:
    """Convert pixel-space boxes into Ultralytics YOLO label lines."""
    if image_width <= 0 or image_height <= 0:
        return []

    lines: list[str] = []
    for box in boxes:
        class_id = DEFECT_CLASS_NAMES.index(box.label)
        width = max(box.x2 - box.x1, 1)
        height = max(box.y2 - box.y1, 1)
        center_x = box.x1 + width / 2
        center_y = box.y1 + height / 2
        lines.append(
            " ".join(
                [
                    str(class_id),
                    f"{center_x / image_width:.6f}",
                    f"{center_y / image_height:.6f}",
                    f"{width / image_width:.6f}",
                    f"{height / image_height:.6f}",
                ]
            )
        )
    return lines


def _read_label(candidate: dict[str, Any]) -> str:
    raw = candidate.get("label") or candidate.get("class_name") or candidate.get("defect")
    if not raw:
        return ""
    return str(raw).strip().lower().replace(" ", "_")


def _read_coords(candidate: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if all(key in candidate for key in ("x1", "y1", "x2", "y2")):
        return _to_int_tuple(candidate["x1"], candidate["y1"], candidate["x2"], candidate["y2"])

    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return _to_int_tuple(*bbox)

    return None


def _to_int_tuple(*values: Any) -> tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
