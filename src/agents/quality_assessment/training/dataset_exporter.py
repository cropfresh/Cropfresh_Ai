"""Dataset export helpers for CropFresh quality grading and YOLO defects."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger
from PIL import Image

from src.agents.quality_assessment.training.annotations import to_yolo_label_lines
from src.agents.quality_assessment.training.manifest_schema import (
    VisionManifestRow,
    coerce_manifest_row,
    load_manifest_rows,
)
from src.agents.quality_assessment.training.models import DatasetBuildSummary
from src.agents.quality_assessment.yolo_detector import DEFECT_CLASS_NAMES


def export_quality_dataset(
    rows: list[VisionManifestRow | dict],
    output_dir: str | Path,
) -> DatasetBuildSummary:
    """Write classification, detection, JSONL manifest, and dataset YAML outputs."""
    root = Path(output_dir)
    manifest_path = root / "manifest.jsonl"
    summary = DatasetBuildSummary()
    _ensure_layout(root)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for candidate in rows:
            summary.records_seen += 1
            try:
                row = coerce_manifest_row(candidate)
                if row.split is None:
                    raise ValueError("Manifest row must have split assigned before export")
                source_path = _resolve_local_source(row.source_uri)
                exported = _export_row(row, source_path, root)
            except Exception as exc:  # noqa: BLE001
                summary.records_skipped += 1
                logger.warning("Skipping dataset row during export: {}", exc)
                continue

            handle.write(json.dumps(exported, ensure_ascii=True) + "\n")
            summary.classification_images += 1
            summary.detection_images += 1
            summary.images_written += 2

    _write_yolo_dataset_yaml(root / "detection" / "dataset.yaml")
    return summary


def export_quality_dataset_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> DatasetBuildSummary:
    """Load a manifest file from disk, then export the normalized dataset layout."""
    return export_quality_dataset(load_manifest_rows(manifest_path), output_dir)


def _export_row(row: VisionManifestRow, source_path: Path, root: Path) -> dict:
    suffix = source_path.suffix.lower() or ".jpg"
    class_rel = Path("classification") / row.split / row.grade / f"{row.image_id}{suffix}"
    detect_image_rel = Path("detection") / "images" / row.split / f"{row.image_id}{suffix}"
    detect_label_rel = Path("detection") / "labels" / row.split / f"{row.image_id}.txt"
    (root / class_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / detect_image_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / detect_label_rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, root / class_rel)
    shutil.copy2(source_path, root / detect_image_rel)
    width, height = Image.open(source_path).size
    yolo_lines = to_yolo_label_lines(row.to_bounding_boxes(), width, height)
    (root / detect_label_rel).write_text("\n".join(yolo_lines), encoding="utf-8")
    payload = row.to_dict()
    payload.update(
        {
            "classification_image_path": class_rel.as_posix(),
            "detection_image_path": detect_image_rel.as_posix(),
            "detection_label_path": detect_label_rel.as_posix(),
        }
    )
    return payload


def _ensure_layout(root: Path) -> None:
    for split in ("train", "val", "test"):
        for branch in ("classification", "detection/images", "detection/labels"):
            (root / branch / split).mkdir(parents=True, exist_ok=True)


def _resolve_local_source(source_uri: str) -> Path:
    path = Path(source_uri)
    if source_uri.startswith(("http://", "https://", "s3://")):
        raise ValueError("Remote source_uri values are not supported in backend-only export")
    if not path.exists():
        raise FileNotFoundError(f"Source image not found: {source_uri}")
    return path


def _write_yolo_dataset_yaml(dataset_yaml: Path) -> None:
    root = dataset_yaml.parent
    names = "\n".join(f"  {index}: {label}" for index, label in enumerate(DEFECT_CLASS_NAMES))
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names,
            ]
        ),
        encoding="utf-8",
    )
