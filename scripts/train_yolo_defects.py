"""Train and export a CropFresh YOLO defect detector."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.agents.quality_assessment.training.model_contracts import (
    load_validated_onnx_session,
    validate_yolo_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export CropFresh YOLO defect detector")
    parser.add_argument("--data-config", required=True, type=Path, help="YOLO dataset.yaml produced by export_quality_dataset.py")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Ultralytics checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/cropfresh-yolo")
    parser.add_argument("--name", default="defect-detector")
    parser.add_argument("--output", type=Path, default=Path("models/vision/yolov26n_agri_defects.onnx"))
    args = parser.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    trainer = YOLO(args.base_model)
    trainer.train(
        data=str(args.data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )
    best_weights = Path(trainer.trainer.best)
    exporter = YOLO(str(best_weights))
    exported = Path(exporter.export(format="onnx", imgsz=args.imgsz, opset=17))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), args.output)
    load_validated_onnx_session(args.output, validate_yolo_session)
    print(f"Exported validated YOLO detector -> {args.output}")


if __name__ == "__main__":
    main()
