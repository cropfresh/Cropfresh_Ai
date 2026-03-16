"""Train and export a CropFresh YOLO defect detector."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.agents.quality_assessment.training.model_contracts import (
    load_validated_onnx_session,
    validate_yolo_session,
)
from src.agents.quality_assessment.training.yolo_reporting import (
    assert_minimum_metrics,
    extract_validation_metrics,
    write_yolo_report,
)
from src.shared.logger import setup_logger

logger = setup_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export CropFresh YOLO defect detector")
    parser.add_argument(
        "--data-config",
        required=True,
        type=Path,
        help="YOLO dataset.yaml produced by export_quality_dataset.py",
    )
    parser.add_argument(
        "--base-model", default="yolov8n.pt", help="Ultralytics checkpoint to fine-tune"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", default="runs/cropfresh-yolo")
    parser.add_argument("--name", default="defect-detector")
    parser.add_argument("--min-map50", type=float, default=0.7)
    parser.add_argument("--min-precision", type=float, default=0.7)
    parser.add_argument("--min-recall", type=float, default=0.7)
    parser.add_argument(
        "--output", type=Path, default=Path("models/vision/yolov26n_agri_defects.onnx")
    )
    parser.add_argument("--report", type=Path, help="Optional JSON metrics report path")
    args = parser.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    trainer = YOLO(args.base_model)
    trainer.train(
        data=str(args.data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    best_weights = Path(trainer.trainer.best)
    validator = YOLO(str(best_weights))
    metrics = extract_validation_metrics(
        validator.val(data=str(args.data_config), imgsz=args.imgsz)
    )
    report_path = args.report or args.output.with_suffix(".metrics.json")
    write_yolo_report(
        report_path,
        metrics=metrics,
        artifact_path=args.output,
        data_config=args.data_config,
        training_config={
            "base_model": args.base_model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "project": args.project,
            "name": args.name,
            "best_weights": str(best_weights),
        },
    )
    assert_minimum_metrics(
        metrics,
        min_map50=args.min_map50,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    exported = Path(validator.export(format="onnx", imgsz=args.imgsz, opset=17))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), args.output)
    load_validated_onnx_session(args.output, validate_yolo_session)
    logger.info(
        "Exported validated YOLO detector to {} with map50={:.4f} precision={:.4f} recall={:.4f}",
        args.output,
        metrics.map50,
        metrics.precision,
        metrics.recall,
    )


if __name__ == "__main__":
    main()
