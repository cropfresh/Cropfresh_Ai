"""Thin CLI for commodity-conditioned DINOv2 grade training."""

from __future__ import annotations

import argparse
import pathlib

from src.agents.quality_assessment.training.dinov2_training import (
    evaluate_checkpoint,
    require_torch,
    train_manifest_model,
)


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train CropFresh DINOv2 grade classifier")
    parser.add_argument(
        "--manifest",
        required=True,
        type=pathlib.Path,
        help="Canonical JSON/JSONL manifest with split assignments",
    )
    parser.add_argument(
        "--output", default="models/vision/dinov2_grade_classifier.onnx", type=pathlib.Path
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--unfreeze-backbone-layers", default=1, type=int)
    parser.add_argument("--no-balance-train", action="store_true")
    parser.add_argument("--report", type=pathlib.Path, help="Optional JSON metrics report path")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=pathlib.Path, help="Checkpoint used with --eval-only")
    args = parser.parse_args()

    if args.eval_only:
        if not args.checkpoint:
            parser.error("--eval-only requires --checkpoint")
        evaluate_checkpoint(
            args.manifest,
            args.checkpoint,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            report_path=args.report,
        )
        return

    train_manifest_model(
        args.manifest,
        args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        unfreeze_backbone_layers=args.unfreeze_backbone_layers,
        balance_train=not args.no_balance_train,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
