"""Thin CLI for ResNet similarity training and export."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.quality_assessment.training.similarity_training import (
    run_similarity_training,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CropFresh ResNet similarity model")
    parser.add_argument("--data", type=Path, default=Path("data/similarity/"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("models/vision/resnet50_produce_similarity.onnx"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_similarity_training(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        margin=args.margin,
    )
