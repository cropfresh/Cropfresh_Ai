"""Artifact helpers for reproducible DINO training runs."""

from __future__ import annotations

import pathlib
import random

import numpy as np


def set_training_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible training runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_onnx(model, output_path: pathlib.Path, device=None) -> None:
    """Export the trained DINO model with pixel_values + commodity_id inputs."""
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_pixels = torch.randn(1, 3, 224, 224).to(device or "cpu")
    dummy_commodity = torch.zeros(1, dtype=torch.long).to(device or "cpu")
    model.eval()
    torch.onnx.export(
        model,
        (dummy_pixels, dummy_commodity),
        str(output_path),
        input_names=["pixel_values", "commodity_id"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "commodity_id": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
