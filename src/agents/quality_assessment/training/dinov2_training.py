"""Training and export orchestration for commodity-conditioned DINO grading."""

from __future__ import annotations

import pathlib
import sys

from src.agents.quality_assessment.training.commodity_registry import COMMODITY_SPECS
from src.agents.quality_assessment.training.dinov2_data import build_dataloaders
from src.agents.quality_assessment.training.dinov2_model import build_grade_model


def require_torch() -> None:
    """Fail early with a clear message when training deps are unavailable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch/transformers are required for DINO training.", file=sys.stderr)
        sys.exit(1)


def train_manifest_model(
    manifest_path: pathlib.Path,
    output_path: pathlib.Path,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 16,
) -> None:
    """Train a commodity-conditioned DINO classifier from a manifest file."""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_grade_model(
        num_commodities=max((spec.commodity_id for spec in COMMODITY_SPECS.values()), default=0) + 1
    ).to(device)
    train_loader, val_loader = build_dataloaders(manifest_path, batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        list(model.head.parameters()) + list(model.commodity_embedding.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_ckpt = output_path.parent / "dinov2-best.pt"
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, commodity_ids, labels in train_loader:
            images, commodity_ids, labels = images.to(device), commodity_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, commodity_ids), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = _evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{epochs} loss={total_loss:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    export_onnx(model, output_path, device)


def evaluate_checkpoint(manifest_path: pathlib.Path, checkpoint: pathlib.Path, batch_size: int = 16) -> None:
    """Evaluate a saved DINO checkpoint against the manifest val split."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_grade_model(
        num_commodities=max((spec.commodity_id for spec in COMMODITY_SPECS.values()), default=0) + 1
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, val_loader = build_dataloaders(manifest_path, batch_size=batch_size)
    print(f"Validation accuracy: {_evaluate_model(model, val_loader, device):.3f}")


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
        dynamic_axes={"pixel_values": {0: "batch"}, "commodity_id": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def _evaluate_model(model, loader, device) -> float:
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, commodity_ids, labels in loader:
            preds = model(images.to(device), commodity_ids.to(device)).argmax(dim=1)
            labels = labels.to(device)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total else 0.0
