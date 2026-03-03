"""
Training script for DINOv2 ViT-S/14 + MLP head grade classifier.

This is a reference implementation — run locally or on a GPU server.
The exported ONNX file is placed in models/vision/ where CropVisionPipeline
will load it at startup.

Requirements (install separately via pip / uv):
    torch>=2.1, torchvision>=0.16, transformers>=4.38, Pillow>=10.0.0

Usage:
    # Train and export
    python scripts/train_dinov2_classifier.py \\
        --data data/grading/ \\
        --epochs 20 \\
        --output models/vision/dinov2_grade_classifier.onnx

    # Evaluate existing checkpoint
    python scripts/train_dinov2_classifier.py \\
        --data data/grading/ --eval-only --checkpoint runs/best.pt

Dataset layout expected under --data:
    data/grading/
        train/A+/<images>
        train/A/<images>
        train/B/<images>
        train/C/<images>
        val/A+/<images>
        ...
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _require_torch():
    """Fail early with a clear message if torch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "ERROR: torch not installed. Install training extras:\n"
            "  uv pip install torch torchvision transformers\n"
            "or: pip install 'cropfresh-service-ai[ml]'",
            file=sys.stderr,
        )
        sys.exit(1)


# * ─── model definition ───────────────────────────────────────────────────────


def build_model(num_classes: int = 4):
    """
    Frozen DINOv2 ViT-S/14 backbone + trainable MLP classification head.

    Only the head trains — freezing the backbone lets us reach high accuracy
    with only 1,000–5,000 labelled produce images per commodity.
    Architecture: GlobalAvgPool → Linear(384, 128) → ReLU → Dropout(0.3) → Linear(128, C)
    """
    import torch.nn as nn
    from transformers import AutoModel

    class GradeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # * facebook/dinov2-small outputs 384-dim CLS token features.
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            for param in self.backbone.parameters():
                param.requires_grad = False

            self.head = nn.Sequential(
                nn.Linear(384, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, pixel_values):
            # pooler_output: (batch, 384) — CLS token after pooling
            features = self.backbone(pixel_values=pixel_values).pooler_output
            return self.head(features)

    return GradeClassifier()


# * ─── data loading ───────────────────────────────────────────────────────────


def build_dataloaders(data_root: pathlib.Path, batch_size: int = 16):
    """Build standard ImageFolder dataloaders with ImageNet augmentation."""
    from torchvision import datasets, transforms

    GRADE_TO_IDX = {"A+": 0, "A": 1, "B": 2, "C": 3}

    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfm)
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=val_tfm)

    import torch
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


# * ─── training loop ──────────────────────────────────────────────────────────


def train(
    data_root: pathlib.Path,
    output_path: pathlib.Path,
    epochs: int = 20,
    lr: float = 3e-4,
):
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = build_model().to(device)
    train_loader, val_loader = build_dataloaders(data_root)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt = output_path.parent / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── validation ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch:02d}/{epochs}  loss={total_loss:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint saved → {best_ckpt}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    # Load best weights before exporting
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    export_onnx(model, output_path, device)


# * ─── ONNX export ────────────────────────────────────────────────────────────


def export_onnx(model, output_path: pathlib.Path, device=None):
    """Export trained model to ONNX opset 17 for onnxruntime CPU inference."""
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device or "cpu")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"Exported ONNX model → {output_path} ({size_mb:.1f} MB)")


# * ─── evaluation ─────────────────────────────────────────────────────────────


def evaluate(data_root: pathlib.Path, checkpoint: pathlib.Path):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, val_loader = build_dataloaders(data_root)
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    print(f"Validation accuracy: {correct / total:.3f} ({correct}/{total})")


# * ─── CLI entry point ────────────────────────────────────────────────────────


def main():
    _require_torch()

    parser = argparse.ArgumentParser(
        description="Train DINOv2 grade classifier for Cropfresh AI."
    )
    parser.add_argument("--data",       required=True, type=pathlib.Path, help="Dataset root (train/val subfolders)")
    parser.add_argument("--output",     default="models/vision/dinov2_grade_classifier.onnx", type=pathlib.Path)
    parser.add_argument("--epochs",     default=20, type=int)
    parser.add_argument("--lr",         default=3e-4, type=float)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", type=pathlib.Path, help="Checkpoint for --eval-only")
    args = parser.parse_args()

    if args.eval_only:
        if not args.checkpoint:
            parser.error("--eval-only requires --checkpoint")
        evaluate(args.data, args.checkpoint)
    else:
        train(args.data, args.output, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
