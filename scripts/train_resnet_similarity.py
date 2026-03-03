"""
Fine-tune ResNet50 with triplet loss for produce image similarity.
=================================================================
Trains a contrastive embedding model that maps produce images to a
128-dim L2-normalised space. Same-batch image pairs have high cosine
similarity; different-batch pairs have low cosine similarity.

Requirements:
    pip install torch torchvision
    (onnxruntime is the runtime dependency — training only needs torch)

Dataset directory layout:
    data/similarity/
        train/
            triplets.csv    # columns: anchor, positive, negative (image paths)
        val/
            triplets.csv

Run:
    python scripts/train_resnet_similarity.py \\
        --data data/similarity/ \\
        --epochs 20 \\
        --batch-size 32 \\
        --output models/vision/resnet50_produce_similarity.onnx
"""

# * TRAINING SCRIPT — RESNET SIMILARITY
# NOTE: This is a reference/offline script. ONNX export produces the file
#       consumed by ResNetSimilarityEngine in similarity.py at runtime.
# TODO: Replace CSV triplet format with proper dataloader once dataset tooling is ready.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# * ═══════════════════════════════════════════════════════════════
# * Model Definition
# * ═══════════════════════════════════════════════════════════════

def build_model(embed_dim: int = 128):
    """
    ResNet50 backbone + narrow projection head for embedding similarity.

    Architecture:
        ResNet50 (pretrained, no final classifier) → GAP → (2048,)
        → Linear(2048, embed_dim) → L2Norm → (embed_dim,)
    """
    import torch.nn as nn
    from torchvision import models

    class _L2Norm(nn.Module):
        """Apply L2 normalisation along dim=1."""
        def forward(self, x):
            import torch.nn.functional as F
            return F.normalize(x, p=2, dim=1)

    class ProduceSimilarityNet(nn.Module):
        """ResNet50 backbone with a contrastive embedding head."""

        def __init__(self) -> None:
            super().__init__()
            backbone = models.resnet50(weights="IMAGENET1K_V2")
            # * Drop original FC classifier — keep everything up to global avg pool
            self.features   = nn.Sequential(*list(backbone.children())[:-1])
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, embed_dim),
                _L2Norm(),
            )

        def forward(self, x):
            feat = self.features(x)     # (B, 2048, 1, 1)
            return self.projection(feat) # (B, embed_dim)

    return ProduceSimilarityNet()


# * ═══════════════════════════════════════════════════════════════
# * Dataset
# * ═══════════════════════════════════════════════════════════════

def build_dataset(triplets_csv: Path):
    """
    Load a triplet dataset from a CSV with columns: anchor, positive, negative.

    Each row is a tuple of image file paths relative to the CSV directory.
    """
    from PIL import Image
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class TripletDataset(Dataset):
        """Image triplet dataset from CSV file."""

        def __init__(self, csv_path: Path) -> None:
            base = csv_path.parent
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                self.rows = [
                    (base / r["anchor"], base / r["positive"], base / r["negative"])
                    for r in reader
                ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            anchor_p, positive_p, negative_p = self.rows[idx]
            anchor   = _transform(Image.open(anchor_p).convert("RGB"))
            positive = _transform(Image.open(positive_p).convert("RGB"))
            negative = _transform(Image.open(negative_p).convert("RGB"))
            return anchor, positive, negative

    return TripletDataset(triplets_csv)


# * ═══════════════════════════════════════════════════════════════
# * Training Loop
# * ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, margin: float = 0.3):
    """
    Single epoch of triplet loss training.

    Loss = max(0, margin + d(anchor, positive) − d(anchor, negative))
    where d = 1 − cosine_similarity.

    Returns:
        Mean loss for the epoch.
    """
    import torch
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    for anchor, positive, negative in loader:
        optimizer.zero_grad()
        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        # * Cosine distances (1 − similarity) ∈ [0, 2]
        d_pos = 1.0 - F.cosine_similarity(emb_a, emb_p)
        d_neg = 1.0 - F.cosine_similarity(emb_a, emb_n)
        loss  = torch.clamp(d_pos - d_neg + margin, min=0.0).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# * ═══════════════════════════════════════════════════════════════
# * ONNX Export
# * ═══════════════════════════════════════════════════════════════

def export_onnx(model, output_path: Path, embed_dim: int = 128) -> None:
    """
    Export the trained model to ONNX format for runtime inference.

    Input:  (1, 3, 224, 224) float32 — ImageNet-normalised RGB image
    Output: (1, embed_dim) float32   — L2-normalised embedding
    """
    import torch

    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported → {output_path}  (embed_dim={embed_dim})")


# * ═══════════════════════════════════════════════════════════════
# * CLI Entry Point
# * ═══════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet50 with triplet loss for produce similarity"
    )
    parser.add_argument("--data",       type=Path, default=Path("data/similarity/"),
                        help="Root directory with train/ and val/ triplet CSVs")
    parser.add_argument("--epochs",     type=int,  default=20)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--embed-dim",  type=int,  default=128)
    parser.add_argument("--margin",     type=float, default=0.3)
    parser.add_argument("--output",     type=Path,
                        default=Path("models/vision/resnet50_produce_similarity.onnx"))
    return parser.parse_args()


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = build_model(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_csv = args.data / "train" / "triplets.csv"
    dataset   = build_dataset(train_csv)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, margin=args.margin)
        print(f"Epoch {epoch:>3}/{args.epochs}  loss={loss:.4f}")

    export_onnx(model.cpu(), args.output, embed_dim=args.embed_dim)
