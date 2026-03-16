"""Training and export helpers for ResNet similarity models."""

from __future__ import annotations

from pathlib import Path

from src.agents.quality_assessment.training.similarity_dataset import build_triplet_dataset
from src.agents.quality_assessment.training.similarity_model import build_similarity_model


def run_similarity_training(
    data_root: Path,
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    embed_dim: int = 128,
    margin: float = 0.3,
) -> None:
    """Train the ResNet similarity model from triplet CSV inputs."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_similarity_model(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = build_triplet_dataset(data_root / "train" / "triplets.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            emb_a = model(anchor.to(device))
            emb_p = model(positive.to(device))
            emb_n = model(negative.to(device))
            loss = torch.clamp((1.0 - F.cosine_similarity(emb_a, emb_p)) - (1.0 - F.cosine_similarity(emb_a, emb_n)) + margin, min=0.0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:03d}/{epochs} loss={total_loss / max(len(loader), 1):.4f}")
    export_similarity_onnx(model.cpu(), output_path, embed_dim=embed_dim)


def export_similarity_onnx(model, output_path: Path, embed_dim: int = 128) -> None:
    """Export the trained similarity model as embedding(batch,128)."""
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        str(output_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
