"""Manifest-backed dataloaders for commodity-conditioned DINO training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.agents.quality_assessment.training.manifest_schema import load_manifest_rows

GRADE_TO_INDEX = {"A+": 0, "A": 1, "B": 2, "C": 3}


def build_dataloaders(manifest_path: Path, batch_size: int = 16):
    """Build manifest-backed train/val dataloaders for DINO training."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    rows = load_manifest_rows(manifest_path)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class ManifestDataset(Dataset):
        def __init__(self, dataset_rows, transform) -> None:
            self.rows = dataset_rows
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(Path(row.source_uri)).convert("RGB")
            return self.transform(image), torch.tensor(row.commodity_id), torch.tensor(GRADE_TO_INDEX[row.grade])

    return (
        DataLoader(ManifestDataset(train_rows, train_transform), batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(ManifestDataset(val_rows, val_transform), batch_size=batch_size, shuffle=False, num_workers=2),
    )
