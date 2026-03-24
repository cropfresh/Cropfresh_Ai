"""Triplet dataset helpers for ResNet similarity training."""

from __future__ import annotations

import csv
from pathlib import Path


def build_triplet_dataset(triplets_csv: Path):
    """Load a triplet CSV into a torchvision dataset."""
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class TripletDataset(Dataset):
        def __init__(self, csv_path: Path) -> None:
            base = csv_path.parent
            with csv_path.open(encoding="utf-8") as handle:
                self.rows = [
                    (base / row["anchor"], base / row["positive"], base / row["negative"])
                    for row in csv.DictReader(handle)
                ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            anchor_p, positive_p, negative_p = self.rows[index]
            return (
                transform(Image.open(anchor_p).convert("RGB")),
                transform(Image.open(positive_p).convert("RGB")),
                transform(Image.open(negative_p).convert("RGB")),
            )

    return TripletDataset(triplets_csv)
