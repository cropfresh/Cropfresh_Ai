"""Manifest-backed dataloaders and label balancing for DINO training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.agents.quality_assessment.training.manifest_schema import (
    VisionManifestRow,
    load_manifest_rows,
)

GRADE_LABELS = ("A+", "A", "B", "C")
GRADE_TO_INDEX = {label: index for index, label in enumerate(GRADE_LABELS)}


@dataclass(slots=True)
class DinoDataBundle:
    """Training-ready dataloaders plus label-distribution metadata."""

    train_loader: Any
    val_loader: Any
    class_weights: Any
    train_label_counts: dict[str, int]
    val_label_counts: dict[str, int]


def build_dataloaders(
    manifest_path: Path,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    balance_train: bool = True,
) -> DinoDataBundle:
    """Build manifest-backed train/val dataloaders for DINO training."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    rows = load_manifest_rows(manifest_path)
    train_rows, val_rows = _split_rows(rows)
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_label_counts = _count_labels(train_rows)
    val_label_counts = _count_labels(val_rows)
    class_weights = _build_class_weights(train_label_counts)
    sampler = _build_sampler(train_rows, generator) if balance_train else None
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return DinoDataBundle(
        train_loader=DataLoader(
            ManifestDataset(train_rows, train_transform),
            sampler=sampler,
            shuffle=sampler is None,
            generator=generator,
            **loader_kwargs,
        ),
        val_loader=DataLoader(
            ManifestDataset(val_rows, val_transform), shuffle=False, **loader_kwargs
        ),
        class_weights=class_weights,
        train_label_counts=train_label_counts,
        val_label_counts=val_label_counts,
    )


class ManifestDataset:
    """Minimal manifest-backed dataset for DINO grade training."""

    def __init__(self, dataset_rows: list[VisionManifestRow], transform: Any) -> None:
        self.rows = dataset_rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        image = Image.open(Path(row.source_uri)).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(row.commodity_id, dtype=torch.long),
            torch.tensor(GRADE_TO_INDEX[row.grade], dtype=torch.long),
        )


def _split_rows(
    rows: list[VisionManifestRow],
) -> tuple[list[VisionManifestRow], list[VisionManifestRow]]:
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    if not train_rows:
        raise ValueError("Manifest must contain at least one train row")
    if not val_rows:
        raise ValueError("Manifest must contain at least one val row")
    return train_rows, val_rows


def _count_labels(rows: list[VisionManifestRow]) -> dict[str, int]:
    counts = Counter(row.grade for row in rows)
    return {label: counts.get(label, 0) for label in GRADE_LABELS}


def _build_class_weights(train_label_counts: dict[str, int]):
    import torch

    non_zero = [count for count in train_label_counts.values() if count > 0]
    if not non_zero:
        raise ValueError("Training split does not contain any labeled rows")
    total = sum(non_zero)
    weights = []
    for label in GRADE_LABELS:
        count = train_label_counts[label]
        weight = total / (len(non_zero) * count) if count else 0.0
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.float32)
    active = tensor > 0
    tensor[active] = tensor[active] / tensor[active].mean()
    return tensor


def _build_sampler(rows: list[VisionManifestRow], generator):
    import torch
    from torch.utils.data import WeightedRandomSampler

    grade_counts = Counter(row.grade for row in rows)
    sample_weights = [1.0 / grade_counts[row.grade] for row in rows]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
