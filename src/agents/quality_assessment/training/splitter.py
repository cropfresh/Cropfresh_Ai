"""Deterministic train/val/test split assignment for quality datasets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from math import floor

from src.agents.quality_assessment.training.manifest_schema import VisionManifestRow


def assign_grouped_splits(
    rows: list[VisionManifestRow],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[VisionManifestRow]:
    """Assign deterministic splits while keeping farm and lot groups intact."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if not rows:
        return []

    grouped: dict[str, list[VisionManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    ordered_keys = sorted(grouped, key=lambda key: _stable_sort_key(key, seed))
    split_map = _build_split_map(ordered_keys, train_ratio, val_ratio)
    return [row.with_split(split_map[_group_key(row)]) for row in rows]


def _group_key(row: VisionManifestRow) -> str:
    farm_part = row.farm_id.strip() or "unknown-farm"
    lot_part = row.lot_id.strip() or "unknown-lot"
    return farm_part if farm_part != "unknown-farm" else lot_part


def _stable_sort_key(group_key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()


def _build_split_map(ordered_keys: list[str], train_ratio: float, val_ratio: float) -> dict[str, str]:
    total = len(ordered_keys)
    train_count = floor(total * train_ratio)
    val_count = floor(total * val_ratio)
    if total >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        if train_count + val_count >= total:
            val_count = max(1, total - train_count - 1)
    test_start = min(train_count + val_count, total)

    split_map: dict[str, str] = {}
    for index, key in enumerate(ordered_keys):
        split_map[key] = "train" if index < train_count else "val" if index < test_start else "test"
    return split_map


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
