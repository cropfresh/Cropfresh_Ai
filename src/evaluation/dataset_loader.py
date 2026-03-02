"""
Dataset Loader — load / validate / merge golden Q&A datasets.

Responsibility: pure I/O layer, no evaluation logic here.

Usage:
    loader = DatasetLoader()
    items = loader.load("datasets/agronomy_qa.json")
    all_items = loader.load_all()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from loguru import logger

from src.evaluation.models import GoldenItem


# Path of the datasets sub-directory, relative to this file's location
_DATASETS_DIR = Path(__file__).parent / "datasets"
_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"

# Ordered list of domain dataset files
_DOMAIN_FILES: list[str] = [
    "agronomy_qa.json",
    "commerce_qa.json",
    "platform_qa.json",
    "multilingual_qa.json",
]


class DatasetLoader:
    """
    Load and validate golden datasets from JSON files.

    Each JSON file must be a list of objects matching the GoldenItem schema.
    """

    def __init__(self, datasets_dir: Path | None = None) -> None:
        self.datasets_dir = datasets_dir or _DATASETS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> list[GoldenItem]:
        """Load a single dataset file and return validated GoldenItems."""
        resolved = Path(path) if Path(path).is_absolute() else self.datasets_dir / path
        return self._read_and_validate(resolved)

    def load_all(self) -> list[GoldenItem]:
        """Load every domain dataset and return the combined list."""
        combined: list[GoldenItem] = []
        for fname in _DOMAIN_FILES:
            try:
                combined.extend(self.load(fname))
            except FileNotFoundError:
                logger.warning(f"Dataset file not found, skipping: {fname}")
        logger.info(f"Loaded {len(combined)} golden items from {len(_DOMAIN_FILES)} domain files")
        return combined

    def load_golden(self) -> list[GoldenItem]:
        """Load the combined golden_dataset.json file."""
        return self._read_and_validate(_GOLDEN_PATH)

    def save_golden(self, items: Sequence[GoldenItem], path: Path | None = None) -> None:
        """Serialize items back to JSON (useful for dataset generation scripts)."""
        out_path = path or _GOLDEN_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [item.model_dump() for item in items]
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(data)} golden items to {out_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_and_validate(self, path: Path) -> list[GoldenItem]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        items: list[GoldenItem] = []
        for idx, entry in enumerate(raw):
            try:
                items.append(GoldenItem(**entry))
            except Exception as exc:
                logger.warning(f"Skipping malformed entry #{idx} in {path}: {exc}")

        logger.debug(f"Loaded {len(items)} items from {path.name}")
        return items
