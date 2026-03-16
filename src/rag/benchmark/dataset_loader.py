from __future__ import annotations

import json
from pathlib import Path

from src.rag.benchmark.models import GoldenEntry

DATASETS_DIR = Path(__file__).parent / "datasets"
DATASET_FILES = {
    "core_live": "core_live.json",
    "full": "full.json",
}


class BenchmarkDatasetLoader:
    """Load benchmark datasets from JSON files."""

    def __init__(self, datasets_dir: Path | None = None):
        self.datasets_dir = datasets_dir or DATASETS_DIR

    def load(self, subset: str = "full") -> list[GoldenEntry]:
        filename = DATASET_FILES.get(subset, f"{subset}.json")
        dataset_path = self.datasets_dir / filename
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")
        raw_entries = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [GoldenEntry(**entry) for entry in raw_entries]
