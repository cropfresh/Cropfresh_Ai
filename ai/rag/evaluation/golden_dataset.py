"""Dataset helpers for the JSON-backed benchmark datasets."""

from ai.rag.evaluation.dataset_loader import BenchmarkDatasetLoader
from ai.rag.evaluation.models import GoldenEntry


def get_golden_dataset(subset: str = "full") -> list[GoldenEntry]:
    """Load a named benchmark dataset."""
    return BenchmarkDatasetLoader().load(subset)


def get_by_category(category: str, subset: str = "full") -> list[GoldenEntry]:
    """Filter a named dataset by category."""
    return [entry for entry in get_golden_dataset(subset=subset) if entry.category == category]
