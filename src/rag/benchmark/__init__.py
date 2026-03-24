"""Convenience exports for the JSON-backed RAG benchmark package."""

from src.rag.benchmark.dataset_loader import BenchmarkDatasetLoader
from src.rag.benchmark.golden_dataset import GoldenEntry, get_by_category, get_golden_dataset
from src.rag.benchmark.guardrail import EvalGuardrail, GuardrailResult, GuardrailThresholds
from src.rag.benchmark.metrics import EvalMetrics, RAGEvaluator
from src.rag.benchmark.reference_resolver import ReferenceResolver

__all__ = [
    "BenchmarkDatasetLoader",
    "RAGEvaluator",
    "EvalMetrics",
    "GoldenEntry",
    "get_golden_dataset",
    "get_by_category",
    "EvalGuardrail",
    "GuardrailResult",
    "GuardrailThresholds",
    "ReferenceResolver",
]
