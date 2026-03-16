"""Convenience exports for the JSON-backed RAG benchmark package."""

from ai.rag.evaluation.dataset_loader import BenchmarkDatasetLoader
from ai.rag.evaluation.golden_dataset import GoldenEntry, get_by_category, get_golden_dataset
from ai.rag.evaluation.guardrail import EvalGuardrail, GuardrailResult, GuardrailThresholds
from ai.rag.evaluation.metrics import EvalMetrics, RAGEvaluator
from ai.rag.evaluation.reference_resolver import ReferenceResolver

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
