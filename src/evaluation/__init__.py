"""
src/evaluation — RAGAS Evaluation Framework for CropFresh RAG Pipeline.

Modules
-------
models          — shared Pydantic/dataclass types (GoldenItem, EvalResults, …)
dataset_loader  — load/save golden dataset JSON files
ragas_evaluator — core RAGAS metrics engine (dual-mode: real + heuristic)
report_generator— markdown + JSON report writer
eval_runner     — batch orchestrator and CLI entry-point
"""

from src.evaluation.dataset_loader import DatasetLoader
from src.evaluation.eval_runner import EvalRunner
from src.evaluation.models import EvalReportMeta, EvalResults, GoldenItem, PerQuestionScore
from src.evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGPipeline,
    RAGResponse,
    create_ragas_evaluator,
)
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.voice_benchmark_models import (
    VoiceBenchmarkArtifact,
    VoiceBenchmarkEntry,
    VoiceBenchmarkObservation,
    VoiceBenchmarkReport,
)
from src.evaluation.voice_benchmark_runner import VoiceBenchmarkRunner

__all__ = [
    # Models
    "GoldenItem",
    "PerQuestionScore",
    "EvalResults",
    "EvalReportMeta",
    # Dataset I/O
    "DatasetLoader",
    # Evaluator
    "RAGASEvaluator",
    "RAGPipeline",
    "RAGResponse",
    "create_ragas_evaluator",
    # Report
    "ReportGenerator",
    # Runner
    "EvalRunner",
    "VoiceBenchmarkEntry",
    "VoiceBenchmarkObservation",
    "VoiceBenchmarkArtifact",
    "VoiceBenchmarkReport",
    "VoiceBenchmarkRunner",
]
