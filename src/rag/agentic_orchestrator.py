"""Backward compatibility redirect for the split ``src.rag.agentic`` package."""

from src.rag.agentic.evaluator import AgenticSelfEvaluator
from src.rag.agentic.models import (
    Draft,
    EvalGate,
    OrchestratorResult,
    RetrievalPlan,
    ToolCall,
)
from src.rag.agentic.orchestrator import AgenticOrchestrator
from src.rag.agentic.planner import RetrievalPlanner
from src.rag.agentic.speculative import SpeculativeDraftEngine

__all__ = [
    "AgenticOrchestrator",
    "AgenticSelfEvaluator",
    "Draft",
    "EvalGate",
    "OrchestratorResult",
    "RetrievalPlan",
    "RetrievalPlanner",
    "SpeculativeDraftEngine",
    "ToolCall",
]
