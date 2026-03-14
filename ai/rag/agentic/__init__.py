"""
Agentic RAG Orchestrator Package.

Re-exports main classes for backward compatibility.
"""

from ai.rag.agentic.evaluator import AgenticSelfEvaluator
from ai.rag.agentic.models import (
    Draft,
    EvalGate,
    OrchestratorResult,
    RetrievalPlan,
    ToolCall,
)
from ai.rag.agentic.orchestrator import AgenticOrchestrator
from ai.rag.agentic.planner import RetrievalPlanner
from ai.rag.agentic.speculative import SpeculativeDraftEngine

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
