"""
Agentic RAG Orchestrator — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `ai.rag.agentic` package.
! Import from `ai.rag.agentic` directly in new code.
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
