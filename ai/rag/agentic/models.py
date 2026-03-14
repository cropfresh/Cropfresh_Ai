"""
Data models for the Agentic RAG Orchestrator.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool call in a retrieval plan."""
    tool_name: str = Field(description="Name of the tool to call")
    params: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    can_parallelize: bool = Field(
        default=True,
        description="True if this call can run in parallel with other parallelizable calls"
    )
    priority: int = Field(default=1, description="Execution priority (lower = run first)")


class RetrievalPlan(BaseModel):
    """
    A structured retrieval plan from the RetrievalPlanner.

    Contains an ordered list of tool calls to execute,
    with parallel execution hints.
    """
    plan: list[ToolCall] = Field(description="Ordered list of tool calls to execute")
    confidence_threshold: float = Field(
        default=0.75,
        description="Minimum answer confidence before returning (0–1)"
    )
    query: str = Field(description="Original query this plan was generated for")
    plan_reasoning: str = Field(default="", description="Planner's reasoning for LangSmith")


class EvalGate(BaseModel):
    """Result of the AgenticSelfEvaluator confidence check."""
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence 0–1")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Faithfulness to retrieved docs")
    relevance: float = Field(ge=0.0, le=1.0, description="Answer relevancy to query")
    should_retry: bool = Field(description="True if confidence < threshold")
    reason: str = Field(description="Evaluation explanation for logging")


class Draft(BaseModel):
    """A single speculative draft from a drafter LLM."""
    content: str = Field(description="Generated draft text")
    source_doc_indices: list[int] = Field(
        default_factory=list,
        description="Which document subset indices this draft used"
    )
    drafter_model: str = Field(default="groq/llama-3.1-8b-instant")
    generation_ms: float = Field(default=0.0)


class OrchestratorResult(BaseModel):
    """Final result from the AgenticOrchestrator."""
    answer: str
    retrieved_documents: list[Any] = Field(default_factory=list)
    plan_used: Optional[RetrievalPlan] = None
    eval_gate: Optional[EvalGate] = None
    retry_count: int = 0
    total_latency_ms: float = 0.0
    estimated_cost_inr: float = 0.0
    tools_called: list[str] = Field(default_factory=list)
