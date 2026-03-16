"""
RAG Graph State — TypedDict for LangGraph state machine (ADR-010 Phase 3).

Defines the shared state object passed through all LangGraph nodes.
Each key is updated by specific nodes as the pipeline progresses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class RAGGraphState(TypedDict, total=False):
    """Shared state for the LangGraph RAG pipeline.

    Keys are updated by individual nodes:
    - rewrite_node  → rewritten_queries, rewrite_strategy
    - retrieve_node → documents
    - grade_node    → relevant_documents, needs_web_search
    - generate_node → answer, generation_model
    - cite_node     → cited_answer, sources
    - gate_node     → is_approved, safety_level, grounding_score
    - evaluate_node → faithfulness, relevance, confidence, should_retry
    """

    # ── Input ──────────────────────────────────────────────────────────
    query: str
    context: str
    has_image: bool
    knowledge_base: Any
    llm: Any
    web_search_tool: Any
    route: str
    route_reason: str
    tool_calls: list[str]

    # ── Query Rewriting (Phase 1) ──────────────────────────────────────
    rewritten_queries: list[str]
    rewrite_strategy: str
    hyde_document: str

    # ── Retrieval ──────────────────────────────────────────────────────
    documents: list[Any]
    retrieval_source: str  # "vector" | "graph" | "web" | "hybrid"

    # ── Grading ────────────────────────────────────────────────────────
    relevant_documents: list[Any]
    needs_web_search: bool
    grading_scores: list[float]

    # ── Generation ─────────────────────────────────────────────────────
    answer: str
    generation_model: str

    # ── Citation (Phase 1) ─────────────────────────────────────────────
    cited_answer: str
    sources: list[dict[str, Any]]
    citation_count: int

    # ── Confidence Gate (Phase 1) ──────────────────────────────────────
    is_approved: bool
    safety_level: str
    grounding_score: float
    decline_reason: str

    # ── Self-Evaluation ────────────────────────────────────────────────
    faithfulness: float
    relevance: float
    confidence: float
    should_retry: bool
    eval_reason: str

    # ── Control Flow ───────────────────────────────────────────────────
    retry_count: int
    max_retries: int
    web_search_attempted: bool
    current_node: str
    error: str


class GraphRunResult(BaseModel):
    """Final output from a compiled RAG graph run."""

    answer: str = ""
    cited_answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    is_approved: bool = False
    safety_level: str = "safe"
    confidence: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    retry_count: int = 0
    grounding_score: float = 0.0
    documents_used: int = 0
    rewrite_strategy: str = "none"
    route: str = ""
    route_reason: str = ""
    tool_calls: list[str] = Field(default_factory=list)
    documents: list[Any] = Field(default_factory=list)
    error: str = ""
