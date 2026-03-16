from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeResponse(BaseModel):
    """User-facing knowledge response."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    query_type: str = ""
    steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSourceDetail(BaseModel):
    """Resolved source details for benchmark/debug runs."""

    source: str
    title: str = ""
    timestamp: str = ""
    is_fresh: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkDebugResult(BaseModel):
    """Internal debug payload used by the benchmark runner."""

    answer: str
    raw_answer: str = ""
    sources: list[str] = Field(default_factory=list)
    source_details: list[BenchmarkSourceDetail] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    route: str = ""
    tool_calls: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    retry_count: int = 0
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
