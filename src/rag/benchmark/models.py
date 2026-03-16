from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GoldenEntry(BaseModel):
    """Benchmark dataset entry for static or live evaluation."""

    id: str
    query: str
    category: str
    mode: Literal["static", "live"] = "static"
    ground_truth: str = ""
    contexts: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    language: str = "en"
    reference_resolver: str = ""
    resolver_params: dict[str, Any] = Field(default_factory=dict)


class ResolvedReference(BaseModel):
    """Resolved reference answer and contexts for a benchmark entry."""

    ground_truth: str
    contexts: list[str] = Field(default_factory=list)
    resolved_at: datetime = Field(default_factory=_utcnow)
    freshness_ok: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveRunExtras(BaseModel):
    """Extra deterministic benchmark signals beyond semantic RAGAS metrics."""

    citation_coverage: float = 0.0
    freshness_compliance: float = 0.0
    per_category: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
