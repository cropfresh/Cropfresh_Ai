"""
Shared Pydantic models for the RAGAS Evaluation Framework.

Defines the data structures used across the evaluation pipeline:
- GoldenItem: a single Q&A entry in the golden dataset
- EvalResults: aggregate RAGAS metric scores for a full run
- EvalReport: final report metadata + per-question breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------


class GoldenItem(BaseModel):
    """A single Q&A pair used as ground truth for RAG evaluation."""

    question: str
    ground_truth: str
    contexts: list[str] = Field(default_factory=list)
    agent_domain: str = "general"  # agronomy | commerce | platform | multilingual
    difficulty: str = "medium"     # easy | medium | hard
    language: str = "en"


# ---------------------------------------------------------------------------
# Evaluation Results
# ---------------------------------------------------------------------------


class PerQuestionScore(BaseModel):
    """Metric scores for one Q&A pair."""

    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def average(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4


@dataclass
class EvalResults:
    """Aggregate scores returned after a full evaluation run."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    per_question: list[PerQuestionScore] = field(default_factory=list)
    dataset_path: str = ""
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_score(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4

    def worst_questions(self, n: int = 5) -> list[PerQuestionScore]:
        """Return the n questions with the lowest average score."""
        return sorted(self.per_question, key=lambda q: q.average)[:n]

    def meets_targets(self) -> dict[str, bool]:
        """Check whether each metric meets the baseline target."""
        return {
            "faithfulness": self.faithfulness > 0.80,
            "answer_relevancy": self.answer_relevancy > 0.75,
            "context_precision": self.context_precision > 0.70,
            "context_recall": self.context_recall > 0.70,
        }


# ---------------------------------------------------------------------------
# Report metadata
# ---------------------------------------------------------------------------


class EvalReportMeta(BaseModel):
    """Lightweight metadata written alongside the markdown report."""

    generated_at: str
    dataset_path: str
    num_questions: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    all_targets_met: bool
    per_question: list[dict[str, Any]] = Field(default_factory=list)
