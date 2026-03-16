"""Shared models and constants for enhanced document grading."""

from __future__ import annotations

from pydantic import BaseModel

from src.rag.knowledge_base import Document


class GradeResult(BaseModel):
    """Result of grading a single retrieved document."""

    document_id: str
    is_relevant: bool
    score: float = 0.0
    reasoning: str = ""
    time_penalty_applied: bool = False


class GradingResult(BaseModel):
    """Aggregate grading result across a retrieved document set."""

    relevant_documents: list[Document]
    irrelevant_count: int
    needs_web_search: bool
    total_graded: int


DOC_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

Rate the document's relevance on a scale from 0.0 to 1.0:
  1.0 = Directly answers the question with specific information
  0.7 = Contains relevant information but not a direct answer
  0.4 = Tangentially related but may be useful
  0.1 = Barely related or mostly irrelevant
  0.0 = Completely irrelevant

Respond with ONLY JSON:
{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}"""

MARKET_KEYWORDS = {
    "price",
    "rate",
    "cost",
    "mandi",
    "market",
    "sell",
    "buy",
    "today",
    "current",
    "ಬೆಲೆ",
    "ದರ",
}

MARKET_DOC_MAX_AGE_SECONDS = 7 * 24 * 3600
