"""Compatibility exports for the app-facing RAG grading surface."""

from ai.rag.grader import (
    DOC_GRADER_PROMPT,
    MARKET_DOC_MAX_AGE_SECONDS,
    MARKET_KEYWORDS,
    DocumentGrader,
    GradeResult,
    GradingResult,
    HallucinationChecker,
)

__all__ = [
    "DOC_GRADER_PROMPT",
    "MARKET_DOC_MAX_AGE_SECONDS",
    "MARKET_KEYWORDS",
    "DocumentGrader",
    "GradeResult",
    "GradingResult",
    "HallucinationChecker",
]
