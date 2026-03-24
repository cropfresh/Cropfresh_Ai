"""
Confidence Gate — "I Don't Know" Safety Fallback
=================================================
Prevents hallucinated answers for safety-critical agricultural queries.

Features:
  - Topic safety classification (safe / safety-critical / platform)
  - Grounding score calculation (% of claims supported by docs)
  - Confidence-based gating with configurable thresholds
  - Decline response for ungrounded or unsafe answers

Reference: ADR-010 — Advanced Agentic RAG System
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from src.rag.language_support import detect_response_language, get_localized_message


class SafetyLevel(str, Enum):
    """Safety classification for agricultural queries."""
    SAFE = "safe"
    SAFETY_CRITICAL = "safety_critical"
    PLATFORM = "platform"


class GatedAnswer(BaseModel):
    """Result of the confidence gate check."""
    answer: str
    is_approved: bool = True
    safety_level: SafetyLevel = SafetyLevel.SAFE
    grounding_score: float = 1.0
    confidence_threshold: float = 0.70
    decline_reason: Optional[str] = None


#! Safety-critical keywords that require higher confidence
SAFETY_CRITICAL_KEYWORDS = {
    # Pesticide / chemical dosage
    "pesticide", "insecticide", "fungicide", "herbicide",
    "spray", "dose", "dosage", "ml per litre", "grams per litre",
    "poison", "toxic", "chemical",
    # Financial / legal
    "loan", "subsidy", "insurance", "legal", "contract",
    "compensation", "payment", "bank",
    # Health
    "disease", "infection", "health risk", "contamination",
}

PLATFORM_KEYWORDS = {
    "register", "sign up", "login", "account", "password",
    "app", "download", "cropfresh", "support", "help",
}

# Kannada safety keywords
SAFETY_CRITICAL_KEYWORDS_KN = {
    "ಔಷಧ", "ಕೀಟನಾಶಕ", "ವಿಷ", "ಸಾಲ", "ವಿಮೆ",
}

class ConfidenceGate:
    """
    Gates RAG answers based on grounding confidence and safety level.

    Thresholds:
      - Safety-critical queries: grounding ≥ 0.85
      - General agronomy: grounding ≥ 0.70
      - Platform FAQ: grounding ≥ 0.60
    """

    THRESHOLDS = {
        SafetyLevel.SAFETY_CRITICAL: 0.85,
        SafetyLevel.SAFE: 0.70,
        SafetyLevel.PLATFORM: 0.60,
    }

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def gate(
        self,
        query: str,
        answer: str,
        documents: list[Any],
        faithfulness: float = 0.80,
        relevance: float = 0.80,
    ) -> GatedAnswer:
        """
        Check if an answer should be approved or declined.

        Args:
            query: Original user query
            answer: Generated answer to evaluate
            documents: Source documents used
            faithfulness: Faithfulness score from evaluator
            relevance: Relevance score from evaluator

        Returns:
            GatedAnswer with approval status
        """
        safety_level = self.classify_safety(query)
        response_language = detect_response_language(query)
        threshold = self.THRESHOLDS[safety_level]

        grounding = self._calculate_grounding(answer, documents)

        # Combine evaluator scores with grounding
        combined = (faithfulness * 0.4) + (relevance * 0.3) + (grounding * 0.3)

        is_approved = combined >= threshold

        if not is_approved:
            logger.warning(
                f"Answer DECLINED: safety={safety_level.value} | "
                f"combined={combined:.2f} < threshold={threshold}"
            )
            return GatedAnswer(
                answer=self._decline_response(safety_level, response_language),
                is_approved=False,
                safety_level=safety_level,
                grounding_score=grounding,
                confidence_threshold=threshold,
                decline_reason=(
                    f"Combined confidence {combined:.2f} below "
                    f"{safety_level.value} threshold {threshold}"
                ),
            )

        logger.info(
            f"Answer APPROVED: safety={safety_level.value} | "
            f"combined={combined:.2f} ≥ threshold={threshold}"
        )
        return GatedAnswer(
            answer=answer,
            is_approved=True,
            safety_level=safety_level,
            grounding_score=grounding,
            confidence_threshold=threshold,
        )

    def _decline_response(self, safety_level: SafetyLevel, language: str) -> str:
        key_map = {
            SafetyLevel.SAFETY_CRITICAL: "decline_safety_critical",
            SafetyLevel.SAFE: "decline_safe",
            SafetyLevel.PLATFORM: "decline_platform",
        }
        return get_localized_message(key_map[safety_level], language)

    def classify_safety(self, query: str) -> SafetyLevel:
        """Classify query safety level using keyword matching."""
        q_lower = query.lower()

        if any(kw in q_lower for kw in SAFETY_CRITICAL_KEYWORDS):
            return SafetyLevel.SAFETY_CRITICAL

        # Kannada safety keywords
        if any(kw in query for kw in SAFETY_CRITICAL_KEYWORDS_KN):
            return SafetyLevel.SAFETY_CRITICAL

        if any(kw in q_lower for kw in PLATFORM_KEYWORDS):
            return SafetyLevel.PLATFORM

        return SafetyLevel.SAFE

    def _calculate_grounding(
        self,
        answer: str,
        documents: list[Any],
    ) -> float:
        """Calculate what fraction of answer claims are grounded in docs."""
        if not answer or not documents:
            return 0.0

        answer_words = set(re.sub(r"[^\w\s]", "", answer.lower()).split())
        #! Filter stop words to avoid inflated scores
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                       "to", "for", "of", "and", "or", "it", "this", "that", "with"}
        answer_words -= stop_words

        doc_words: set[str] = set()
        for doc in documents:
            text = getattr(doc, "text", str(doc))
            doc_words.update(re.sub(r"[^\w\s]", "", text.lower()).split())
        doc_words -= stop_words

        if not answer_words:
            return 1.0

        grounded = len(answer_words & doc_words)
        return grounded / len(answer_words)
