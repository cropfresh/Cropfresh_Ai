"""Enhanced relevance grading with freshness-aware fallback logic."""

from __future__ import annotations

import time

from loguru import logger

from src.rag.grading.models import (
    DOC_GRADER_PROMPT,
    MARKET_DOC_MAX_AGE_SECONDS,
    MARKET_KEYWORDS,
    GradeResult,
    GradingResult,
)
from src.rag.knowledge_base import Document


class DocumentGrader:
    """Score document relevance and decide whether web fallback is needed."""

    def __init__(self, llm=None, relevance_threshold: float = 0.5):
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    async def grade_document(self, document: Document, query: str) -> GradeResult:
        if self.llm is None:
            return self._grade_simple(document, query)
        return await self._grade_with_llm(document, query)

    async def _grade_with_llm(self, document: Document, query: str) -> GradeResult:
        import json

        from src.orchestrator.llm_provider import LLMMessage

        messages = [
            LLMMessage(role="system", content=DOC_GRADER_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Document:\n{document.text[:1000]}\n\n"
                    f"User Question: {query}\n\n"
                    "Rate this document's relevance (0.0 to 1.0)."
                ),
            ),
        ]

        try:
            response = await self.llm.generate(messages, temperature=0.0, max_tokens=80)
            result = json.loads(response.content)
            score = float(result.get("score", 0.0))
            score, penalty_applied = self._apply_time_decay(score, query, document)
            return GradeResult(
                document_id=document.id,
                is_relevant=score >= 0.4,
                score=score,
                reasoning=result.get("reasoning", "LLM grading"),
                time_penalty_applied=penalty_applied,
            )
        except Exception as exc:
            logger.warning("LLM grading failed: {}", exc)
            return self._grade_simple(document, query)

    def _grade_simple(self, document: Document, query: str) -> GradeResult:
        query_words = set(query.lower().split())
        doc_words = set(document.text.lower().split())
        overlap = len(query_words & doc_words)
        score = overlap / len(query_words) if query_words else 0.0

        if document.score is not None:
            score = (score + document.score) / 2

        score, penalty_applied = self._apply_time_decay(score, query, document)
        return GradeResult(
            document_id=document.id,
            is_relevant=score > 0.3,
            score=score,
            reasoning=f"Keyword overlap: {overlap} words",
            time_penalty_applied=penalty_applied,
        )

    def _apply_time_decay(
        self,
        score: float,
        query: str,
        document: Document,
    ) -> tuple[float, bool]:
        if not any(keyword in query.lower() for keyword in MARKET_KEYWORDS):
            return score, False

        metadata = getattr(document, "metadata", {}) or {}
        doc_timestamp = metadata.get("created_at") or metadata.get("timestamp")
        if not isinstance(doc_timestamp, (int, float)):
            return score, False

        try:
            age_seconds = time.time() - doc_timestamp
        except (TypeError, ValueError):
            return score, False

        if age_seconds <= MARKET_DOC_MAX_AGE_SECONDS:
            return score, False

        decayed = score * 0.5
        logger.debug(
            "Time-decay applied: {:.2f} -> {:.2f} (age: {:.0f}h)",
            score,
            decayed,
            age_seconds / 3600,
        )
        return decayed, True

    async def grade_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> GradingResult:
        if not documents:
            return GradingResult(
                relevant_documents=[],
                irrelevant_count=0,
                needs_web_search=True,
                total_graded=0,
            )

        logger.info("Grading {} documents for relevance", len(documents))
        relevant_docs: list[Document] = []
        irrelevant_count = 0

        for doc in documents:
            result = await self.grade_document(doc, query)
            if result.is_relevant:
                relevant_docs.append(doc)
                logger.debug("Document {}... is RELEVANT", doc.id[:8])
            else:
                irrelevant_count += 1
                logger.debug("Document {}... is NOT RELEVANT", doc.id[:8])

        relevance_ratio = len(relevant_docs) / len(documents)
        needs_web_search = relevance_ratio < self.relevance_threshold
        if needs_web_search:
            logger.info("Low relevance - triggering web search fallback")

        return GradingResult(
            relevant_documents=relevant_docs,
            irrelevant_count=irrelevant_count,
            needs_web_search=needs_web_search,
            total_graded=len(documents),
        )
