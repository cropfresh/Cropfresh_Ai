"""
Document Grader
================
Corrective RAG (CRAG) document relevance grading.

Evaluates retrieved documents for relevance to the query.
If documents are not relevant, triggers web search fallback.

Enhancements (ADR-010):
  - Continuous 0–1 relevance scoring (not binary)
  - Time-decay penalty for stale market/price documents
  - Category-aware grading thresholds
"""


import time

from loguru import logger
from pydantic import BaseModel

from src.rag.knowledge_base import Document


class GradeResult(BaseModel):
    """Result of document grading."""

    document_id: str
    is_relevant: bool
    score: float = 0.0
    reasoning: str = ""
    time_penalty_applied: bool = False


class GradingResult(BaseModel):
    """Result of grading multiple documents."""

    relevant_documents: list[Document]
    irrelevant_count: int
    needs_web_search: bool
    total_graded: int


# System prompt for document grading (ADR-010: continuous scoring)
DOC_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

Rate the document's relevance on a scale from 0.0 to 1.0:
  1.0 = Directly answers the question with specific information
  0.7 = Contains relevant information but not a direct answer
  0.4 = Tangentially related but may be useful
  0.1 = Barely related or mostly irrelevant
  0.0 = Completely irrelevant

Respond with ONLY JSON:
{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}"""

#! Keywords that indicate market/price queries needing fresh data
MARKET_KEYWORDS = {
    "price", "rate", "cost", "mandi", "market", "sell",
    "buy", "today", "current", "ಬೆಲೆ", "ದರ",
}

# Max age (seconds) before time-decay kicks in for market docs
MARKET_DOC_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days


class DocumentGrader:
    """
    CRAG Document Grader.

    Evaluates document relevance using LLM.
    Triggers web search when documents are not relevant.

    Usage:
        grader = DocumentGrader(llm=llm_provider)
        result = await grader.grade_documents(docs, query)
        if result.needs_web_search:
            # Fallback to web search
    """

    def __init__(self, llm=None, relevance_threshold: float = 0.5):
        """
        Initialize document grader.

        Args:
            llm: LLM provider for grading
            relevance_threshold: Minimum ratio of relevant docs to avoid web search
        """
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    async def grade_document(
        self,
        document: Document,
        query: str,
    ) -> GradeResult:
        """
        Grade a single document for relevance.

        Args:
            document: Document to evaluate
            query: User query

        Returns:
            GradeResult with relevance decision
        """
        if self.llm is None:
            # Fallback to simple relevance check
            return self._grade_simple(document, query)

        return await self._grade_with_llm(document, query)

    async def _grade_with_llm(
        self,
        document: Document,
        query: str,
    ) -> GradeResult:
        """Grade document using LLM with continuous scoring."""
        import json

        from src.orchestrator.llm_provider import LLMMessage

        user_content = f"""Document:
{document.text[:1000]}

User Question: {query}

Rate this document's relevance (0.0 to 1.0)."""

        messages = [
            LLMMessage(role="system", content=DOC_GRADER_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=80,
            )

            result = json.loads(response.content)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "LLM grading")

            # Apply time-decay for stale market documents
            score, penalty_applied = self._apply_time_decay(
                score, query, document,
            )

            return GradeResult(
                document_id=document.id,
                is_relevant=score >= 0.4,
                score=score,
                reasoning=reasoning,
                time_penalty_applied=penalty_applied,
            )

        except Exception as e:
            logger.warning(f"LLM grading failed: {e}")
            return self._grade_simple(document, query)

    def _grade_simple(
        self,
        document: Document,
        query: str,
    ) -> GradeResult:
        """
        Simple keyword-based relevance check.

        Used as fallback when LLM is unavailable.
        """
        query_words = set(query.lower().split())
        doc_words = set(document.text.lower().split())

        # Calculate overlap
        overlap = len(query_words & doc_words)
        score = overlap / len(query_words) if query_words else 0

        # Use document's similarity score if available
        if document.score is not None:
            score = (score + document.score) / 2

        # Apply time-decay for stale market documents
        score, penalty_applied = self._apply_time_decay(
            score, query, document,
        )

        is_relevant = score > 0.3

        return GradeResult(
            document_id=document.id,
            is_relevant=is_relevant,
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
        """
        Apply time-decay penalty for stale market/price documents.

        Market docs older than 7 days get score × 0.5.
        """
        q_lower = query.lower()
        is_market_query = any(kw in q_lower for kw in MARKET_KEYWORDS)

        if not is_market_query:
            return score, False

        metadata = getattr(document, "metadata", {}) or {}
        doc_timestamp = metadata.get("created_at") or metadata.get("timestamp")

        if doc_timestamp is None:
            return score, False

        try:
            if isinstance(doc_timestamp, (int, float)):
                age_seconds = time.time() - doc_timestamp
            else:
                return score, False

            if age_seconds > MARKET_DOC_MAX_AGE_SECONDS:
                decayed = score * 0.5
                logger.debug(
                    f"Time-decay applied: {score:.2f} → {decayed:.2f} "
                    f"(age: {age_seconds / 3600:.0f}h)"
                )
                return decayed, True
        except (TypeError, ValueError):
            pass

        return score, False

    async def grade_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> GradingResult:
        """
        Grade multiple documents and determine if web search is needed.

        Args:
            documents: List of documents to grade
            query: User query

        Returns:
            GradingResult with filtered docs and web search flag
        """
        if not documents:
            return GradingResult(
                relevant_documents=[],
                irrelevant_count=0,
                needs_web_search=True,  # No docs = need web search
                total_graded=0,
            )

        logger.info(f"Grading {len(documents)} documents for relevance")

        relevant_docs = []
        irrelevant_count = 0

        for doc in documents:
            result = await self.grade_document(doc, query)

            if result.is_relevant:
                relevant_docs.append(doc)
                logger.debug(f"Document {doc.id[:8]}... is RELEVANT")
            else:
                irrelevant_count += 1
                logger.debug(f"Document {doc.id[:8]}... is NOT RELEVANT")

        # Decide if web search is needed
        # If more than half of docs are irrelevant, trigger web search
        relevance_ratio = len(relevant_docs) / len(documents) if documents else 0
        needs_web_search = relevance_ratio < self.relevance_threshold

        if needs_web_search:
            logger.info("Low relevance - triggering web search fallback")

        return GradingResult(
            relevant_documents=relevant_docs,
            irrelevant_count=irrelevant_count,
            needs_web_search=needs_web_search,
            total_graded=len(documents),
        )


class HallucinationChecker:
    """
    Self-RAG Hallucination Checker.

    Validates that generated answers are grounded in retrieved documents.
    """

    HALLUCINATION_PROMPT = """You are a grader checking if an LLM generation is grounded in and supported by a set of retrieved documents.

Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the documents.
'yes' means the answer is supported by the facts in the documents.
'no' means the answer contains information not found in the documents.

Respond with only JSON:
{"score": "yes", "reasoning": "brief explanation"} or {"score": "no", "reasoning": "what was not grounded"}"""

    def __init__(self, llm=None):
        """Initialize hallucination checker."""
        self.llm = llm

    async def check(
        self,
        answer: str,
        documents: list[Document],
        query: str,
    ) -> tuple[bool, str]:
        """
        Check if answer is grounded in documents.

        Args:
            answer: Generated answer to check
            documents: Source documents
            query: Original query

        Returns:
            (is_grounded, reasoning)
        """
        if self.llm is None:
            return True, "No LLM available for hallucination check"

        import json

        from src.orchestrator.llm_provider import LLMMessage

        # Format documents
        docs_text = "\n\n".join([f"Document {i+1}:\n{d.text}" for i, d in enumerate(documents)])

        user_content = f"""Documents:
{docs_text}

User Question: {query}

LLM Answer: {answer}

Is this answer grounded in the documents?"""

        messages = [
            LLMMessage(role="system", content=self.HALLUCINATION_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=100,
            )

            result = json.loads(response.content)
            is_grounded = result.get("score", "yes").lower() == "yes"
            reasoning = result.get("reasoning", "")

            return is_grounded, reasoning

        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return True, f"Check failed: {e}"
