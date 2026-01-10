"""
Document Grader
================
Corrective RAG (CRAG) document relevance grading.

Evaluates retrieved documents for relevance to the query.
If documents are not relevant, triggers web search fallback.
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.rag.knowledge_base import Document


class GradeResult(BaseModel):
    """Result of document grading."""
    
    document_id: str
    is_relevant: bool
    score: float = 0.0
    reasoning: str = ""


class GradingResult(BaseModel):
    """Result of grading multiple documents."""
    
    relevant_documents: list[Document]
    irrelevant_count: int
    needs_web_search: bool
    total_graded: int


# System prompt for document grading
DOC_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant.

Respond with only JSON:
{"score": "yes"} or {"score": "no"}"""


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
        """Grade document using LLM."""
        import json
        from src.orchestrator.llm_provider import LLMMessage
        
        user_content = f"""Document:
{document.text}

User Question: {query}

Is this document relevant to the question?"""
        
        messages = [
            LLMMessage(role="system", content=DOC_GRADER_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]
        
        try:
            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=50,
            )
            
            result = json.loads(response.content)
            is_relevant = result.get("score", "no").lower() == "yes"
            
            return GradeResult(
                document_id=document.id,
                is_relevant=is_relevant,
                score=1.0 if is_relevant else 0.0,
                reasoning="LLM grading",
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
        
        is_relevant = score > 0.3
        
        return GradeResult(
            document_id=document.id,
            is_relevant=is_relevant,
            score=score,
            reasoning=f"Keyword overlap: {overlap} words",
        )
    
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
