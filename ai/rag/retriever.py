"""
RAG Retriever
=============
Multi-source retrieval with query rewriting and context handling.

Features:
- Knowledge base retrieval
- Query rewriting for clarity
- Query decomposition for complex questions
- Context-aware retrieval
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.rag.knowledge_base import Document, KnowledgeBase, SearchResult


class RetrievalResult(BaseModel):
    """Result of retrieval operation."""
    
    documents: list[Document]
    query: str
    rewritten_query: Optional[str] = None
    source: str = "knowledge_base"
    retrieval_time_ms: float = 0.0


# Query rewriting prompt
REWRITE_PROMPT = """You are a query rewriter for an agricultural knowledge system.

Your task is to rewrite user questions to be clearer and more specific for retrieval.
Keep the meaning the same but improve clarity.

Original query: {query}

Provide ONLY the rewritten query, nothing else."""


class RAGRetriever:
    """
    Multi-source RAG Retriever.
    
    Handles retrieval from knowledge base with query optimization.
    
    Usage:
        retriever = RAGRetriever(kb, llm)
        docs = await retriever.retrieve("How to grow tomatoes?")
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm=None,
    ):
        """
        Initialize retriever.
        
        Args:
            knowledge_base: Qdrant knowledge base
            llm: LLM for query rewriting
        """
        self.knowledge_base = knowledge_base
        self.llm = llm
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        rewrite: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            category: Optional category filter
            rewrite: Whether to rewrite query first
            
        Returns:
            RetrievalResult with documents
        """
        import time
        start_time = time.time()
        
        # Optionally rewrite query
        search_query = query
        rewritten = None
        if rewrite and self.llm:
            rewritten = await self._rewrite_query(query)
            if rewritten:
                search_query = rewritten
                logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
        
        # Search knowledge base
        result = await self.knowledge_base.search(
            query=search_query,
            top_k=top_k,
            category=category,
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            documents=result.documents,
            query=query,
            rewritten_query=rewritten,
            source="knowledge_base",
            retrieval_time_ms=retrieval_time,
        )
    
    async def _rewrite_query(self, query: str) -> Optional[str]:
        """Rewrite query for better retrieval."""
        from src.orchestrator.llm_provider import LLMMessage
        
        try:
            messages = [
                LLMMessage(role="user", content=REWRITE_PROMPT.format(query=query)),
            ]
            
            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=100,
            )
            
            rewritten = response.content.strip()
            # Only use if different and not too long
            if rewritten and rewritten != query and len(rewritten) < len(query) * 2:
                return rewritten
            return None
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return None
    
    async def retrieve_with_decomposition(
        self,
        query: str,
        sub_queries: list[str],
        top_k_per_query: int = 3,
    ) -> RetrievalResult:
        """
        Retrieve documents for decomposed sub-queries.
        
        Retrieves for each sub-query and combines results.
        
        Args:
            query: Original query
            sub_queries: List of sub-queries
            top_k_per_query: Docs per sub-query
            
        Returns:
            Combined RetrievalResult
        """
        import time
        start_time = time.time()
        
        all_docs = []
        seen_ids = set()
        
        for sub_q in sub_queries:
            result = await self.knowledge_base.search(
                query=sub_q,
                top_k=top_k_per_query,
            )
            
            # Deduplicate
            for doc in result.documents:
                if doc.id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc.id)
        
        # Sort by score
        all_docs.sort(key=lambda d: d.score or 0, reverse=True)
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            documents=all_docs,
            query=query,
            source="knowledge_base_decomposed",
            retrieval_time_ms=retrieval_time,
        )


# Query decomposition prompt
DECOMPOSE_PROMPT = """Break down this complex question into simpler sub-questions that can be answered independently.

Complex Question: {query}

Return a JSON array of 2-4 simpler sub-questions:
["sub-question 1", "sub-question 2", "sub-question 3"]

Only return the JSON array, nothing else."""


async def decompose_query(query: str, llm) -> list[str]:
    """
    Decompose complex query into sub-queries.
    
    Args:
        query: Complex query
        llm: LLM provider
        
    Returns:
        List of sub-queries
    """
    import json
    from src.orchestrator.llm_provider import LLMMessage
    
    try:
        messages = [
            LLMMessage(role="user", content=DECOMPOSE_PROMPT.format(query=query)),
        ]
        
        response = await llm.generate(
            messages,
            temperature=0.0,
            max_tokens=200,
        )
        
        sub_queries = json.loads(response.content)
        if isinstance(sub_queries, list):
            return sub_queries
        return [query]
        
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return [query]
