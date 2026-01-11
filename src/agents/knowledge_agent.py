"""
Knowledge Agent
================
High-level interface for the agentic RAG system.

Provides simple API for querying agricultural knowledge.
"""

from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


class KnowledgeResponse(BaseModel):
    """Response from knowledge agent."""
    
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    query_type: str = ""
    steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeAgent:
    """
    Knowledge Agent for CropFresh AI.
    
    High-level interface to the agentic RAG system.
    Handles agricultural knowledge queries with intelligent routing,
    document grading, and self-correction.
    
    Usage:
        agent = KnowledgeAgent(llm=provider)
        await agent.initialize()
        response = await agent.answer("How to grow tomatoes?")
        print(response.answer)
    """
    
    def __init__(
        self,
        llm=None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: str = "",
    ):
        """
        Initialize Knowledge Agent.
        
        Args:
            llm: LLM provider for generation
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            qdrant_api_key: Qdrant API key for cloud
        """
        self.llm = llm
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self._knowledge_base = None
        self._initialized = False
    
    @property
    def knowledge_base(self):
        """Get knowledge base instance."""
        if self._knowledge_base is None:
            from src.rag.knowledge_base import KnowledgeBase
            self._knowledge_base = KnowledgeBase(
                host=self.qdrant_host,
                port=self.qdrant_port,
                api_key=self.qdrant_api_key,
            )
        return self._knowledge_base
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge agent.
        
        Creates Qdrant collection if needed.
        
        Returns:
            True if initialized successfully
        """
        try:
            success = await self.knowledge_base.initialize()
            self._initialized = success
            if success:
                logger.info("Knowledge Agent initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Agent: {e}")
            return False
    
    async def answer(
        self,
        question: str,
        context: str = "",
    ) -> KnowledgeResponse:
        """
        Answer a knowledge question using agentic RAG.
        
        Args:
            question: User question
            context: Optional additional context
            
        Returns:
            KnowledgeResponse with answer and metadata
        """
        from src.rag.graph import run_agentic_rag
        
        if not self._initialized:
            await self.initialize()
        
        # Include context in question if provided
        full_question = f"{question}\n\nContext: {context}" if context else question
        
        try:
            # Run agentic RAG pipeline
            result = await run_agentic_rag(
                question=full_question,
                knowledge_base=self.knowledge_base,
                llm=self.llm,
            )
            
            # Extract sources from documents
            sources = []
            for doc in result.get("documents", []):
                if hasattr(doc, "source") and doc.source:
                    sources.append(doc.source)
            
            return KnowledgeResponse(
                answer=result.get("final_answer") or result.get("generation", "I couldn't find an answer."),
                sources=list(set(sources)),  # Deduplicate
                confidence=0.8 if result.get("final_answer") else 0.5,
                query_type=result.get("query_type", ""),
                steps=result.get("steps", []),
                metadata={
                    "documents_retrieved": len(result.get("documents", [])),
                    "web_search_used": result.get("web_search") == "Yes",
                    "retry_count": result.get("retry_count", 0),
                },
            )
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return KnowledgeResponse(
                answer=f"Sorry, I encountered an error: {str(e)}",
                confidence=0.0,
                steps=["error"],
            )
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        Simple semantic search without generation.
        
        Args:
            query: Search query
            top_k: Number of results
            category: Optional category filter
            
        Returns:
            List of matching documents
        """
        if not self._initialized:
            await self.initialize()
        
        result = await self.knowledge_base.search(
            query=query,
            top_k=top_k,
            category=category,
        )
        
        return [
            {
                "text": doc.text,
                "source": doc.source,
                "category": doc.category,
                "score": doc.score,
            }
            for doc in result.documents
        ]
    
    async def ingest_documents(
        self,
        documents: list[dict],
    ) -> int:
        """
        Ingest documents into knowledge base.
        
        Args:
            documents: List of dicts with text, source, category
            
        Returns:
            Number of documents ingested
        """
        from src.rag.knowledge_base import Document
        
        if not self._initialized:
            await self.initialize()
        
        docs = [
            Document(
                text=d["text"],
                source=d.get("source", ""),
                category=d.get("category", ""),
                metadata=d.get("metadata", {}),
            )
            for d in documents
        ]
        
        return await self.knowledge_base.add_documents(docs)
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return self.knowledge_base.get_stats()
