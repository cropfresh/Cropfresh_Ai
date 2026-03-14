"""
Base Agent Retrievals
=====================
Retrieval functionalities for Base Agent.
"""

from typing import Optional
from loguru import logger


class RetrievalMixin:
    """Mixin for knowledge base retrieval functionalities."""
    
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents from knowledge base.
        """
        if not self.knowledge_base:
            return []
        
        try:
            search_categories = categories or self.config.kb_categories
            
            result = await self.knowledge_base.search(
                query=query,
                top_k=top_k,
                category=search_categories[0] if search_categories else None,
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
            
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    def format_context(self, documents: list[dict]) -> str:
        """
        Format retrieved documents for LLM context.
        """
        if not documents:
            return "No relevant documents found."
        
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "Unknown")
            text = doc.get("text", "")
            score = doc.get("score", 0)
            
            parts.append(f"[Document {i}] (Source: {source}, Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(parts)

    def _extract_sources(self, documents: list[dict]) -> list[str]:
        """Extract unique source references from documents."""
        sources = []
        seen = set()
        
        for doc in documents:
            source = doc.get("source", "")
            if source and source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
