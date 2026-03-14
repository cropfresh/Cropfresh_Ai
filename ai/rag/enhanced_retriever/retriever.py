"""
Unified Enhanced Retriever
"""

from typing import Optional

from loguru import logger

from .mmr_retriever import MMRRetriever
from .models import EnhancedRetrieverConfig, RetrievalResult, RetrievalStrategy
from .parent_retriever import ParentDocumentRetriever
from .sentence_retriever import SentenceWindowRetriever


class EnhancedRetriever:
    """
    Unified Enhanced Retriever supporting multiple strategies.

    Usage:
        retriever = EnhancedRetriever(embedding_manager)
        await retriever.add_documents(documents)

        # Use specific strategy
        results = await retriever.retrieve(
            "query",
            strategy=RetrievalStrategy.PARENT_DOCUMENT
        )
    """

    def __init__(
        self,
        embedding_manager,
        config: Optional[EnhancedRetrieverConfig] = None,
    ):
        """Initialize enhanced retriever."""
        self.embedding_manager = embedding_manager
        self.config = config or EnhancedRetrieverConfig()

        # Initialize strategy-specific retrievers
        self.parent_retriever = ParentDocumentRetriever(embedding_manager, config)
        self.sentence_retriever = SentenceWindowRetriever(embedding_manager, config)
        self.mmr_retriever = MMRRetriever(embedding_manager, config)

        logger.info("EnhancedRetriever initialized with all strategies")

    async def add_documents(self, documents: list):
        """Add documents to all retrievers."""
        await self.parent_retriever.add_documents(documents)
        await self.sentence_retriever.add_documents(documents)
        await self.mmr_retriever.add_documents(documents)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.MMR,
    ) -> RetrievalResult:
        """
        Retrieve using specified strategy.

        Args:
            query: Search query
            top_k: Number of results
            strategy: Retrieval strategy to use

        Returns:
            RetrievalResult
        """
        if strategy == RetrievalStrategy.PARENT_DOCUMENT:
            return await self.parent_retriever.retrieve(query, top_k)
        elif strategy == RetrievalStrategy.SENTENCE_WINDOW:
            return await self.sentence_retriever.retrieve(query, top_k)
        elif strategy == RetrievalStrategy.MMR:
            return await self.mmr_retriever.retrieve(query, top_k)
        else:
            # Default to MMR
            return await self.mmr_retriever.retrieve(query, top_k)


# Factory function
def create_enhanced_retriever(
    embedding_manager,
    config: Optional[EnhancedRetrieverConfig] = None,
) -> EnhancedRetriever:
    """Create an enhanced retriever."""
    return EnhancedRetriever(embedding_manager, config)
