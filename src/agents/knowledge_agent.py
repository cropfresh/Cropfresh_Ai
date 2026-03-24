"""High-level interface for the agentic RAG system."""

from typing import Optional

from loguru import logger

from src.agents.knowledge_mapping import build_source_details, extract_citations
from src.agents.knowledge_models import BenchmarkDebugResult, KnowledgeResponse
from src.agents.knowledge_runtime import configure_benchmark_embeddings


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
        self._initialize_attempted = False

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
        self._initialize_attempted = True
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
        debug_result = await self.answer_with_debug(question=question, context=context)
        return KnowledgeResponse(
            answer=debug_result.answer,
            sources=debug_result.sources,
            confidence=debug_result.confidence,
            query_type=debug_result.route,
            steps=debug_result.metadata.get("steps", []),
            metadata=debug_result.metadata,
        )

    async def answer_with_debug(
        self,
        question: str,
        context: str = "",
    ) -> BenchmarkDebugResult:
        """Answer a query while returning benchmark/debug details."""
        from src.rag.graph import run_agentic_rag

        if not self._initialized and not self._initialize_attempted:
            await self.initialize()
        configure_benchmark_embeddings(self)

        full_question = f"{question}\n\nContext: {context}" if context else question
        try:
            result = await run_agentic_rag(
                question=full_question,
                knowledge_base=self.knowledge_base,
                llm=self.llm,
            )
            documents = result.get("documents", [])
            source_details = build_source_details(documents)
            sources = [detail.source for detail in source_details if detail.source]
            answer = result.get("final_answer") or result.get("generation", "I couldn't find an answer.")
            return BenchmarkDebugResult(
                answer=answer,
                raw_answer=result.get("generation", ""),
                sources=list(dict.fromkeys(sources)),
                source_details=source_details,
                contexts=[getattr(doc, "text", str(doc)) for doc in documents],
                route=result.get("query_type", ""),
                tool_calls=result.get("tool_calls", []),
                confidence=float(result.get("confidence", 0.0)),
                retry_count=int(result.get("retry_count", 0)),
                citations=extract_citations(answer),
                metadata={
                    "documents_retrieved": len(documents),
                    "web_search_used": result.get("web_search") == "Yes",
                    "retry_count": result.get("retry_count", 0),
                    "steps": result.get("steps", []),
                    "route_reason": result.get("route_reason", ""),
                },
            )
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return BenchmarkDebugResult(
                answer=f"Sorry, I encountered an error: {str(e)}",
                route="error",
                metadata={"steps": ["error"]},
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
        if not self._initialized and not self._initialize_attempted:
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

        if not self._initialized and not self._initialize_attempted:
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
