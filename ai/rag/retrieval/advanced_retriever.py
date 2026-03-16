"""
Advanced Retriever Coordinator — Ties Phase 4 modules together (ADR-010).

Orchestrates: query decomposition → retrieval → time-aware reranking.
Used by the LangGraph retrieve_node as an upgraded retrieval backend.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ai.rag.retrieval.query_decomposer import DecomposedQuery, QueryDecomposer
from ai.rag.retrieval.time_aware import FreshnessCategory, TimeAwareRetriever


class AdvancedRetrievalResult(BaseModel):
    """Result from the advanced retrieval coordinator."""

    documents: list[Any] = Field(default_factory=list)
    sub_queries_used: list[str] = Field(default_factory=list)
    freshness_category: str = "evergreen"
    total_retrieved: int = 0
    unique_after_dedup: int = 0
    time_adjusted: bool = False


class AdvancedRetriever:
    """Coordinates decomposition, retrieval, and time-aware reranking.

    Pipeline:
      1. QueryDecomposer splits multi-part queries
      2. Each sub-query retrieves independently
      3. Results are deduplicated by doc ID
      4. TimeAwareRetriever adjusts scores for freshness
    """

    def __init__(
        self,
        knowledge_base: Any = None,
        llm: Any = None,
    ):
        self.kb = knowledge_base
        self.decomposer = QueryDecomposer(llm=llm)
        self.time_ranker = TimeAwareRetriever()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> AdvancedRetrievalResult:
        """Execute advanced retrieval pipeline.

        Args:
            query: User query (may be multi-part).
            top_k: Number of results to return.

        Returns:
            AdvancedRetrievalResult with deduplicated, reranked docs.
        """
        # Step 1: Decompose query
        decomposed = await self.decomposer.decompose(query)
        sub_queries = decomposed.sub_queries or [query]

        logger.info(
            f"AdvancedRetriever: {len(sub_queries)} sub-queries | "
            f"multi_part={decomposed.is_multi_part}"
        )

        # Step 2: Retrieve for each sub-query
        all_docs: list[Any] = []
        seen_ids: set[str] = set()

        for sq in sub_queries[:4]:  # ? Limit to 4 sub-queries
            docs = await self._retrieve_single(sq, top_k=top_k)
            for doc in docs:
                doc_id = getattr(doc, "id", id(doc))
                if doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)

        # Step 3: Time-aware reranking
        category = self.time_ranker.classify_query(query)
        time_adjusted = category != FreshnessCategory.EVERGREEN

        if time_adjusted and all_docs:
            ranked = self.time_ranker.adjust_scores(all_docs, query)
            # Reorder docs by adjusted score
            id_to_doc = {getattr(d, "id", id(d)): d for d in all_docs}
            all_docs = [
                id_to_doc[r.doc_id]
                for r in ranked[:top_k]
                if r.doc_id in id_to_doc
            ]

        result = AdvancedRetrievalResult(
            documents=all_docs[:top_k],
            sub_queries_used=sub_queries,
            freshness_category=category.value,
            total_retrieved=len(seen_ids),
            unique_after_dedup=len(all_docs),
            time_adjusted=time_adjusted,
        )

        logger.info(
            f"AdvancedRetriever: {result.unique_after_dedup} unique docs | "
            f"freshness={category.value} | adjusted={time_adjusted}"
        )
        return result

    async def _retrieve_single(
        self, query: str, top_k: int = 5,
    ) -> list[Any]:
        """Retrieve documents for a single query."""
        if self.kb is None:
            return []

        try:
            result = await self.kb.search(query, top_k=top_k)
            return result.documents if hasattr(result, "documents") else []
        except Exception as e:
            logger.warning(f"AdvancedRetriever single retrieval failed: {e}")
            return []
