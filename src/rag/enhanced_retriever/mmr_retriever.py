"""
MMR Retriever
"""

import uuid
from typing import Optional

import numpy as np
from loguru import logger

from .models import DocumentNode, EnhancedRetrieverConfig, RetrievalResult, RetrievalStrategy


class MMRRetriever:
    """
    Maximum Marginal Relevance (MMR) Retriever.

    Balances relevance and diversity by selecting results that are:
    1. Relevant to the query
    2. Different from already selected results

    Prevents redundant information in retrieved context.
    """

    def __init__(
        self,
        embedding_manager,
        config: Optional[EnhancedRetrieverConfig] = None,
    ):
        """Initialize MMR retriever."""
        self.embedding_manager = embedding_manager
        self.config = config or EnhancedRetrieverConfig()

        self.documents: list[DocumentNode] = []

        logger.info("MMRRetriever initialized")

    async def add_documents(self, documents: list) -> int:
        """Add documents to the retriever."""
        for doc in documents:
            text = doc.text if hasattr(doc, 'text') else str(doc)
            doc_id = doc.id if hasattr(doc, 'id') else str(uuid.uuid4())[:8]

            embedding = self.embedding_manager.embed_query(text)

            node = DocumentNode(
                id=doc_id,
                text=text,
                embedding=embedding,
                source_doc_id=doc_id,
            )
            self.documents.append(node)

        return len(self.documents)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        lambda_param: Optional[float] = None,
    ) -> RetrievalResult:
        """
        MMR retrieval: balance relevance and diversity.

        Args:
            query: Search query
            top_k: Number of results
            lambda_param: Balance parameter (0=diversity, 1=relevance)

        Returns:
            RetrievalResult with diverse documents
        """
        import time
        start_time = time.time()

        if not self.documents:
            return RetrievalResult(
                nodes=[],
                strategy_used=RetrievalStrategy.MMR,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        lambda_param = lambda_param or self.config.mmr_lambda

        # Embed query
        query_embedding = np.array(self.embedding_manager.embed_query(query))

        # Calculate similarity for all documents
        doc_similarities = []
        for doc in self.documents:
            if doc.embedding:
                doc_vec = np.array(doc.embedding)
                similarity = np.dot(query_embedding, doc_vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_vec) + 1e-8
                )
                doc_similarities.append((doc, similarity))

        # Sort by similarity to get initial candidates
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = doc_similarities[:self.config.mmr_fetch_k]

        # MMR selection
        selected = []
        selected_embeddings = []

        while len(selected) < top_k and candidates:
            best_score = -float('inf')
            best_idx = 0

            for i, (doc, relevance) in enumerate(candidates):
                # Calculate max similarity to already selected
                if selected_embeddings:
                    doc_vec = np.array(doc.embedding)
                    similarities_to_selected = [
                        np.dot(doc_vec, sel_emb) / (
                            np.linalg.norm(doc_vec) * np.linalg.norm(sel_emb) + 1e-8
                        )
                        for sel_emb in selected_embeddings
                    ]
                    max_sim_to_selected = max(similarities_to_selected)
                else:
                    max_sim_to_selected = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Add best document
            best_doc, _ = candidates[best_idx]
            selected.append(best_doc)
            selected_embeddings.append(np.array(best_doc.embedding))
            candidates.pop(best_idx)

        # Calculate diversity score
        diversity = 0.0
        if len(selected_embeddings) > 1:
            pairwise_sims = []
            for i in range(len(selected_embeddings)):
                for j in range(i + 1, len(selected_embeddings)):
                    sim = np.dot(selected_embeddings[i], selected_embeddings[j]) / (
                        np.linalg.norm(selected_embeddings[i]) * np.linalg.norm(selected_embeddings[j]) + 1e-8
                    )
                    pairwise_sims.append(sim)
            diversity = 1 - (sum(pairwise_sims) / len(pairwise_sims)) if pairwise_sims else 1.0

        return RetrievalResult(
            nodes=selected,
            strategy_used=RetrievalStrategy.MMR,
            num_unique_sources=len(set(n.source_doc_id for n in selected)),
            avg_similarity=sum(s for _, s in doc_similarities[:top_k]) / min(top_k, len(doc_similarities)) if doc_similarities else 0,
            diversity_score=diversity,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
