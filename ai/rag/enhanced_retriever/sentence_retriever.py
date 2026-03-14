"""
Sentence Window Retriever
"""

import uuid
from typing import Optional

import numpy as np
from loguru import logger

from .models import DocumentNode, EnhancedRetrieverConfig, RetrievalResult, RetrievalStrategy


class SentenceWindowRetriever:
    """
    Sentence Window Retrieval.

    Strategy:
    1. Index individual sentences
    2. On match, expand to include surrounding sentences
    3. Return expanded context

    Provides precise matching with broader context.
    """

    def __init__(
        self,
        embedding_manager,
        config: Optional[EnhancedRetrieverConfig] = None,
    ):
        """Initialize sentence window retriever."""
        self.embedding_manager = embedding_manager
        self.config = config or EnhancedRetrieverConfig()

        # Storage
        self.sentences: list[DocumentNode] = []
        self.doc_sentences: dict[str, list[int]] = {}  # doc_id -> sentence indices

        logger.info("SentenceWindowRetriever initialized")

    async def add_documents(self, documents: list) -> int:
        """Add documents split into sentences."""
        total_sentences = 0

        for doc in documents:
            text = doc.text if hasattr(doc, 'text') else str(doc)
            doc_id = doc.id if hasattr(doc, 'id') else str(uuid.uuid4())[:8]

            # Split into sentences
            sentences = self._split_into_sentences(text)
            doc_start_idx = len(self.sentences)

            for s_idx, sent_text in enumerate(sentences):
                embedding = self.embedding_manager.embed_query(sent_text)

                node = DocumentNode(
                    id=f"{doc_id}_s{s_idx}",
                    text=sent_text,
                    embedding=embedding,
                    source_doc_id=doc_id,
                    node_type="sentence",
                    metadata={"sentence_index": s_idx},
                )
                self.sentences.append(node)
                total_sentences += 1

            self.doc_sentences[doc_id] = list(range(doc_start_idx, len(self.sentences)))

        logger.info(f"Added {total_sentences} sentences from {len(documents)} documents")
        return total_sentences

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve with sentence window expansion.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            RetrievalResult with expanded context
        """
        import time
        start_time = time.time()

        if not self.sentences:
            return RetrievalResult(
                nodes=[],
                strategy_used=RetrievalStrategy.SENTENCE_WINDOW,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Embed query
        query_embedding = np.array(self.embedding_manager.embed_query(query))

        # Score all sentences
        scored = []
        for idx, sent in enumerate(self.sentences):
            if sent.embedding:
                sent_vec = np.array(sent.embedding)
                similarity = np.dot(query_embedding, sent_vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(sent_vec) + 1e-8
                )
                scored.append((idx, sent, similarity))

        # Sort by similarity
        scored.sort(key=lambda x: x[2], reverse=True)

        # Expand windows for top results
        result_nodes = []
        expanded_contexts = []
        window_size = self.config.window_size

        seen_ranges = set()

        for idx, sent, score in scored[:top_k * 2]:
            doc_id = sent.source_doc_id
            doc_indices = self.doc_sentences.get(doc_id, [])

            if not doc_indices:
                continue

            # Get position within document
            doc_pos = idx - doc_indices[0]

            # Calculate window range
            start = max(0, doc_pos - window_size)
            end = min(len(doc_indices), doc_pos + window_size + 1)

            range_key = (doc_id, start, end)
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)

            # Get window sentences
            window_sentences = []
            for i in range(start, end):
                if doc_indices[0] + i < len(self.sentences):
                    window_sentences.append(self.sentences[doc_indices[0] + i].text)

            expanded_text = " ".join(window_sentences)
            expanded_contexts.append(expanded_text)

            # Create window node
            window_node = DocumentNode(
                id=f"{doc_id}_w{start}_{end}",
                text=expanded_text,
                source_doc_id=doc_id,
                node_type="window",
                metadata={
                    "center_sentence": sent.text,
                    "window_start": start,
                    "window_end": end,
                    "similarity": score,
                },
            )
            result_nodes.append(window_node)

            if len(result_nodes) >= top_k:
                break

        return RetrievalResult(
            nodes=result_nodes,
            strategy_used=RetrievalStrategy.SENTENCE_WINDOW,
            expanded_context="\n\n---\n\n".join(expanded_contexts),
            num_unique_sources=len(set(n.source_doc_id for n in result_nodes)),
            avg_similarity=sum(x[2] for x in scored[:top_k]) / min(top_k, len(scored)) if scored else 0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
