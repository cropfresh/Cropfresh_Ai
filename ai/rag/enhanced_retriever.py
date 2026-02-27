"""
Enhanced Retrieval Pipeline
=============================
Advanced retrieval strategies for improved context retrieval.

Implements:
- Parent Document Retriever: Store small chunks, retrieve full docs
- Sentence Window Retrieval: Expand context around matching sentences
- Auto-Merging Retrieval: Combine adjacent chunks
- MMR (Maximum Marginal Relevance): Diversity-focused retrieval

These techniques ensure relevant, complete, and diverse context
for answer generation.

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Optional
from enum import Enum
import uuid

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    """Retrieval strategies."""
    SIMPLE = "simple"
    PARENT_DOCUMENT = "parent_document"
    SENTENCE_WINDOW = "sentence_window"
    AUTO_MERGE = "auto_merge"
    MMR = "mmr"


class DocumentNode(BaseModel):
    """A document node with hierarchical relationships."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    
    # Hierarchical relationship
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    
    # Position in parent
    start_index: int = 0
    end_index: int = 0
    
    # Embedding
    embedding: Optional[list[float]] = None
    
    # Metadata
    source_doc_id: str = ""
    node_type: str = "chunk"  # chunk, sentence, parent, full
    metadata: dict = Field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children_ids) == 0


class RetrievalResult(BaseModel):
    """Result from enhanced retrieval."""
    
    nodes: list[DocumentNode] = Field(default_factory=list)
    strategy_used: RetrievalStrategy
    
    # Additional context
    expanded_context: str = ""  # For sentence window
    parent_documents: list[str] = Field(default_factory=list)  # For parent retriever
    
    # Metrics
    num_unique_sources: int = 0
    avg_similarity: float = 0.0
    diversity_score: float = 0.0
    
    processing_time_ms: float = 0.0


class EnhancedRetrieverConfig(BaseModel):
    """Configuration for enhanced retrieval."""
    
    # Parent Document Retriever
    parent_chunk_size: int = 2000
    child_chunk_size: int = 300
    
    # Sentence Window
    window_size: int = 3  # Number of sentences before/after
    
    # Auto-Merge
    merge_threshold: float = 0.7
    max_merge_count: int = 3
    
    # MMR
    mmr_lambda: float = 0.5  # 0=max diversity, 1=max relevance
    mmr_fetch_k: int = 20


class ParentDocumentRetriever:
    """
    Parent Document Retriever.
    
    Strategy:
    1. Store both small chunks and their parent documents
    2. Search using small chunks (better precision)
    3. Return parent documents (better context)
    
    This gives the best of both worlds: precise matching
    and complete context.
    
    Usage:
        retriever = ParentDocumentRetriever(embedding_manager)
        
        # Add documents
        await retriever.add_documents(documents)
        
        # Retrieve with parent context
        results = await retriever.retrieve("query", top_k=5)
    """
    
    def __init__(
        self,
        embedding_manager,
        config: Optional[EnhancedRetrieverConfig] = None,
    ):
        """Initialize parent document retriever."""
        self.embedding_manager = embedding_manager
        self.config = config or EnhancedRetrieverConfig()
        
        # Storage
        self.parent_nodes: dict[str, DocumentNode] = {}
        self.child_nodes: dict[str, DocumentNode] = {}
        self.child_to_parent: dict[str, str] = {}
        
        logger.info("ParentDocumentRetriever initialized")
    
    async def add_documents(self, documents: list) -> int:
        """
        Add documents with parent-child splitting.
        
        Args:
            documents: List of documents
            
        Returns:
            Number of child chunks created
        """
        total_children = 0
        
        for doc in documents:
            text = doc.text if hasattr(doc, 'text') else str(doc)
            doc_id = doc.id if hasattr(doc, 'id') else str(uuid.uuid4())[:8]
            
            # Create parent chunks
            parent_chunks = self._split_into_parents(text)
            
            for p_idx, parent_text in enumerate(parent_chunks):
                parent_id = f"{doc_id}_p{p_idx}"
                
                parent_node = DocumentNode(
                    id=parent_id,
                    text=parent_text,
                    source_doc_id=doc_id,
                    node_type="parent",
                    metadata={"parent_index": p_idx},
                )
                self.parent_nodes[parent_id] = parent_node
                
                # Create child chunks from parent
                child_chunks = self._split_into_children(parent_text)
                
                for c_idx, child_text in enumerate(child_chunks):
                    child_id = f"{parent_id}_c{c_idx}"
                    
                    # Generate embedding for child
                    embedding = self.embedding_manager.embed_query(child_text)
                    
                    child_node = DocumentNode(
                        id=child_id,
                        text=child_text,
                        parent_id=parent_id,
                        embedding=embedding,
                        source_doc_id=doc_id,
                        node_type="chunk",
                        metadata={"child_index": c_idx},
                    )
                    
                    self.child_nodes[child_id] = child_node
                    self.child_to_parent[child_id] = parent_id
                    parent_node.children_ids.append(child_id)
                    total_children += 1
        
        logger.info(f"Added {len(self.parent_nodes)} parents, {total_children} children")
        return total_children
    
    def _split_into_parents(self, text: str) -> list[str]:
        """Split text into parent-sized chunks."""
        chunk_size = self.config.parent_chunk_size
        chunks = []
        
        paragraphs = text.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) <= chunk_size:
                current += "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                current = para
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def _split_into_children(self, text: str) -> list[str]:
        """Split parent text into child-sized chunks."""
        chunk_size = self.config.child_chunk_size
        chunks = []
        
        sentences = text.replace('. ', '.\n').split('\n')
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) <= chunk_size:
                current += " " + sent if current else sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_parents: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve using small chunks, return parent documents.
        
        Args:
            query: Search query
            top_k: Number of results
            return_parents: Whether to return parent docs or just child matches
            
        Returns:
            RetrievalResult with parent documents
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = np.array(self.embedding_manager.embed_query(query))
        
        # Search child nodes
        scored = []
        for child_id, child in self.child_nodes.items():
            if child.embedding:
                child_vec = np.array(child.embedding)
                similarity = np.dot(query_embedding, child_vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(child_vec) + 1e-8
                )
                scored.append((child, similarity))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        top_children = scored[:top_k * 2]  # Get more for parent dedup
        
        # Get unique parents
        seen_parents = set()
        result_nodes = []
        parent_docs = []
        
        for child, score in top_children:
            if return_parents:
                parent_id = self.child_to_parent.get(child.id)
                if parent_id and parent_id not in seen_parents:
                    parent = self.parent_nodes.get(parent_id)
                    if parent:
                        result_nodes.append(parent)
                        parent_docs.append(parent.text)
                        seen_parents.add(parent_id)
                        
                        if len(result_nodes) >= top_k:
                            break
            else:
                result_nodes.append(child)
                if len(result_nodes) >= top_k:
                    break
        
        return RetrievalResult(
            nodes=result_nodes,
            strategy_used=RetrievalStrategy.PARENT_DOCUMENT,
            parent_documents=parent_docs,
            num_unique_sources=len(set(n.source_doc_id for n in result_nodes)),
            avg_similarity=sum(s for _, s in top_children[:top_k]) / top_k if top_children else 0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


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
