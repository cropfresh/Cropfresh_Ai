"""
Hybrid Search Module
====================
BM25 sparse retrieval combined with dense vector search using Reciprocal Rank Fusion.

Features:
- BM25 sparse indexing and search
- Dense + Sparse hybrid retrieval
- Reciprocal Rank Fusion (RRF) for score combination
- Configurable fusion parameters
"""

import math
from collections import defaultdict
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.rag.knowledge_base import Document


class BM25Index:
    """
    BM25 Sparse Index for keyword-based retrieval.
    
    Implements Okapi BM25 algorithm for term frequency-based ranking.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        
        # Index structures
        self.documents: dict[str, Document] = {}  # doc_id -> Document
        self.doc_lengths: dict[str, int] = {}  # doc_id -> word count
        self.avg_doc_length: float = 0.0
        self.term_doc_freq: dict[str, int] = defaultdict(int)  # term -> doc count
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {doc_id: freq}
        self.total_docs: int = 0
        
        self._is_initialized = False
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization - split on whitespace and lowercase."""
        import re
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def index_documents(self, documents: list[Document]) -> int:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Number of documents indexed
        """
        logger.info(f"Indexing {len(documents)} documents for BM25...")
        
        for doc in documents:
            doc_id = doc.id
            tokens = self._tokenize(doc.text)
            
            # Store document
            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = len(tokens)
            
            # Build inverted index
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            for term, freq in term_freq.items():
                if doc_id not in self.inverted_index[term]:
                    self.term_doc_freq[term] += 1
                self.inverted_index[term][doc_id] = freq
        
        self.total_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(self.total_docs, 1)
        self._is_initialized = True
        
        logger.info(f"BM25 index built: {self.total_docs} docs, {len(self.inverted_index)} terms")
        return len(documents)
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """
        Search using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self._is_initialized:
            logger.warning("BM25 index not initialized")
            return []
        
        query_tokens = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            # IDF calculation
            doc_freq = self.term_doc_freq[term]
            idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            
            # Score each document containing this term
            for doc_id, term_freq in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_id] += idf * (numerator / denominator)
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(self.documents[doc_id], score) for doc_id, score in sorted_docs]
    
    def clear(self):
        """Clear the index."""
        self.documents.clear()
        self.doc_lengths.clear()
        self.term_doc_freq.clear()
        self.inverted_index.clear()
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self._is_initialized = False


class HybridSearchResult(BaseModel):
    """Result of hybrid search operation."""
    
    documents: list[Document]
    query: str
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    search_time_ms: float = 0.0


class HybridRetriever:
    """
    Hybrid Retriever combining dense vectors with BM25 sparse search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine ranking signals.
    
    Usage:
        retriever = HybridRetriever(knowledge_base)
        await retriever.initialize()
        results = await retriever.search("tomato farming tips")
    """
    
    def __init__(
        self,
        knowledge_base,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            knowledge_base: Qdrant knowledge base for dense search
            rrf_k: RRF ranking constant (default 60)
            dense_weight: Weight for dense search scores
            sparse_weight: Weight for sparse search scores
        """
        self.knowledge_base = knowledge_base
        self.bm25_index = BM25Index()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize hybrid retriever by building BM25 index from knowledge base.
        
        Returns:
            True if initialized successfully
        """
        try:
            # Get all documents from knowledge base
            stats = self.knowledge_base.get_stats()
            if stats.get("total_documents", 0) == 0:
                logger.warning("Knowledge base is empty, BM25 index will be empty")
                self._initialized = True
                return True
            
            # Note: We'll build BM25 index from documents as they're ingested
            # For now, mark as initialized
            self._initialized = True
            logger.info("Hybrid retriever initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            return False
    
    def index_documents(self, documents: list[Document]) -> int:
        """
        Index documents for BM25 search.
        
        Args:
            documents: Documents to index
            
        Returns:
            Number of documents indexed
        """
        return self.bm25_index.index_documents(documents)
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        mode: str = "hybrid",  # "dense", "sparse", "hybrid"
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Optional category filter
            mode: Search mode - "dense", "sparse", or "hybrid"
            
        Returns:
            HybridSearchResult with fused documents
        """
        import time
        start_time = time.time()
        
        dense_results: list[tuple[Document, float]] = []
        sparse_results: list[tuple[Document, float]] = []
        
        # Dense search
        if mode in ("dense", "hybrid"):
            try:
                kb_result = await self.knowledge_base.search(
                    query=query,
                    top_k=top_k * 2,  # Get more for fusion
                    category=category,
                )
                dense_results = [(doc, doc.score or 0.0) for doc in kb_result.documents]
            except Exception as e:
                logger.warning(f"Dense search failed: {e}")
        
        # Sparse search (BM25)
        if mode in ("sparse", "hybrid") and self.bm25_index._is_initialized:
            try:
                sparse_results = self.bm25_index.search(query, top_k=top_k * 2)
            except Exception as e:
                logger.warning(f"Sparse search failed: {e}")
        
        # Fuse results
        if mode == "hybrid" and dense_results and sparse_results:
            fused_docs = self._rrf_fusion(dense_results, sparse_results, top_k)
        elif dense_results:
            fused_docs = [doc for doc, _ in dense_results[:top_k]]
        elif sparse_results:
            fused_docs = [doc for doc, _ in sparse_results[:top_k]]
        else:
            fused_docs = []
        
        search_time = (time.time() - start_time) * 1000
        
        return HybridSearchResult(
            documents=fused_docs,
            query=query,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused_docs),
            search_time_ms=search_time,
        )
    
    def _rrf_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        top_k: int,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion to combine dense and sparse results.
        
        RRF score = Σ weight / (k + rank)
        
        Args:
            dense_results: Dense search results with scores
            sparse_results: Sparse search results with scores
            top_k: Number of results to return
            
        Returns:
            Fused list of documents
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}
        
        # Score dense results
        for rank, (doc, _) in enumerate(dense_results, start=1):
            rrf_scores[doc.id] += self.dense_weight / (self.rrf_k + rank)
            doc_map[doc.id] = doc
        
        # Score sparse results
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc.id] += self.sparse_weight / (self.rrf_k + rank)
            doc_map[doc.id] = doc
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return top-k documents with updated scores
        result = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.score = rrf_scores[doc_id]
            result.append(doc)
        
        logger.debug(f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse -> {len(result)} fused")
        return result


# Singleton instance
_hybrid_retriever: Optional[HybridRetriever] = None


async def get_hybrid_retriever(knowledge_base) -> HybridRetriever:
    """
    Get or create singleton hybrid retriever instance.
    
    Args:
        knowledge_base: Knowledge base instance
        
    Returns:
        Initialized HybridRetriever
    """
    global _hybrid_retriever
    
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(knowledge_base)
        await _hybrid_retriever.initialize()
    
    return _hybrid_retriever
