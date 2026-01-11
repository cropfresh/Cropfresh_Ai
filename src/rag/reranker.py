"""
Cross-Encoder Re-ranker
=======================
Re-ranks retrieved documents using cross-encoder models for improved precision.

Features:
- Cross-encoder scoring for query-document pairs
- Batch processing for efficiency
- Lightweight model options (MiniLM ~80MB)
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel

from src.rag.knowledge_base import Document


class RerankedResult(BaseModel):
    """Result of re-ranking operation."""
    
    documents: list[Document]
    query: str
    original_count: int
    reranked_count: int
    model_name: str
    rerank_time_ms: float = 0.0


class CrossEncoderReranker:
    """
    Cross-Encoder Re-ranker for precision boosting.
    
    Uses cross-encoder models to score query-document pairs directly,
    which is more accurate than bi-encoder similarity for ranking.
    
    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, documents, top_k=5)
    """
    
    # Available lightweight models
    MODELS = {
        "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~80MB, fast
        "accurate": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # ~120MB, better
        "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # multilingual
    }
    
    def __init__(
        self,
        model_name: str = "default",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Model key from MODELS or full HuggingFace model name
            device: Device to run model on ("cpu" or "cuda")
            batch_size: Batch size for scoring
        """
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._initialized = False
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            self._initialized = True
            logger.info("Cross-encoder model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> RerankedResult:
        """
        Re-rank documents using cross-encoder scoring.
        
        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top results to return
            
        Returns:
            RerankedResult with re-ranked documents
        """
        import time
        start_time = time.time()
        
        if not documents:
            return RerankedResult(
                documents=[],
                query=query,
                original_count=0,
                reranked_count=0,
                model_name=self.model_name,
                rerank_time_ms=0.0,
            )
        
        # Load model if needed
        self._load_model()
        
        # Create query-document pairs
        pairs = [(query, doc.text[:512]) for doc in documents]  # Truncate for efficiency
        
        # Score in batches
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Update document scores and return top-k
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc.score = float(score)
            reranked.append(doc)
        
        rerank_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Re-ranked {len(documents)} docs -> top {len(reranked)} in {rerank_time:.1f}ms")
        
        return RerankedResult(
            documents=reranked,
            query=query,
            original_count=len(documents),
            reranked_count=len(reranked),
            model_name=self.model_name,
            rerank_time_ms=rerank_time,
        )
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: list[Document],
        threshold: float = 0.0,
        top_k: int = 10,
    ) -> RerankedResult:
        """
        Re-rank with score threshold filtering.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            threshold: Minimum score threshold
            top_k: Maximum results to return
            
        Returns:
            RerankedResult with filtered and re-ranked documents
        """
        result = self.rerank(query, documents, top_k=len(documents))
        
        # Filter by threshold
        filtered = [doc for doc in result.documents if (doc.score or 0) >= threshold][:top_k]
        
        return RerankedResult(
            documents=filtered,
            query=query,
            original_count=result.original_count,
            reranked_count=len(filtered),
            model_name=self.model_name,
            rerank_time_ms=result.rerank_time_ms,
        )


class LightweightReranker:
    """
    Lightweight re-ranker using simple heuristics when model loading is not desired.
    
    Useful for:
    - Quick testing without GPU
    - Low-memory environments
    - When cross-encoder models are overkill
    """
    
    def __init__(self):
        """Initialize lightweight reranker."""
        self.model_name = "heuristic"
    
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> RerankedResult:
        """
        Re-rank using keyword overlap and position heuristics.
        
        Scoring based on:
        - Query term presence in document
        - Term position (earlier = better)
        - Exact phrase matching
        """
        import time
        import re
        start_time = time.time()
        
        if not documents:
            return RerankedResult(
                documents=[],
                query=query,
                original_count=0,
                reranked_count=0,
                model_name=self.model_name,
            )
        
        query_terms = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        
        scored_docs = []
        for doc in documents:
            text_lower = doc.text.lower()
            
            # Base score from original retrieval
            score = doc.score or 0.5
            
            # Boost for query term presence
            term_matches = sum(1 for term in query_terms if term in text_lower)
            term_boost = term_matches / max(len(query_terms), 1) * 0.3
            
            # Boost for exact phrase match
            if query.lower() in text_lower:
                term_boost += 0.2
            
            # Boost for terms appearing early in document
            first_100 = text_lower[:100]
            early_matches = sum(1 for term in query_terms if term in first_100)
            early_boost = early_matches / max(len(query_terms), 1) * 0.1
            
            final_score = min(score + term_boost + early_boost, 1.0)
            scored_docs.append((doc, final_score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return top-k
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc.score = score
            reranked.append(doc)
        
        rerank_time = (time.time() - start_time) * 1000
        
        return RerankedResult(
            documents=reranked,
            query=query,
            original_count=len(documents),
            reranked_count=len(reranked),
            model_name=self.model_name,
            rerank_time_ms=rerank_time,
        )


# Factory function
def get_reranker(
    model_type: str = "auto",
    device: str = "cpu",
) -> CrossEncoderReranker | LightweightReranker:
    """
    Get appropriate reranker based on available resources.
    
    Args:
        model_type: "cross-encoder", "lightweight", or "auto"
        device: Device for model ("cpu" or "cuda")
        
    Returns:
        Reranker instance
    """
    if model_type == "lightweight":
        return LightweightReranker()
    
    if model_type == "cross-encoder":
        return CrossEncoderReranker(device=device)
    
    # Auto-detect
    try:
        import sentence_transformers  # noqa
        return CrossEncoderReranker(device=device)
    except ImportError:
        logger.warning("sentence-transformers not installed, using lightweight reranker")
        return LightweightReranker()
