"""
Embedding Manager
=================
BGE-M3 multilingual embeddings for semantic search.

Features:
- 1024-dimensional dense vectors
- Multilingual support (English + Indian languages)
- Query instruction prefixing for better retrieval
- Batch processing for efficiency
"""

from functools import lru_cache
from typing import Optional

from loguru import logger
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    vectors: list[list[float]]
    model: str
    dimensions: int


class EmbeddingManager:
    """
    BGE-M3 Embedding Manager.
    
    Uses sentence-transformers for high-quality multilingual embeddings.
    Optimized for agricultural domain with support for Indian languages.
    
    Usage:
        manager = EmbeddingManager()
        vectors = manager.embed_documents(["How to grow tomatoes?"])
        query_vec = manager.embed_query("tomato cultivation tips")
    """
    
    # Query instruction prefix for BGE models
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: HuggingFace model name
            device: "cpu" or "cuda"
            cache_dir: Optional cache directory for models
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._dimensions = 1024  # BGE-M3 dimensions
        
        logger.info(f"Initializing EmbeddingManager with {model_name} on {device}")
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            # Update dimensions from model
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimensions: {self._dimensions}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: uv sync --extra ml")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed documents (passages to be searched).
        
        Args:
            texts: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.debug(f"Embedding {len(texts)} documents")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # For cosine similarity
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query with instruction prefix.
        
        For BGE models, queries should have an instruction prefix
        for optimal retrieval performance.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        # Add instruction prefix for BGE models
        if "bge" in self.model_name.lower():
            query = f"{self.QUERY_INSTRUCTION}{query}"
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        
        return embedding.tolist()
    
    def embed_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed multiple queries.
        
        Args:
            queries: List of query texts
            batch_size: Batch size for encoding
            
        Returns:
            List of query embedding vectors
        """
        if not queries:
            return []
        
        # Add instruction prefix for BGE models
        if "bge" in self.model_name.lower():
            queries = [f"{self.QUERY_INSTRUCTION}{q}" for q in queries]
        
        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        
        return embeddings.tolist()
    
    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Since vectors are normalized, this is just a dot product.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        return float(np.dot(vec1, vec2))


@lru_cache(maxsize=1)
def get_embedding_manager(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> EmbeddingManager:
    """
    Get cached embedding manager instance.
    
    Uses LRU cache to avoid reloading model multiple times.
    """
    return EmbeddingManager(model_name=model_name, device=device)
