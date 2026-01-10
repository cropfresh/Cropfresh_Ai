"""
Knowledge Base
===============
Qdrant vector database wrapper for agricultural knowledge storage and retrieval.

Features:
- Automatic collection creation with optimal settings
- Batch document ingestion with embeddings
- Dense vector search with filtering
- Hybrid search (dense + sparse) for better recall
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for storage and retrieval."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    source: str = ""
    category: str = ""  # agronomy, market, platform, regulatory
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Populated during retrieval
    score: Optional[float] = None


class SearchResult(BaseModel):
    """Search result with relevance score."""
    
    documents: list[Document]
    query: str
    total_found: int
    search_time_ms: float


class KnowledgeBase:
    """
    Qdrant-based Knowledge Base for CropFresh AI.
    
    Stores agricultural knowledge with semantic search capabilities.
    
    Categories:
    - agronomy: Crop guides, pest management, farming practices
    - market: Mandi info, pricing, trading rules
    - platform: CropFresh features, FAQs, user guides
    - regulatory: APMC rules, certifications, compliance
    
    Usage:
        kb = KnowledgeBase()
        await kb.initialize()
        await kb.add_documents([Document(text="How to grow tomatoes...")])
        results = await kb.search("tomato cultivation")
    """
    
    COLLECTION_NAME = "agri_knowledge"
    VECTOR_SIZE = 1024  # BGE-M3 dimensions
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize Knowledge Base.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: Optional API key
            collection_name: Override default collection name
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name or self.COLLECTION_NAME
        self._client = None
        self._embedding_manager = None
    
    @property
    def client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            self._connect()
        return self._client
    
    @property
    def embedding_manager(self):
        """Lazy load embedding manager."""
        if self._embedding_manager is None:
            from src.rag.embeddings import get_embedding_manager
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager
    
    def _connect(self):
        """Connect to Qdrant."""
        try:
            from qdrant_client import QdrantClient
            
            # Check if using cloud (URL contains qdrant.io or starts with https)
            is_cloud = (
                "qdrant.io" in self.host or 
                "cloud" in self.host or
                self.host.startswith("https://")
            )
            
            if is_cloud:
                # Cloud connection - use URL format
                url = self.host if self.host.startswith("https://") else f"https://{self.host}"
                # Ensure port is included if not already
                if ":6333" not in url and ":443" not in url:
                    url = f"{url}:6333"
                    
                logger.info(f"Connecting to Qdrant Cloud at {url}")
                self._client = QdrantClient(
                    url=url,
                    api_key=self.api_key,
                )
            elif self.host == ":memory:":
                # In-memory mode for testing/Colab
                logger.info("Using in-memory Qdrant")
                self._client = QdrantClient(":memory:")
            else:
                # Local connection
                logger.info(f"Connecting to local Qdrant at {self.host}:{self.port}")
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                )
            logger.info("Qdrant connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge base.
        
        Creates collection if it doesn't exist.
        
        Returns:
            True if initialized successfully
        """
        from qdrant_client.models import Distance, VectorParams
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    async def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 32,
    ) -> int:
        """
        Add documents to the knowledge base.
        
        Generates embeddings and stores in Qdrant.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of documents added
        """
        from qdrant_client.models import PointStruct
        
        if not documents:
            return 0
        
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        # Generate embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.embedding_manager.embed_documents(texts, batch_size=batch_size)
        
        # Create points
        points = []
        for doc, embedding in zip(documents, embeddings):
            point = PointStruct(
                id=doc.id,
                vector=embedding,
                payload={
                    "text": doc.text,
                    "source": doc.source,
                    "category": doc.category,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                },
            )
            points.append(point)
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Successfully added {len(documents)} documents")
        return len(documents)
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> SearchResult:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Optional category filter
            score_threshold: Minimum relevance score
            
        Returns:
            SearchResult with matching documents
        """
        import time
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        
        start_time = time.time()
        
        # Generate query embedding
        query_vector = self.embedding_manager.embed_query(query)
        
        # Build filter
        search_filter = None
        if category:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category),
                    )
                ]
            )
        
        # Search - try new API first, fall back to legacy
        try:
            # New Qdrant client API (1.7+)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=score_threshold,
            ).points
        except (AttributeError, TypeError):
            # Legacy Qdrant client API
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )
        
        # Convert to documents
        documents = []
        for hit in results:
            doc = Document(
                id=str(hit.id),
                text=hit.payload.get("text", ""),
                source=hit.payload.get("source", ""),
                category=hit.payload.get("category", ""),
                metadata=hit.payload.get("metadata", {}),
                score=hit.score,
            )
            documents.append(doc)
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResult(
            documents=documents,
            query=query,
            total_found=len(documents),
            search_time_ms=search_time,
        )
    
    async def delete_documents(
        self,
        document_ids: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> int:
        """
        Delete documents from knowledge base.
        
        Args:
            document_ids: Specific document IDs to delete
            category: Delete all documents in category
            
        Returns:
            Number of documents deleted
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList
        
        if document_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=document_ids),
            )
            return len(document_ids)
        
        elif category:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category),
                        )
                    ]
                ),
            )
            # We don't know exact count for filter delete
            return -1
        
        return 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            # Handle different Qdrant client versions
            vectors_count = getattr(info, 'vectors_count', None)
            points_count = getattr(info, 'points_count', None)
            
            # Newer versions might have different structure
            if vectors_count is None and hasattr(info, 'indexed_vectors_count'):
                vectors_count = info.indexed_vectors_count
            if points_count is None and hasattr(info, 'vectors_count'):
                points_count = info.vectors_count
            
            return {
                "collection": self.collection_name,
                "vectors_count": vectors_count or 0,
                "points_count": points_count or 0,
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}
