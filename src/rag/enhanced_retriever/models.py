"""
Enhanced Retrieval Models
"""

import uuid
from enum import Enum
from typing import Optional

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
