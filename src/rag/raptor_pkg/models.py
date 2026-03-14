"""
RAPTOR data models and configuration.
"""

from datetime import datetime
from enum import Enum
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class NodeLevel(int, Enum):
    """RAPTOR tree node levels."""
    LEAF = 0
    CLUSTER_1 = 1
    CLUSTER_2 = 2
    CLUSTER_3 = 3
    ROOT = 4


class RAPTORNode(BaseModel):
    """Node in the RAPTOR tree structure."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    level: int = 0  # 0 = leaf (original chunk), higher = more abstract
    
    embedding: Optional[list[float]] = None
    
    # Tree relationships
    children: list[str] = Field(default_factory=list)  # Child node IDs
    parent: Optional[str] = None  # Parent node ID
    
    # Metadata
    source_doc_ids: list[str] = Field(default_factory=list)
    chunk_indices: list[int] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.level == 0
    
    @property
    def is_summary(self) -> bool:
        """Check if this is a summary node."""
        return self.level > 0
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (~4 chars per token)."""
        return len(self.text) // 4


class RAPTORTreeStats(BaseModel):
    """Statistics about the RAPTOR tree."""
    
    total_nodes: int = 0
    nodes_per_level: dict[int, int] = Field(default_factory=dict)
    max_depth: int = 0
    total_documents: int = 0
    avg_children_per_node: float = 0.0
    avg_summary_length: int = 0


class ClusterInfo(BaseModel):
    """Information about a cluster during tree building."""
    
    cluster_id: int
    node_ids: list[str]
    centroid: list[float] = Field(default_factory=list)
    within_cluster_variance: float = 0.0


class RAPTORConfig(BaseModel):
    """Configuration for RAPTOR indexing."""
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Clustering settings
    max_cluster_size: int = 10
    min_cluster_size: int = 2
    n_clusters_ratio: float = 0.3  # Reduce nodes by this ratio each level
    use_umap: bool = True
    umap_n_components: int = 10
    umap_n_neighbors: int = 10
    
    # Summarization settings
    max_summary_length: int = 300
    summarization_model: str = "default"
    
    # Tree building
    max_levels: int = 4
    stop_at_single_cluster: bool = True
