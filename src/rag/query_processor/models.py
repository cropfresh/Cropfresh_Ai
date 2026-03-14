"""
Query Processor Models
"""

from enum import Enum
from pydantic import BaseModel, Field


class QueryExpansionType(str, Enum):
    """Types of query expansion."""
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    STEP_BACK = "step_back"
    DECOMPOSE = "decompose"
    REWRITE = "rewrite"


class ExpandedQuery(BaseModel):
    """Result of query expansion."""
    
    original_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    hypothetical_doc: str = ""  # For HyDE
    step_back_query: str = ""   # For step-back prompting
    sub_queries: list[str] = Field(default_factory=list)  # For decomposition
    rewritten_query: str = ""
    
    # Metadata
    expansion_type: QueryExpansionType
    model_used: str = ""
    processing_time_ms: float = 0.0
    
    @property
    def all_queries(self) -> list[str]:
        """Get all generated queries for retrieval."""
        queries = [self.original_query]
        queries.extend(self.expanded_queries)
        queries.extend(self.sub_queries)
        if self.step_back_query:
            queries.append(self.step_back_query)
        if self.rewritten_query:
            queries.append(self.rewritten_query)
        return list(set(queries))


class QueryProcessorConfig(BaseModel):
    """Configuration for query processing."""
    
    # HyDE settings
    hyde_enabled: bool = True
    hyde_num_docs: int = 1
    hyde_max_length: int = 300
    
    # Multi-query settings
    multi_query_enabled: bool = True
    multi_query_count: int = 3
    
    # Step-back prompting
    step_back_enabled: bool = True
    
    # Decomposition
    decompose_enabled: bool = True
    max_sub_queries: int = 3
    
    # Rewriting
    rewrite_enabled: bool = True
