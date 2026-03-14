"""
Contextual Chunker Models
=========================
Data schemas for context-enriched document chunking.
"""

from datetime import datetime
import uuid
from pydantic import BaseModel, Field


class EnrichedChunk(BaseModel):
    """A document chunk enriched with contextual information."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    
    context: str = ""
    section_header: str = ""
    document_title: str = ""
    document_source: str = ""
    
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def enriched_text(self) -> str:
        parts = []
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.section_header:
            parts.append(f"Section: {self.section_header}")
        if self.document_title:
            parts.append(f"Document: {self.document_title}")
        parts.append(self.text)
        return "\n\n".join(parts)
    
    @property
    def token_estimate(self) -> int:
        return len(self.enriched_text) // 4


class ChunkingConfig(BaseModel):
    """Configuration for contextual chunking."""
    
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    
    add_context: bool = True
    context_max_length: int = 150
    propagate_headers: bool = True
    
    extract_entities: bool = True
    entity_types: list[str] = Field(default_factory=lambda: [
        "CROP", "PEST", "DISEASE", "CHEMICAL", "LOCATION", "PRICE"
    ])
    
    use_semantic_boundaries: bool = True
    use_llm_context: bool = True
