"""
Contextual Chunking
===================
Enhanced document chunking with context enrichment for improved retrieval.

Features:
- Context-enriched chunks with document metadata
- LLM-generated context summaries
- Section header extraction and propagation
- Entity extraction for each chunk
- Semantic boundary detection

Reference: Anthropic Contextual Retrieval (2024)

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Optional
import re
import uuid

from loguru import logger
from pydantic import BaseModel, Field


class EnrichedChunk(BaseModel):
    """A document chunk enriched with contextual information."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    
    # Context enrichment
    context: str = ""  # LLM-generated or extracted context
    section_header: str = ""  # Parent section header
    document_title: str = ""
    document_source: str = ""
    
    # Position information
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    
    # Extracted entities
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    
    # Metadata
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def enriched_text(self) -> str:
        """
        Get the chunk text with context prepended.
        
        This enhanced text is used for embedding, providing
        better context for similarity matching.
        """
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
        """Estimate token count (~4 chars per token)."""
        return len(self.enriched_text) // 4


class ChunkingConfig(BaseModel):
    """Configuration for contextual chunking."""
    
    # Size settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    
    # Context settings
    add_context: bool = True
    context_max_length: int = 150
    propagate_headers: bool = True
    
    # Entity extraction
    extract_entities: bool = True
    entity_types: list[str] = Field(default_factory=lambda: [
        "CROP", "PEST", "DISEASE", "CHEMICAL", "LOCATION", "PRICE"
    ])
    
    # Semantic chunking
    use_semantic_boundaries: bool = True
    
    # LLM settings
    use_llm_context: bool = True


class ContextualChunker:
    """
    Context-enriched document chunking.
    
    Each chunk is augmented with:
    - Document title and source
    - Section headers (if any)
    - Preceding context summary
    - Key entities mentioned
    
    This context helps the retrieval system understand
    what each chunk is about, even without seeing the
    full document.
    
    Usage:
        chunker = ContextualChunker(llm=llm)
        
        chunks = await chunker.chunk_with_context(document)
        
        # Use enriched text for embedding
        for chunk in chunks:
            embedding = embed(chunk.enriched_text)
    """
    
    # Context generation prompt
    CONTEXT_PROMPT = """You are generating context for a document chunk to help a search system understand it better.

Given the document information and a specific chunk from it, write a brief context statement (2-3 sentences) that explains:
1. What document this is from and its topic
2. What specific subject this chunk covers
3. Any key entities or concepts mentioned

Document Title: {title}
Document Source: {source}
Document Summary: {summary}

Chunk:
{chunk}

Brief Context (2-3 sentences):"""
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "CROP": [
            r'\b(tomato|potato|onion|rice|wheat|maize|cotton|sugarcane|mango|banana|chilli|turmeric)\b',
            r'\b(टमाटर|आलू|प्याज|चावल|गेहूं|मक्का|कपास)\b',  # Hindi
        ],
        "PEST": [
            r'\b(aphid|whitefly|thrips|mite|borer|caterpillar|beetle|worm|nematode)\b',
            r'\b(pest|insect|bug|larvae)\b',
        ],
        "DISEASE": [
            r'\b(blight|wilt|rust|rot|mosaic|mildew|canker|scab|leaf spot)\b',
            r'\b(fungal|bacterial|viral|disease|infection)\b',
        ],
        "CHEMICAL": [
            r'\b(neem|pesticide|fungicide|insecticide|herbicide|fertilizer|urea|DAP|potash)\b',
        ],
        "LOCATION": [
            r'\b(Karnataka|Maharashtra|Andhra Pradesh|Tamil Nadu|Kerala|Gujarat|Punjab|Haryana)\b',
            r'\b(Kolar|Nashik|Pune|Bangalore|Hyderabad|Chennai)\b',
        ],
        "PRICE": [
            r'₹\s*[\d,]+',
            r'\b\d+\s*(?:per|/)\s*(?:kg|quintal|tonne)\b',
        ],
    }
    
    def __init__(
        self,
        llm=None,
        config: Optional[ChunkingConfig] = None,
    ):
        """
        Initialize contextual chunker.
        
        Args:
            llm: LLM for context generation (optional)
            config: Chunking configuration
        """
        self.llm = llm
        self.config = config or ChunkingConfig()
        
        # Compile entity patterns
        self._compiled_patterns = {
            entity_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for entity_type, patterns in self.ENTITY_PATTERNS.items()
        }
        
        logger.info("ContextualChunker initialized")
    
    async def chunk_with_context(
        self,
        document,
        document_title: str = "",
        document_source: str = "",
        document_summary: str = "",
    ) -> list[EnrichedChunk]:
        """
        Split document and add context to each chunk.
        
        Args:
            document: Document object or text string
            document_title: Optional title override
            document_source: Optional source override
            document_summary: Document summary for context
            
        Returns:
            List of EnrichedChunk with context
        """
        # Get text and metadata
        if hasattr(document, 'text'):
            text = document.text
            title = document_title or document.metadata.get("title", "")
            source = document_source or document.metadata.get("source", "")
        else:
            text = str(document)
            title = document_title
            source = document_source
        
        # Extract section headers if enabled
        section_headers = []
        if self.config.propagate_headers:
            section_headers = self._extract_section_headers(text)
        
        # Split into chunks
        if self.config.use_semantic_boundaries:
            raw_chunks = self._semantic_chunk(text)
        else:
            raw_chunks = self._simple_chunk(text)
        
        # Enrich each chunk
        enriched_chunks = []
        total_chunks = len(raw_chunks)
        
        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            # Find applicable section header
            section = self._find_section_for_position(section_headers, start_char)
            
            # Extract entities
            entities = []
            if self.config.extract_entities:
                entities = self._extract_entities(chunk_text)
            
            # Generate context
            context = ""
            if self.config.add_context:
                if self.config.use_llm_context and self.llm:
                    context = await self._generate_llm_context(
                        chunk_text, title, source, document_summary
                    )
                else:
                    context = self._generate_simple_context(
                        chunk_text, title, source, section, idx, total_chunks
                    )
            
            enriched_chunks.append(EnrichedChunk(
                text=chunk_text,
                context=context,
                section_header=section,
                document_title=title,
                document_source=source,
                chunk_index=idx,
                total_chunks=total_chunks,
                start_char=start_char,
                end_char=end_char,
                entities=entities,
                keywords=self._extract_keywords(chunk_text),
            ))
        
        logger.info(f"Created {len(enriched_chunks)} enriched chunks")
        return enriched_chunks
    
    def _simple_chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Simple character-based chunking with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                search_start = max(end - 100, start)
                best_break = end
                
                for break_char in ['. ', '.\n', '! ', '? ', '\n\n']:
                    pos = text.rfind(break_char, search_start, end + 50)
                    if pos > start:
                        best_break = pos + len(break_char)
                        break
                
                end = best_break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append((chunk_text, start, end))
            
            # Move to next chunk with overlap
            start = end - self.config.chunk_overlap
            start = max(start, end - self.config.chunk_size // 2)  # Prevent infinite loop
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> list[tuple[str, int, int]]:
        """
        Semantic boundary-aware chunking.
        
        Respects:
        - Paragraph boundaries
        - Section headers
        - List items
        - Sentence endings
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                char_pos += 2  # Account for removed newlines
                continue
            
            # Check if adding this paragraph exceeds limit
            if len(current_chunk) + len(para) + 2 > self.config.chunk_size:
                # Save current chunk if not empty
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append((
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk),
                    ))
                
                # Start new chunk
                current_chunk = para
                current_start = char_pos
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = char_pos
            
            char_pos += len(para) + 2
        
        # Don't forget last chunk
        if len(current_chunk) >= self.config.min_chunk_size:
            chunks.append((
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk),
            ))
        
        return chunks
    
    def _extract_section_headers(self, text: str) -> list[tuple[str, int]]:
        """Extract section headers and their positions."""
        headers = []
        
        # Common header patterns
        patterns = [
            r'^(#{1,3})\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+):?\s*$',  # ALL CAPS HEADERS
            r'^(\d+\.\s+.+)$',  # Numbered sections
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                header_text = match.group(0).strip()
                header_text = re.sub(r'^#+\s*', '', header_text)  # Remove markdown #
                headers.append((header_text, match.start()))
        
        # Sort by position
        headers.sort(key=lambda x: x[1])
        
        return headers
    
    def _find_section_for_position(
        self,
        headers: list[tuple[str, int]],
        position: int,
    ) -> str:
        """Find the section header that applies to a given position."""
        applicable_header = ""
        
        for header_text, header_pos in headers:
            if header_pos <= position:
                applicable_header = header_text
            else:
                break
        
        return applicable_header
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []
        
        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = f"{entity_type}:{match.group()}"
                    if entity not in entities:
                        entities.append(entity)
        
        return entities[:20]  # Limit to prevent bloat
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract key phrases from text."""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will',
            'they', 'their', 'there', 'which', 'about', 'would', 'could',
            'should', 'does', 'these', 'those', 'after', 'before',
        }
        
        filtered = [w for w in words if w not in stop_words]
        
        # Count frequencies
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:max_keywords]]
    
    async def _generate_llm_context(
        self,
        chunk_text: str,
        title: str,
        source: str,
        summary: str,
    ) -> str:
        """Generate context using LLM."""
        try:
            prompt = self.CONTEXT_PROMPT.format(
                title=title or "Unknown",
                source=source or "Unknown",
                summary=summary or "Agricultural knowledge document",
                chunk=chunk_text[:1000],  # Limit chunk size
            )
            
            response = await self.llm.agenerate([prompt])
            context = response.generations[0][0].text.strip()
            
            # Truncate if too long
            if len(context) > self.config.context_max_length * 4:
                context = context[:self.config.context_max_length * 4] + "..."
            
            return context
            
        except Exception as e:
            logger.warning(f"LLM context generation failed: {e}")
            return self._generate_simple_context(chunk_text, title, source, "", 0, 1)
    
    def _generate_simple_context(
        self,
        chunk_text: str,
        title: str,
        source: str,
        section: str,
        index: int,
        total: int,
    ) -> str:
        """Generate simple rule-based context."""
        parts = []
        
        if title:
            parts.append(f"From '{title}'")
        
        if source:
            parts.append(f"source: {source}")
        
        if section:
            parts.append(f"in section: {section}")
        
        # Add position context
        if total > 1:
            if index == 0:
                parts.append("(beginning of document)")
            elif index == total - 1:
                parts.append("(end of document)")
            else:
                parts.append(f"(part {index + 1} of {total})")
        
        return ". ".join(parts) if parts else ""


# Factory function
def create_contextual_chunker(
    llm=None,
    config: Optional[ChunkingConfig] = None,
) -> ContextualChunker:
    """
    Create a configured contextual chunker.
    
    Args:
        llm: LLM for context generation
        config: Chunking configuration
        
    Returns:
        ContextualChunker instance
    """
    return ContextualChunker(llm=llm, config=config)


# Utility function for batch processing
async def enrich_documents(
    documents: list,
    llm=None,
    config: Optional[ChunkingConfig] = None,
) -> list[EnrichedChunk]:
    """
    Convenience function to chunk multiple documents with context.
    
    Args:
        documents: List of documents
        llm: LLM for context generation
        config: Chunking configuration
        
    Returns:
        Flat list of all EnrichedChunk from all documents
    """
    chunker = create_contextual_chunker(llm=llm, config=config)
    
    all_chunks = []
    for doc in documents:
        chunks = await chunker.chunk_with_context(doc)
        all_chunks.extend(chunks)
    
    return all_chunks
