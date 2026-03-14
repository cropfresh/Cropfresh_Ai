"""
Contextual Chunker Core
=======================
Main class for context-enriched document chunking.
"""

from typing import Optional
from loguru import logger

from .models import ChunkingConfig, EnrichedChunk
from .extractor import ExtractorMixin
from .splitter import SplitterMixin
from .llm_context import LLMContextMixin


class ContextualChunker(ExtractorMixin, SplitterMixin, LLMContextMixin):
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
    """

    def __init__(
        self,
        llm=None,
        config: Optional[ChunkingConfig] = None,
    ):
        self.llm = llm
        self.config = config or ChunkingConfig()
        super().__init__()
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
        if hasattr(document, 'text'):
            text = document.text
            title = document_title or document.metadata.get("title", "")
            source = document_source or document.metadata.get("source", "")
        else:
            text = str(document)
            title = document_title
            source = document_source
        
        section_headers = []
        if self.config.propagate_headers:
            section_headers = self._extract_section_headers(text)
        
        if self.config.use_semantic_boundaries:
            raw_chunks = self._semantic_chunk(text)
        else:
            raw_chunks = self._simple_chunk(text)
        
        enriched_chunks = []
        total_chunks = len(raw_chunks)
        
        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            section = self._find_section_for_position(section_headers, start_char)
            
            entities = []
            if self.config.extract_entities:
                entities = self._extract_entities(chunk_text)
            
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
