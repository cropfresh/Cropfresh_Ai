"""
Main Query Processor Class
"""

import time
import asyncio
from typing import Optional
from loguru import logger
from typing import Any

from .models import ExpandedQuery, QueryExpansionType, QueryProcessorConfig
from .hyde import hyde_expand
from .multi_query import multi_query_expand
from .step_back import step_back_expand
from .decompose import decompose_query
from .rewrite import rewrite_query


class AdvancedQueryProcessor:
    """
    Advanced Query Processing for improved retrieval.
    """
    
    def __init__(
        self,
        llm: Any = None,
        config: Optional[QueryProcessorConfig] = None,
    ):
        """
        Initialize advanced query processor.
        
        Args:
            llm: Language model for query processing
            config: Query processor configuration
        """
        self.llm = llm
        self.config = config or QueryProcessorConfig()
        
        logger.info("AdvancedQueryProcessor initialized")
    
    async def process_query(
        self,
        query: str,
        techniques: Optional[list[QueryExpansionType]] = None,
    ) -> ExpandedQuery:
        """
        Apply multiple query processing techniques.
        """
        start_time = time.time()
        
        result = ExpandedQuery(
            original_query=query,
            expansion_type=QueryExpansionType.MULTI_QUERY,
        )
        
        # Determine which techniques to apply
        if techniques is None:
            techniques = []
            if self.config.multi_query_enabled:
                techniques.append(QueryExpansionType.MULTI_QUERY)
            if self.config.hyde_enabled:
                techniques.append(QueryExpansionType.HYDE)
            if self.config.step_back_enabled:
                techniques.append(QueryExpansionType.STEP_BACK)
            if self.config.decompose_enabled:
                techniques.append(QueryExpansionType.DECOMPOSE)
            if self.config.rewrite_enabled:
                techniques.append(QueryExpansionType.REWRITE)
        
        # Apply techniques in parallel
        tasks = []
        task_types = []
        
        for technique in techniques:
            if technique == QueryExpansionType.MULTI_QUERY:
                tasks.append(self.multi_query_expand(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.HYDE:
                tasks.append(self.hyde_expand(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.STEP_BACK:
                tasks.append(self.step_back_expand(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.DECOMPOSE:
                tasks.append(self.decompose_query(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.REWRITE:
                tasks.append(self.rewrite_query(query))
                task_types.append(technique)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for technique, res in zip(task_types, results):
                if isinstance(res, Exception):
                    logger.warning(f"Query expansion failed for {technique}: {res}")
                    continue
                
                # We expect the helper functions to return an ExpandedQuery from which we can pull the values
                if technique == QueryExpansionType.MULTI_QUERY:
                    result.expanded_queries = res.expanded_queries
                elif technique == QueryExpansionType.HYDE:
                    result.hypothetical_doc = res.hypothetical_doc
                elif technique == QueryExpansionType.STEP_BACK:
                    result.step_back_query = res.step_back_query
                elif technique == QueryExpansionType.DECOMPOSE:
                    result.sub_queries = res.sub_queries
                elif technique == QueryExpansionType.REWRITE:
                    result.rewritten_query = res.rewritten_query
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query processed: {len(result.all_queries)} total queries generated")
        return result
    
    async def hyde_expand(self, query: str) -> ExpandedQuery:
        """HyDE (Hypothetical Document Embeddings)."""
        return await hyde_expand(self.llm, self.config, query)
    
    async def multi_query_expand(self, query: str) -> ExpandedQuery:
        """Generate multiple query variations."""
        return await multi_query_expand(self.llm, self.config, query)
    
    async def step_back_expand(self, query: str) -> ExpandedQuery:
        """Step-Back Prompting."""
        return await step_back_expand(self.llm, query)
    
    async def decompose_query(self, query: str) -> ExpandedQuery:
        """Query Decomposition."""
        return await decompose_query(self.llm, self.config, query)
    
    async def rewrite_query(self, query: str) -> ExpandedQuery:
        """Query Rewriting."""
        return await rewrite_query(self.llm, query)
