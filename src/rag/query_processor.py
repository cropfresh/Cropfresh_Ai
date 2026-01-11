"""
Advanced Query Processing
=========================
Enhanced query processing techniques for improved retrieval.

Implements:
- HyDE (Hypothetical Document Embeddings)
- Multi-Query Expansion
- Step-Back Prompting
- Query Decomposition
- Query Rewriting

These techniques improve retrieval by generating better
search representations from user queries.

Reference: HyDE paper (Gao et al., 2023)

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Optional
from enum import Enum
import asyncio

from loguru import logger
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


class AdvancedQueryProcessor:
    """
    Advanced Query Processing for improved retrieval.
    
    Techniques:
    1. HyDE: Generate hypothetical answer, embed that instead of query
    2. Multi-Query: Generate variations of the query
    3. Step-Back: Generate higher-level abstraction of the query
    4. Decompose: Break complex queries into sub-queries
    5. Rewrite: Optimize query for retrieval
    
    Usage:
        processor = AdvancedQueryProcessor(llm)
        
        # HyDE expansion
        hyde_result = await processor.hyde_expand("What's the best fertilizer for tomatoes?")
        
        # Multi-query expansion
        multi_result = await processor.multi_query_expand("tomato pest control")
        
        # Full pipeline
        expanded = await processor.process_query("How to increase tomato yield?")
    """
    
    # Prompts for different techniques
    HYDE_PROMPT = """You are an agricultural expert writing a detailed answer.
Write a comprehensive paragraph that would answer this question about farming in India.
Include specific details, numbers, and practical recommendations.
Do not mention that this is hypothetical.

Question: {query}

Answer:"""

    MULTI_QUERY_PROMPT = """You are an AI assistant helping farmers find information.
Generate {count} different search queries that would help find information for this user question.
Each query should approach the topic from a different angle or focus on different aspects.
Make queries specific and searchable.

Original question: {query}

Generate {count} alternative search queries, one per line:"""

    STEP_BACK_PROMPT = """You are an AI assistant that reformulates specific questions into more general ones.
Given a specific question about agriculture, generate a broader, more abstract question 
that could provide background knowledge helpful for answering the original.

Original question: {query}

Step-back question (more general):"""

    DECOMPOSE_PROMPT = """You are an AI assistant that breaks down complex questions.
If the question is simple, just output the original question.
If complex, break it into 2-3 simpler sub-questions that together answer the original.

Original question: {query}

Sub-questions (one per line, or just the original if simple):"""

    REWRITE_PROMPT = """You are an AI assistant optimizing search queries.
Rewrite this question to be more effective for searching an agricultural knowledge base.
Remove filler words, add relevant agricultural terms, and make it more specific.

Original: {query}

Optimized search query:"""
    
    def __init__(
        self,
        llm=None,
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
        
        Args:
            query: User query
            techniques: Specific techniques to apply (None = all enabled)
            
        Returns:
            ExpandedQuery with all expansions
        """
        import time
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
                tasks.append(self._multi_query_expand_internal(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.HYDE:
                tasks.append(self._hyde_expand_internal(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.STEP_BACK:
                tasks.append(self._step_back_internal(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.DECOMPOSE:
                tasks.append(self._decompose_internal(query))
                task_types.append(technique)
            elif technique == QueryExpansionType.REWRITE:
                tasks.append(self._rewrite_internal(query))
                task_types.append(technique)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for technique, res in zip(task_types, results):
                if isinstance(res, Exception):
                    logger.warning(f"Query expansion failed for {technique}: {res}")
                    continue
                
                if technique == QueryExpansionType.MULTI_QUERY:
                    result.expanded_queries = res
                elif technique == QueryExpansionType.HYDE:
                    result.hypothetical_doc = res
                elif technique == QueryExpansionType.STEP_BACK:
                    result.step_back_query = res
                elif technique == QueryExpansionType.DECOMPOSE:
                    result.sub_queries = res
                elif technique == QueryExpansionType.REWRITE:
                    result.rewritten_query = res
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query processed: {len(result.all_queries)} total queries generated")
        return result
    
    async def hyde_expand(self, query: str) -> ExpandedQuery:
        """
        HyDE (Hypothetical Document Embeddings).
        
        Generates a hypothetical answer to the query, which is then
        used for embedding-based retrieval instead of the raw query.
        This often finds more relevant documents because the hypothetical
        answer is more similar to actual documents than the question.
        
        Args:
            query: User query
            
        Returns:
            ExpandedQuery with hypothetical document
        """
        import time
        start_time = time.time()
        
        hypothetical_doc = await self._hyde_expand_internal(query)
        
        return ExpandedQuery(
            original_query=query,
            hypothetical_doc=hypothetical_doc,
            expansion_type=QueryExpansionType.HYDE,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def _hyde_expand_internal(self, query: str) -> str:
        """Generate hypothetical document for HyDE."""
        if self.llm is None:
            return self._rule_based_hyde(query)
        
        try:
            prompt = self.HYDE_PROMPT.format(query=query)
            response = await self.llm.agenerate([prompt])
            hypothetical = response.generations[0][0].text.strip()
            
            # Truncate if too long
            if len(hypothetical) > self.config.hyde_max_length * 4:
                hypothetical = hypothetical[:self.config.hyde_max_length * 4]
            
            return hypothetical
            
        except Exception as e:
            logger.warning(f"HyDE expansion failed: {e}")
            return self._rule_based_hyde(query)
    
    def _rule_based_hyde(self, query: str) -> str:
        """Simple rule-based hypothetical document generation."""
        # Create a template-based hypothetical answer
        query_lower = query.lower()
        
        if "how" in query_lower or "what" in query_lower:
            return f"To address '{query}', farmers should consider several factors. " \
                   f"Based on agricultural best practices in India, the recommended approach involves " \
                   f"proper planning, selecting appropriate varieties, and following scientific methods. " \
                   f"Key considerations include soil preparation, irrigation management, and pest control."
        elif "when" in query_lower:
            return f"The timing for '{query}' depends on the crop season and local conditions. " \
                   f"In India, Kharif season (June-October) and Rabi season (October-March) have different cycles. " \
                   f"Consult local KVK for region-specific guidance."
        elif "why" in query_lower:
            return f"Understanding '{query}' is crucial for successful farming. " \
                   f"The reasons involve multiple factors including soil health, climate, and market conditions. " \
                   f"Scientific research has established clear guidelines for farmers."
        else:
            return f"Regarding '{query}', agricultural experts recommend following established guidelines. " \
                   f"Proper implementation involves understanding crop requirements, local conditions, and market dynamics. " \
                   f"Farmers should consult agricultural extension services for personalized advice."
    
    async def multi_query_expand(self, query: str) -> ExpandedQuery:
        """
        Generate multiple query variations.
        
        Creates different phrasings and perspectives of the same query
        to improve recall by searching with multiple queries.
        
        Args:
            query: User query
            
        Returns:
            ExpandedQuery with multiple queries
        """
        import time
        start_time = time.time()
        
        expanded = await self._multi_query_expand_internal(query)
        
        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded,
            expansion_type=QueryExpansionType.MULTI_QUERY,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def _multi_query_expand_internal(self, query: str) -> list[str]:
        """Generate multiple query variations."""
        if self.llm is None:
            return self._rule_based_multi_query(query)
        
        try:
            count = self.config.multi_query_count
            prompt = self.MULTI_QUERY_PROMPT.format(query=query, count=count)
            response = await self.llm.agenerate([prompt])
            
            # Parse response into queries
            raw_queries = response.generations[0][0].text.strip().split('\n')
            queries = []
            
            for q in raw_queries:
                # Clean up numbering and whitespace
                q = q.strip()
                if q.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                    q = q[2:].strip()
                if q and q != query:
                    queries.append(q)
            
            return queries[:count]
            
        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}")
            return self._rule_based_multi_query(query)
    
    def _rule_based_multi_query(self, query: str) -> list[str]:
        """Simple rule-based query expansion."""
        query_lower = query.lower()
        expansions = []
        
        # Add synonym-based variations
        synonyms = {
            "tomato": ["tamatar", "tomatoes"],
            "onion": ["pyaj", "onions"],
            "potato": ["aloo", "potatoes"],
            "pest": ["insects", "bugs", "infestation"],
            "disease": ["infection", "blight", "rot"],
            "fertilizer": ["manure", "nutrients", "NPK"],
            "price": ["rate", "cost", "market value"],
            "yield": ["production", "harvest", "output"],
            "grow": ["cultivate", "plant", "farm"],
            "best": ["recommended", "ideal", "optimal"],
        }
        
        for word, syns in synonyms.items():
            if word in query_lower:
                for syn in syns[:1]:
                    expansions.append(query_lower.replace(word, syn))
        
        # Add perspective variations
        if "how to" in query_lower:
            expansions.append(query_lower.replace("how to", "best practices for"))
            expansions.append(query_lower.replace("how to", "guide for"))
        
        if "what is" in query_lower:
            expansions.append(query_lower.replace("what is", "explain"))
        
        # Add location context
        if not any(loc in query_lower for loc in ["india", "karnataka", "maharashtra"]):
            expansions.append(query + " in India")
        
        return expansions[:self.config.multi_query_count]
    
    async def step_back_expand(self, query: str) -> ExpandedQuery:
        """
        Step-Back Prompting.
        
        Generates a more abstract, higher-level question that can
        provide background knowledge helpful for answering the original.
        
        Args:
            query: User query
            
        Returns:
            ExpandedQuery with step-back question
        """
        import time
        start_time = time.time()
        
        step_back = await self._step_back_internal(query)
        
        return ExpandedQuery(
            original_query=query,
            step_back_query=step_back,
            expansion_type=QueryExpansionType.STEP_BACK,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def _step_back_internal(self, query: str) -> str:
        """Generate step-back question."""
        if self.llm is None:
            return self._rule_based_step_back(query)
        
        try:
            prompt = self.STEP_BACK_PROMPT.format(query=query)
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.warning(f"Step-back expansion failed: {e}")
            return self._rule_based_step_back(query)
    
    def _rule_based_step_back(self, query: str) -> str:
        """Simple rule-based step-back question."""
        query_lower = query.lower()
        
        # Extract crop if mentioned
        crops = ["tomato", "onion", "potato", "rice", "wheat", "cotton", "maize"]
        mentioned_crop = None
        for crop in crops:
            if crop in query_lower:
                mentioned_crop = crop
                break
        
        # Generate abstract question based on topic
        if "pest" in query_lower or "insect" in query_lower:
            if mentioned_crop:
                return f"What are the general principles of integrated pest management for {mentioned_crop}?"
            return "What are the general principles of integrated pest management in vegetable crops?"
        
        elif "disease" in query_lower or "blight" in query_lower:
            return "What are the common causes and prevention methods for plant diseases?"
        
        elif "price" in query_lower or "market" in query_lower:
            return "What factors affect agricultural commodity prices in Indian markets?"
        
        elif "yield" in query_lower or "production" in query_lower:
            return "What are the key factors that determine crop yield and productivity?"
        
        elif "fertilizer" in query_lower or "nutrient" in query_lower:
            return "What are the nutrient requirements and fertilizer management principles for crops?"
        
        elif "water" in query_lower or "irrigat" in query_lower:
            return "What are the principles of efficient irrigation management in agriculture?"
        
        else:
            if mentioned_crop:
                return f"What are the complete cultivation practices for {mentioned_crop} in India?"
            return "What are the fundamental principles of successful agricultural practices in India?"
    
    async def decompose_query(self, query: str) -> ExpandedQuery:
        """
        Query Decomposition.
        
        Breaks complex queries into simpler sub-queries that
        can be answered individually and combined.
        
        Args:
            query: User query
            
        Returns:
            ExpandedQuery with sub-queries
        """
        import time
        start_time = time.time()
        
        sub_queries = await self._decompose_internal(query)
        
        return ExpandedQuery(
            original_query=query,
            sub_queries=sub_queries,
            expansion_type=QueryExpansionType.DECOMPOSE,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def _decompose_internal(self, query: str) -> list[str]:
        """Decompose query into sub-queries."""
        if self.llm is None:
            return self._rule_based_decompose(query)
        
        try:
            prompt = self.DECOMPOSE_PROMPT.format(query=query)
            response = await self.llm.agenerate([prompt])
            
            # Parse sub-queries
            raw = response.generations[0][0].text.strip().split('\n')
            sub_queries = []
            
            for q in raw:
                q = q.strip()
                if q.startswith(('1.', '2.', '3.', '-', '*')):
                    q = q[2:].strip()
                if q and q != query:
                    sub_queries.append(q)
            
            return sub_queries[:self.config.max_sub_queries]
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return self._rule_based_decompose(query)
    
    def _rule_based_decompose(self, query: str) -> list[str]:
        """Simple rule-based query decomposition."""
        query_lower = query.lower()
        
        # Check for compound queries (and, also, as well as)
        if " and " in query_lower:
            parts = query_lower.split(" and ")
            return [p.strip().capitalize() + "?" for p in parts if len(p) > 10]
        
        # Check for multi-aspect queries
        if "how to" in query_lower and ("when" in query_lower or "where" in query_lower):
            return [
                query_lower.split("when")[0].strip() + "?",
                "When is the best time for " + query_lower.split("how to")[-1].split("when")[0].strip() + "?",
            ]
        
        # Default: return empty (query is simple)
        return []
    
    async def rewrite_query(self, query: str) -> ExpandedQuery:
        """
        Query Rewriting.
        
        Optimizes the query for better retrieval by:
        - Removing filler words
        - Adding relevant context
        - Standardizing terminology
        
        Args:
            query: User query
            
        Returns:
            ExpandedQuery with rewritten query
        """
        import time
        start_time = time.time()
        
        rewritten = await self._rewrite_internal(query)
        
        return ExpandedQuery(
            original_query=query,
            rewritten_query=rewritten,
            expansion_type=QueryExpansionType.REWRITE,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def _rewrite_internal(self, query: str) -> str:
        """Rewrite query for better retrieval."""
        if self.llm is None:
            return self._rule_based_rewrite(query)
        
        try:
            prompt = self.REWRITE_PROMPT.format(query=query)
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return self._rule_based_rewrite(query)
    
    def _rule_based_rewrite(self, query: str) -> str:
        """Simple rule-based query rewriting."""
        # Remove common filler phrases
        fillers = [
            "can you tell me",
            "i want to know",
            "please help me",
            "i need to understand",
            "what is the answer to",
            "could you explain",
        ]
        
        rewritten = query.lower()
        for filler in fillers:
            rewritten = rewritten.replace(filler, "").strip()
        
        # Standardize terminology
        replacements = {
            "tamatar": "tomato",
            "pyaj": "onion",
            "aloo": "potato",
            "gehu": "wheat",
            "dhan": "rice",
            "kapas": "cotton",
        }
        
        for hindi, english in replacements.items():
            rewritten = rewritten.replace(hindi, english)
        
        # Capitalize first letter
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]
        
        return rewritten


# Factory function
def create_query_processor(
    llm=None,
    config: Optional[QueryProcessorConfig] = None,
) -> AdvancedQueryProcessor:
    """
    Create an advanced query processor.
    
    Args:
        llm: Language model for processing
        config: Processor configuration
        
    Returns:
        AdvancedQueryProcessor instance
    """
    return AdvancedQueryProcessor(llm=llm, config=config)
