"""
Instructed Retriever
====================
LLM-guided retrieval with instruction-following.

Instead of raw query -> retrieval, uses:
1. Query analysis to generate retrieval instructions
2. Dynamic filter/boost selection
3. Result ranking based on instruction adherence
"""

from datetime import datetime
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field


class RetrievalInstruction(BaseModel):
    """Instructions for retrieval."""
    original_query: str
    refined_query: str = ""
    keywords: list[str] = Field(default_factory=list)
    required_topics: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    time_range: Optional[str] = None  # "recent", "historical", "all"
    source_types: list[str] = Field(default_factory=list)
    min_confidence: float = 0.5
    max_results: int = 10


INSTRUCTION_PROMPT = """Analyze this query and generate retrieval instructions.

Query: {query}

Generate instructions for finding relevant documents:
1. List key search keywords
2. Identify required topics that must be present
3. Identify topics to exclude
4. Suggest time relevance (recent/historical/all)
5. Suggest source types (articles, data, guides)

Respond with JSON:
{{
    "refined_query": "clearer version of query",
    "keywords": ["keyword1", "keyword2"],
    "required_topics": ["must have topic"],
    "excluded_topics": ["avoid these"],
    "time_range": "recent|historical|all",
    "source_types": ["articles", "data", "guides"],
    "min_confidence": 0.5
}}"""


class InstructedRetriever:
    """
    LLM-instructed retrieval for better precision.
    
    Usage:
        retriever = InstructedRetriever(llm=groq, kb=knowledge_base)
        
        results = await retriever.retrieve(
            "What are the current tomato prices in Karnataka?"
        )
    """
    
    def __init__(
        self,
        llm=None,
        knowledge_base=None,
        enable_instruction_generation: bool = True,
    ):
        """
        Initialize instructed retriever.
        
        Args:
            llm: LLM for instruction generation
            knowledge_base: Knowledge base for retrieval
            enable_instruction_generation: Use LLM for instructions
        """
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.enable_instructions = enable_instruction_generation
    
    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.5,
    ) -> list[dict]:
        """
        Retrieve documents with instruction guidance.
        
        Args:
            query: User query
            max_results: Maximum results
            min_score: Minimum relevance score
            
        Returns:
            List of relevant documents
        """
        # Step 1: Generate instructions
        if self.enable_instructions and self.llm:
            instructions = await self._generate_instructions(query)
        else:
            instructions = RetrievalInstruction(
                original_query=query,
                refined_query=query,
                keywords=self._extract_keywords(query),
            )
        
        # Step 2: Execute retrieval with instructions
        results = await self._execute_retrieval(instructions)
        
        # Step 3: Re-rank based on instruction adherence
        ranked_results = self._rank_by_instructions(results, instructions)
        
        # Step 4: Filter and return
        filtered = [
            r for r in ranked_results
            if r.get("score", 0) >= min_score
        ][:max_results]
        
        logger.info(
            "Instructed retrieval: {} -> {} results",
            query[:30], len(filtered)
        )
        
        return filtered
    
    async def _generate_instructions(self, query: str) -> RetrievalInstruction:
        """Generate retrieval instructions via LLM."""
        try:
            import json
            
            prompt = INSTRUCTION_PROMPT.format(query=query)
            response = await self.llm.agenerate(prompt)
            
            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            return RetrievalInstruction(
                original_query=query,
                refined_query=data.get("refined_query", query),
                keywords=data.get("keywords", []),
                required_topics=data.get("required_topics", []),
                excluded_topics=data.get("excluded_topics", []),
                time_range=data.get("time_range", "all"),
                source_types=data.get("source_types", []),
                min_confidence=data.get("min_confidence", 0.5),
            )
            
        except Exception as e:
            logger.warning("Instruction generation failed: {}", str(e))
            return RetrievalInstruction(
                original_query=query,
                refined_query=query,
                keywords=self._extract_keywords(query),
            )
    
    async def _execute_retrieval(
        self,
        instructions: RetrievalInstruction,
    ) -> list[dict]:
        """Execute retrieval with instructions."""
        if not self.knowledge_base:
            return []
        
        # Use refined query
        query = instructions.refined_query or instructions.original_query
        
        # Build filters from instructions
        filters = {}
        if instructions.source_types:
            filters["source_type"] = instructions.source_types
        if instructions.time_range == "recent":
            filters["recency"] = "30d"
        
        try:
            results = await self.knowledge_base.search(
                query=query,
                limit=instructions.max_results * 2,  # Over-fetch for filtering
                filters=filters if filters else None,
            )
            
            # Convert to dicts
            return [
                {
                    "content": r.content if hasattr(r, 'content') else str(r),
                    "score": r.score if hasattr(r, 'score') else 0.5,
                    "metadata": r.metadata if hasattr(r, 'metadata') else {},
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error("Retrieval failed: {}", str(e))
            return []
    
    def _rank_by_instructions(
        self,
        results: list[dict],
        instructions: RetrievalInstruction,
    ) -> list[dict]:
        """Re-rank results based on instruction adherence."""
        for result in results:
            content = result.get("content", "").lower()
            base_score = result.get("score", 0.5)
            
            # Boost for required topics
            topic_boost = 0
            for topic in instructions.required_topics:
                if topic.lower() in content:
                    topic_boost += 0.1
            
            # Penalty for excluded topics
            topic_penalty = 0
            for topic in instructions.excluded_topics:
                if topic.lower() in content:
                    topic_penalty += 0.2
            
            # Boost for keywords
            keyword_boost = 0
            for kw in instructions.keywords:
                if kw.lower() in content:
                    keyword_boost += 0.05
            
            # Calculate final score
            final_score = base_score + topic_boost - topic_penalty + keyword_boost
            result["score"] = min(1.0, max(0.0, final_score))
        
        # Sort by score
        return sorted(results, key=lambda r: r["score"], reverse=True)
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Simple keyword extraction."""
        # Remove common words
        stopwords = {"what", "how", "is", "the", "a", "an", "in", "for", "to", "of", "and", "are"}
        words = query.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]
