"""
Strategy Selector
=================
Dynamic retrieval strategy selection based on query analysis.

Strategies:
- Dense: Semantic similarity (default)
- Sparse: BM25 keyword matching
- Hybrid: Dense + Sparse with RRF
- Graph: Relationship-based retrieval
- Multi-hop: Iterative refinement
"""

from enum import Enum
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    DENSE = "dense"           # Semantic embedding similarity
    SPARSE = "sparse"         # BM25 keyword matching
    HYBRID = "hybrid"         # Dense + Sparse fusion
    GRAPH = "graph"           # Neo4j relationship traversal
    MULTI_HOP = "multi_hop"   # Iterative retrieval
    INSTRUCTED = "instructed" # LLM-guided retrieval


class QueryAnalysis(BaseModel):
    """Analysis of a query for strategy selection."""
    query: str
    has_entities: bool = False
    has_relationships: bool = False
    is_factual: bool = False
    is_complex: bool = False
    requires_recency: bool = False
    keyword_density: float = 0.0
    estimated_specificity: float = 0.5


class StrategySelection(BaseModel):
    """Selected strategy with parameters."""
    primary_strategy: RetrievalStrategy
    fallback_strategy: Optional[RetrievalStrategy] = None
    confidence: float = 0.5
    parameters: dict = Field(default_factory=dict)
    reasoning: str = ""


class StrategySelector:
    """
    Selects optimal retrieval strategy based on query characteristics.
    
    Usage:
        selector = StrategySelector()
        
        selection = await selector.select(
            "What is the relationship between tomato prices and monsoon?"
        )
        
        print(selection.primary_strategy)  # GRAPH
    """
    
    # Strategy selection weights
    STRATEGY_INDICATORS = {
        RetrievalStrategy.GRAPH: {
            "keywords": ["relationship", "how does", "affect", "impact", "between", "connect"],
            "requires_entities": True,
        },
        RetrievalStrategy.SPARSE: {
            "keywords": ["definition", "what is", "meaning of", "exact"],
            "high_keyword_density": True,
        },
        RetrievalStrategy.MULTI_HOP: {
            "keywords": ["compare", "versus", "difference", "all", "comprehensive"],
            "is_complex": True,
        },
        RetrievalStrategy.INSTRUCTED: {
            "keywords": ["specific", "particular", "only", "exactly", "must have"],
            "requires_precision": True,
        },
    }
    
    def __init__(self, llm=None):
        """
        Initialize strategy selector.
        
        Args:
            llm: Optional LLM for advanced analysis
        """
        self.llm = llm
    
    async def select(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> StrategySelection:
        """
        Select optimal retrieval strategy.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            StrategySelection with recommendations
        """
        # Analyze query
        analysis = self._analyze_query(query)
        
        # Score each strategy
        scores = {}
        
        for strategy in RetrievalStrategy:
            score = self._score_strategy(strategy, analysis)
            scores[strategy] = score
        
        # Select best strategy
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]
        
        # Determine fallback
        fallback = None
        if best_score < 0.7:
            # Use hybrid as fallback if uncertain
            fallback = RetrievalStrategy.HYBRID
        
        # Build parameters
        params = self._get_strategy_params(best_strategy, analysis)
        
        selection = StrategySelection(
            primary_strategy=best_strategy,
            fallback_strategy=fallback,
            confidence=best_score,
            parameters=params,
            reasoning=self._get_reasoning(best_strategy, analysis),
        )
        
        logger.info(
            "Strategy selected: {} (conf: {:.2f}) for: {}",
            best_strategy.value, best_score, query[:40]
        )
        
        return selection
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query characteristics."""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Check for entities (capitalized words, quoted terms)
        has_entities = any(
            word[0].isupper() for word in query.split() if word
        ) or '"' in query
        
        # Check for relationship indicators
        relationship_words = ["between", "affect", "impact", "cause", "lead to", "relationship"]
        has_relationships = any(w in query_lower for w in relationship_words)
        
        # Check if factual
        factual_starters = ["what is", "define", "when did", "who is"]
        is_factual = any(query_lower.startswith(s) for s in factual_starters)
        
        # Check complexity
        is_complex = len(words) > 10 or "and" in query_lower or "," in query
        
        # Check recency
        recency_words = ["current", "today", "now", "latest", "recent", "2024", "2025", "2026"]
        requires_recency = any(w in query_lower for w in recency_words)
        
        # Keyword density (non-stopword ratio)
        stopwords = {"the", "a", "an", "is", "are", "what", "how", "in", "for", "to", "of"}
        keywords = [w for w in words if w not in stopwords]
        keyword_density = len(keywords) / max(len(words), 1)
        
        return QueryAnalysis(
            query=query,
            has_entities=has_entities,
            has_relationships=has_relationships,
            is_factual=is_factual,
            is_complex=is_complex,
            requires_recency=requires_recency,
            keyword_density=keyword_density,
        )
    
    def _score_strategy(
        self,
        strategy: RetrievalStrategy,
        analysis: QueryAnalysis,
    ) -> float:
        """Score a strategy for the query."""
        score = 0.5  # Base score
        
        if strategy == RetrievalStrategy.DENSE:
            # Good general purpose
            score = 0.6
            if not analysis.is_factual:
                score += 0.1
        
        elif strategy == RetrievalStrategy.SPARSE:
            # Good for exact keyword matching
            score = 0.4
            if analysis.is_factual:
                score += 0.2
            if analysis.keyword_density > 0.7:
                score += 0.2
        
        elif strategy == RetrievalStrategy.HYBRID:
            # Good balance
            score = 0.65
            # Hybrid is good when uncertain
        
        elif strategy == RetrievalStrategy.GRAPH:
            # Good for relationships
            score = 0.3
            if analysis.has_relationships:
                score += 0.4
            if analysis.has_entities:
                score += 0.2
        
        elif strategy == RetrievalStrategy.MULTI_HOP:
            # Good for complex queries
            score = 0.3
            if analysis.is_complex:
                score += 0.3
        
        elif strategy == RetrievalStrategy.INSTRUCTED:
            # Good for precision
            score = 0.4
            # Boost if query has specific requirements
            if '"' in analysis.query or "specific" in analysis.query.lower():
                score += 0.3
        
        return min(1.0, score)
    
    def _get_strategy_params(
        self,
        strategy: RetrievalStrategy,
        analysis: QueryAnalysis,
    ) -> dict:
        """Get parameters for selected strategy."""
        params = {}
        
        if strategy == RetrievalStrategy.HYBRID:
            # Balance between dense and sparse
            params["dense_weight"] = 0.6
            params["sparse_weight"] = 0.4
            params["fusion"] = "rrf"
        
        elif strategy == RetrievalStrategy.GRAPH:
            params["max_hops"] = 2
            params["relationship_types"] = ["RELATED_TO", "AFFECTS", "CONTAINS"]
        
        elif strategy == RetrievalStrategy.MULTI_HOP:
            params["max_iterations"] = 3
            params["refinement_threshold"] = 0.7
        
        if analysis.requires_recency:
            params["time_filter"] = "recent"
        
        return params
    
    def _get_reasoning(
        self,
        strategy: RetrievalStrategy,
        analysis: QueryAnalysis,
    ) -> str:
        """Generate reasoning for strategy selection."""
        reasons = []
        
        if strategy == RetrievalStrategy.GRAPH:
            reasons.append("Query involves relationships between entities")
        elif strategy == RetrievalStrategy.SPARSE:
            reasons.append("Query is factual with specific keywords")
        elif strategy == RetrievalStrategy.HYBRID:
            reasons.append("Query benefits from both semantic and keyword matching")
        elif strategy == RetrievalStrategy.MULTI_HOP:
            reasons.append("Complex query requires iterative refinement")
        elif strategy == RetrievalStrategy.INSTRUCTED:
            reasons.append("Query has specific precision requirements")
        else:
            reasons.append("General semantic similarity is appropriate")
        
        if analysis.requires_recency:
            reasons.append("Recency filter applied")
        
        return "; ".join(reasons)
