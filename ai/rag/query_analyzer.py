"""
Query Analyzer
==============
Adaptive RAG query classification and routing.

Analyzes incoming queries to determine the best retrieval strategy:
- VECTOR_SEARCH: Knowledge base lookup
- WEB_SEARCH: Real-time data (prices, weather)
- DECOMPOSE: Complex multi-part queries
- DIRECT: Simple factual, no retrieval needed
"""

from enum import Enum
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries for routing."""
    
    VECTOR_SEARCH = "vector"      # Use knowledge base
    WEB_SEARCH = "web"            # Real-time data needed
    DECOMPOSE = "decompose"       # Break into sub-queries
    DIRECT = "direct"             # Answer directly, no retrieval


class QueryCategory(str, Enum):
    """Domain categories for targeted retrieval."""
    
    AGRONOMY = "agronomy"         # Crop guides, farming practices
    MARKET = "market"             # Prices, mandis, trading
    PLATFORM = "platform"         # CropFresh features, FAQs
    REGULATORY = "regulatory"     # APMC rules, certifications
    GENERAL = "general"           # General agriculture


class QueryAnalysis(BaseModel):
    """Result of query analysis."""
    
    original_query: str
    query_type: QueryType
    category: QueryCategory
    sub_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.8
    
    # Extracted entities
    crops: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    time_sensitive: bool = False


# System prompt for query classification
QUERY_ANALYZER_PROMPT = """You are a query analyzer for CropFresh AI, an agricultural marketplace platform.

Analyze the user query and classify it into one of these types:
1. **vector** - Questions about farming practices, crop cultivation, pest management, platform features
2. **web** - Questions requiring real-time data: current prices, weather, news
3. **decompose** - Complex questions with multiple parts that need to be broken down
4. **direct** - Simple greetings, thanks, or questions you can answer directly

Also identify:
- Category: agronomy, market, platform, regulatory, or general
- Any crop names mentioned
- Any location/state/district mentioned
- Whether the query is time-sensitive (needs current data)

Respond in JSON format:
{
    "query_type": "vector|web|decompose|direct",
    "category": "agronomy|market|platform|regulatory|general",
    "sub_queries": ["sub-query 1", "sub-query 2"],  // Only if type is "decompose"
    "reasoning": "Brief explanation of classification",
    "crops": ["tomato", "potato"],  // Extracted crop names
    "locations": ["Karnataka", "Kolar"],  // Extracted locations
    "time_sensitive": false  // true if needs current data
}"""


class QueryAnalyzer:
    """
    Adaptive RAG Query Analyzer.
    
    Uses LLM to classify queries and determine optimal retrieval strategy.
    
    Usage:
        analyzer = QueryAnalyzer(llm=llm_provider)
        result = await analyzer.analyze("How to grow tomatoes in Karnataka?")
        print(result.query_type)  # QueryType.VECTOR_SEARCH
    """
    
    def __init__(self, llm=None):
        """
        Initialize query analyzer.
        
        Args:
            llm: LLM provider for classification
        """
        self.llm = llm
    
    async def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and determine routing.
        
        Args:
            query: User query text
            
        Returns:
            QueryAnalysis with classification and extracted entities
        """
        if self.llm is None:
            # Fallback to rule-based classification
            return self._analyze_rule_based(query)
        
        return await self._analyze_with_llm(query)
    
    async def _analyze_with_llm(self, query: str) -> QueryAnalysis:
        """Analyze query using LLM."""
        import json
        from src.orchestrator.llm_provider import LLMMessage
        
        messages = [
            LLMMessage(role="system", content=QUERY_ANALYZER_PROMPT),
            LLMMessage(role="user", content=query),
        ]
        
        try:
            response = await self.llm.generate(
                messages,
                temperature=0.0,  # Deterministic classification
                max_tokens=500,
            )
            
            # Parse JSON response
            result = json.loads(response.content)
            
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType(result.get("query_type", "vector")),
                category=QueryCategory(result.get("category", "general")),
                sub_queries=result.get("sub_queries", []),
                reasoning=result.get("reasoning", ""),
                crops=result.get("crops", []),
                locations=result.get("locations", []),
                time_sensitive=result.get("time_sensitive", False),
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._analyze_rule_based(query)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._analyze_rule_based(query)
    
    def _analyze_rule_based(self, query: str) -> QueryAnalysis:
        """
        Simple rule-based classification as fallback.
        
        Uses keyword matching for basic routing.
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = QueryType.VECTOR_SEARCH
        time_sensitive = False
        
        # Web search indicators
        web_keywords = [
            "current price", "today price", "weather", "forecast",
            "latest", "news", "happening", "right now", "current market"
        ]
        if any(kw in query_lower for kw in web_keywords):
            query_type = QueryType.WEB_SEARCH
            time_sensitive = True
        
        # Complex query indicators
        complex_keywords = [
            "compare", "difference between", "step by step",
            "multiple", "and also", "as well as", "both"
        ]
        if any(kw in query_lower for kw in complex_keywords):
            query_type = QueryType.DECOMPOSE
        
        # Direct answer indicators
        direct_keywords = [
            "hello", "hi", "thanks", "thank you", "bye", "goodbye",
            "who are you", "what is cropfresh", "help"
        ]
        if any(kw in query_lower for kw in direct_keywords):
            query_type = QueryType.DIRECT
        
        # Detect category
        category = QueryCategory.GENERAL
        
        agronomy_keywords = [
            "grow", "plant", "cultivat", "harvest", "pest", "disease",
            "fertilizer", "soil", "seed", "irrigation", "organic"
        ]
        if any(kw in query_lower for kw in agronomy_keywords):
            category = QueryCategory.AGRONOMY
        
        market_keywords = [
            "price", "mandi", "sell", "buy", "market", "rate", "export"
        ]
        if any(kw in query_lower for kw in market_keywords):
            category = QueryCategory.MARKET
        
        platform_keywords = [
            "cropfresh", "app", "account", "register", "login", "feature"
        ]
        if any(kw in query_lower for kw in platform_keywords):
            category = QueryCategory.PLATFORM
        
        # Extract crops (simple approach)
        crop_list = [
            "tomato", "potato", "onion", "carrot", "cabbage", "cauliflower",
            "rice", "wheat", "maize", "cotton", "sugarcane", "mango", "banana",
            "apple", "grapes", "pomegranate", "chilli", "brinjal", "okra"
        ]
        crops = [c for c in crop_list if c in query_lower]
        
        # Extract locations (Karnataka focus)
        location_list = [
            "karnataka", "kolar", "bangalore", "bengaluru", "mysore", "mysuru",
            "hubli", "dharwad", "belgaum", "belagavi", "shimoga", "shivamogga",
            "india", "maharashtra", "tamil nadu", "kerala", "andhra pradesh"
        ]
        locations = [loc.title() for loc in location_list if loc in query_lower]
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            category=category,
            reasoning="Rule-based classification",
            crops=crops,
            locations=locations,
            time_sensitive=time_sensitive,
        )
