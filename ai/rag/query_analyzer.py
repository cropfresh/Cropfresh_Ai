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


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 05: Adaptive Query Router — 8-Strategy Routing System (ADR-008)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
from typing import Optional


class RetrievalRoute(str, Enum):
    """
    8 retrieval routing strategies for the Adaptive Query Router.

    Each strategy maps to a different tool set and cost tier.
    Note: Named 'RetrievalRoute' (not 'RetrievalStrategy') to avoid collision
    with the existing RetrievalStrategy enum in enhanced_retriever.py.
    """
    DIRECT_LLM       = "direct_llm"        # No retrieval — greetings, definitions
    VECTOR_ONLY      = "vector_only"        # KB dense + sparse search
    GRAPH_TRAVERSAL  = "graph_traversal"    # Neo4j entity/relational query
    LIVE_PRICE_API   = "live_price_api"     # eNAM real-time mandi prices
    WEATHER_API      = "weather_api"        # IMD agro-met forecast
    BROWSER_SCRAPE   = "browser_scrape"     # Live web news/scheme scrape
    MULTIMODAL       = "multimodal"         # Vision + RAG (crop photo)
    FULL_AGENTIC     = "full_agentic"       # Full orchestrator: all tools


# Cost reference per strategy (INR, approximate at current Groq pricing)
ROUTE_COST_MAP: dict[RetrievalRoute, float] = {
    RetrievalRoute.DIRECT_LLM:       0.03,
    RetrievalRoute.VECTOR_ONLY:      0.12,
    RetrievalRoute.GRAPH_TRAVERSAL:  0.15,
    RetrievalRoute.LIVE_PRICE_API:   0.05,
    RetrievalRoute.WEATHER_API:      0.05,
    RetrievalRoute.BROWSER_SCRAPE:   0.25,
    RetrievalRoute.MULTIMODAL:       0.35,
    RetrievalRoute.FULL_AGENTIC:     0.55,
}


class RoutingDecision(BaseModel):
    """
    Output of the AdaptiveQueryRouter for a single query.

    Contains both the routing decision and observability signals
    for LangSmith logging and cost monitoring.
    """
    strategy: RetrievalRoute
    confidence: float = Field(ge=0.0, le=1.0, description="Router confidence 0–1")
    reason: str = Field(description="Human-readable routing reason for logging")
    estimated_cost_inr: float = Field(description="Estimated INR cost for this route")
    entities: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Extracted entities: crops, locations, schemes etc."
    )
    requires_live_data: bool = False
    requires_image: bool = False
    pre_filter_matched: bool = False  # True if caught by fast rule-based filter


# LLM prompt for query classification — used by Groq 8B router
ADAPTIVE_ROUTER_PROMPT = """You are a routing classifier for CropFresh, an AI assistant for Indian farmers.

Given a user query, classify it into EXACTLY ONE of these retrieval routes:
1. "direct_llm"       — Greetings, who-am-I, platform FAQ, yes/no, simple definitions. No retrieval needed.
2. "vector_only"      — Farming practice, crop cultivation, pest/disease, organic methods, soil science, schemes (general info).
3. "graph_traversal"  — Queries about specific farmers, buyers, supply chain relationships, "which farmers in X district grow Y".
4. "live_price_api"   — Queries about current/today/aaj mandi prices, commodity rates, market price. Time-sensitive.
5. "weather_api"      — Weather forecast, rainfall, temperature, monsoon, barish, agro-met advisory.
6. "browser_scrape"   — Latest scheme updates, newly launched government programmes, novel disease alerts, pesticide bans. Must be live.
7. "multimodal"       — User attached a crop photo or mentions analysing an image.
8. "full_agentic"     — Complex multi-factor decisions: "should I sell now or store?", combining price + weather + farming advice.

Respond ONLY with valid JSON, no markdown:
{
  "route": "<one of the 8 values above>",
  "confidence": <0.0-1.0>,
  "reason": "<one sentence explaining the choice>",
  "entities": {
    "crops": ["tomato"],
    "locations": ["Hubli"],
    "schemes": []
  },
  "requires_live_data": <true|false>,
  "requires_image": <true|false>
}"""


class AdaptiveQueryRouter:
    """
    8-Strategy Adaptive Query Router (ADR-008).

    Entry point for every query in the CropFresh AI pipeline.
    Routes queries to the most cost-efficient retrieval strategy.

    Two-stage routing:
    1. Rule-based pre-filter (0ms, free) — catches obvious cases
    2. Groq Llama-3.1-8B LLM classifier (~80ms, ~₹0.001) — for ambiguous queries

    Feature flag: Set USE_ADAPTIVE_ROUTER=true in .env to enable (default: off).

    Expected cost reduction: ₹0.44 → ₹0.22 avg/query (–52%)

    Usage:
        router = AdaptiveQueryRouter()
        decision = await router.route("tomato price today in Hubli?")
        print(decision.strategy)   # RetrievalRoute.LIVE_PRICE_API
        print(decision.estimated_cost_inr)  # 0.05
    """

    # ── Rule-based pre-filter sets (exact & substring matching) ──────────────

    GREETING_TOKENS = {
        "hello", "hi", "hey", "namaste", "namaskar", "vanakkam", "sat sri akal",
        "thanks", "thank you", "shukriya", "dhanyavad", "bye", "goodbye",
        "who are you", "what is cropfresh", "what can you do",
        "help me", "help", "how are you",
    }

    PRICE_SUBSTRINGS = {
        "price", "rate", "mandi rate", "today rate", "current price",
        "what is the price", "how much for", "selling price", "bhaav",
        "market rate", "sabzi mandi", "fasal ka bhaav", "aaj ka bhav",
        "mandi bhav",
    }

    PRICE_TIME_TOKENS = {
        "today", "now", "current", "aaj", "abhi", "is waqt", "live",
        "real-time", "realtime", "latest price",
    }

    WEATHER_SUBSTRINGS = {
        "weather", "rain", "rainfall", "forecast", "temperature", "humidity",
        "monsoon", "barish", "garmi", "sardi", "climate", "cyclone",
        "warm", "cold", "dry spell", "imd", "mausam",
    }

    IMAGE_TRIGGERS = {
        "[image", "photo of", "image of", "picture of", "pic of",
        "look at this", "attached image", "crop photo",
    }

    def __init__(self, llm=None):
        """
        Initialize the adaptive router.

        Args:
            llm: LLM provider instance (uses Groq Llama-3.1-8B-Instant if provided).
                 Falls back to rule-based only if None.
        """
        self.llm = llm
        self._enabled = os.getenv("USE_ADAPTIVE_ROUTER", "false").lower() == "true"

        if not self._enabled:
            logger.info(
                "AdaptiveQueryRouter: USE_ADAPTIVE_ROUTER=false — "
                "router instantiated but route() returns VECTOR_ONLY by default. "
                "Set USE_ADAPTIVE_ROUTER=true to enable."
            )
        else:
            logger.info(
                f"AdaptiveQueryRouter: ENABLED | "
                f"llm={'LLM' if llm else 'rule-based only'}"
            )

    async def route(self, query: str, has_image: bool = False) -> RoutingDecision:
        """
        Route a query to the optimal retrieval strategy.

        Args:
            query: User query text
            has_image: True if the user attached an image

        Returns:
            RoutingDecision with strategy, confidence, cost signal, and entities
        """
        if not self._enabled:
            # Feature flag OFF — safe default: use existing vector pipeline
            return RoutingDecision(
                strategy=RetrievalRoute.VECTOR_ONLY,
                confidence=1.0,
                reason="Adaptive router disabled (USE_ADAPTIVE_ROUTER=false)",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.VECTOR_ONLY],
                pre_filter_matched=True,
            )

        # Stage 1: Rule-based pre-filter (0ms, free)
        pre_result = self._prefilter(query, has_image)
        if pre_result is not None:
            logger.debug(
                f"AdaptiveRouter pre-filter match | "
                f"strategy={pre_result.strategy} | query={query[:60]}..."
            )
            return pre_result

        # Stage 2: LLM classifier (Groq 8B-instant, ~80ms)
        if self.llm is not None:
            return await self._llm_classify(query, has_image)

        # Stage 3: Rule-based fallback if no LLM available
        return self._rule_classify(query, has_image)

    def _prefilter(self, query: str, has_image: bool) -> Optional[RoutingDecision]:
        """
        Fast rule-based routing for unambiguous cases.

        Returns a RoutingDecision for obvious queries, None if ambiguous.
        Average latency: < 1ms (pure Python string operations).
        """
        q = query.lower().strip()

        # Image attached → always multimodal
        if has_image or any(t in q for t in self.IMAGE_TRIGGERS):
            return RoutingDecision(
                strategy=RetrievalRoute.MULTIMODAL,
                confidence=0.97,
                reason="Image attachment detected",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.MULTIMODAL],
                requires_image=True,
                pre_filter_matched=True,
            )

        # Greeting or platform FAQ
        q_words = set(q.split())
        if any(token in q for token in self.GREETING_TOKENS) or q_words & self.GREETING_TOKENS:
            return RoutingDecision(
                strategy=RetrievalRoute.DIRECT_LLM,
                confidence=0.98,
                reason="Greeting or FAQ detected by rule",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.DIRECT_LLM],
                pre_filter_matched=True,
            )

        # Live price query: price keyword + time token
        has_price = any(p in q for p in self.PRICE_SUBSTRINGS)
        has_time = any(t in q for t in self.PRICE_TIME_TOKENS)
        if has_price and has_time:
            return RoutingDecision(
                strategy=RetrievalRoute.LIVE_PRICE_API,
                confidence=0.92,
                reason="Live price query: price keyword + time indicator",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.LIVE_PRICE_API],
                requires_live_data=True,
                pre_filter_matched=True,
            )

        # Weather query
        if any(w in q for w in self.WEATHER_SUBSTRINGS):
            return RoutingDecision(
                strategy=RetrievalRoute.WEATHER_API,
                confidence=0.90,
                reason="Weather or forecast query detected",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.WEATHER_API],
                requires_live_data=True,
                pre_filter_matched=True,
            )

        return None  # Ambiguous — pass to LLM classifier

    async def _llm_classify(self, query: str, has_image: bool) -> RoutingDecision:
        """
        Classify query using Groq Llama-3.1-8B-Instant.

        ~80ms latency, ~₹0.001 per call.
        Falls back to rule-based if LLM fails.
        """
        try:
            from src.orchestrator.llm_provider import LLMMessage

            messages = [
                LLMMessage(role="system", content=ADAPTIVE_ROUTER_PROMPT),
                LLMMessage(
                    role="user",
                    content=f"Query: {query}\nHas image: {has_image}"
                ),
            ]

            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=200,
            )

            result = json.loads(response.content)

            route = RetrievalRoute(result.get("route", "vector_only"))
            confidence = float(result.get("confidence", 0.80))
            reason = result.get("reason", "LLM classification")
            entities = result.get("entities", {})
            requires_live_data = bool(result.get("requires_live_data", False))
            requires_image = has_image or bool(result.get("requires_image", False))

            logger.debug(
                f"AdaptiveRouter LLM classified | "
                f"route={route} | confidence={confidence:.2f} | query={query[:60]}..."
            )

            return RoutingDecision(
                strategy=route,
                confidence=confidence,
                reason=reason,
                estimated_cost_inr=ROUTE_COST_MAP[route],
                entities=entities,
                requires_live_data=requires_live_data,
                requires_image=requires_image,
                pre_filter_matched=False,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"AdaptiveRouter: LLM returned invalid JSON: {e}")
            return self._rule_classify(query, has_image)

        except Exception as e:
            logger.warning(f"AdaptiveRouter: LLM classify failed: {e}")
            return self._rule_classify(query, has_image)

    def _rule_classify(self, query: str, has_image: bool) -> RoutingDecision:
        """
        Rule-based fallback classification for when no LLM is available.

        More thorough than pre-filter but still heuristic-based.
        """
        q = query.lower()

        # Check for graph traversal patterns
        graph_patterns = [
            "which farmers", "who grows", "find farmers", "list buyers",
            "show me farmers", "supply chain", "how many farmers",
        ]
        if any(p in q for p in graph_patterns):
            return RoutingDecision(
                strategy=RetrievalRoute.GRAPH_TRAVERSAL,
                confidence=0.72,
                reason="Graph traversal pattern detected (rule fallback)",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.GRAPH_TRAVERSAL],
            )

        # Complex decision queries
        complex_patterns = [
            "should i sell", "is it better to", "compare", "vs ", "versus",
            "sell or store", "hold or sell", "when should i", "market trend",
        ]
        if any(p in q for p in complex_patterns):
            return RoutingDecision(
                strategy=RetrievalRoute.FULL_AGENTIC,
                confidence=0.70,
                reason="Complex decision query detected (rule fallback)",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.FULL_AGENTIC],
            )

        # Browser scrape patterns
        browser_patterns = [
            "latest scheme", "new scheme", "new policy", "recently launched",
            "just announced", "news", "update", "ban", "banned pesticide",
        ]
        if any(p in q for p in browser_patterns):
            return RoutingDecision(
                strategy=RetrievalRoute.BROWSER_SCRAPE,
                confidence=0.68,
                reason="Live news/scheme query detected (rule fallback)",
                estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.BROWSER_SCRAPE],
                requires_live_data=True,
            )

        # Default: vector search (safest fallback)
        return RoutingDecision(
            strategy=RetrievalRoute.VECTOR_ONLY,
            confidence=0.60,
            reason="Default vector search (no pattern matched, rule fallback)",
            estimated_cost_inr=ROUTE_COST_MAP[RetrievalRoute.VECTOR_ONLY],
        )

