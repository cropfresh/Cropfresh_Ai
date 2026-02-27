"""
Graph RAG Retriever
===================
Neo4j-based retrieval for relationship-aware context augmentation.

Features:
- Entity extraction from queries
- Graph traversal for relationships (farmers, crops, buyers)
- Context templates for LLM augmentation
- Combines with vector search results
"""

import re
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


class GraphContext(BaseModel):
    """Graph context extracted from Neo4j."""
    
    entities: list[str] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    farmers: list[dict[str, Any]] = Field(default_factory=list)
    buyers: list[dict[str, Any]] = Field(default_factory=list)
    crops: list[dict[str, Any]] = Field(default_factory=list)
    supply_chain: Optional[dict[str, Any]] = None
    context_text: str = ""
    query_time_ms: float = 0.0


class EntityExtractor:
    """
    Extract agricultural entities from user queries.
    
    Identifies:
    - Crop names (tomato, onion, rice, etc.)
    - Locations (districts, states)
    - Entity types (farmer, buyer, supplier)
    """
    
    # Common crops in Karnataka/India
    CROPS = {
        "tomato", "onion", "potato", "rice", "wheat", "maize", "corn",
        "mango", "banana", "grape", "pomegranate", "coconut",
        "sugarcane", "cotton", "chilli", "pepper", "turmeric",
        "groundnut", "sunflower", "ragi", "jowar", "bajra",
        "cabbage", "cauliflower", "carrot", "beans", "brinjal",
    }
    
    # Karnataka districts
    DISTRICTS = {
        "bangalore", "bengaluru", "mysore", "mysuru", "kolar", "tumkur",
        "bellary", "ballari", "belgaum", "belagavi", "hubli", "dharwad",
        "mangalore", "mangaluru", "shimoga", "shivamogga", "davangere",
        "hassan", "mandya", "raichur", "bidar", "gulbarga", "kalaburagi",
        "chitradurga", "chikmagalur", "udupi", "kodagu", "coorg",
    }
    
    # Entity types
    ENTITY_KEYWORDS = {
        "farmer": ["farmer", "farmers", "grower", "growers", "cultivator"],
        "buyer": ["buyer", "buyers", "trader", "traders", "merchant", "retailer"],
        "supplier": ["supplier", "suppliers", "vendor", "vendors"],
    }
    
    def extract(self, query: str) -> dict[str, list[str]]:
        """
        Extract entities from query.
        
        Args:
            query: User query text
            
        Returns:
            Dict with entity types as keys and found entities as values
        """
        query_lower = query.lower()
        words = set(re.sub(r'[^\w\s]', ' ', query_lower).split())
        
        entities = {
            "crops": [],
            "districts": [],
            "entity_types": [],
        }
        
        # Find crops
        for crop in self.CROPS:
            if crop in words or crop in query_lower:
                entities["crops"].append(crop.title())
        
        # Find districts
        for district in self.DISTRICTS:
            if district in words or district in query_lower:
                entities["districts"].append(district.title())
        
        # Find entity types
        for entity_type, keywords in self.ENTITY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                entities["entity_types"].append(entity_type)
        
        return entities


class GraphRetriever:
    """
    Graph-based retriever using Neo4j for relationship context.
    
    Retrieves:
    - Farmers growing specific crops
    - Buyers interested in crops
    - Supply chain relationships
    - Location-based connections
    
    Usage:
        retriever = GraphRetriever(neo4j_client)
        context = await retriever.retrieve("farmers growing tomatoes in Kolar")
    """
    
    def __init__(self, neo4j_client=None):
        """
        Initialize graph retriever.
        
        Args:
            neo4j_client: Neo4jClient instance (lazy loaded if None)
        """
        self._client = neo4j_client
        self.entity_extractor = EntityExtractor()
        self._initialized = False
    
    @property
    def client(self):
        """Lazy load Neo4j client."""
        if self._client is None:
            try:
                from src.db.neo4j_client import get_neo4j
                self._client = get_neo4j()
                self._initialized = True
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
                return None
        return self._client
    
    async def retrieve(self, query: str) -> GraphContext:
        """
        Retrieve graph context for a query.
        
        Args:
            query: User query
            
        Returns:
            GraphContext with extracted relationships
        """
        import time
        start_time = time.time()
        
        # Extract entities
        entities = self.entity_extractor.extract(query)
        
        context = GraphContext(
            entities=entities.get("crops", []) + entities.get("districts", []),
        )
        
        if not self.client:
            context.context_text = "Graph database not available."
            return context
        
        try:
            # Query based on extracted entities
            crops = entities.get("crops", [])
            districts = entities.get("districts", [])
            entity_types = entities.get("entity_types", [])
            
            # Get farmers for crops
            if crops and ("farmer" in entity_types or not entity_types):
                for crop in crops[:3]:  # Limit to avoid too many queries
                    farmers = self.client.get_farmers_for_crop(
                        crop_name=crop,
                        district=districts[0] if districts else None,
                        limit=5,
                    )
                    context.farmers.extend(farmers)
            
            # Get supply chain for crops
            if crops:
                for crop in crops[:2]:
                    supply_info = self.client.get_supply_chain(crop)
                    if supply_info:
                        context.supply_chain = supply_info
                        break
            
            # Build context text
            context.context_text = self._build_context_text(context, crops, districts)
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            context.context_text = f"Graph query failed: {str(e)}"
        
        context.query_time_ms = (time.time() - start_time) * 1000
        return context
    
    def _build_context_text(
        self,
        context: GraphContext,
        crops: list[str],
        districts: list[str],
    ) -> str:
        """Build human-readable context text from graph data."""
        parts = []
        
        # Farmer information
        if context.farmers:
            farmer_info = []
            for f in context.farmers[:5]:
                name = f.get("name", "Unknown")
                district = f.get("district", "Unknown")
                farmer_crops = f.get("crops", [])
                if farmer_crops:
                    farmer_info.append(f"- {name} ({district}): grows {', '.join(farmer_crops)}")
                else:
                    farmer_info.append(f"- {name} ({district})")
            
            if farmer_info:
                crop_str = crops[0] if crops else "crops"
                parts.append(f"**Farmers growing {crop_str}:**\n" + "\n".join(farmer_info))
        
        # Supply chain
        if context.supply_chain:
            sc = context.supply_chain
            parts.append(
                f"**Supply Chain for {crops[0] if crops else 'crop'}:**\n"
                f"- Farmers: {sc.get('farmer_count', 0)}\n"
                f"- Buyers: {sc.get('buyer_count', 0)}\n"
                f"- Total transactions: {sc.get('transaction_count', 0)}"
            )
        
        if not parts:
            return ""
        
        return "\n\n".join(parts)
    
    async def get_recommendations(
        self,
        buyer_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get farmer recommendations for a buyer.
        
        Args:
            buyer_id: Buyer ID
            limit: Max recommendations
            
        Returns:
            List of recommended farmers
        """
        if not self.client:
            return []
        
        try:
            return self.client.get_buyer_recommendations(buyer_id, limit)
        except Exception as e:
            logger.error(f"Recommendation query failed: {e}")
            return []


class GraphAugmentedRetriever:
    """
    Combines vector retrieval with graph context for richer responses.
    
    Pipeline:
    1. Extract entities from query
    2. Retrieve from vector store (existing RAG)
    3. Retrieve from graph (relationships)
    4. Merge context for LLM
    """
    
    def __init__(self, knowledge_base, neo4j_client=None):
        """
        Initialize graph-augmented retriever.
        
        Args:
            knowledge_base: Vector knowledge base
            neo4j_client: Neo4j client (optional)
        """
        self.knowledge_base = knowledge_base
        self.graph_retriever = GraphRetriever(neo4j_client)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_graph: bool = True,
    ) -> dict[str, Any]:
        """
        Retrieve from both vector and graph sources.
        
        Args:
            query: User query
            top_k: Number of vector results
            include_graph: Whether to include graph context
            
        Returns:
            Combined retrieval result
        """
        result = {
            "query": query,
            "documents": [],
            "graph_context": None,
            "combined_context": "",
        }
        
        # Vector retrieval
        try:
            vector_result = await self.knowledge_base.search(query, top_k=top_k)
            result["documents"] = vector_result.documents
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
        
        # Graph retrieval
        if include_graph:
            try:
                graph_context = await self.graph_retriever.retrieve(query)
                result["graph_context"] = graph_context
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")
        
        # Combine context
        result["combined_context"] = self._combine_context(result)
        
        return result
    
    def _combine_context(self, result: dict[str, Any]) -> str:
        """Combine vector and graph context into a single text."""
        parts = []
        
        # Document context
        if result["documents"]:
            doc_texts = [doc.text for doc in result["documents"][:3]]
            parts.append("**Retrieved Knowledge:**\n" + "\n\n".join(doc_texts))
        
        # Graph context
        if result["graph_context"] and result["graph_context"].context_text:
            parts.append(result["graph_context"].context_text)
        
        return "\n\n---\n\n".join(parts)


# Graph context prompt template
GRAPH_CONTEXT_PROMPT = """You have access to the following relationship data from our agricultural network:

{graph_context}

Use this information to provide specific, actionable answers about farmers, crops, and buyers in the CropFresh network."""
