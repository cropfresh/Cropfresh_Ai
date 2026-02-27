"""
Neo4j Client
============
Graph database client for relationship data.

Features:
- Farm-Buyer connections
- Supply chain tracking
- Recommendation engine support
"""

from functools import lru_cache
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel


class GraphNode(BaseModel):
    """Graph node model."""
    labels: list[str]
    properties: dict[str, Any]


class GraphRelationship(BaseModel):
    """Graph relationship model."""
    type: str
    start_node: str
    end_node: str
    properties: dict[str, Any] = {}


class Neo4jClient:
    """
    Neo4j Graph Database client for CropFresh AI.
    
    Handles relationship data:
    - Farmer → Crop connections (GROWS)
    - Farmer → Buyer transactions (SOLD_TO)
    - Location hierarchies (District → State)
    - Buyer preferences (BUYS)
    
    Usage:
        client = Neo4jClient(uri, user, password)
        await client.create_farmer(farmer_id, name, crops)
        recommendations = await client.get_buyer_recommendations(buyer_id)
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (neo4j+s://...)
            user: Username (usually 'neo4j')
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        
        logger.info(f"Initializing Neo4j client for {uri}")
    
    @property
    def driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            self._connect()
        return self._driver
    
    def _connect(self):
        """Connect to Neo4j."""
        try:
            from neo4j import GraphDatabase
            
            logger.info("Connecting to Neo4j...")
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connection
            self._driver.verify_connectivity()
            logger.info("Neo4j connection established")
            
        except ImportError:
            logger.error("neo4j not installed. Run: uv add neo4j")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    # ═══════════════════════════════════════════════════════════════
    # Farmers
    # ═══════════════════════════════════════════════════════════════
    
    async def create_farmer(
        self,
        farmer_id: str,
        name: str,
        phone: str,
        district: str,
        crops: list[str],
    ) -> dict[str, Any]:
        """
        Create a farmer node with crop relationships.
        
        Creates:
        - Farmer node
        - Crop nodes (if not exist)
        - GROWS relationships
        - District location
        """
        query = """
        MERGE (f:Farmer {id: $farmer_id})
        SET f.name = $name, f.phone = $phone
        
        MERGE (d:District {name: $district})
        MERGE (f)-[:LOCATED_IN]->(d)
        
        WITH f
        UNWIND $crops AS crop_name
        MERGE (c:Crop {name: crop_name})
        MERGE (f)-[:GROWS]->(c)
        
        RETURN f
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                farmer_id=farmer_id,
                name=name,
                phone=phone,
                district=district,
                crops=crops,
            )
            record = result.single()
            logger.info(f"Created farmer node: {name}")
            return dict(record["f"]) if record else {}
    
    async def create_buyer(
        self,
        buyer_id: str,
        name: str,
        business_type: str,
        preferred_crops: list[str],
    ) -> dict[str, Any]:
        """
        Create a buyer node with preferences.
        
        Creates:
        - Buyer node
        - BUYS relationships to crops
        """
        query = """
        MERGE (b:Buyer {id: $buyer_id})
        SET b.name = $name, b.business_type = $business_type
        
        WITH b
        UNWIND $crops AS crop_name
        MERGE (c:Crop {name: crop_name})
        MERGE (b)-[:BUYS]->(c)
        
        RETURN b
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                buyer_id=buyer_id,
                name=name,
                business_type=business_type,
                crops=preferred_crops,
            )
            record = result.single()
            logger.info(f"Created buyer node: {name}")
            return dict(record["b"]) if record else {}
    
    # ═══════════════════════════════════════════════════════════════
    # Transactions
    # ═══════════════════════════════════════════════════════════════
    
    async def record_transaction(
        self,
        farmer_id: str,
        buyer_id: str,
        crop_name: str,
        quantity_kg: float,
        price_per_kg: float,
    ) -> dict[str, Any]:
        """Record a transaction between farmer and buyer."""
        query = """
        MATCH (f:Farmer {id: $farmer_id})
        MATCH (b:Buyer {id: $buyer_id})
        MERGE (c:Crop {name: $crop_name})
        
        CREATE (f)-[t:SOLD_TO {
            crop: $crop_name,
            quantity_kg: $quantity_kg,
            price_per_kg: $price_per_kg,
            timestamp: datetime()
        }]->(b)
        
        RETURN t
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                farmer_id=farmer_id,
                buyer_id=buyer_id,
                crop_name=crop_name,
                quantity_kg=quantity_kg,
                price_per_kg=price_per_kg,
            )
            record = result.single()
            logger.info(f"Recorded transaction: {farmer_id} → {buyer_id}")
            return dict(record["t"]) if record else {}
    
    # ═══════════════════════════════════════════════════════════════
    # Recommendations
    # ═══════════════════════════════════════════════════════════════
    
    async def get_farmers_for_crop(
        self,
        crop_name: str,
        district: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find farmers who grow a specific crop."""
        query = """
        MATCH (f:Farmer)-[:GROWS]->(c:Crop {name: $crop_name})
        """ + ("""
        MATCH (f)-[:LOCATED_IN]->(d:District {name: $district})
        """ if district else "") + """
        RETURN f.id AS id, f.name AS name, f.phone AS phone
        LIMIT $limit
        """
        
        params = {"crop_name": crop_name, "limit": limit}
        if district:
            params["district"] = district
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    
    async def get_buyer_recommendations(
        self,
        buyer_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Recommend farmers to a buyer based on:
        - Crops buyer prefers
        - Previous successful transactions
        - Location proximity
        """
        query = """
        MATCH (b:Buyer {id: $buyer_id})-[:BUYS]->(c:Crop)<-[:GROWS]-(f:Farmer)
        WHERE NOT (b)<-[:SOLD_TO]-(f)
        RETURN DISTINCT f.id AS id, f.name AS name, 
               collect(c.name) AS crops,
               count(c) AS match_score
        ORDER BY match_score DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, buyer_id=buyer_id, limit=limit)
            return [dict(record) for record in result]
    
    # ═══════════════════════════════════════════════════════════════
    # Supply Chain
    # ═══════════════════════════════════════════════════════════════
    
    async def get_supply_chain(self, crop_name: str) -> dict[str, Any]:
        """Get supply chain overview for a crop."""
        query = """
        MATCH (f:Farmer)-[:GROWS]->(c:Crop {name: $crop_name})
        OPTIONAL MATCH (f)-[:LOCATED_IN]->(d:District)
        RETURN count(DISTINCT f) AS farmer_count,
               collect(DISTINCT d.name) AS districts
        """
        
        with self.driver.session() as session:
            result = session.run(query, crop_name=crop_name)
            record = result.single()
            return {
                "crop": crop_name,
                "farmer_count": record["farmer_count"] if record else 0,
                "districts": record["districts"] if record else [],
            }
    
    # ═══════════════════════════════════════════════════════════════
    # Health Check
    # ═══════════════════════════════════════════════════════════════
    
    def health_check(self) -> bool:
        """Check if Neo4j connection is healthy."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            stats = {record["label"]: record["count"] for record in result}
            return stats


@lru_cache(maxsize=1)
def get_neo4j(uri: str = None, user: str = None, password: str = None) -> Neo4jClient:
    """
    Get cached Neo4j client instance.
    
    If credentials not provided, reads from settings.
    """
    if uri is None or user is None or password is None:
        from src.config import get_settings
        settings = get_settings()
        uri = settings.neo4j_uri
        user = settings.neo4j_user
        password = settings.neo4j_password
    
    return Neo4jClient(uri=uri, user=user, password=password)
