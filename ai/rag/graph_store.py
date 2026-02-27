"""
Neo4j Graph Store
=================
Graph database wrapper for knowledge graph operations.

Features:
- Connection management
- cypher query execution
- Knowledge graph construction
- Graph traversal and retrieval
"""

from typing import Any, Dict, List, Optional
from loguru import logger
from neo4j import GraphDatabase, Driver

class GraphStore:
    """
    Neo4j Graph Store for CropFresh AI.
    
    Manages knowledge graph connections and operations.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "cropfresh123",
        database: str = "neo4j",
    ):
        """
        Initialize Graph Store.
        
        Args:
            uri: Neo4j URI
            username: Auth username
            password: Auth password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        
    @property
    def driver(self) -> Driver:
        """Lazy load Neo4j driver."""
        if self._driver is None:
            self._connect()
        return self._driver
        
    def _connect(self):
        """Connect to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Verify connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
            
    def close(self):
        """Close connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute Cypher query.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            List of records as dictionaries
        """
        if params is None:
            params = {}
            
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {query} Error: {e}")
            raise

    def add_entity(self, label: str, properties: Dict[str, Any]) -> str:
        """
        Add a node to the graph.
        
        Args:
            label: Node label (e.g., Crop, Market, Disease)
            properties: Node properties
            
        Returns:
            Properties of created node
        """
        query = f"""
        CREATE (n:{label} $props)
        RETURN n
        """
        result = self.query(query, {"props": properties})
        return str(result[0]['n'])

    def add_relation(
        self, 
        source_label: str, 
        source_props: Dict[str, Any],
        target_label: str,
        target_props: Dict[str, Any],
        relation_type: str,
        rel_props: Dict[str, Any] = None
    ):
        """
        Create a relationship between two nodes.
        Nodes are identified by matching ALL properties in source_props/target_props.
        
        Args:
            source_label: Label of source node
            source_props: Properties to match source node
            target_label: Label of target node
            target_props: Properties to match target node
            relation_type: Type of relationship
            rel_props: Properties for the relationship
        """
        if rel_props is None:
            rel_props = {}
            
        # Build dynamic match clauses (simplified)
        # Note: In production, passing dicts as params is safer/cleaner than string manipulation
        query = f"""
        MATCH (a:{source_label}), (b:{target_label})
        WHERE a += $source_props AND b += $target_props
        MERGE (a)-[r:{relation_type}]->(b)
        SET r += $rel_props
        RETURN r
        """
        
        params = {
            "source_props": source_props,
            "target_props": target_props,
            "rel_props": rel_props
        }
        
        return self.query(query, params)

    def get_context(self, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve context subgraph for an entity.
        
        Args:
            entity_name: Name property of the entity
            depth: Traversal depth
            
        Returns:
            List of paths/relations
        """
        query = f"""
        MATCH p=(n {{name: $name}})-[*1..{depth}]-(m)
        RETURN p
        """
        return self.query(query, {"name": entity_name})
