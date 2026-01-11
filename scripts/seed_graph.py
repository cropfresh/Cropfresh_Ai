"""
Graph Seeder
============
Seed Neo4j database with initial agricultural knowledge.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.graph_store import GraphStore
from loguru import logger

async def seed_graph():
    """Seed Graph Database from agronomy.json."""
    
    # Initialize store
    store = GraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="cropfresh123"
    )
    
    try:
        # Load knowledge data
        data_path = Path(__file__).parent.parent / "data" / "knowledge" / "agronomy.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.info(f"Loaded {len(data)} items from agronomy.json")
        
        # Clear existing data (optional, be careful in prod)
        store.query("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing graph data")
        
        count = 0
        for item in data:
            category = item.get("category", "General")
            source = item.get("source", "Unknown")
            text = item.get("text", "")
            
            # Simple extraction heuristic for demo
            # In real system, use LLM to extract entities
            
            # Create KnowledgeNode
            props = {
                "text": text,
                "source": source,
                "category": category,
                "type": "KnowledgeChunk"
            }
            
            # Create node
            store.add_entity("KnowledgeChunk", props)
            
            # Create Category node and relation
            cat_props = {"name": category}
            store.query("MERGE (c:Category {name: $name})", cat_props)
            
            store.query("""
                MATCH (k:KnowledgeChunk {text: $text}), (c:Category {name: $cat})
                MERGE (k)-[:BELONGS_TO]->(c)
            """, {"text": text, "cat": category})
            
            # Extract keywords/entities (Simple rule based or LLM based)
            # For now, let's link Sources
            store.query("MERGE (s:Source {name: $name})", {"name": source})
            store.query("""
                MATCH (k:KnowledgeChunk {text: $text}), (s:Source {name: $source})
                MERGE (k)-[:PROVIDED_BY]->(s)
            """, {"text": text, "source": source})
            
            count += 1
            
        logger.info(f"Seeded {count} knowledge chunks into graph")
        
        # Verify
        result = store.query("MATCH (n) RETURN count(n) as count")
        logger.info(f"Total nodes in graph: {result[0]['count']}")
        
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise
    finally:
        store.close()

if __name__ == "__main__":
    asyncio.run(seed_graph())
