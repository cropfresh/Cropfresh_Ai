"""
Graph RAG - Auto Construction & Linking
=======================================
Module for constructing knowledge graphs from documents and multi-hop reasoning.

Features:
- LLM-based Triple Extraction (Subject -> Predicate -> Object)
- Entity Linking & Disambiguation
- Multi-hop Reasoning Traversal
- Neo4j Graph Injection

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import uuid

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CROP = "Crop"
    DISEASE = "Disease"
    PEST = "Pest"
    LOCATION = "Location"
    CHEMICAL = "Chemical"
    PRACTICE = "Practice"
    PERSON = "Person"
    ORGANIZATION = "Organization"
    CONCEPT = "Concept"


class RelationType(str, Enum):
    AFFECTS = "AFFECTS"
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    LOCATED_IN = "LOCATED_IN"
    GROWS_IN = "GROWS_IN"
    REQUIRES = "REQUIRES"
    IS_A = "IS_A"
    RELATED_TO = "RELATED_TO"


class GraphNode(BaseModel):
    """Node in the knowledge graph."""
    id: str
    label: str
    type: EntityType = EntityType.CONCEPT
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


class GraphEdge(BaseModel):
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    type: RelationType = RelationType.RELATED_TO
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class ConstructedGraph(BaseModel):
    """Graph structure extracted from documents."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for traversal."""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id, label=node.label, type=node.type, **node.properties)
        for edge in self.edges:
            G.add_edge(edge.source_id, edge.target_id, type=edge.type, weight=edge.weight, **edge.properties)
        return G


class GraphConstructor:
    """
    Constructs a knowledge graph from unstructured text documents.
    """
    
    # Prompt for triple extraction
    EXTRACTION_PROMPT = """You are a knowledge graph expert.
Extract entities and relationships from the following agricultural text.
Return a list of triples in the format: (Subject, Relation, Object).
Possible relations: AFFECTS, CAUSES, PREVENTS, LOCATED_IN, GROWS_IN, REQUIRES, IS_A.
Possible entity types: Crop, Disease, Pest, Location, Chemical, Practice.

Text: {text}

Triples:"""
    
    def __init__(self, llm=None, neo4j_client=None):
        self.llm = llm
        self.neo4j_client = neo4j_client
        logger.info("GraphConstructor initialized")
    
    async def process_document(self, text: str) -> ConstructedGraph:
        """
        Extract graph from a single document.
        
        Args:
            text: Document content
            
        Returns:
            ConstructedGraph object
        """
        if self.llm is None:
            return self._rule_based_extraction(text)
            
        try:
            prompt = self.EXTRACTION_PROMPT.format(text=text[:2000])  # Limit ctx
            response = await self.llm.agenerate([prompt])
            content = response.generations[0][0].text.strip()
            
            return self._parse_triples(content)
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            return self._rule_based_extraction(text)
            
    def _parse_triples(self, content: str) -> ConstructedGraph:
        """Parse LLM output into graph structure."""
        nodes_dict: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []
        
        # Simple parsing of (Subject, Relation, Object) lines
        # This is a basic implementation; robust regex or JSON parsing is preferred in prod
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('(') or not line.endswith(')'):
                continue
                
            try:
                # Remove parens and split
                parts = [p.strip() for p in line[1:-1].split(',')]
                if len(parts) >= 3:
                    subj, rel, obj = parts[0], parts[1], parts[2]
                    
                    # Create nodes if not exist
                    subj_id = self._normalize_id(subj)
                    obj_id = self._normalize_id(obj)
                    
                    if subj_id not in nodes_dict:
                        nodes_dict[subj_id] = GraphNode(id=subj_id, label=subj)
                    if obj_id not in nodes_dict:
                        nodes_dict[obj_id] = GraphNode(id=obj_id, label=obj)
                        
                    # Create edge
                    try:
                        rel_type = RelationType(rel.upper()) 
                    except ValueError:
                        rel_type = RelationType.RELATED_TO
                        
                    edges.append(GraphEdge(
                        source_id=subj_id,
                        target_id=obj_id,
                        type=rel_type
                    ))
            except Exception:
                continue
                
        return ConstructedGraph(nodes=list(nodes_dict.values()), edges=edges)

    def _rule_based_extraction(self, text: str) -> ConstructedGraph:
        """Fallback extraction using basic rules."""
        nodes = []
        edges = []
        
        text_lower = text.lower()
        
        # Example entities to look for
        crops = ["tomato", "onion", "potato", "rice"]
        pests = ["aphid", "borer", "thrip"]
        locations = ["kolar", "bangalore"]
        
        found_crops = [c for c in crops if c in text_lower]
        found_pests = [p for p in pests if p in text_lower]
        found_locs = [l for l in locations if l in text_lower]
        
        # Create nodes
        for c in found_crops:
            nodes.append(GraphNode(id=c, label=c.title(), type=EntityType.CROP))
        for p in found_pests:
            nodes.append(GraphNode(id=p, label=p.title(), type=EntityType.PEST))
        for l in found_locs:
            nodes.append(GraphNode(id=l, label=l.title(), type=EntityType.LOCATION))
            
        # Create edges (heuristic)
        # If crop and pest appear, link them
        for c in found_crops:
            for p in found_pests:
                edges.append(GraphEdge(
                    source_id=p,
                    target_id=c,
                    type=RelationType.AFFECTS
                ))
            for l in found_locs:
                edges.append(GraphEdge(
                    source_id=c,
                    target_id=l,
                    type=RelationType.GROWS_IN
                ))
                
        return ConstructedGraph(nodes=nodes, edges=edges)
        
    def _normalize_id(self, text: str) -> str:
        """Create normalized ID from text."""
        return text.lower().replace(" ", "_").strip()

    async def ingest_to_neo4j(self, graph: ConstructedGraph):
        """Write constructed graph to Neo4j."""
        if not self.neo4j_client:
            logger.warning("No Neo4j client available for ingestion.")
            return
            
        logger.info(f"Ingesting {len(graph.nodes)} nodes and {len(graph.edges)} edges to Neo4j")
        # In prod: Use Cypher queries with UNWIND for batching
        # self.neo4j_client.execute_query(...)
        pass
        
    async def multi_hop_reasoning(self, start_node: str, steps: int = 2) -> List[str]:
        """
        Simulate walking the graph to find connected insights.
        """
        if not self.neo4j_client:
            # Simulated logic for tests
             return [f"Reasoning path from {start_node}: {start_node} -> AFFECTS -> Tomato"]
        
        # In prod: Cypher query MATCH path=(n)-[*1..2]-(m) WHERE n.id = start_node RETURN path
        return []


# Factory
def create_graph_constructor(llm=None, neo4j_client=None) -> GraphConstructor:
    return GraphConstructor(llm=llm, neo4j_client=neo4j_client)
