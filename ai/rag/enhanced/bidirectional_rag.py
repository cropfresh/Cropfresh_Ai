"""
Bidirectional RAG
=================
Knowledge expansion through bidirectional retrieval.

Traditional RAG: Query -> Retrieve -> Generate
Bidirectional: Query <-> Retrieve <-> Expand <-> Generate

Features:
- Forward retrieval (query to docs)
- Backward expansion (docs to related concepts)
- Knowledge gap detection
- Concept linking
"""

from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field


class KnowledgeNode(BaseModel):
    """A node in the knowledge graph."""
    concept: str
    content: str = ""
    source: str = ""
    confidence: float = 0.5
    related_concepts: list[str] = Field(default_factory=list)


class KnowledgeExpansion(BaseModel):
    """Result of knowledge expansion."""
    original_query: str
    retrieved_nodes: list[KnowledgeNode] = Field(default_factory=list)
    expanded_nodes: list[KnowledgeNode] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)
    total_concepts: int = 0


class BidirectionalRAG:
    """
    Bidirectional retrieval for knowledge expansion.
    
    Usage:
        brag = BidirectionalRAG(kb=knowledge_base)
        
        expansion = await brag.expand(
            "What affects tomato yield?"
        )
        
        print(expansion.expanded_nodes)  # Related concepts discovered
    """
    
    def __init__(
        self,
        knowledge_base=None,
        graph_client=None,
        llm=None,
        max_expansion_depth: int = 2,
    ):
        """
        Initialize bidirectional RAG.
        
        Args:
            knowledge_base: Vector store for retrieval
            graph_client: Graph database for relationships
            llm: LLM for concept extraction
            max_expansion_depth: Max levels of expansion
        """
        self.knowledge_base = knowledge_base
        self.graph_client = graph_client
        self.llm = llm
        self.max_depth = max_expansion_depth
    
    async def expand(
        self,
        query: str,
        initial_limit: int = 5,
    ) -> KnowledgeExpansion:
        """
        Perform bidirectional retrieval and expansion.
        
        Args:
            query: User query
            initial_limit: Initial retrieval limit
            
        Returns:
            KnowledgeExpansion with all discovered nodes
        """
        expansion = KnowledgeExpansion(original_query=query)
        seen_concepts = set()
        
        # Step 1: Forward retrieval
        initial_nodes = await self._forward_retrieve(query, initial_limit)
        expansion.retrieved_nodes = initial_nodes
        
        for node in initial_nodes:
            seen_concepts.add(node.concept.lower())
        
        # Step 2: Backward expansion
        for depth in range(self.max_depth):
            new_nodes = await self._backward_expand(
                expansion.retrieved_nodes + expansion.expanded_nodes,
                seen_concepts,
            )
            
            for node in new_nodes:
                if node.concept.lower() not in seen_concepts:
                    expansion.expanded_nodes.append(node)
                    seen_concepts.add(node.concept.lower())
            
            if not new_nodes:
                break
        
        # Step 3: Detect knowledge gaps
        expansion.knowledge_gaps = await self._detect_gaps(
            query, expansion.retrieved_nodes + expansion.expanded_nodes
        )
        
        expansion.total_concepts = len(expansion.retrieved_nodes) + len(expansion.expanded_nodes)
        
        logger.info(
            "Bidirectional expansion: {} initial, {} expanded, {} gaps",
            len(expansion.retrieved_nodes),
            len(expansion.expanded_nodes),
            len(expansion.knowledge_gaps),
        )
        
        return expansion
    
    async def _forward_retrieve(
        self,
        query: str,
        limit: int,
    ) -> list[KnowledgeNode]:
        """Forward retrieval: Query -> Documents."""
        nodes = []
        
        if self.knowledge_base:
            try:
                results = await self.knowledge_base.search(query=query, limit=limit)
                
                for r in results:
                    content = r.content if hasattr(r, 'content') else str(r)
                    concept = self._extract_main_concept(content)
                    
                    nodes.append(KnowledgeNode(
                        concept=concept,
                        content=content,
                        source="knowledge_base",
                        confidence=r.score if hasattr(r, 'score') else 0.5,
                        related_concepts=self._extract_related_concepts(content),
                    ))
                    
            except Exception as e:
                logger.warning("Forward retrieval failed: {}", str(e))
        
        return nodes
    
    async def _backward_expand(
        self,
        nodes: list[KnowledgeNode],
        seen: set,
    ) -> list[KnowledgeNode]:
        """Backward expansion: Documents -> Related concepts."""
        new_nodes = []
        
        # Collect related concepts from existing nodes
        concepts_to_expand = set()
        for node in nodes:
            for concept in node.related_concepts:
                if concept.lower() not in seen:
                    concepts_to_expand.add(concept)
        
        # Limit expansion
        concepts_to_expand = list(concepts_to_expand)[:5]
        
        # Retrieve for each concept
        for concept in concepts_to_expand:
            if self.knowledge_base:
                try:
                    results = await self.knowledge_base.search(
                        query=concept,
                        limit=2,
                    )
                    
                    for r in results:
                        content = r.content if hasattr(r, 'content') else str(r)
                        
                        new_nodes.append(KnowledgeNode(
                            concept=concept,
                            content=content,
                            source="expansion",
                            confidence=(r.score if hasattr(r, 'score') else 0.5) * 0.8,
                            related_concepts=self._extract_related_concepts(content),
                        ))
                        
                except Exception as e:
                    logger.debug("Expansion failed for {}: {}", concept, str(e))
        
        # Also check graph if available
        if self.graph_client:
            for node in nodes[:3]:
                try:
                    if hasattr(self.graph_client, 'get_neighbors'):
                        neighbors = await self.graph_client.get_neighbors(node.concept)
                        
                        for neighbor in neighbors:
                            if neighbor.lower() not in seen:
                                new_nodes.append(KnowledgeNode(
                                    concept=neighbor,
                                    content=f"Related to {node.concept}",
                                    source="graph",
                                    confidence=0.7,
                                ))
                except Exception as e:
                    logger.debug("Graph expansion failed: {}", str(e))
        
        return new_nodes
    
    async def _detect_gaps(
        self,
        query: str,
        nodes: list[KnowledgeNode],
    ) -> list[str]:
        """Detect knowledge gaps."""
        gaps = []
        
        # Extract expected concepts from query
        query_concepts = self._extract_related_concepts(query)
        found_concepts = {node.concept.lower() for node in nodes}
        
        # Find concepts mentioned in query but not found
        for concept in query_concepts:
            if concept.lower() not in found_concepts:
                gaps.append(f"No information found about: {concept}")
        
        # Check for low confidence areas
        low_confidence = [n for n in nodes if n.confidence < 0.4]
        if low_confidence:
            gaps.append(f"Low confidence on: {', '.join(n.concept for n in low_confidence[:3])}")
        
        return gaps
    
    def _extract_main_concept(self, content: str) -> str:
        """Extract main concept from content."""
        # Simple: use first few words or title
        lines = content.strip().split('\n')
        for line in lines[:3]:
            line = line.strip()
            if line and not line.startswith('#'):
                # Get first meaningful phrase
                words = line.split()[:5]
                return ' '.join(words)
        return content[:50] if content else "Unknown"
    
    def _extract_related_concepts(self, content: str) -> list[str]:
        """Extract related concepts from content."""
        concepts = []
        
        # Common agricultural concepts to look for
        agri_concepts = [
            "tomato", "rice", "wheat", "cotton", "onion", "potato",
            "irrigation", "fertilizer", "pest", "disease", "soil",
            "monsoon", "rainfall", "temperature", "harvest", "yield",
            "mandi", "price", "market", "farmer", "crop",
            "karnataka", "maharashtra", "punjab", "uttar pradesh",
        ]
        
        content_lower = content.lower()
        for concept in agri_concepts:
            if concept in content_lower:
                concepts.append(concept.title())
        
        return concepts[:10]  # Limit
