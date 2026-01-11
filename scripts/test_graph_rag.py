"""
Test Graph Construction and Reasoning (Phase 7)
================================================
Tests for auto-construction of knowledge graphs and reasoning.

Run with:
    uv run python scripts/test_graph_rag.py

Author: CropFresh AI Team
"""

import asyncio
from src.rag.graph_constructor import (
    GraphConstructor,
    GraphNode,
    EntityType,
    RelationType,
    ConstructedGraph
)


async def test_rule_based_extraction():
    """Test rule-based graph extraction fallback."""
    print("Testing rule-based graph extraction...")
    
    gc = GraphConstructor()
    text = "Tomato crops in Kolar are often affected by aphids and fruit borers. Farmers in Bangalore also grow potatoes."
    
    graph = await gc.process_document(text)
    
    # Check nodes
    expected_nodes = ["tomato", "kolar", "aphid", "borer", "potato", "bangalore"]
    found_labels = [n.id for n in graph.nodes]
    print(f"Nodes found: {found_labels}")
    
    for en in expected_nodes:
        assert en in found_labels
        
    # Check edges
    print(f"Edges extracted: {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  {edge.source_id} --[{edge.type}]--> {edge.target_id}")
        
    assert len(graph.edges) > 0


async def test_multi_hop_simulation():
    """Test reasoning path simulation."""
    print("\nTesting multi-hop reasoning simulation...")
    
    gc = GraphConstructor()
    # In test mode without Neo4j, it returns a simulated path
    path = await gc.multi_hop_reasoning("aphid", steps=2)
    
    assert len(path) > 0
    print(f"Reasoning path: {path[0]}")


async def main():
    print("=== Phase 7: Enhanced Graph RAG Tests ===")
    await test_rule_based_extraction()
    await test_multi_hop_simulation()
    print("\nAll Graph RAG tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
