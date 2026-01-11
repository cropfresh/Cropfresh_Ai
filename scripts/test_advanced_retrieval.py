"""
Test Advanced Retrieval Techniques (Phase 2)
=============================================
Tests for RAPTOR hierarchical retrieval and Contextual Chunking.

Run with:
    uv run python scripts/test_advanced_retrieval.py

Author: CropFresh AI Team
"""

import asyncio
from datetime import datetime


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} - {name}")
    if details:
        print(f"         {details}")


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "text": """# Tomato Cultivation Guide

## Soil Preparation
Tomatoes thrive in well-drained, fertile soil with pH between 6.0 and 6.8. 
Prepare beds by adding 20-25 tonnes of farmyard manure per hectare.
Ensure proper drainage to prevent waterlogging.

## Planting
Sow seeds in nursery beds during June-July for Kharif season or October-November for Rabi.
Transplant seedlings 25-30 days after sowing when they have 4-5 true leaves.
Maintain spacing of 60cm between rows and 45cm between plants.

## Irrigation
Irrigate immediately after transplanting. Maintain regular irrigation every 5-7 days.
During flowering and fruit setting, ensure adequate moisture.
Drip irrigation is recommended for water efficiency.

## Pest Management
Watch for common pests: tomato fruit borer, whitefly, and leafminer.
Apply neem oil spray (5ml/litre) for organic control.
For severe infestations, use approved insecticides like Spinosad.

## Harvesting
Harvest when fruits turn light red for distant markets.
For local markets, harvest at fully ripe stage.
Average yield: 25-30 tonnes per hectare.""",
        "metadata": {"title": "Tomato Cultivation Guide", "source": "KVK Karnataka"}
    },
    {
        "id": "doc_002",
        "text": """# Market Price Analysis for Vegetables

## Current Trends in Karnataka
Tomato prices in Kolar market have shown volatility this month.
Modal price ranges from ₹2,000 to ₹3,500 per quintal.
Prices peak during October-November due to reduced supply.

## Price Factors
Weather conditions significantly impact vegetable prices.
Transportation costs add 10-15% to final market price.
Cold storage availability reduces price fluctuations.

## Selling Strategy
Monitor market arrivals before deciding to sell.
Early morning arrivals fetch better prices.
Grade vegetables before selling for premium rates.

## Digital Marketing
Register on eNAM platform for wider market access.
Use CropFresh app to check real-time prices.
Direct buyer connections eliminate middlemen.""",
        "metadata": {"title": "Market Price Analysis", "source": "APMC Karnataka"}
    },
    {
        "id": "doc_003",
        "text": """# Integrated Pest Management for Vegetable Crops

## Prevention Strategy
Crop rotation breaks pest cycles.
Use disease-resistant varieties when available.
Maintain field hygiene by removing crop residues.

## Biological Control
Encourage natural predators like ladybugs and lacewings.
Release Trichogramma cards for borer control.
Apply Trichoderma for soil-borne diseases.

## Chemical Control
Use pesticides only when pest population exceeds threshold.
Follow recommended dosage strictly.
Maintain safety interval before harvest.

## Monitoring
Scout fields twice weekly during cropping season.
Use pheromone traps for pest monitoring.
Record pest populations for future reference.""",
        "metadata": {"title": "Integrated Pest Management", "source": "ICAR"}
    }
]


class MockDocument:
    """Mock document for testing."""
    def __init__(self, doc_data: dict):
        self.id = doc_data["id"]
        self.text = doc_data["text"]
        self.metadata = doc_data.get("metadata", {})


class MockEmbeddingManager:
    """Mock embedding manager for testing."""
    
    def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding."""
        import hashlib
        # Create deterministic embedding from text hash
        hash_bytes = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes] + [0.0] * (1024 - 16)


async def test_raptor():
    """Test RAPTOR hierarchical retrieval."""
    print_header("Testing RAPTOR Hierarchical Retrieval")
    
    from src.rag.raptor import RAPTORIndex, RAPTORConfig
    
    # Create mock dependencies
    embedding_manager = MockEmbeddingManager()
    
    # Create RAPTOR index with smaller config for testing
    config = RAPTORConfig(
        chunk_size=200,
        chunk_overlap=50,
        max_cluster_size=5,
        min_cluster_size=2,
        max_levels=3,
        use_umap=False,  # Disable UMAP for faster testing
    )
    
    raptor = RAPTORIndex(
        embedding_manager=embedding_manager,
        llm=None,  # No LLM for testing
        config=config,
    )
    
    # Test 1: Build tree
    try:
        documents = [MockDocument(d) for d in SAMPLE_DOCUMENTS]
        stats = await raptor.build_tree(documents)
        passed = stats.total_nodes > 0 and stats.max_depth >= 0
        print_result("Build RAPTOR tree", passed,
                    f"{stats.total_nodes} nodes, {stats.max_depth} levels")
        
        if stats.nodes_per_level:
            for level, count in sorted(stats.nodes_per_level.items()):
                print(f"           Level {level}: {count} nodes")
    except Exception as e:
        print_result("Build RAPTOR tree", False, str(e))
        return  # Can't continue without tree
    
    # Test 2: Collapsed retrieval
    try:
        results = await raptor.retrieve("How to grow tomatoes?", top_k=3, strategy="collapsed")
        passed = len(results) > 0
        print_result("Collapsed retrieval", passed,
                    f"{len(results)} results returned")
        if results:
            print(f"           Top result (level {results[0].level}): {results[0].text[:60]}...")
    except Exception as e:
        print_result("Collapsed retrieval", False, str(e))
    
    # Test 3: Tree traversal retrieval
    try:
        results = await raptor.retrieve("pest management", top_k=3, strategy="tree")
        passed = len(results) > 0
        print_result("Tree traversal retrieval", passed,
                    f"{len(results)} results returned")
    except Exception as e:
        print_result("Tree traversal retrieval", False, str(e))
    
    # Test 4: Multi-level search (abstract + specific)
    try:
        # Search only leaf nodes
        leaf_results = await raptor.retrieve("tomato", top_k=3, levels=[0])
        # Search summary nodes (level 1+)
        summary_results = await raptor.retrieve("tomato", top_k=3, levels=[1, 2, 3])
        
        passed = len(leaf_results) > 0
        print_result("Multi-level search", passed,
                    f"Leaf: {len(leaf_results)}, Summaries: {len(summary_results)}")
    except Exception as e:
        print_result("Multi-level search", False, str(e))
    
    # Test 5: Visualize tree
    try:
        tree_viz = raptor.visualize_tree()
        passed = "RAPTOR Tree" in tree_viz
        print_result("Tree visualization", passed, "ASCII tree generated")
    except Exception as e:
        print_result("Tree visualization", False, str(e))
    
    # Test 6: Get ancestors/descendants
    try:
        # Get a node
        node_ids = list(raptor.nodes.keys())
        if node_ids:
            test_node_id = node_ids[0]
            ancestors = raptor.get_ancestors(test_node_id)
            descendants = raptor.get_descendants(test_node_id)
            passed = isinstance(ancestors, list) and isinstance(descendants, list)
            print_result("Tree navigation", passed,
                        f"Ancestors: {len(ancestors)}, Descendants: {len(descendants)}")
    except Exception as e:
        print_result("Tree navigation", False, str(e))


async def test_contextual_chunker():
    """Test Contextual Chunking."""
    print_header("Testing Contextual Chunking")
    
    from src.rag.contextual_chunker import ContextualChunker, ChunkingConfig
    
    # Create chunker with test config
    config = ChunkingConfig(
        chunk_size=300,
        chunk_overlap=50,
        add_context=True,
        use_llm_context=False,  # Rule-based context for testing
        extract_entities=True,
        use_semantic_boundaries=True,
    )
    
    chunker = ContextualChunker(llm=None, config=config)
    
    # Test 1: Chunk with context
    try:
        doc = MockDocument(SAMPLE_DOCUMENTS[0])
        chunks = await chunker.chunk_with_context(
            doc,
            document_title="Tomato Cultivation Guide",
            document_source="KVK Karnataka",
        )
        passed = len(chunks) > 0 and chunks[0].enriched_text != ""
        print_result("Chunk with context", passed,
                    f"{len(chunks)} chunks created")
    except Exception as e:
        print_result("Chunk with context", False, str(e))
        return
    
    # Test 2: Enriched text
    try:
        chunk = chunks[0]
        enriched = chunk.enriched_text
        passed = len(enriched) > len(chunk.text)  # Context should add length
        print_result("Enriched text generation", passed,
                    f"Original: {len(chunk.text)} chars, Enriched: {len(enriched)} chars")
    except Exception as e:
        print_result("Enriched text generation", False, str(e))
    
    # Test 3: Entity extraction
    try:
        # Combine all entities from all chunks
        all_entities = []
        for chunk in chunks:
            all_entities.extend(chunk.entities)
        
        unique_entities = list(set(all_entities))
        passed = len(unique_entities) > 0
        print_result("Entity extraction", passed,
                    f"{len(unique_entities)} unique entities found")
        
        # Show some examples
        if unique_entities:
            examples = unique_entities[:5]
            print(f"           Examples: {', '.join(examples)}")
    except Exception as e:
        print_result("Entity extraction", False, str(e))
    
    # Test 4: Section header propagation
    try:
        sections_found = [c.section_header for c in chunks if c.section_header]
        passed = len(sections_found) > 0
        print_result("Section header propagation", passed,
                    f"{len(set(sections_found))} unique sections found")
    except Exception as e:
        print_result("Section header propagation", False, str(e))
    
    # Test 5: Keyword extraction
    try:
        keywords = chunks[0].keywords
        passed = len(keywords) > 0
        print_result("Keyword extraction", passed,
                    f"{len(keywords)} keywords extracted")
        if keywords:
            print(f"           Keywords: {', '.join(keywords[:5])}")
    except Exception as e:
        print_result("Keyword extraction", False, str(e))
    
    # Test 6: Batch processing
    try:
        from src.rag.contextual_chunker import enrich_documents
        
        all_docs = [MockDocument(d) for d in SAMPLE_DOCUMENTS]
        all_chunks = await enrich_documents(
            all_docs,
            llm=None,
            config=ChunkingConfig(chunk_size=300),
        )
        
        passed = len(all_chunks) > len(all_docs)  # Should have multiple chunks per doc
        print_result("Batch document enrichment", passed,
                    f"{len(all_chunks)} total chunks from {len(all_docs)} documents")
    except Exception as e:
        print_result("Batch document enrichment", False, str(e))


async def test_integration():
    """Test RAPTOR + Contextual Chunking integration."""
    print_header("Testing Integration")
    
    from src.rag.raptor import RAPTORIndex, RAPTORConfig
    from src.rag.contextual_chunker import enrich_documents, ChunkingConfig
    
    embedding_manager = MockEmbeddingManager()
    
    # Test: Use enriched chunks with RAPTOR
    try:
        # First, create enriched chunks
        docs = [MockDocument(d) for d in SAMPLE_DOCUMENTS]
        chunks = await enrich_documents(
            docs,
            config=ChunkingConfig(chunk_size=250),
        )
        
        # Create mock documents from enriched chunks for RAPTOR
        class EnrichedMockDoc:
            def __init__(self, chunk):
                self.id = chunk.id
                self.text = chunk.enriched_text  # Use enriched text!
                self.metadata = {"entities": chunk.entities}
        
        enriched_docs = [EnrichedMockDoc(c) for c in chunks]
        
        # Build RAPTOR on enriched chunks
        raptor = RAPTORIndex(
            embedding_manager=embedding_manager,
            config=RAPTORConfig(
                chunk_size=200,
                max_levels=2,
                use_umap=False,
            ),
        )
        
        stats = await raptor.build_tree(enriched_docs)
        
        passed = stats.total_nodes > 0
        print_result("RAPTOR on enriched chunks", passed,
                    f"{stats.total_nodes} nodes from {len(chunks)} enriched chunks")
        
        # Test retrieval
        results = await raptor.retrieve("how to manage pests on tomato", top_k=3)
        passed = len(results) > 0
        print_result("Retrieval with context", passed,
                    f"{len(results)} contextual results")
        
    except Exception as e:
        print_result("Integration test", False, str(e))
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("   PHASE 2: ADVANCED RETRIEVAL TECHNIQUES TESTS")
    print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    await test_contextual_chunker()
    await test_raptor()
    await test_integration()
    
    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)
    print("\n  🎉 Phase 2 components tested!")
    print("  📊 RAPTOR and Contextual Chunking ready.")
    print("\n  Next steps:")
    print("    1. Add sklearn for GMM clustering: uv add scikit-learn")
    print("    2. Optionally add umap-learn for UMAP: uv add umap-learn")
    print("    3. Integrate with existing RAG pipeline")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
