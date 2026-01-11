"""
Test Enhanced Query Processing and Retrieval (Phase 3-4)
=========================================================
Tests for HyDE, Multi-Query, Step-Back, Enhanced Retrievers.

Run with:
    uv run python scripts/test_query_retrieval.py

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
SAMPLE_DOCS = [
    {
        "id": "doc1",
        "text": """Tomato Cultivation in Karnataka

Tomatoes are one of the most important vegetable crops in Karnataka. The state produces over 500,000 tonnes annually. 
Key districts include Kolar, Chikballapur, and Bangalore Rural.

Soil Requirements:
Tomatoes prefer well-drained loamy soil with pH 6.0-7.0. Add organic matter before planting. 
Ensure proper drainage to prevent root rot.

Climate:
Optimal temperature is 21-24°C. Temperatures above 35°C can cause flower drop.
Frost is damaging to the crop.

Varieties:
Popular varieties include Arka Vikas, PKM-1, and hybrid varieties like Lakshmi and Namdhari-62."""
    },
    {
        "id": "doc2", 
        "text": """Integrated Pest Management for Vegetables

IPM is a sustainable approach to managing pests. It combines biological, cultural, and chemical methods.

Key Principles:
1. Monitor pest populations regularly using traps and scouting
2. Use biological control agents like Trichogramma wasps
3. Apply pesticides only when pest levels exceed economic threshold
4. Rotate pesticides to prevent resistance

Common Vegetable Pests:
- Aphids: Control with neem oil or ladybug release
- Whiteflies: Use yellow sticky traps and neem spray
- Fruit borer: Release Trichogramma, spray Bt if needed
- Thrips: Blue sticky traps, spinosad spray"""
    },
    {
        "id": "doc3",
        "text": """Market Price Trends in Indian Agriculture

Agricultural commodity prices fluctuate based on supply, demand, and seasonal factors.

Key Price Factors:
1. Monsoon performance affects Kharif crop supply
2. Cold storage capacity influences off-season availability
3. Transportation costs add 10-15% to final price
4. Government MSP provides price floor for some crops

Market Infrastructure:
APMCs regulate wholesale trading across India. 
eNAM platform enables online trading across states.
Direct farmer-buyer connect reduces intermediary costs.

Price Discovery:
Mandi prices are announced daily based on arrivals and demand.
Premium prices for graded produce."""
    }
]


class MockDocument:
    """Mock document for testing."""
    def __init__(self, data: dict):
        self.id = data["id"]
        self.text = data["text"]
        self.metadata = {}


class MockEmbeddingManager:
    """Mock embedding manager for testing."""
    
    def embed_query(self, text: str) -> list[float]:
        """Generate deterministic mock embedding from text."""
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Create a 1024-dim embedding from hash
        embedding = []
        for i in range(64):
            # Use different parts of text for variety
            sample = text[i % len(text):(i % len(text)) + 10] if i < len(text) else text[:10]
            h = hashlib.md5(f"{sample}{i}".encode()).digest()
            embedding.extend([float(b) / 255.0 for b in h])
        return embedding


async def test_query_processor():
    """Test Advanced Query Processor."""
    print_header("Testing Advanced Query Processor")
    
    from src.rag.query_processor import AdvancedQueryProcessor, QueryProcessorConfig
    
    config = QueryProcessorConfig(
        hyde_enabled=True,
        multi_query_enabled=True,
        step_back_enabled=True,
        decompose_enabled=True,
        rewrite_enabled=True,
    )
    
    processor = AdvancedQueryProcessor(llm=None, config=config)
    
    # Test 1: HyDE expansion
    try:
        result = await processor.hyde_expand("What is the best fertilizer for tomatoes?")
        passed = len(result.hypothetical_doc) > 0
        print_result("HyDE Expansion", passed,
                    f"Generated {len(result.hypothetical_doc)} char hypothetical doc")
        if result.hypothetical_doc:
            print(f"         Preview: {result.hypothetical_doc[:80]}...")
    except Exception as e:
        print_result("HyDE Expansion", False, str(e))
    
    # Test 2: Multi-query expansion
    try:
        result = await processor.multi_query_expand("tomato pest control methods")
        passed = len(result.expanded_queries) > 0
        print_result("Multi-Query Expansion", passed,
                    f"Generated {len(result.expanded_queries)} alternative queries")
        for q in result.expanded_queries[:2]:
            print(f"         → {q}")
    except Exception as e:
        print_result("Multi-Query Expansion", False, str(e))
    
    # Test 3: Step-back prompting
    try:
        result = await processor.step_back_expand("How to control aphids on tomato plants in Kolar?")
        passed = len(result.step_back_query) > 0
        print_result("Step-Back Prompting", passed,
                    f"Step-back: {result.step_back_query[:60]}...")
    except Exception as e:
        print_result("Step-Back Prompting", False, str(e))
    
    # Test 4: Query decomposition
    try:
        result = await processor.decompose_query("How to grow tomatoes and what is the market price?")
        passed = isinstance(result.sub_queries, list)
        print_result("Query Decomposition", passed,
                    f"Decomposed into {len(result.sub_queries)} sub-queries")
    except Exception as e:
        print_result("Query Decomposition", False, str(e))
    
    # Test 5: Query rewriting
    try:
        result = await processor.rewrite_query("can you tell me how to grow tamatar in india")
        passed = len(result.rewritten_query) > 0
        print_result("Query Rewriting", passed,
                    f"Rewritten: {result.rewritten_query}")
    except Exception as e:
        print_result("Query Rewriting", False, str(e))
    
    # Test 6: Full pipeline
    try:
        result = await processor.process_query("What's the best way to increase tomato yield?")
        passed = len(result.all_queries) > 1
        print_result("Full Pipeline", passed,
                    f"Total {len(result.all_queries)} queries, {result.processing_time_ms:.1f}ms")
    except Exception as e:
        print_result("Full Pipeline", False, str(e))


async def test_parent_document_retriever():
    """Test Parent Document Retriever."""
    print_header("Testing Parent Document Retriever")
    
    from src.rag.enhanced_retriever import ParentDocumentRetriever, EnhancedRetrieverConfig
    
    embedding_manager = MockEmbeddingManager()
    config = EnhancedRetrieverConfig(
        parent_chunk_size=500,
        child_chunk_size=150,
    )
    
    retriever = ParentDocumentRetriever(embedding_manager, config)
    
    # Add documents
    try:
        docs = [MockDocument(d) for d in SAMPLE_DOCS]
        num_children = await retriever.add_documents(docs)
        passed = num_children > 0
        print_result("Add documents", passed,
                    f"Created {len(retriever.parent_nodes)} parents, {num_children} children")
    except Exception as e:
        print_result("Add documents", False, str(e))
        return
    
    # Retrieve with parent context
    try:
        result = await retriever.retrieve("tomato soil requirements", top_k=3)
        passed = len(result.nodes) > 0
        print_result("Retrieve with parents", passed,
                    f"{len(result.nodes)} parent docs, {result.processing_time_ms:.1f}ms")
        
        if result.parent_documents:
            print(f"         Parent preview: {result.parent_documents[0][:80]}...")
    except Exception as e:
        print_result("Retrieve with parents", False, str(e))


async def test_sentence_window_retriever():
    """Test Sentence Window Retriever."""
    print_header("Testing Sentence Window Retriever")
    
    from src.rag.enhanced_retriever import SentenceWindowRetriever, EnhancedRetrieverConfig
    
    embedding_manager = MockEmbeddingManager()
    config = EnhancedRetrieverConfig(window_size=2)
    
    retriever = SentenceWindowRetriever(embedding_manager, config)
    
    # Add documents
    try:
        docs = [MockDocument(d) for d in SAMPLE_DOCS]
        num_sentences = await retriever.add_documents(docs)
        passed = num_sentences > 0
        print_result("Add sentences", passed,
                    f"Indexed {num_sentences} sentences")
    except Exception as e:
        print_result("Add sentences", False, str(e))
        return
    
    # Retrieve with window expansion
    try:
        result = await retriever.retrieve("aphid control", top_k=3)
        passed = len(result.nodes) > 0
        print_result("Sentence window retrieval", passed,
                    f"{len(result.nodes)} windows, {result.processing_time_ms:.1f}ms")
        
        if result.expanded_context:
            print(f"         Expanded context: {result.expanded_context[:100]}...")
    except Exception as e:
        print_result("Sentence window retrieval", False, str(e))


async def test_mmr_retriever():
    """Test MMR Retriever."""
    print_header("Testing MMR (Diversity) Retriever")
    
    from src.rag.enhanced_retriever import MMRRetriever, EnhancedRetrieverConfig
    
    embedding_manager = MockEmbeddingManager()
    config = EnhancedRetrieverConfig(mmr_lambda=0.5)
    
    retriever = MMRRetriever(embedding_manager, config)
    
    # Add documents
    try:
        docs = [MockDocument(d) for d in SAMPLE_DOCS]
        count = await retriever.add_documents(docs)
        passed = count > 0
        print_result("Add documents", passed, f"Added {count} documents")
    except Exception as e:
        print_result("Add documents", False, str(e))
        return
    
    # MMR retrieval
    try:
        result = await retriever.retrieve("agriculture farming", top_k=3)
        passed = len(result.nodes) > 0
        print_result("MMR retrieval", passed,
                    f"{len(result.nodes)} docs, diversity: {result.diversity_score:.2f}")
    except Exception as e:
        print_result("MMR retrieval", False, str(e))
    
    # Compare high vs low diversity
    try:
        high_div = await retriever.retrieve("vegetable farming", top_k=3, lambda_param=0.2)
        low_div = await retriever.retrieve("vegetable farming", top_k=3, lambda_param=0.9)
        
        passed = high_div.diversity_score >= low_div.diversity_score - 0.1  # Allow small variance
        print_result("Diversity tuning", passed,
                    f"λ=0.2: div={high_div.diversity_score:.2f}, λ=0.9: div={low_div.diversity_score:.2f}")
    except Exception as e:
        print_result("Diversity tuning", False, str(e))


async def test_enhanced_retriever():
    """Test Unified Enhanced Retriever."""
    print_header("Testing Unified Enhanced Retriever")
    
    from src.rag.enhanced_retriever import EnhancedRetriever, RetrievalStrategy
    
    embedding_manager = MockEmbeddingManager()
    retriever = EnhancedRetriever(embedding_manager)
    
    # Add documents
    try:
        docs = [MockDocument(d) for d in SAMPLE_DOCS]
        await retriever.add_documents(docs)
        print_result("Initialize all strategies", True, "All retrievers loaded")
    except Exception as e:
        print_result("Initialize all strategies", False, str(e))
        return
    
    # Test each strategy
    strategies = [
        (RetrievalStrategy.PARENT_DOCUMENT, "Parent Document"),
        (RetrievalStrategy.SENTENCE_WINDOW, "Sentence Window"),
        (RetrievalStrategy.MMR, "MMR"),
    ]
    
    for strategy, name in strategies:
        try:
            result = await retriever.retrieve("tomato cultivation", top_k=2, strategy=strategy)
            passed = len(result.nodes) > 0
            print_result(f"{name} strategy", passed,
                        f"{len(result.nodes)} results, {result.processing_time_ms:.1f}ms")
        except Exception as e:
            print_result(f"{name} strategy", False, str(e))


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("   PHASE 3-4: QUERY PROCESSING & ENHANCED RETRIEVAL TESTS")
    print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    await test_query_processor()
    await test_parent_document_retriever()
    await test_sentence_window_retriever()
    await test_mmr_retriever()
    await test_enhanced_retriever()
    
    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)
    print("\n  🎉 Phase 3-4 components tested!")
    print("  📊 Query Processing: HyDE, Multi-Query, Step-Back, Decompose, Rewrite")
    print("  📊 Enhanced Retrieval: Parent Doc, Sentence Window, MMR")
    print("\n  Note: Tests run with mock LLM and embeddings.")
    print("  For production, configure with real LLM for better results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
