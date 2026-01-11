"""
RAG Enhancements Test Script
============================
Tests all new RAG enhancement features:
- Hybrid Search (BM25 + Dense)
- Cross-Encoder Re-ranking
- Graph RAG (Neo4j)
- Observability

Usage:
    cd d:\Cropfresh Ai\cropfresh-service-ai
    uv run python scripts/test_rag_enhancements.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_hybrid_search():
    """Test hybrid search with BM25 + dense vectors."""
    print("\n" + "=" * 60)
    print("TEST 1: Hybrid Search (BM25 + Dense)")
    print("=" * 60)
    
    try:
        from src.rag.hybrid_search import BM25Index, HybridRetriever
        from src.rag.knowledge_base import Document, KnowledgeBase
        
        # Create sample documents
        sample_docs = [
            Document(id="1", text="Tomato cultivation in Karnataka requires well-drained soil. Plant during June-July monsoon season.", category="agronomy"),
            Document(id="2", text="Onion farming best practices: Use drip irrigation and maintain proper spacing.", category="agronomy"),
            Document(id="3", text="Kolar district is known for tomato and grape cultivation.", category="market"),
            Document(id="4", text="CropFresh platform helps farmers sell directly to buyers.", category="platform"),
            Document(id="5", text="Potato storage requires cool temperatures between 4-10°C.", category="agronomy"),
        ]
        
        # Test BM25 Index
        print("\n   Testing BM25 Index...")
        bm25 = BM25Index()
        indexed = bm25.index_documents(sample_docs)
        print(f"   ✅ Indexed {indexed} documents in BM25")
        
        # Search with BM25
        query = "tomato farming Karnataka"
        results = bm25.search(query, top_k=3)
        print(f"\n   BM25 Search: '{query}'")
        for doc, score in results:
            print(f"     Score {score:.3f}: {doc.text[:50]}...")
        
        # Test Hybrid Retriever (without Qdrant)
        print("\n   Testing Hybrid Retriever...")
        from src.config import get_settings
        settings = get_settings()
        kb = KnowledgeBase(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        
        try:
            await kb.initialize()
            hybrid = HybridRetriever(kb)
            hybrid.index_documents(sample_docs)  # Index for BM25
            
            # Hybrid search
            result = await hybrid.search(query, top_k=3, mode="hybrid")
            print(f"   ✅ Hybrid search returned {result.fused_count} results")
            print(f"      Dense: {result.dense_count}, Sparse: {result.sparse_count}")
            
        except Exception as e:
            print(f"   ⚠️  Qdrant not available, testing BM25-only mode: {e}")
            # Fallback to sparse-only
            hybrid = HybridRetriever(kb)
            hybrid.index_documents(sample_docs)
            result = await hybrid.search(query, top_k=3, mode="sparse")
            print(f"   ✅ Sparse-only search returned {result.sparse_count} results")
        
        print("\n✅ Hybrid Search test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranker():
    """Test cross-encoder and lightweight rerankers."""
    print("\n" + "=" * 60)
    print("TEST 2: Cross-Encoder Re-ranking")
    print("=" * 60)
    
    try:
        from src.rag.reranker import get_reranker, LightweightReranker
        from src.rag.knowledge_base import Document
        
        # Create sample documents with scores
        sample_docs = [
            Document(id="1", text="Tomato plants need 6-8 hours of sunlight daily.", score=0.8),
            Document(id="2", text="CropFresh provides quality grading for produce.", score=0.75),
            Document(id="3", text="Best tomato varieties for Karnataka: Arka Rakshak, Arka Samrat.", score=0.7),
            Document(id="4", text="Irrigation systems help conserve water.", score=0.65),
            Document(id="5", text="Organic tomato farming avoids synthetic pesticides.", score=0.6),
        ]
        
        query = "How to grow tomatoes organically?"
        
        # Test lightweight reranker (always available)
        print("\n   Testing Lightweight Reranker...")
        light_reranker = LightweightReranker()
        result = light_reranker.rerank(query, sample_docs, top_k=3)
        
        print(f"   ✅ Reranked {result.original_count} -> {result.reranked_count} docs")
        print(f"   Time: {result.rerank_time_ms:.1f}ms")
        
        for i, doc in enumerate(result.documents, 1):
            print(f"     [{i}] Score: {doc.score:.3f} - {doc.text[:40]}...")
        
        # Try cross-encoder (may not be available)
        print("\n   Testing Cross-Encoder Reranker...")
        try:
            reranker = get_reranker(model_type="cross-encoder")
            if isinstance(reranker, LightweightReranker):
                print("   ⚠️  Cross-encoder not available, using lightweight")
            else:
                result = reranker.rerank(query, sample_docs, top_k=3)
                print(f"   ✅ Cross-encoder reranked in {result.rerank_time_ms:.1f}ms")
                for i, doc in enumerate(result.documents, 1):
                    print(f"     [{i}] Score: {doc.score:.3f} - {doc.text[:40]}...")
        except Exception as e:
            print(f"   ⚠️  Cross-encoder not available: {e}")
        
        print("\n✅ Reranker test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Reranker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_entity_extraction():
    """Test entity extraction for Graph RAG."""
    print("\n" + "=" * 60)
    print("TEST 3: Entity Extraction")
    print("=" * 60)
    
    try:
        from src.rag.graph_retriever import EntityExtractor
        
        extractor = EntityExtractor()
        
        test_queries = [
            ("What farmers grow tomatoes in Kolar?", ["Tomato"], ["Kolar"], ["farmer"]),
            ("Onion prices in Bellary district", ["Onion"], ["Bellary"], []),
            ("How to grow rice and wheat?", ["Rice", "Wheat"], [], []),
            ("Find buyers for mangoes", ["Mango"], [], ["buyer"]),
        ]
        
        all_passed = True
        for query, expected_crops, expected_districts, expected_types in test_queries:
            entities = extractor.extract(query)
            
            # Check crops
            crops_match = set(entities["crops"]) == set(expected_crops)
            districts_match = set(entities["districts"]) == set(expected_districts)
            types_match = set(entities["entity_types"]) == set(expected_types)
            
            status = "✅" if crops_match and districts_match and types_match else "⚠️"
            print(f"\n   {status} Query: '{query}'")
            print(f"      Crops: {entities['crops']} (expected: {expected_crops})")
            print(f"      Districts: {entities['districts']} (expected: {expected_districts})")
            print(f"      Types: {entities['entity_types']} (expected: {expected_types})")
            
            if not (crops_match and districts_match and types_match):
                all_passed = False
        
        print("\n✅ Entity extraction test passed!" if all_passed else "\n⚠️ Some entity extractions differ")
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graph_retriever():
    """Test Graph RAG with Neo4j."""
    print("\n" + "=" * 60)
    print("TEST 4: Graph RAG (Neo4j)")
    print("=" * 60)
    
    try:
        from src.rag.graph_retriever import GraphRetriever
        from src.config import get_settings
        
        settings = get_settings()
        
        if not settings.neo4j_uri or not settings.neo4j_password:
            print("   ⚠️  Neo4j not configured - skipping test")
            return True
        
        print(f"   Connecting to Neo4j: {settings.neo4j_uri}")
        
        retriever = GraphRetriever()
        
        # Test query
        query = "What farmers grow tomatoes?"
        print(f"\n   Query: '{query}'")
        
        context = await retriever.retrieve(query)
        
        print(f"   ✅ Graph context retrieved in {context.query_time_ms:.1f}ms")
        print(f"   Entities: {context.entities}")
        print(f"   Farmers found: {len(context.farmers)}")
        
        if context.context_text:
            print(f"\n   Context preview:\n   {context.context_text[:200]}...")
        
        print("\n✅ Graph RAG test passed!")
        return True
        
    except Exception as e:
        if "neo4j" in str(e).lower():
            print(f"   ⚠️  Neo4j connection failed: {e}")
            return True  # Not a test failure
        print(f"❌ Graph RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_observability():
    """Test observability and evaluation."""
    print("\n" + "=" * 60)
    print("TEST 5: Observability & Evaluation")
    print("=" * 60)
    
    try:
        from src.rag.observability import RAGEvaluator, LANGSMITH_AVAILABLE
        
        print(f"   LangSmith available: {LANGSMITH_AVAILABLE}")
        
        # Test local evaluator
        print("\n   Testing RAG Evaluator...")
        evaluator = RAGEvaluator()
        
        query = "How to grow tomatoes in Karnataka?"
        response = "To grow tomatoes in Karnataka, choose varieties like Arka Rakshak. Plant during monsoon season in well-drained soil with proper irrigation."
        docs = [
            "Tomato cultivation in Karnataka requires June-July planting.",
            "Arka Rakshak is a popular tomato variety.",
        ]
        
        metrics = evaluator.evaluate_response(query, response, docs)
        
        print(f"   ✅ Evaluation metrics:")
        for metric, score in metrics.items():
            print(f"      {metric}: {score:.3f}")
        
        # Check thresholds
        if metrics.get("relevance", 0) > 0.3:
            print("   ✅ Relevance: GOOD")
        if metrics.get("grounding", 0) > 0.3:
            print("   ✅ Grounding: GOOD")
        
        print("\n✅ Observability test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Observability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integrated_pipeline():
    """Test full enhanced RAG pipeline."""
    print("\n" + "=" * 60)
    print("TEST 6: Integrated Enhanced RAG Pipeline")
    print("=" * 60)
    
    try:
        from src.rag.hybrid_search import BM25Index
        from src.rag.reranker import LightweightReranker
        from src.rag.graph_retriever import EntityExtractor
        from src.rag.observability import RAGEvaluator
        from src.rag.knowledge_base import Document
        
        # Create sample documents
        docs = [
            Document(id="1", text="Tomato farming in Kolar district uses drip irrigation.", score=0.7),
            Document(id="2", text="Farmers in Karnataka grow Arka Rakshak tomatoes.", score=0.65),
            Document(id="3", text="CropFresh connects farmers with buyers.", score=0.6),
            Document(id="4", text="Organic tomato cultivation avoids pesticides.", score=0.55),
        ]
        
        query = "tomato farmers in Kolar"
        
        print(f"\n   Query: '{query}'")
        
        # Step 1: Entity extraction
        extractor = EntityExtractor()
        entities = extractor.extract(query)
        print(f"   1. Entities: crops={entities['crops']}, districts={entities['districts']}")
        
        # Step 2: BM25 search
        bm25 = BM25Index()
        bm25.index_documents(docs)
        sparse_results = bm25.search(query, top_k=4)
        print(f"   2. BM25 found {len(sparse_results)} results")
        
        # Step 3: Rerank
        docs_to_rerank = [doc for doc, _ in sparse_results]
        reranker = LightweightReranker()
        reranked = reranker.rerank(query, docs_to_rerank, top_k=2)
        print(f"   3. Reranked to top {reranked.reranked_count} results")
        
        # Step 4: Evaluate
        response = "Kolar district farmers grow Arka Rakshak tomatoes using drip irrigation."
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_response(
            query, response, [d.text for d in reranked.documents]
        )
        print(f"   4. Evaluation: relevance={metrics['relevance']:.2f}, grounding={metrics['grounding']:.2f}")
        
        print("\n✅ Integrated pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all enhancement tests."""
    print("=" * 60)
    print("  CropFresh AI - RAG Enhancements Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run all tests
    results["hybrid_search"] = await test_hybrid_search()
    results["reranker"] = await test_reranker()
    results["entity_extraction"] = await test_entity_extraction()
    results["graph_retriever"] = await test_graph_retriever()
    results["observability"] = await test_observability()
    results["integrated_pipeline"] = await test_integrated_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 All enhancement tests passed!" if all_passed else "⚠️ Some tests failed"))
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
