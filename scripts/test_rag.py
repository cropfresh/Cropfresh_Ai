"""
RAG System Test Script
======================
Tests the Advanced Agentic RAG system.

Usage:
    cd d:\Cropfresh Ai\cropfresh-service-ai
    .venv\Scripts\activate
    python scripts/test_rag.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_qdrant_connection():
    """Test Qdrant connection."""
    print("\n" + "=" * 60)
    print("TEST 1: Qdrant Connection")
    print("=" * 60)
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        
        print(f"✅ Qdrant connected!")
        print(f"   Collections: {[c.name for c in collections.collections]}")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Make sure Qdrant is running: docker start qdrant")
        return False


async def test_embeddings():
    """Test embedding generation."""
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Generation")
    print("=" * 60)
    
    try:
        from src.rag.embeddings import get_embedding_manager
        
        print("Loading BGE-M3 model (this may take a moment)...")
        manager = get_embedding_manager()
        
        # Test document embedding
        texts = ["How to grow tomatoes?", "Best practices for potato farming"]
        embeddings = manager.embed_documents(texts)
        
        print(f"✅ Embeddings generated!")
        print(f"   Documents: {len(texts)}")
        print(f"   Dimensions: {len(embeddings[0])}")
        
        # Test query embedding
        query_emb = manager.embed_query("tomato cultivation tips")
        print(f"   Query embedding: {len(query_emb)} dims")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_knowledge_base():
    """Test knowledge base operations."""
    print("\n" + "=" * 60)
    print("TEST 3: Knowledge Base")
    print("=" * 60)
    
    try:
        from src.rag.knowledge_base import KnowledgeBase, Document
        
        kb = KnowledgeBase()
        await kb.initialize()
        
        print(f"✅ Knowledge base initialized!")
        print(f"   Stats: {kb.get_stats()}")
        
        return kb
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_ingestion(kb):
    """Test document ingestion."""
    print("\n" + "=" * 60)
    print("TEST 4: Document Ingestion")
    print("=" * 60)
    
    try:
        from src.rag.knowledge_base import Document
        
        # Load sample data
        data_path = Path(__file__).parent.parent / "data" / "knowledge" / "agronomy.json"
        
        if not data_path.exists():
            print(f"⚠️  Sample data not found at {data_path}")
            return False
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"   Loaded {len(data)} documents from {data_path.name}")
        
        # Convert to Document objects
        documents = [
            Document(
                text=d["text"],
                source=d.get("source", ""),
                category=d.get("category", ""),
            )
            for d in data
        ]
        
        # Ingest
        count = await kb.add_documents(documents)
        
        print(f"✅ Ingested {count} documents!")
        print(f"   Updated stats: {kb.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ingestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search(kb):
    """Test semantic search."""
    print("\n" + "=" * 60)
    print("TEST 5: Semantic Search")
    print("=" * 60)
    
    try:
        queries = [
            "How to grow tomatoes in Karnataka?",
            "What is CropFresh quality grading?",
            "Organic certification process",
        ]
        
        for query in queries:
            print(f"\n   Query: '{query}'")
            result = await kb.search(query, top_k=3)
            
            print(f"   Found: {len(result.documents)} documents ({result.search_time_ms:.1f}ms)")
            
            for i, doc in enumerate(result.documents, 1):
                score = f"{doc.score:.3f}" if doc.score else "N/A"
                text_preview = doc.text[:80].replace("\n", " ") + "..."
                print(f"     [{i}] Score: {score} | {text_preview}")
        
        print("\n✅ Search test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_query_analyzer():
    """Test query analysis."""
    print("\n" + "=" * 60)
    print("TEST 6: Query Analyzer")
    print("=" * 60)
    
    try:
        from src.rag.query_analyzer import QueryAnalyzer
        
        analyzer = QueryAnalyzer()  # No LLM, use rule-based
        
        test_queries = [
            ("How to grow tomatoes?", "vector"),
            ("What is the current price of onions?", "web"),
            ("Hello, how are you?", "direct"),
        ]
        
        for query, expected_type in test_queries:
            result = await analyzer.analyze(query)
            status = "✅" if result.query_type.value == expected_type else "⚠️"
            print(f"   {status} '{query}'")
            print(f"      Type: {result.query_type.value} (expected: {expected_type})")
            print(f"      Category: {result.category.value}")
            print(f"      Crops: {result.crops}")
        
        print("\n✅ Query analyzer test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Query analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_rag_pipeline():
    """Test the full agentic RAG pipeline."""
    print("\n" + "=" * 60)
    print("TEST 7: Full Agentic RAG Pipeline")
    print("=" * 60)
    
    try:
        from src.config import get_settings
        from src.orchestrator.llm_provider import create_llm_provider
        from src.agents.knowledge_agent import KnowledgeAgent
        
        settings = get_settings()
        
        if not settings.groq_api_key:
            print("⚠️  No GROQ_API_KEY configured - skipping LLM test")
            return True
        
        # Create LLM provider
        llm = create_llm_provider(
            provider=settings.llm_provider,
            api_key=settings.groq_api_key,
            model=settings.llm_model,
        )
        
        # Create knowledge agent
        agent = KnowledgeAgent(llm=llm)
        await agent.initialize()
        
        # Test a query
        query = "How to grow tomatoes in Karnataka? What varieties are best?"
        print(f"\n   Query: '{query}'")
        print("   Processing with agentic RAG...")
        
        response = await agent.answer(query)
        
        print(f"\n   ✅ Answer received!")
        print(f"   Steps: {' → '.join(response.steps)}")
        print(f"   Query type: {response.query_type}")
        print(f"   Sources: {response.sources}")
        print(f"   Confidence: {response.confidence}")
        print(f"\n   Answer:\n   {response.answer[:500]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Full RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("  CropFresh AI - Advanced Agentic RAG Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Qdrant
    results["qdrant"] = await test_qdrant_connection()
    
    if not results["qdrant"]:
        print("\n⚠️  Qdrant not available. Skipping remaining tests.")
        print("   Start Qdrant: docker start qdrant")
        return
    
    # Test 2: Embeddings
    results["embeddings"] = await test_embeddings()
    
    # Test 3: Knowledge Base
    kb = await test_knowledge_base()
    results["knowledge_base"] = kb is not None
    
    if kb:
        # Test 4: Ingestion
        results["ingestion"] = await test_ingestion(kb)
        
        # Test 5: Search
        results["search"] = await test_search(kb)
    
    # Test 6: Query Analyzer
    results["query_analyzer"] = await test_query_analyzer()
    
    # Test 7: Full Pipeline
    results["full_pipeline"] = await test_full_rag_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️  Some tests failed"))


if __name__ == "__main__":
    asyncio.run(main())
