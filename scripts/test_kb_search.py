"""Quick test for knowledge base search."""
import asyncio
from src.rag.knowledge_base import KnowledgeBase

async def test():
    kb = KnowledgeBase()
    await kb.initialize()
    
    # Test search
    result = await kb.search("How to grow tomatoes", top_k=3)
    print(f"Search found {len(result.documents)} documents")
    for doc in result.documents:
        title = doc.metadata.get("title", doc.source)
        print(f"  - {title} (score: {doc.score:.2f})")
        print(f"    {doc.text[:100]}...")
    
    # Test stats
    stats = kb.get_stats()
    print(f"\nStats: {stats}")

asyncio.run(test())
