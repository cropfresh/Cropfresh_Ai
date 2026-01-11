"""Quick test for knowledge base search."""
import asyncio
from src.rag.knowledge_base import KnowledgeBase
from src.config import get_settings

async def test():
    settings = get_settings()
    kb = KnowledgeBase(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
    )
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
