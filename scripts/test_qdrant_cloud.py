"""Test Qdrant Cloud connection — reads credentials from .env."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

host = os.getenv("QDRANT_HOST", "")
port = int(os.getenv("QDRANT_PORT", "6333"))
api_key = os.getenv("QDRANT_API_KEY", "")

if not host or not api_key:
    print("❌ QDRANT_HOST or QDRANT_API_KEY not set in .env")
    sys.exit(1)

print("🔗 Connecting to Qdrant Cloud...")
print(f"   Host: {host}")

client = QdrantClient(
    url=f"{host}:{port}",
    api_key=api_key,
)

print("✅ Connected successfully!")

collections = client.get_collections()
existing = [c.name for c in collections.collections]
print(f"📚 Collections: {existing}")

# Try to get or create the agri_knowledge collection
from qdrant_client.models import Distance, VectorParams

collection_name = "agri_knowledge"

if collection_name not in existing:
    print(f"🆕 Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{collection_name}' created!")
else:
    print(f"✅ Collection '{collection_name}' already exists!")

# Get collection info
info = client.get_collection(collection_name)
print(f"📊 Points count: {info.points_count}")
print(f"📊 Status: {info.status}")
print(f"\n🎉 Qdrant Cloud test complete!")
