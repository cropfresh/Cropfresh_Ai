"""Test Qdrant Cloud connection."""
from qdrant_client import QdrantClient

print("🔗 Connecting to Qdrant Cloud...")

client = QdrantClient(
    url="https://33941042-8b02-48b3-8f31-f7a0fc3ebef3.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.OP8l8EX3Rf4ncdh_eDUfOoAgUMpAgm9HcCeWsJsxyzc",
)

print("✅ Connected successfully!")

collections = client.get_collections()
print(f"📚 Collections: {collections}")

# Try to get or create the agri_knowledge collection
from qdrant_client.models import Distance, VectorParams

collection_name = "agri_knowledge"
existing = [c.name for c in collections.collections]
print(f"📋 Existing collections: {existing}")

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
print(f"📊 Collection info: {info}")
