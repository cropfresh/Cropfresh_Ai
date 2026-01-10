"""Test Neo4j AuraDB connection and create sample data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings

settings = get_settings()

print("🔗 Testing Neo4j AuraDB Connection...")
print(f"   URI: {settings.neo4j_uri}")

if not settings.neo4j_uri or not settings.neo4j_password:
    print("❌ Neo4j credentials not set in .env")
    sys.exit(1)

try:
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    
    # Test connectivity
    driver.verify_connectivity()
    print("✅ Neo4j connection successful!")
    
    # Create sample data
    print("\n📊 Creating sample graph data...")
    
    with driver.session() as session:
        # Create sample farmers and crops
        session.run("""
            // Create sample farmers
            MERGE (f1:Farmer {id: 'farmer-001', name: 'Ramesh Kumar', phone: '+919876543210'})
            MERGE (f2:Farmer {id: 'farmer-002', name: 'Lakshmi Devi', phone: '+919876543211'})
            MERGE (f3:Farmer {id: 'farmer-003', name: 'Venkatesh Gowda', phone: '+919876543212'})
            
            // Create crops
            MERGE (tomato:Crop {name: 'Tomato', category: 'Vegetables'})
            MERGE (onion:Crop {name: 'Onion', category: 'Vegetables'})
            MERGE (potato:Crop {name: 'Potato', category: 'Vegetables'})
            MERGE (rice:Crop {name: 'Rice', category: 'Grains'})
            
            // Create districts
            MERGE (kolar:District {name: 'Kolar'})
            MERGE (bellary:District {name: 'Bellary'})
            MERGE (mysore:District {name: 'Mysore'})
            
            // Create relationships
            MERGE (f1)-[:GROWS]->(tomato)
            MERGE (f1)-[:GROWS]->(onion)
            MERGE (f2)-[:GROWS]->(potato)
            MERGE (f2)-[:GROWS]->(tomato)
            MERGE (f3)-[:GROWS]->(rice)
            
            MERGE (f1)-[:LOCATED_IN]->(kolar)
            MERGE (f2)-[:LOCATED_IN]->(bellary)
            MERGE (f3)-[:LOCATED_IN]->(mysore)
            
            // Create sample buyers
            MERGE (b1:Buyer {id: 'buyer-001', name: 'Fresh Mart', business_type: 'Retail'})
            MERGE (b2:Buyer {id: 'buyer-002', name: 'Veggie Express', business_type: 'Wholesale'})
            
            MERGE (b1)-[:BUYS]->(tomato)
            MERGE (b1)-[:BUYS]->(onion)
            MERGE (b2)-[:BUYS]->(potato)
            MERGE (b2)-[:BUYS]->(tomato)
        """)
        
        # Get stats
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        
        print("\n📈 Graph Statistics:")
        for record in result:
            print(f"   {record['label']}: {record['count']}")
    
    driver.close()
    print("\n🎉 Neo4j setup complete!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
