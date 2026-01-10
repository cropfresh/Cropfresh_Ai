"""Test Supabase connection."""
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings

settings = get_settings()

print("🔗 Testing Supabase Connection...")
print(f"   URL: {settings.supabase_url}")

if not settings.supabase_url or not settings.supabase_key:
    print("❌ Supabase credentials not set in .env")
    sys.exit(1)

try:
    from src.db.supabase_client import SupabaseClient
    
    client = SupabaseClient(
        url=settings.supabase_url,
        key=settings.supabase_key,
    )
    
    if client.health_check():
        print("✅ Supabase connection successful!")
    else:
        print("⚠️ Supabase connection may have issues")
    
    # Note: Tables need to be created first via Supabase Dashboard
    print("\n📋 Next steps:")
    print("   1. Go to Supabase Dashboard → SQL Editor")
    print("   2. Run the schema from implementation_plan.md")
    print("   3. Then test CRUD operations")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)
