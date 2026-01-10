"""
Qdrant Data Population Script
=============================
Populates Qdrant with sample agricultural knowledge for testing.

Usage:
    cd d:\\Cropfresh Ai\\cropfresh-service-ai
    .venv\\Scripts\\python scripts\\populate_qdrant.py

Author: CropFresh AI Team
Version: 2.0.0
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Sample agricultural knowledge base data
AGRONOMY_DATA = [
    {
        "title": "Tomato Cultivation in Karnataka",
        "category": "agronomy",
        "content": """Tomato (Solanum lycopersicum) is one of the most important vegetable crops in Karnataka.

**Recommended Varieties:**
- Arka Vikas: High yielding, heat tolerant, suitable for summer
- Arka Abhijit: Disease resistant, good for monsoon season
- Pusa Ruby: Early maturing, good for fresh market
- Arka Rakshak: Triple disease resistant (ToLCV, bacterial wilt, early blight)

**Growing Season:**
- Kharif (June-July): High risk due to heavy rains, use raised beds
- Rabi (October-November): Best season, optimal temperatures
- Summer (January-February): Requires irrigation, heat-tolerant varieties

**Soil Requirements:**
- Well-drained sandy loam to clay loam
- pH: 6.0-7.0
- Rich in organic matter (add 10-15 tons/acre FYM)

**Spacing:**
- Row to row: 60-75 cm
- Plant to plant: 45-60 cm
- Plants per acre: ~8,000-10,000

**Irrigation:**
- Drip irrigation recommended (saves 40% water)
- Critical stages: Flowering, fruit set, fruit development
- Frequency: Every 2-3 days in summer, 5-7 days in winter

**Expected Yield:** 15-25 tons/acre (good management)

**Common Issues:**
- Fruit cracking (irregular watering)
- Blossom end rot (calcium deficiency)
- Sun scald (excessive heat)""",
        "source": "UAS Bangalore Agricultural Extension"
    },
    {
        "title": "Pest Management in Tomato",
        "category": "agronomy",
        "content": """**Major Pests in Tomato:**

1. **Fruit Borer (Helicoverpa armigera)**
   - Symptoms: Holes in fruits, frass visible
   - Management:
     - Install pheromone traps (5/acre)
     - Spray Bacillus thuringiensis (Bt) @ 1g/L
     - Use NPV (Nuclear Polyhedrosis Virus) @ 250 LE/acre
     - Chemical: Spinosad 45 SC @ 0.3 ml/L

2. **Whitefly (Bemisia tabaci)**
   - Symptoms: Leaf yellowing, sticky honeydew, sooty mold
   - Vector for ToLCV (Tomato Leaf Curl Virus)
   - Management:
     - Yellow sticky traps (10/acre)
     - Neem oil @ 5 ml/L weekly spray
     - Imidacloprid 17.8 SL @ 0.3 ml/L

3. **Thrips**
   - Symptoms: Silvery patches on leaves, curled leaves
   - Management:
     - Blue sticky traps
     - Spinosad @ 0.5 ml/L
     - Fipronil 5 SC @ 1.5 ml/L

4. **Red Spider Mite**
   - Symptoms: Fine webbing, bronzing of leaves
   - Management:
     - Spray water to increase humidity
     - Propargite 57 EC @ 2 ml/L
     - Abamectin 1.9 EC @ 0.5 ml/L

**IPM Strategy:**
- Regular scouting (twice weekly)
- Use resistant varieties
- Maintain field hygiene
- Rotate chemicals to prevent resistance""",
        "source": "ICAR-IIHR Pest Management Guidelines"
    },
    {
        "title": "Onion Cultivation Best Practices",
        "category": "agronomy",
        "content": """**Onion (Allium cepa) Cultivation Guide for Karnataka**

**Varieties:**
- Kharif: Arka Niketan, Arka Kalyan, Bhima Super
- Rabi: Bellary Red, Arka Pragati, Pusa Red
- Late Kharif: N-53, Agrifound Light Red

**Nursery Management:**
- Seed rate: 8-10 kg/acre
- Nursery bed: 3m x 1m raised beds
- Transplant after 6-8 weeks (seedling height 15-20 cm)

**Field Preparation:**
- Deep ploughing + 2-3 harrowing
- Add FYM: 10-12 tons/acre
- Ridges and furrows at 30cm spacing

**Planting:**
- Spacing: 15 x 10 cm
- Planting depth: 2-3 cm
- Best transplanting time: Evening hours

**Fertilizer Schedule:**
- Basal: DAP 100 kg + MOP 75 kg/acre
- 30 DAT: Urea 50 kg/acre
- 45 DAT: Urea 50 kg/acre

**Irrigation:**
- Light but frequent irrigation
- Critical: Bulb initiation, bulb development
- Stop irrigation 15 days before harvest

**Harvesting:**
- When 50-75% tops fall naturally
- Cure in shade for 3-5 days
- Store in well-ventilated structures

**Yield:** 120-150 quintals/acre""",
        "source": "NHRDF Onion Production Guide"
    },
    {
        "title": "Organic Farming Practices",
        "category": "agronomy",
        "content": """**Organic Farming Methods for Vegetables**

**Soil Health:**
1. Composting
   - Heap method: Layer green + brown material
   - Vermicompost: Use Eisenia fetida earthworms
   - Application: 2-3 tons/acre

2. Green Manuring
   - Grow Dhaincha, Sunnhemp before main crop
   - Incorporate at 50% flowering (45 days)
   - Adds 25-30 kg N/acre

3. Biofertilizers
   - Rhizobium (for legumes): 200g/acre
   - Azospirillum: 2 kg/acre
   - Phosphobacteria: 2 kg/acre
   - Trichoderma: 2.5 kg/acre mixed with FYM

**Pest Management (Organic):**
1. Neem formulations
   - Neem oil: 3-5 ml/L
   - Neem seed kernel extract: 5%
   - Azadirachtin 0.15% EC: 2.5 ml/L

2. Botanical pesticides
   - Panchagavya: 3% spray (fermented cow products)
   - Garlic-chilli extract: 2% spray
   - Ginger extract: 5% spray

3. Biological control
   - Trichogramma cards: 1 lakh eggs/acre
   - Chrysoperla: 5000 eggs/acre
   - Ladybird beetles for aphid control

**Certification:**
- Register with NPOP (National Programme for Organic Production)
- Conversion period: 2-3 years
- Premium price: 20-50% above conventional""",
        "source": "Karnataka State Organic Mission"
    },
    {
        "title": "Drip Irrigation for Vegetables",
        "category": "agronomy",
        "content": """**Drip Irrigation Systems for Vegetable Crops**

**Benefits:**
- 30-50% water saving compared to flood irrigation
- 20-30% yield increase
- Reduced weed growth
- Fertigation possible (fertilizer through drip)
- Reduced disease incidence

**System Components:**
1. Water source (bore well, tank)
2. Pump (1-3 HP based on area)
3. Filters (sand + screen/disc)
4. Main line (PVC 63-75 mm)
5. Sub-main (PVC 40-50 mm)
6. Laterals (16 mm LLDPE)
7. Emitters/drippers (2-4 LPH)

**Dripper Spacing:**
- Tomato: 40-50 cm
- Brinjal: 50-60 cm
- Chilli: 30-40 cm
- Cucumber: 50-60 cm

**Operating Hours:**
| Crop | Stage | Hours/day |
|------|-------|-----------|
| Tomato | Vegetative | 1-1.5 |
| Tomato | Flowering | 2-2.5 |
| Tomato | Fruiting | 2.5-3 |

**Maintenance:**
- Daily: Check pressure, clean filters
- Weekly: Flush laterals
- Monthly: Check emitter discharge
- Annually: Replace filters, check valves

**Fertigation Schedule (Tomato):**
- Vegetative: 19:19:19 @ 2 kg/acre/day
- Flowering: 13:0:45 @ 2 kg/acre/day
- Fruiting: 0:0:50 @ 2 kg/acre/day

**Subsidy:**
- Available through PM-KUSUM, PMKSY
- 55-80% subsidy for small farmers
- Apply through: https://pmksy.gov.in""",
        "source": "IIHR Water Management Division"
    }
]

MARKET_DATA = [
    {
        "title": "Understanding Mandi Pricing",
        "category": "market",
        "content": """**How Agricultural Market (Mandi) Pricing Works**

**Price Components:**
1. **Modal Price**: Most frequent price in market
2. **Minimum Price**: Lowest traded price
3. **Maximum Price**: Highest traded price

**Factors Affecting Prices:**
- Season (supply from harvest)
- Quality grade (A/B/C)
- Market arrivals (tons per day)
- Storage availability
- Weather conditions
- Festival demand

**Karnataka Major Mandis:**
1. Kolar (Tomato hub) - 500+ MT daily
2. Ramanagara (Silk & vegetables)
3. Hubballi (North Karnataka hub)
4. Belgaum (Border trade)
5. Yeshwanthpur (Bangalore wholesale)

**Quality Grading:**
- Grade A (Premium): <5% defects, uniform size, fresh
- Grade B (Standard): 5-15% defects, mixed sizes
- Grade C (Economy): >15% defects, suitable for processing

**Best Selling Practices:**
1. Harvest early morning
2. Clean and grade properly
3. Use proper packaging (crates vs gunny bags)
4. Reach market by 6-8 AM
5. Build relationships with commission agents
6. Track prices on Agmarknet before selling

**AISP (All-Inclusive Sourcing Price):**
For buyers on CropFresh:
- Farmer Payout: Base price × Quantity
- Logistics: ₹2-4/kg based on distance
- Handling: ₹0.50/kg
- Platform Fee: 4-8% based on quantity

**Price Trend Analysis:**
- Tomato: Low in Dec-Jan (peak harvest), High in June-July (off-season)
- Onion: Low in Nov-Dec, High in June-Aug (storage period)
- Potato: Low in Feb-Mar, High in Sep-Oct""",
        "source": "Agmarknet Market Intelligence"
    },
    {
        "title": "When to Sell Vegetables",
        "category": "market",
        "content": """**Sell vs Hold Decision Guide**

**SELL When:**
1. Price is 20%+ above modal price
2. Quality is deteriorating (perishables)
3. Storage costs exceed expected gains
4. Bulk arrivals expected (after rain breaks)
5. Market is at seasonal peak

**HOLD When:**
1. Price is below modal price
2. Good storage available (onion, potato)
3. Festival demand approaching
4. Off-season premium expected
5. Quality can be maintained

**Seasonal Price Patterns:**

**Tomato:**
| Month | Price Trend | Action |
|-------|-------------|--------|
| Dec-Jan | Low (peak supply) | HOLD if storage possible |
| Feb-Mar | Moderate | SELL gradually |
| Apr-May | Rising | SELL before summer drop |
| Jun-Aug | High (off-season) | SELL - best prices |
| Sep-Nov | Declining | SELL early |

**Onion:**
- Store for 3-4 months for 50-100% price increase
- Best storage: Well-ventilated, elevated structures
- Weight loss: ~20-30% during storage

**Potato:**
- Cold storage essential (₹60-80/quintal/month)
- Can store 6-8 months
- Sell before next harvest starts

**Quick Tips:**
- Check Agmarknet/e-NAM before selling
- Compare prices across nearby mandis
- Consider transport cost vs price difference
- Build relationships with multiple buyers
- Use CropFresh for transparent pricing""",
        "source": "NABARD Market Linkage Guidelines"
    }
]

PLATFORM_DATA = [
    {
        "title": "CropFresh Farmer Registration Guide",
        "category": "platform",
        "content": """**How to Register as a Farmer on CropFresh**

**Step 1: Download the App**
- Play Store: Search "CropFresh Farmer"
- App Store: Search "CropFresh"
- Or visit: https://cropfresh.ai/download

**Step 2: Choose Account Type**
- Select "I am a Farmer"
- Enter mobile number
- Verify with OTP (6-digit code)

**Step 3: KYC Verification**
- Aadhaar Number: For identity verification
- Bank Account: IFSC code + Account number
- Land Documents (optional): For premium verification

**Step 4: Profile Setup**
- Farm location (village, taluk, district)
- Farm size (in acres)
- Crops you grow
- Preferred language (English/Hindi/Kannada)

**Verification Timeline:**
- Basic: Instant (can browse prices)
- Full KYC: 24-48 hours (can sell produce)

**Benefits After Registration:**
✅ View live market prices
✅ List your produce with photos
✅ Receive bids from verified buyers
✅ Direct payments to bank (T+2)
✅ Access to Prashna Krishi AI assistant
✅ Price alerts on your crops

**Support:**
📞 Helpline: 1800-XXX-XXXX (toll-free)
📧 Email: support@cropfresh.ai
💬 WhatsApp: +91-XXXXX-XXXXX""",
        "source": "CropFresh User Guide v2.0"
    },
    {
        "title": "Understanding CropFresh Quality Grades",
        "category": "platform",
        "content": """**CropFresh Quality Grading System**

**Grade A - Premium Quality**
- Appearance: Uniform size, shape, color
- Defects: Less than 5%
- Freshness: Harvested within 24 hours
- Packaging: Clean crates/boxes
- Price: 15-25% premium over Grade B

**Grade B - Standard Quality**
- Appearance: Slight variations acceptable
- Defects: 5-15%
- Freshness: Harvested within 48 hours
- Packaging: Clean gunny bags acceptable
- Price: Market modal price

**Grade C - Economy Quality**
- Appearance: Mixed sizes, minor blemishes
- Defects: 15-25%
- Best for: Processing, local retail
- Price: 10-20% below Grade B

**Digital Twin QR Code:**
Every produce batch gets a unique QR code containing:
- Farm location (GPS coordinates)
- Farmer details (verified)
- Harvest date and time
- Quality grade
- Lab test results (if applicable)
- Logistics chain

**How Quality is Assessed:**
1. Photo upload by farmer
2. AI-assisted grading (initial)
3. Physical inspection at collection
4. Buyer verification at delivery

**Quality Disputes:**
- Report within 4 hours of delivery
- Photo evidence required
- Refund/credit within 48 hours
- Fair resolution guaranteed""",
        "source": "CropFresh Quality Standards"
    },
    {
        "title": "CropFresh Payment & Settlements",
        "category": "platform",
        "content": """**Payment Process on CropFresh**

**For Farmers (Receiving Payments):**

**Settlement Timeline:**
- T+0: Order confirmed, produce handed over
- T+1: Quality verified at delivery
- T+2: Payment credited to bank account

**Payment Methods:**
- Direct bank transfer (NEFT/IMPS)
- UPI (for amounts <₹1 lakh)

**Transaction View:**
In app: Profile → My Transactions
Shows:
- Order ID
- Quantity sold
- Price per kg
- Total amount
- Deductions (if any)
- Net payout
- Settlement status

**Common Deductions:**
- Quality downgrade: Based on actual vs declared grade
- Weight variance: If >2% difference
- Damage during transport: Rare, insurance covered

**Tips for Faster Payment:**
✅ Ensure correct bank details
✅ Accurate quantity declaration
✅ Proper grading before listing
✅ Use recommended packaging

**Payment Issues:**
If payment not received by T+3:
1. Check app for settlement status
2. Verify bank account details
3. Contact support with Order ID
4. Escalation: payments@cropfresh.ai

**For Buyers (Making Payments):**
- Prepaid wallet: 5% bonus on top-up
- Credit line: Available for verified businesses
- Payment terms: Advance or COD based on history""",
        "source": "CropFresh Finance & Settlements"
    }
]

GENERAL_DATA = [
    {
        "title": "About CropFresh",
        "category": "general",
        "content": """**CropFresh: Empowering Indian Agriculture**

**Mission:**
Connect farmers directly with buyers, eliminating middlemen and ensuring fair prices for all.

**Founded:** 2024
**Headquarters:** Bangalore, Karnataka

**What We Offer:**

**For Farmers:**
🌾 Direct market access to 10,000+ buyers
💰 Fair, transparent pricing (no hidden cuts)
📱 Easy-to-use mobile app in local languages
🤖 AI-powered Prashna Krishi assistant
🚚 Hassle-free logistics
💳 Fast payments (T+2 settlement)

**For Buyers:**
🥬 Fresh produce directly from farms
✅ Quality-graded with Digital Twin traceability
📊 Transparent AISP pricing
🔍 Farm-level visibility
📦 Reliable delivery

**Technology:**
- AI grading for quality assessment
- Blockchain-inspired traceability
- Voice assistants in Hindi, Kannada, Telugu
- Real-time price intelligence

**Impact (2025):**
- 50,000+ registered farmers
- 5,000+ active buyers
- ₹500 Cr+ traded volume
- 100+ districts covered

**Awards:**
- Agritech Innovation Award 2025
- Best Farmer Empowerment Platform

**Coverage:**
Currently: Karnataka, Andhra Pradesh, Tamil Nadu
Coming soon: Maharashtra, Gujarat, UP""",
        "source": "CropFresh Corporate Overview"
    },
    {
        "title": "Prashna Krishi AI Assistant",
        "category": "general",
        "content": """**Prashna Krishi (प्रश्न कृषि) - Your AI Farming Assistant**

**What is Prashna Krishi?**
An AI-powered assistant that helps farmers with:
- Crop cultivation advice
- Pest and disease identification
- Market prices and sell/hold recommendations
- CropFresh app guidance
- Voice support in local languages

**How to Access:**
- In CropFresh app: Tap the chat icon
- WhatsApp: +91-XXXXX-XXXXX
- Voice call: Available in app

**Languages Supported:**
- English
- Hindi (हिंदी)
- Kannada (ಕನ್ನಡ)
- Telugu (తెలుగు) - Coming soon

**What You Can Ask:**

📘 **Farming Advice:**
"How to grow tomatoes?"
"What fertilizer for onion?"
"Pest on my cabbage leaves"

💰 **Market Prices:**
"Current tomato price in Kolar?"
"Should I sell my onion now?"
"Calculate AISP for 200kg potato"

📱 **App Help:**
"How to register?"
"Where is my payment?"
"How to list my produce?"

**Tips for Best Results:**
- Be specific about your crop and location
- Share photos for pest identification
- Mention quantity for price calculations

**Prashna Krishi is available 24/7, even offline!**
Basic features work without internet.""",
        "source": "CropFresh Product Documentation"
    }
]


async def populate_qdrant():
    """Populate Qdrant with sample agricultural data."""
    print("=" * 60)
    print("  CropFresh AI - Qdrant Data Population")
    print("=" * 60)
    
    try:
        from src.config import get_settings
        from src.rag.knowledge_base import KnowledgeBase, Document
        
        settings = get_settings()
        print(f"\n📦 Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}...")
        
        # Initialize knowledge base with API key for cloud
        kb = KnowledgeBase(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        await kb.initialize()
        print("✅ Connected to Qdrant")
        
        # Combine all data
        all_data = AGRONOMY_DATA + MARKET_DATA + PLATFORM_DATA + GENERAL_DATA
        print(f"\n📄 Preparing {len(all_data)} documents for ingestion...")
        
        # Create documents
        documents = []
        for item in all_data:
            doc = Document(
                text=f"# {item['title']}\n\n{item['content']}",
                source=item["source"],
                category=item["category"],
                metadata={
                    "title": item["title"],
                    "category": item["category"],
                },
            )
            documents.append(doc)
        
        # Ingest documents
        print("\n🔄 Ingesting documents (this may take a minute)...")
        success = await kb.add_documents(documents)
        
        if success:
            print(f"✅ Successfully ingested {len(documents)} documents!")
        else:
            print("⚠️  Some documents may have failed to ingest")
        
        # Get stats
        stats = kb.get_stats()
        print(f"\n📊 Knowledge Base Stats:")
        print(f"   Total documents: {stats.get('total_documents', 'unknown')}")
        print(f"   Status: {stats.get('status', 'unknown')}")
        
        # Test search
        print("\n🔍 Testing search...")
        test_queries = [
            ("How to grow tomatoes?", "agronomy"),
            ("Current market prices", "market"),
            ("How to register on CropFresh", "platform"),
        ]
        
        for query, expected_cat in test_queries:
            result = await kb.search(query, top_k=2)
            if result.documents:
                top_doc = result.documents[0]
                status = "✅" if top_doc.category == expected_cat else "⚠️"
                print(f"   {status} '{query}' → {top_doc.category} (score: {top_doc.score:.2f})")
            else:
                print(f"   ❌ '{query}' → No results")
        
        print("\n🎉 Qdrant population complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("   1. Ensure Qdrant is running: docker start qdrant")
        print("   2. Or start fresh: docker run -d -p 6333:6333 --name qdrant qdrant/qdrant")
        print("   3. Check QDRANT_HOST and QDRANT_PORT in .env")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(populate_qdrant())
    sys.exit(0 if success else 1)
