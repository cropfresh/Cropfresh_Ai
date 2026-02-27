# Data Flow — CropFresh AI

## How Data Moves Through the System

### 1. Data Collection (Scrapers → Storage)
```
APMC Websites ──┐
eNAM Portal ────┤──→ src/scrapers/ ──→ Supabase (prices table)
IMD Weather ────┤                  ──→ Qdrant (embeddings)
Agri News ──────┘                  ──→ Neo4j (relationships)
```

### 2. Farmer Crop Listing Flow
```
Farmer (WhatsApp/App)
    │
    ├── Voice message ──→ src/voice/stt.py ──→ Text (Kannada)
    ├── Crop photo ──→ AI Quality Agent ──→ Grade (A/B/C)
    │
    ▼
Crop Listing Agent (src/agents/)
    │
    ├── Classify crop type
    ├── Assess quality
    ├── Predict fair price
    │
    ▼
    ├── Supabase: listings table (structured data)
    ├── Qdrant: crop embedding (semantic search)
    └── Neo4j: farmer→crop→mandi graph
```

### 3. Buyer Search & Matching
```
Buyer search query
    │
    ▼
Buyer Matching Agent
    │
    ├── Qdrant: semantic similarity search
    ├── Neo4j: graph-based matching (proximity, history)
    ├── Scorer: relevance scoring
    │
    ▼
Matched listings (ranked) → Buyer notification
```

### 4. Price Prediction Pipeline
```
Real-time APMC data ──┐
Historical prices ────┤──→ Price Prediction Agent
Weather forecast ─────┤        │
Seasonal patterns ────┘        ▼
                          Price recommendation (₹/kg)
```
