# Database Schema — CropFresh AI

## Supabase (PostgreSQL)

### users
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| phone | VARCHAR(15) | Phone number (unique) |
| name | VARCHAR(100) | Full name |
| role | ENUM | farmer, buyer, admin |
| district | VARCHAR(50) | Karnataka district |
| language | VARCHAR(10) | Preferred language (kn/en/hi) |
| created_at | TIMESTAMP | Registration time |

### crops
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| name_en | VARCHAR(100) | English name |
| name_kn | VARCHAR(100) | Kannada name |
| category | VARCHAR(50) | vegetable, fruit, grain, spice |
| season | VARCHAR(50) | kharif, rabi, zaid |

### listings
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| farmer_id | UUID | → users.id |
| crop_id | UUID | → crops.id |
| quantity_kg | DECIMAL | Available quantity |
| price_per_kg | DECIMAL | Asking price |
| grade | ENUM | A, B, C |
| image_urls | TEXT[] | Crop photos |
| status | ENUM | active, sold, expired |
| created_at | TIMESTAMP | Listing time |

### orders
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| listing_id | UUID | → listings.id |
| buyer_id | UUID | → users.id |
| quantity_kg | DECIMAL | Ordered quantity |
| total_price | DECIMAL | Total amount |
| payment_status | ENUM | pending, paid, refunded |
| delivery_status | ENUM | pending, in_transit, delivered |

### prices
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| crop_id | UUID | → crops.id |
| mandi | VARCHAR(100) | APMC market name |
| price_min | DECIMAL | Min price/kg |
| price_max | DECIMAL | Max price/kg |
| price_modal | DECIMAL | Modal price/kg |
| date | DATE | Price date |

### agent_logs
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| agent_type | VARCHAR(50) | Agent name |
| input | JSONB | Agent input |
| output | JSONB | Agent output |
| latency_ms | INTEGER | Execution time |
| token_usage | JSONB | Token counts |
| created_at | TIMESTAMP | Log time |

## Qdrant Collections
- `crop_embeddings` — Crop listing vector embeddings
- `knowledge_base` — Agricultural knowledge vectors
- `price_history` — Historical price embeddings
- `farmer_queries` — Previous farmer query embeddings

## Neo4j Graph
- **Nodes**: Farmer, Buyer, Crop, Mandi, District
- **Relationships**: GROWS, BUYS, LISTED_AT, LOCATED_IN, MATCHED_WITH
