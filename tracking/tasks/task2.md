# Task 2: Implement Matchmaking / Buyer Matching Agent

> **Priority:** 🔴 P0 | **Phase:** 1 | **Effort:** 3–4 days  
> **Files:** `src/agents/buyer_matching/agent.py`, `src/agents/supervisor_agent.py`  
> **Score Target:** 9/10 — Multi-factor scoring with geospatial optimization
> **Status:** ✅ Completed (2026-03-01)

---

## 📌 Problem Statement

`src/agents/buyer_matching/agent.py` exists (9.7KB) but lacks real matching logic. The business model requires intelligent farmer→buyer matching based on proximity, quality fit, demand signals, and price alignment.

---

## 🔬 Research Findings

### Optimal Matching Algorithm Architecture
1. **Multi-factor weighted scoring**: Combine 5+ signals into a single match score
2. **Geospatial proximity**: Haversine distance for GPS-based farm-to-buyer distance
3. **Demand signal matching**: Buyer order history → commodity preference scoring
4. **Quality grade alignment**: A+ buyer gets A+ farmers first
5. **Price fit**: `|buyer_budget - farmer_asking_price| / farmer_asking_price`

### Scoring Formula (Research-Backed)
```
match_score = (
    w1 × proximity_score +          # 30% — closer = better
    w2 × quality_match_score +      # 25% — grade alignment
    w3 × price_fit_score +          # 20% — budget alignment
    w4 × demand_signal_score +      # 15% — historical preference
    w5 × reliability_score          # 10% — farmer quality_score
)
```

### Geospatial Libraries
- **Haversine formula**: Fast, no external dependency, ±0.5% accuracy at <100km
- **GeoPandas + Rtree**: For spatial indexing when matching 1000+ farmers
- **HDBSCAN**: Density-based clustering for grouping nearby farms

### Advanced Patterns
- **Market Microstructure**: Order book matching (buy/sell sides) adapted for produce
- **Munkres (Hungarian) Algorithm**: Optimal bipartite matching when batch-processing
- **Genetic Algorithm**: For large-scale multi-objective optimization
- **Real-time scoring cache**: Redis-backed score cache with 5-minute TTL

---

## 🏗️ Implementation Spec

### 1. Match Score Calculator
```python
@dataclass
class MatchCandidate:
    listing_id: str
    farmer_id: str
    buyer_id: str
    match_score: float           # 0.0–1.0
    proximity_km: float
    quality_match: float         # 0.0–1.0
    price_fit: float             # 0.0–1.0
    demand_signal: float         # 0.0–1.0
    reliability: float           # 0.0–1.0
    estimated_delivery_hours: float
    estimated_logistics_cost: float

class MatchingEngine:
    """
    Multi-factor matching engine for CropFresh marketplace.
    
    Weights are tuned for Indian agricultural context:
    - Proximity matters most (perishables, logistics cost)
    - Quality grade must match buyer expectations
    - Price fit prevents rejected orders
    """
    
    WEIGHTS = {
        'proximity': 0.30,
        'quality': 0.25,
        'price_fit': 0.20,
        'demand_signal': 0.15,
        'reliability': 0.10,
    }
    
    def calculate_proximity_score(
        self, 
        farmer_lat: float, farmer_lon: float,
        buyer_lat: float, buyer_lon: float,
        max_distance_km: float = 100.0,
    ) -> float:
        """
        Haversine-based proximity score.
        Score = 1.0 at 0km, 0.0 at max_distance_km.
        Non-linear decay: closer distances score disproportionately higher.
        """
        distance = haversine(farmer_lat, farmer_lon, buyer_lat, buyer_lon)
        if distance >= max_distance_km:
            return 0.0
        # Exponential decay: score = e^(-distance/30)
        return math.exp(-distance / (max_distance_km * 0.3))
    
    def calculate_quality_match(
        self,
        listing_grade: str,       # 'A+', 'A', 'B', 'C'
        buyer_min_grade: str,     # Buyer's minimum acceptable grade
    ) -> float:
        """
        Grade alignment scoring.
        Perfect match = 1.0, one grade above = 0.9, below minimum = 0.0
        """
        grade_order = {'A+': 4, 'A': 3, 'B': 2, 'C': 1}
        listing_val = grade_order.get(listing_grade, 1)
        buyer_min_val = grade_order.get(buyer_min_grade, 1)
        
        if listing_val < buyer_min_val:
            return 0.0  # Below minimum — no match
        elif listing_val == buyer_min_val:
            return 1.0  # Exact match
        else:
            return 0.9  # Above minimum (slight penalty to prefer exact)
    
    def calculate_price_fit(
        self,
        asking_price: float,
        buyer_budget: float,
    ) -> float:
        """
        Price alignment. Score = 1.0 when asking ≤ budget.
        Penalize when asking > budget (but still show if within 15%).
        """
        if asking_price <= buyer_budget:
            return 1.0
        overshoot = (asking_price - buyer_budget) / buyer_budget
        if overshoot > 0.15:
            return 0.0  # Too expensive
        return max(0.0, 1.0 - (overshoot * 5))  # Linear penalty
    
    def calculate_demand_signal(
        self,
        commodity: str,
        buyer_order_history: list[dict],
    ) -> float:
        """
        How much this buyer historically orders this commodity.
        Based on frequency and recency of past orders.
        """
        relevant_orders = [o for o in buyer_order_history if o['commodity'] == commodity]
        if not relevant_orders:
            return 0.1  # Baseline for new commodities
        
        frequency_score = min(1.0, len(relevant_orders) / 10)
        recency_days = (datetime.now() - relevant_orders[-1]['date']).days
        recency_score = max(0.0, 1.0 - recency_days / 90)
        
        return 0.6 * frequency_score + 0.4 * recency_score
```

### 2. Batch Matching with Ranking
```python
async def find_matches(
    self,
    listing_id: str,
    max_results: int = 10,
    min_score: float = 0.3,
) -> list[MatchCandidate]:
    """
    Find top buyers for a listing.
    
    Algorithm:
    1. Fetch listing details (commodity, grade, location, price)
    2. Fetch eligible buyers (active, in range, matching commodity preference)
    3. Score each buyer × listing pair
    4. Sort by score descending, return top N
    
    Performance: O(N×M) for N listings × M buyers.
    For large scale: pre-filter with R-tree spatial index.
    """
```

### 3. Reverse Matching: Buyer Seeks Farmers
```python
async def find_farmers_for_buyer(
    self,
    buyer_id: str,
    commodity: str,
    quantity_needed_kg: float,
    max_price_per_kg: float,
    max_results: int = 10,
) -> list[MatchCandidate]:
    """
    Reverse matching: buyer specifies needs, find matching listings.
    Useful for scheduled procurement (e.g., restaurant daily orders).
    """
```

### 4. Integration with Supervisor
```python
# In supervisor_agent.py — add to routing table:
MATCHING_INTENTS = [
    "find buyer", "match buyer", "who wants to buy",
    "find farmer", "find supplier", "I need tomatoes",
    "buyer matching", "sell my produce",
]
```

---

## ✅ Acceptance Criteria (9/10 Score)

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Multi-factor scoring with 5 weighted signals | 25% |
| 2 | Haversine geospatial distance calculation correct | 15% |
| 3 | Quality grade alignment enforced (below min = 0 score) | 15% |
| 4 | Returns ≥1 valid match in unit test with mock data | 15% |
| 5 | Reverse matching (buyer → farmers) works | 10% |
| 6 | Wired into supervisor routing for matching intents | 10% |
| 7 | Redis cache for recent match results (5-min TTL) | 10% |

---

## 🧪 Test Cases

```python
def test_proximity_nearby_scores_high():
    """Farm 5km from buyer should score > 0.85"""
    engine = MatchingEngine()
    score = engine.calculate_proximity_score(12.97, 77.59, 12.98, 77.60)
    assert score > 0.85

def test_below_grade_returns_zero():
    """Buyer wants A, farmer has C → match score = 0"""
    engine = MatchingEngine()
    score = engine.calculate_quality_match('C', 'A')
    assert score == 0.0

def test_batch_matching_returns_ranked():
    """find_matches returns results sorted by score descending"""
    results = await engine.find_matches(listing_id="test")
    scores = [r.match_score for r in results]
    assert scores == sorted(scores, reverse=True)
```

---

## 📚 Dependencies
- `src/db/postgres_client.py` → `listings`, `buyers` tables
- `src/agents/pricing_agent.py` → for logistics cost estimates
- `redis` → match result caching

---

## ✅ Completion Update (2026-03-01)

### Implemented
- Upgraded `src/agents/buyer_matching/agent.py` with a production-ready matching engine:
  - `MatchingEngine` with 5 weighted signals:
    - proximity (30%)
    - quality (25%)
    - price fit (20%)
    - demand signal (15%)
    - reliability (10%)
  - Haversine proximity with non-linear/exponential decay scoring.
  - Grade alignment enforcement (`below minimum grade => 0.0` quality score).
  - Price fit scoring with budget overshoot penalties.
  - Demand signal scoring using buyer order history frequency + recency.
  - Reliability factor integrated into final score.
- Added matching workflows:
  - `find_matches(listing_id, max_results, min_score)` for listing → buyers.
  - `find_farmers_for_buyer(...)` for buyer → listings reverse matching.
- Added transparent candidate output fields:
  - `match_score`, `proximity_km`, `quality_match`, `price_fit`, `demand_signal`, `reliability`,
    `estimated_delivery_hours`, `estimated_logistics_cost`.
- Implemented caching for recent match results:
  - Redis-backed cache when available (`redis_url`)
  - local in-memory 5-minute TTL fallback
- Wired supervisor integration:
  - Added buyer matching intents and routing behavior in `src/agents/supervisor_agent.py`.
  - Registered `buyer_matching_agent` in both chat route initializers:
    - `src/api/routes/chat.py`
    - `src/api/routers/chat.py`

### Validation
- Updated tests:
  - `tests/unit/test_buyer_matching.py`
  - `tests/unit/test_supervisor_routing.py`
- Test run result:
  - `uv run pytest tests/unit/test_buyer_matching.py tests/unit/test_supervisor_routing.py`
  - **28 passed**

### Acceptance Criteria Outcome
| # | Criterion | Status |
|---|-----------|--------|
| 1 | Multi-factor scoring with 5 weighted signals | ✅ |
| 2 | Haversine geospatial distance calculation correct | ✅ |
| 3 | Quality grade alignment enforced (below min = 0 score) | ✅ |
| 4 | Returns ≥1 valid match in unit test with mock data | ✅ |
| 5 | Reverse matching (buyer → farmers) works | ✅ |
| 6 | Wired into supervisor routing for matching intents | ✅ |
| 7 | Redis cache for recent match results (5-min TTL) | ✅ |
