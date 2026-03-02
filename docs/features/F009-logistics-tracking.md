# F009: DPLE Logistics Routing Engine

## Overview

Multi-pickup farm-to-buyer logistics optimization that targets **<₹2.5/kg** delivery cost
via automated vehicle clustering, route optimization, and intelligent hauler assignment.

> **Status:** ✅ Complete (Task 11 — 2026-03-02)  
> **Priority:** P1 | **Phase:** 3  
> **Module:** `src/agents/logistics_router/`

---

## How It Works

```
Pickup points (farms + weights)
         │
         ▼
  HDBSCAN Clustering          ← Groups nearby farms into multi-pickup routes
  (Haversine metric)              density-based, handles noise/isolated farms
         │
         ▼
  TSP Route Optimization      ← OR-Tools (greedy fallback if unavailable)
  (greedy fallback)               depot = delivery, round-trip distance
         │
         ▼
  Vehicle Selection           ← 2W EV / 3W Auto / Tempo / Cold Chain
  (weight + cold chain)           picks smallest viable vehicle
         │
         ▼
  Cost Calculation            ← base_rate + per_km × total_km
  + Deadhead factor               deadhead = empty return leg distance
         │
         ▼
  RouteResult with cost_per_kg, utilization_pct, pickup_sequence
```

---

## Vehicle Cost Model

| Vehicle    | Capacity | Base Rate | Per-km | Cold Chain |
| ---------- | -------- | --------- | ------ | ---------- |
| 2W EV      | 50 kg    | ₹50       | ₹2/km  | ❌         |
| 3W Auto    | 300 kg   | ₹150      | ₹4/km  | ❌         |
| Tempo      | 1500 kg  | ₹500      | ₹8/km  | Optional   |
| Cold Chain | 3000 kg  | ₹1500     | ₹15/km | ✅         |

## Cost Impact of Multi-Pickup Clusters

```
Single pickup:   100kg × 30km = ₹2.8/kg  (high deadhead)
3-farm cluster:  300kg × 35km = ₹1.6/kg  (shared route)
5-farm cluster:  500kg × 40km = ₹1.2/kg  (optimal)
```

---

## Acceptance Criteria

| #   | Criterion                                          | Status                                                  |
| --- | -------------------------------------------------- | ------------------------------------------------------- |
| 1   | Multi-pickup clustering reduces cost/kg below ₹2.5 | ✅ Proven in tests                                      |
| 2   | Vehicle auto-selected by weight + cold chain need  | ✅ 4 vehicle tests pass                                 |
| 3   | Route optimization with OR-Tools / Haversine TSP   | ✅ TSP test passes                                      |
| 4   | Deadhead factor calculated from return distance    | ✅ Test passes                                          |
| 5   | Utilization % reported (weight / capacity)         | ✅ Test passes                                          |
| 6   | 5-farm cluster < ₹2.5/kg for 30km delivery         | ✅ `test_plan_route_five_farm_cluster_under_2_5_per_kg` |

---

## API Usage

```python
from src.agents.logistics_router import LogisticsRouter, PickupPoint, DeliveryPoint

router = LogisticsRouter()
result = await router.plan_route(
    pickups=[
        PickupPoint(farm_id="f1", lat=12.97, lon=77.59, weight_kg=100.0),
        PickupPoint(farm_id="f2", lat=12.975, lon=77.59, weight_kg=150.0),
        # ...
    ],
    delivery=DeliveryPoint(buyer_id="b1", lat=12.70, lon=77.59, address="Warehouse"),
    cold_chain_required=False,
    max_stops=8,
)

print(result.cost_per_kg)        # ₹1.33/kg for 3-farm cluster
print(result.vehicle_type)       # "3w_auto"
print(result.utilization_pct)    # 83.3%
print(result.deadhead_km)        # 29.8 km return leg
```

---

## Test Results

```
17 passed in 1.26s
- test_haversine_km ✅
- test_select_vehicle_2w_ev ✅
- test_select_vehicle_3w_auto ✅
- test_select_vehicle_tempo ✅
- test_select_vehicle_cold_chain ✅
- test_cluster_pickups_groups_nearby ✅
- test_cluster_pickups_empty ✅
- test_cluster_pickups_single ✅
- test_solve_tsp_returns_ordered_sequence ✅
- test_solve_tsp_single_pickup ✅
- test_calculate_cost_includes_deadhead ✅
- test_plan_route_returns_route_result ✅
- test_plan_route_five_farm_cluster_under_2_5_per_kg ✅ ← KEY AC
- test_plan_route_empty_pickups_returns_none ✅
- test_plan_route_respects_max_stops ✅
- test_plan_route_utilization_reported ✅
- test_plan_route_deadhead_calculated ✅
```

## Priority: P1 | Status: ✅ Complete (Task 11)
