# DPLE Logistics Router — Agent Spec

> **Module:** `src/agents/logistics_router/`  
> **Task:** 11 | **Status:** ✅ Complete (2026-03-02)  
> **Phase:** 3 — DPLE Logistics Routing

---

## Purpose

Implements the **Dynamic Pickup-and-Load Efficiency (DPLE)** multi-farm logistics routing
engine. Plans optimal pickup routes for haulers collecting produce from multiple farms in a
single trip, achieving the CropFresh business target of **<₹2.5/kg** logistics cost via
vehicle utilization and route sharing.

---

## Module Structure

```
src/agents/logistics_router/
├── __init__.py       # Public API: LogisticsRouter, PickupPoint, DeliveryPoint, RouteResult
├── models.py         # Data classes: PickupPoint, DeliveryPoint, RouteResult
├── geo.py            # haversine_km(), build_distance_matrix()
├── clustering.py     # cluster_pickups() — HDBSCAN (haversine metric)
├── vehicle.py        # VehicleConfig, VEHICLES list, select_vehicle()
├── cost.py           # calculate_cost() — base_rate + per_km + deadhead
├── routing.py        # solve_tsp() — OR-Tools → greedy fallback
└── engine.py         # LogisticsRouter.plan_route() — orchestrates all above
```

---

## Algorithm

1. **Cluster** nearby pickup points using HDBSCAN (`metric='haversine'`)
2. **Route** within best cluster using OR-Tools TSP (greedy fallback)
3. **Select** smallest vehicle that fits total weight (+ cold chain flag)
4. **Cost** = `base_rate + per_km_rate × total_km`; deadhead = return leg
5. **Return** `RouteResult` ranked by `cost_per_kg` (lowest = best)

Engine always evaluates the **full pickup set** as a candidate alongside HDBSCAN
sub-clusters, ensuring the optimal split is chosen.

---

## Key Configuration

| Parameter             | Default | Purpose                             |
| --------------------- | ------- | ----------------------------------- |
| `min_cluster_size`    | 2       | Min farms per HDBSCAN cluster       |
| `max_stops`           | 8       | Max pickup stops per route          |
| `cold_chain_required` | False   | Forces cold chain vehicle selection |

---

## Vehicle Fleet

| Type         | Capacity | Base   | Per-km | Cold Chain |
| ------------ | -------- | ------ | ------ | ---------- |
| `2w_ev`      | 50 kg    | ₹50    | ₹2/km  | ❌         |
| `3w_auto`    | 300 kg   | ₹150   | ₹4/km  | ❌         |
| `tempo`      | 1,500 kg | ₹500   | ₹8/km  | Optional   |
| `cold_chain` | 3,000 kg | ₹1,500 | ₹15/km | ✅         |

---

## Known Compatibility Notes

- `sklearn 1.8.x`: `cluster_selection_epsilon` has a Cython bug with `metric='haversine'`.
  Omitted from `HDBSCAN()` config; all-noise fallback groups isolated farms as one cluster.
- OR-Tools (≥9.10): Use `pywrapcp.DefaultRoutingSearchParameters()` (module-level function),
  not the deprecated `routing.DefaultSearchParameters()` method.
- OR-Tools arc cost: Use Haversine summation over the returned route, not `GetArcCostForVehicle`.

---

## Tests

```
tests/unit/test_logistics_router.py  — 17 tests, all passing
```
