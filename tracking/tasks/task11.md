# Task 11: Implement DPLE Logistics Routing Engine ✅ COMPLETE

> **Priority:** 🟠 P1 | **Phase:** 3 | **Effort:** 4–5 days  
> **Status:** ✅ **Completed — 2026-03-02**  
> **Files:** `src/agents/logistics_router/` (package: 7 modules)  
> **Score Target:** 9/10 — Achieve <₹2.5/kg logistics cost via multi-pickup clustering  
> **Tests:** 17/17 unit tests pass

---

## ✅ Completion Evidence

| #   | Criterion                                             | Evidence                                                                  | Result  |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------- | ------- |
| 1   | Multi-pickup clustering reduces cost/kg below ₹2.5    | `test_plan_route_five_farm_cluster_under_2_5_per_kg`: `cost_per_kg < 2.5` | ✅ Pass |
| 2   | Vehicle auto-selected by weight + cold chain need     | 4 vehicle selection tests pass (2W EV / 3W Auto / Tempo / Cold Chain)     | ✅ Pass |
| 3   | Route optimization with OR-Tools / Haversine fallback | `test_solve_tsp_returns_ordered_sequence` + greedy fallback tested        | ✅ Pass |
| 4   | Deadhead factor calculated from return distance       | `test_plan_route_deadhead_calculated`: `deadhead_km >= 0`                 | ✅ Pass |
| 5   | Utilization % reported (weight / capacity)            | `test_plan_route_utilization_reported`: `0 < utilization_pct <= 100`      | ✅ Pass |
| 6   | 5-farm cluster < ₹2.5/kg for 30km delivery            | The key acceptance test above                                             | ✅ Pass |

### Bugs Fixed During Implementation

| Bug                                                                                 | Module          | Fix                                                           |
| ----------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------- |
| `sklearn 1.8.x` HDBSCAN `cluster_selection_epsilon` Cython TypeError with haversine | `clustering.py` | Removed `cluster_selection_epsilon`; added all-noise fallback |
| `routing.DefaultSearchParameters()` removed in newer OR-Tools                       | `routing.py`    | Updated to `pywrapcp.DefaultRoutingSearchParameters()`        |
| `GetArcCostForVehicle` returning 0 for total distance                               | `routing.py`    | Replaced with explicit Haversine summation over route stops   |

## 📌 Problem Statement

No logistics routing exists. Business model targets <₹2.5/kg logistics cost and >70% vehicle utilization through multi-pickup clustering and intelligent hauler assignment.

---

## 🔬 Research Findings

### Approach: Cluster-First, Route-Second

1. **HDBSCAN clustering**: Group nearby farms into pickup clusters (density-based, handles noise)
2. **OR-Tools VRP**: Optimize route within each cluster (Capacitated VRP with time windows)
3. **Hauler matching**: Assign vehicle type based on total weight + commodity type

### Vehicle Types & Cost Model

| Vehicle    | Capacity | Base Rate (₹/trip) | Per-km Rate | Cold Chain |
| ---------- | -------- | ------------------ | ----------- | ---------- |
| 2W EV      | 50 kg    | ₹50                | ₹2/km       | ❌         |
| 3W Auto    | 300 kg   | ₹150               | ₹4/km       | ❌         |
| Tempo      | 1500 kg  | ₹500               | ₹8/km       | Optional   |
| Cold Chain | 3000 kg  | ₹1500              | ₹15/km      | ✅         |

### Multi-Pickup Savings Model

```
Single pickup:  100kg × 30km = ₹2.8/kg (high deadhead)
3-farm cluster: 300kg × 35km = ₹1.6/kg (shared route, high utilization)
5-farm cluster: 500kg × 40km = ₹1.2/kg (optimal — below ₹2.5 target)
```

---

## 🏗️ Implementation Spec

```python
@dataclass
class RouteResult:
    route_id: str
    pickup_sequence: list[dict]    # Ordered farm stops
    total_distance_km: float
    total_weight_kg: float
    vehicle_type: str
    estimated_cost: float
    cost_per_kg: float
    utilization_pct: float
    estimated_duration_hours: float
    deadhead_km: float              # Empty return distance

class LogisticsRouter:
    """
    DPLE Logistics Routing Engine.

    Algorithm:
    1. Cluster nearby farms (HDBSCAN, min_cluster_size=2)
    2. For each cluster, solve CVRP (OR-Tools)
    3. Assign optimal vehicle type based on total weight
    4. Calculate cost with deadhead factor
    5. Return ranked routes by cost_per_kg
    """

    async def plan_route(
        self,
        pickups: list[PickupPoint],    # Farm locations + weights
        delivery: DeliveryPoint,        # Buyer location
        max_stops: int = 8,
        cold_chain_required: bool = False,
    ) -> RouteResult:
        """Plan optimal multi-pickup route."""

        # Step 1: Cluster nearby pickups
        clusters = self._cluster_pickups(pickups)

        # Step 2: For best cluster, solve TSP
        route = self._solve_tsp(clusters[0], delivery)

        # Step 3: Assign vehicle
        total_weight = sum(p.weight_kg for p in route.pickups)
        vehicle = self._select_vehicle(total_weight, cold_chain_required)

        # Step 4: Calculate cost
        cost = self._calculate_cost(route, vehicle)

        return RouteResult(...)

    def _cluster_pickups(self, pickups: list[PickupPoint]) -> list[list[PickupPoint]]:
        """HDBSCAN clustering of GPS coordinates."""
        from sklearn.cluster import HDBSCAN
        coords = np.array([[p.lat, p.lon] for p in pickups])
        # Convert to radians for Haversine metric
        coords_rad = np.radians(coords)
        clusterer = HDBSCAN(
            min_cluster_size=2,
            metric='haversine',
            cluster_selection_epsilon=0.005,  # ~500m
        )
        labels = clusterer.fit_predict(coords_rad)
        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(pickups[i])
        return list(clusters.values())

    def _solve_tsp(self, pickups, delivery):
        """Solve TSP using OR-Tools for optimal pickup sequence."""
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
        # Build distance matrix
        locations = [delivery] + pickups
        distance_matrix = self._build_distance_matrix(locations)
        # Solve
        manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        # ... OR-Tools CVRP setup
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                   | Weight |
| --- | ----------------------------------------------------------- | ------ |
| 1   | Multi-pickup clustering reduces cost/kg below ₹2.5          | 25%    |
| 2   | Vehicle type auto-selected by weight + cold chain need      | 20%    |
| 3   | Route optimization with OR-Tools or Haversine TSP           | 20%    |
| 4   | Deadhead factor calculated from return distance             | 15%    |
| 5   | Utilization % reported (weight / capacity)                  | 10%    |
| 6   | Unit test: 5-farm cluster costs < ₹2.5/kg for 30km delivery | 10%    |
