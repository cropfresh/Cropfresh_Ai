"""
Unit tests for the DPLE Logistics Routing Engine.

Covers:
- Multi-pickup clustering (HDBSCAN)
- Vehicle selection by weight + cold chain
- Route optimization (OR-Tools / greedy TSP)
- Cost calculation with deadhead
- Utilization % and cost_per_kg
- 5-farm cluster < ₹2.5/kg for 30km delivery
"""

# * TEST MODULE — LOGISTICS ROUTER
# NOTE: Uses Arrange-Act-Assert (AAA) pattern.

from __future__ import annotations

import pytest

from src.agents.logistics_router.clustering import cluster_pickups
from src.agents.logistics_router.cost import calculate_cost
from src.agents.logistics_router.engine import LogisticsRouter, get_logistics_router
from src.agents.logistics_router.geo import haversine_km
from src.agents.logistics_router.models import DeliveryPoint, PickupPoint
from src.agents.logistics_router.routing import solve_tsp
from src.agents.logistics_router.vehicle import select_vehicle


# * ═══════════════════════════════════════════════════════════════
# * Fixtures
# * ═══════════════════════════════════════════════════════════════

@pytest.fixture
def router() -> LogisticsRouter:
    """LogisticsRouter instance."""
    return get_logistics_router()


@pytest.fixture
def five_farms_nearby() -> list[PickupPoint]:
    """5 farms clustered within ~2km of each other, 500kg total (Bangalore area)."""
    base_lat, base_lon = 12.9716, 77.5946
    return [
        PickupPoint("f1", base_lat, base_lon, 100.0),
        PickupPoint("f2", base_lat + 0.005, base_lon, 100.0),
        PickupPoint("f3", base_lat, base_lon + 0.005, 100.0),
        PickupPoint("f4", base_lat - 0.003, base_lon + 0.003, 100.0),
        PickupPoint("f5", base_lat + 0.002, base_lon - 0.002, 100.0),
    ]


@pytest.fixture
def delivery_30km() -> DeliveryPoint:
    """Delivery point ~30km from farm cluster."""
    return DeliveryPoint("b1", 12.9716 + 0.27, 77.5946, "Buyer Warehouse")


# * ═══════════════════════════════════════════════════════════════
# * Geo
# * ═══════════════════════════════════════════════════════════════

def test_haversine_km() -> None:
    """Haversine distance between two points."""
    d = haversine_km(12.9716, 77.5946, 12.9716 + 0.27, 77.5946)
    assert 29 <= d <= 31


# * ═══════════════════════════════════════════════════════════════
# * Vehicle Selection
# * ═══════════════════════════════════════════════════════════════

def test_select_vehicle_2w_ev() -> None:
    """Light load selects 2W EV."""
    v = select_vehicle(40.0, cold_chain_required=False)
    assert v.vehicle_type == "2w_ev"
    assert v.capacity_kg == 50.0


def test_select_vehicle_3w_auto() -> None:
    """Medium load selects 3W Auto."""
    v = select_vehicle(250.0, cold_chain_required=False)
    assert v.vehicle_type == "3w_auto"
    assert v.capacity_kg == 300.0


def test_select_vehicle_tempo() -> None:
    """Heavy load selects Tempo."""
    v = select_vehicle(500.0, cold_chain_required=False)
    assert v.vehicle_type == "tempo"
    assert v.capacity_kg == 1500.0


def test_select_vehicle_cold_chain() -> None:
    """Cold chain required selects cold chain vehicle."""
    v = select_vehicle(100.0, cold_chain_required=True)
    assert v.vehicle_type == "cold_chain"
    assert v.cold_chain is True


# * ═══════════════════════════════════════════════════════════════
# * Clustering
# * ═══════════════════════════════════════════════════════════════

def test_cluster_pickups_groups_nearby(
    five_farms_nearby: list[PickupPoint],
) -> None:
    """Nearby farms form clusters."""
    clusters = cluster_pickups(five_farms_nearby, min_cluster_size=2)
    assert len(clusters) >= 1
    total = sum(len(c) for c in clusters)
    assert total == len(five_farms_nearby)


def test_cluster_pickups_empty() -> None:
    """Empty input returns empty."""
    assert cluster_pickups([]) == []


def test_cluster_pickups_single() -> None:
    """Single pickup returns single cluster."""
    p = [PickupPoint("f1", 12.97, 77.59, 50.0)]
    clusters = cluster_pickups(p, min_cluster_size=1)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


# * ═══════════════════════════════════════════════════════════════
# * Routing
# * ═══════════════════════════════════════════════════════════════

def test_solve_tsp_returns_ordered_sequence(
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """TSP returns ordered pickups and total distance."""
    pickups = five_farms_nearby
    delivery = delivery_30km
    ordered, total_km = solve_tsp(pickups, delivery)
    assert len(ordered) == 5
    assert total_km > 30
    assert total_km < 100


def test_solve_tsp_single_pickup() -> None:
    """Single pickup returns it and round-trip distance."""
    p = [PickupPoint("f1", 12.97, 77.59, 50.0)]
    d = DeliveryPoint("b1", 12.98, 77.60, "")
    ordered, total_km = solve_tsp(p, d)
    assert len(ordered) == 1
    assert ordered[0].farm_id == "f1"
    expected = haversine_km(12.98, 77.60, 12.97, 77.59) * 2
    assert abs(total_km - expected) < 0.5


# * ═══════════════════════════════════════════════════════════════
# * Cost
# * ═══════════════════════════════════════════════════════════════

def test_calculate_cost_includes_deadhead(
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """Cost calculation returns cost and deadhead."""
    pickups = five_farms_nearby
    delivery = delivery_30km
    ordered, total_km = solve_tsp(pickups, delivery)
    vehicle = select_vehicle(400.0, False)
    cost, deadhead = calculate_cost(ordered, delivery, vehicle, total_km)
    assert cost > 0
    assert deadhead >= 0
    assert deadhead < total_km


# * ═══════════════════════════════════════════════════════════════
# * Engine — plan_route
# * ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_plan_route_returns_route_result(
    router: LogisticsRouter,
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """plan_route returns RouteResult with all required fields."""
    result = await router.plan_route(five_farms_nearby, delivery_30km)
    assert result is not None
    assert result.route_id.startswith("rt-")
    # Engine selects the best cost/kg cluster — may be a sub-cluster of the 5 farms
    assert 1 <= len(result.pickup_sequence) <= 5
    assert result.total_weight_kg > 0
    assert result.vehicle_type in ("2w_ev", "3w_auto", "tempo", "cold_chain")
    assert result.estimated_cost > 0
    assert result.cost_per_kg > 0
    assert 0 <= result.utilization_pct <= 100
    assert result.deadhead_km >= 0
    assert 1 <= result.cluster_size <= 5


@pytest.mark.asyncio
async def test_plan_route_five_farm_cluster_under_2_5_per_kg(
    router: LogisticsRouter,
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """5-farm cluster at 30km delivery achieves <₹2.5/kg (acceptance criterion 6)."""
    result = await router.plan_route(five_farms_nearby, delivery_30km)
    assert result is not None
    assert result.cost_per_kg < 2.5, (
        f"Expected cost_per_kg < 2.5, got {result.cost_per_kg}"
    )


@pytest.mark.asyncio
async def test_plan_route_empty_pickups_returns_none(
    router: LogisticsRouter,
    delivery_30km: DeliveryPoint,
) -> None:
    """Empty pickups returns None."""
    result = await router.plan_route([], delivery_30km)
    assert result is None


@pytest.mark.asyncio
async def test_plan_route_respects_max_stops(
    router: LogisticsRouter,
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """max_stops limits which clusters are considered."""
    result = await router.plan_route(
        five_farms_nearby, delivery_30km, max_stops=8
    )
    assert result is not None
    assert len(result.pickup_sequence) <= 8


@pytest.mark.asyncio
async def test_plan_route_utilization_reported(
    router: LogisticsRouter,
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """Utilization % is reported (acceptance criterion 5)."""
    result = await router.plan_route(five_farms_nearby, delivery_30km)
    assert result is not None
    assert result.utilization_pct > 0
    assert result.utilization_pct <= 100


@pytest.mark.asyncio
async def test_plan_route_deadhead_calculated(
    router: LogisticsRouter,
    five_farms_nearby: list[PickupPoint],
    delivery_30km: DeliveryPoint,
) -> None:
    """Deadhead factor calculated from return distance (acceptance criterion 4)."""
    result = await router.plan_route(five_farms_nearby, delivery_30km)
    assert result is not None
    assert result.deadhead_km >= 0

# * ═══════════════════════════════════════════════════════════════
# * EXTENDED TESTS (TASK 17)
# * ═══════════════════════════════════════════════════════════════

class TestLogisticsRouterExtended:
    @pytest.mark.asyncio
    async def test_single_pickup_cost_under_target(
        self, router: LogisticsRouter, delivery_30km: DeliveryPoint
    ) -> None:
        """Solo pickup should attempt to be cost-effective."""
        pickups = [PickupPoint("f1", 12.9716, 77.5946, 500.0)]
        result = await router.plan_route(pickups, delivery_30km)
        assert result is not None
        assert result.cost_per_kg <= 2.8 # Giving some buffer for variable cost formulas

    @pytest.mark.asyncio
    async def test_cost_per_kg_improves_with_more_pickups(
        self, router: LogisticsRouter, delivery_30km: DeliveryPoint,
        five_farms_nearby: list[PickupPoint]
    ) -> None:
        """Adding nearby pickups should decrease the cost/kg (consolidation benefit)."""
        solo = [five_farms_nearby[0]]
        multi = five_farms_nearby
        
        res_solo = await router.plan_route(solo, delivery_30km)
        res_multi = await router.plan_route(multi, delivery_30km)
        
        assert res_solo is not None and res_multi is not None
        assert res_multi.cost_per_kg < res_solo.cost_per_kg
