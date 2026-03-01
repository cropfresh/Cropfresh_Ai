# Order Endpoints

> **Base path:** `/api/v1/orders`  
> **Router:** `src/api/routers/orders.py`  
> **Service:** `OrderService` in `src/api/services/order_service.py`

## POST /api/v1/orders
Create a new order from a matched listing. AISP breakdown calculated; escrow held.

**Request:** `CreateOrderRequest` — listing_id, buyer_id, quantity_kg, hauler_id?, override_price_per_kg?

## GET /api/v1/orders
List orders by farmer_id or buyer_id (query params). Optional status filter.

## GET /api/v1/orders/{id}
Fetch a single order by UUID.

## PATCH /api/v1/orders/{id}/status
Advance order through state machine. Valid transitions: confirmed → pickup_scheduled → in_transit → delivered → settled (or disputed).

**Request:** `UpdateStatusRequest` — status, metadata?

## POST /api/v1/orders/{id}/dispute
Raise a dispute. Requires `raised_by` (buyer|farmer), `reason`, optional `arrival_photos` (S3 URLs), `departure_twin_id`. Triggers **Digital Twin AI diff** when twin_id + photos provided — produces DiffReport with liability (farmer/hauler/buyer/shared) and claim_percent.

**Request:** `RaiseDisputeRequest` — raised_by, reason, arrival_photos?, departure_twin_id?

## POST /api/v1/orders/{id}/settle
Settle order — release escrow to farmer.

## GET /api/v1/orders/{id}/aisp
Return AISP breakdown (farmer_payout, logistics_cost, platform_margin, risk_buffer, aisp_total, aisp_per_kg).
