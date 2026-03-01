# Task 8: Build Order Management Service

> **Priority:** 🟠 P1 | **Phase:** 2 | **Effort:** 3–4 days  
> **Files:** `src/api/services/order_service.py` [NEW], `src/api/routers/orders.py` [NEW]  
> **Score Target:** 9/10 — Complete order lifecycle with escrow state machine  
> **Status:** ✅ **Completed — 2026-03-01**

---

## 📌 Problem Statement

No order management exists. The business model requires a full order lifecycle: match → order → escrow → pickup → delivery → settlement with AISP breakdown at creation.

---

## 🏗️ Implementation Spec

### Order State Machine
```
confirmed → pickup_scheduled → in_transit → delivered → settled
                                    ↓
                                disputed → ai_analysed → resolved/escalated
                                    ↓
                                    → refunded (escrow release)
```

### Escrow Flow
```
pending → held (at order creation) → released (on delivery) → refunded (on dispute)
```

### Order Service
```python
class OrderService:
    async def create_order(self, data: CreateOrderRequest) -> Order:
        """
        Create order from matched listing + buyer.
        
        Steps:
        1. Validate listing is active
        2. Calculate AISP via PricingAgent
        3. Assign hauler via LogisticsRouter (or manual)
        4. Create order record with escrow=held
        5. Update listing status to 'matched'
        6. Notify farmer + buyer
        """
    
    async def update_status(self, order_id: str, new_status: str, metadata: dict = {}) -> Order:
        """
        Advance order through state machine.
        Validates transition is legal (e.g., can't go from 'confirmed' to 'delivered').
        """
    
    async def raise_dispute(self, order_id: str, dispute_data: dict) -> Dispute:
        """
        Raise dispute with arrival photos.
        Triggers Digital Twin comparison if available.
        AI diff engine analyzes departure vs arrival quality.
        """
    
    async def settle_order(self, order_id: str) -> Order:
        """
        Release escrow to farmer.
        Calculate actual platform margin.
        Update farmer/buyer stats.
        """

VALID_TRANSITIONS = {
    'confirmed': ['pickup_scheduled', 'cancelled'],
    'pickup_scheduled': ['in_transit', 'cancelled'],
    'in_transit': ['delivered', 'disputed'],
    'delivered': ['settled', 'disputed'],
    'disputed': ['ai_analysed'],
    'ai_analysed': ['resolved', 'escalated'],
    'resolved': ['settled'],
    'settled': [],  # Terminal state
}
```

### REST API Endpoints
```
POST   /api/v1/orders                    → Create order
GET    /api/v1/orders/{id}               → Get order details
PATCH  /api/v1/orders/{id}/status        → Update status
GET    /api/v1/orders?farmer_id=X        → Farmer's orders
GET    /api/v1/orders?buyer_id=X         → Buyer's orders
POST   /api/v1/orders/{id}/dispute       → Raise dispute
POST   /api/v1/orders/{id}/settle        → Settle and release escrow
GET    /api/v1/orders/{id}/aisp          → Get AISP breakdown
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight | Status |
|---|-----------|--------|--------|
| 1 | Order creation with AISP calculation | 25% | ✅ Done |
| 2 | State machine enforces valid transitions only | 20% | ✅ Done |
| 3 | Escrow flow: held → released/refunded | 20% | ✅ Done |
| 4 | Dispute flow triggers AI analysis | 15% | ✅ Done |
| 5 | Order history query by farmer and buyer | 10% | ✅ Done |
| 6 | Notification stubs for status changes | 10% | ✅ Done |

---

## 🏁 Completion Evidence — 2026-03-01

### Files Created / Modified

| Action | File | Description |
|--------|------|-------------|
| UPDATED | `src/db/postgres_client.py` | Added 4 new DB methods: `get_order()`, `get_orders_by_farmer()`, `get_orders_by_buyer()`, `update_dispute()` |
| IMPLEMENTED | `src/api/services/order_service.py` | Full `OrderService` with `VALID_TRANSITIONS` state machine (11 statuses), `ESCROW_ON_TRANSITION` map, `create_order`, `update_status`, `raise_dispute`, `settle_order`, `get_order`, `get_orders_by_farmer`, `get_orders_by_buyer`, `get_aisp_breakdown`. Pydantic models: `CreateOrderRequest`, `UpdateStatusRequest`, `RaiseDisputeRequest`, `OrderResponse`, `DisputeResponse`, `AISPBreakdown`. Factory: `get_order_service()` |
| NEW | `src/api/routers/orders.py` | 8 REST endpoints: `POST /orders`, `GET /orders`, `GET /orders/{id}`, `PATCH /orders/{id}/status`, `POST /orders/{id}/dispute`, `POST /orders/{id}/settle`, `GET /orders/{id}/aisp` |
| UPDATED | `src/api/main.py` | Registered orders router at `/api/v1` |
| NEW | `tests/unit/test_order_service.py` | 73 unit tests across 14 test classes — full AC coverage |

### Test Results

```
tests/unit/test_order_service.py — 73 passed in 0.37s
Full suite — 276 passed (0 regressions from 203 baseline)
```

### AC Validation

| # | Acceptance Criterion | Evidence |
|---|----------------------|----------|
| AC1 | Order creation with AISP calculation | `TestCreateOrder` (10 tests) — ratio fallback + PricingAgent delegation both tested. `TestAISPRatioFallback` (4 tests) — 80% farmer / 10% logistics / 6% platform / 4% risk, sum verified. |
| AC2 | State machine enforces valid transitions | `TestValidTransitions` (10 tests) — all 11 statuses checked. `TestUpdateStatus` (7 tests) — illegal transitions raise `ValueError` with descriptive message. |
| AC3 | Escrow flow: held → released / refunded | `TestEscrowOnTransition` (6 tests) — confirmed=held, settled=released, cancelled=refunded, refunded=refunded verified. `TestSettleOrder` confirms `escrow_status=released`. |
| AC4 | Dispute triggers Digital Twin AI diff | `TestRaiseDispute` (7 tests) — `QualityAssessmentAgent.compare_twin()` called when `departure_twin_id` + `arrival_photos` present; skipped without photos. `diff_report` saved to DB. |
| AC5 | Order history by farmer + buyer | `TestGetOrdersByFarmer` (4 tests) + `TestGetOrdersByBuyer` (3 tests) — status filter forwarded, empty result handled, no-DB fallback tested. |
| AC6 | Notification stubs on status changes | `_notify_order_created()` + `_notify_status_change()` stubs fire on every transition (logged as `[NOTIFY-STUB]`). Wiring point clearly marked with `# TODO: Wire to WhatsAppBotAgent`. |

### State Machine — Full Coverage

```
confirmed → pickup_scheduled → in_transit → delivered → settled  ✅
confirmed → cancelled                                             ✅
pickup_scheduled → cancelled                                      ✅
in_transit → disputed → ai_analysed → resolved → settled         ✅
delivered → disputed                                             ✅
ai_analysed → escalated → settled / refunded                     ✅
```

### AISP Breakdown (Ratio-Based Fallback)

| Component | Ratio | Example (100kg × ₹30/kg = ₹3,000) |
|-----------|-------|-------------------------------------|
| Farmer Payout | 80% | ₹2,400 |
| Logistics Cost | 10% | ₹300 |
| Platform Margin | 6% | ₹180 |
| Risk Buffer | 4% | ₹120 |
| **AISP Total** | **100%** | **₹3,000** |
