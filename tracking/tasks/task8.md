# Task 8: Build Order Management Service

> **Priority:** 🟠 P1 | **Phase:** 2 | **Effort:** 3–4 days  
> **Files:** `src/api/services/order_service.py` [NEW], `src/api/routers/orders.py` [NEW]  
> **Score Target:** 9/10 — Complete order lifecycle with escrow state machine

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

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Order creation with AISP calculation | 25% |
| 2 | State machine enforces valid transitions only | 20% |
| 3 | Escrow flow: held → released/refunded | 20% |
| 4 | Dispute flow triggers AI analysis | 15% |
| 5 | Order history query by farmer and buyer | 10% |
| 6 | Notification stubs for status changes | 10% |
