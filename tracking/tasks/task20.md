# Task 20: Production Hardening — Auth, Rate Limiting, Monitoring, Error Handling

> **Priority:** 🟠 P1 | **Phase:** 6 | **Effort:** 4–5 days  
> **Files:** `src/api/main.py`, `src/api/middleware/`, `src/api/routers/health.py`  
> **Score Target:** 9/10 — Production-ready with 99.5% uptime capability

---

## 📌 Problem Statement

The service needs production hardening before the 50-farmer Karnataka pilot. Missing: proper auth enforcement, rate limiting, structured logging, monitoring, and standardized error handling.

---

## 🏗️ Implementation Spec

### 1. Authentication & Authorization
```python
# src/api/middleware/auth.py
class JWTAuthMiddleware:
    """
    JWT-based auth for all /api/v1/ routes.
    
    Excludes: /health, /auth/register, /auth/verify-otp, /docs
    
    Token validation:
    - Decode JWT with HS256
    - Verify expiration
    - Attach user context to request.state
    """
    
    EXEMPT_PATHS = ["/health", "/health/ready", "/auth/", "/docs", "/openapi.json"]

# Role-based access control
class RBACMiddleware:
    ROLE_PERMISSIONS = {
        'farmer': ['listings:read', 'listings:write', 'orders:read', 'voice:use'],
        'buyer': ['listings:read', 'orders:read', 'orders:write', 'matching:use'],
        'admin': ['*'],
    }
```

### 2. Rate Limiting
```python
# src/api/middleware/rate_limiter.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

RATE_LIMITS = {
    '/api/v1/chat': '30/minute',          # LLM calls are expensive
    '/api/v1/voice/process': '20/minute',  # Voice processing
    '/api/v1/query': '60/minute',          # RAG queries
    '/api/v1/listings': '100/minute',      # CRUD operations
    '/api/v1/orders': '50/minute',
    'default': '120/minute',
}
```

### 3. Structured Logging
```python
# src/api/middleware/logging_middleware.py
import structlog

logger = structlog.get_logger()

class RequestLoggingMiddleware:
    """
    Structured JSON logging for every request.
    
    Logs:
    - Request ID (UUID)
    - Method, path, status code
    - Latency (ms)
    - User ID (from JWT)
    - Agent used (if applicable)
    - LLM token count and cost
    """
    
    async def dispatch(self, request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        start = time.time()
        
        response = await call_next(request)
        
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round((time.time() - start) * 1000, 2),
            user_id=getattr(request.state, 'user_id', None),
        )
        
        response.headers["X-Request-ID"] = request_id
        return response
```

### 4. Standardized Error Handling
```python
# src/api/middleware/error_handler.py
class CropFreshException(Exception):
    def __init__(self, code: str, message: str, status: int = 400, details: dict = {}):
        self.code = code
        self.message = message
        self.status = status
        self.details = details

ERROR_CODES = {
    'AUTH_001': 'Invalid or expired token',
    'AUTH_002': 'Insufficient permissions',
    'LISTING_001': 'Listing not found',
    'LISTING_002': 'Listing has expired',
    'ORDER_001': 'Invalid order state transition',
    'ORDER_002': 'Insufficient escrow balance',
    'PRICE_001': 'Price data unavailable for commodity',
    'VOICE_001': 'STT processing failed',
    'RATE_001': 'Rate limit exceeded',
}

# Response format:
{
    "error": {
        "code": "LISTING_001",
        "message": "Listing not found",
        "request_id": "uuid-xxx",
        "timestamp": "2026-03-01T12:00:00Z"
    }
}
```

### 5. Health Checks
```python
# src/api/routers/health.py

@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.5.0"}

@router.get("/health/ready")
async def readiness():
    """Deep health check — tests all dependencies."""
    checks = {
        'database': await check_postgres(),
        'redis': await check_redis(),
        'llm': await check_llm_provider(),
        'vector_db': await check_vector_db(),
    }
    all_ok = all(v['status'] == 'ok' for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }
```

### 6. Prometheus Metrics
```python
# src/api/middleware/metrics.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('cropfresh_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('cropfresh_request_duration_seconds', 'Request latency', ['endpoint'])
LLM_TOKENS = Counter('cropfresh_llm_tokens_total', 'LLM tokens used', ['provider', 'model'])
LLM_COST = Counter('cropfresh_llm_cost_inr', 'LLM cost in INR', ['provider'])
AGENT_CALLS = Counter('cropfresh_agent_calls_total', 'Agent invocations', ['agent'])
```

### 7. Cost Tracking
```python
class CostTracker:
    """Track per-query costs for budget monitoring."""
    
    RATES = {
        'bedrock_claude_sonnet_4_input': 0.003,    # $/1K tokens
        'bedrock_claude_sonnet_4_output': 0.015,
        'groq_llama3_8b_input': 0.00005,
        'groq_llama3_8b_output': 0.00008,
    }
    
    async def track(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        cost_usd = (
            input_tokens * self.RATES[f'{provider}_{model}_input'] / 1000 +
            output_tokens * self.RATES[f'{provider}_{model}_output'] / 1000
        )
        cost_inr = cost_usd * 83  # USD → INR
        LLM_COST.labels(provider=provider).inc(cost_inr)
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | JWT auth on all /api/v1/ routes (with exclusions) | 15% |
| 2 | Rate limiting with per-endpoint configuration | 15% |
| 3 | Structured JSON logging with request IDs | 15% |
| 4 | Standardized error codes and response format | 15% |
| 5 | Deep health check (DB, Redis, LLM, Vector DB) | 10% |
| 6 | Prometheus metrics for requests, latency, LLM cost | 15% |
| 7 | Cost tracking per query (target <₹0.25/query avg) | 10% |
| 8 | All health checks pass on startup | 5% |
