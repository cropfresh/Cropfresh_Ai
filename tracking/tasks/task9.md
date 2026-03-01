# Task 9: Build Farmer/Buyer Registration & Profile Service

> **Priority:** 🟠 P1 | **Phase:** 2 | **Effort:** 2–3 days  
> **Files:** `src/api/services/registration_service.py` [NEW], `src/api/routers/auth.py` [NEW]  
> **Score Target:** 9/10 — OTP-based registration with profile management  
> **Status:** ✅ **Completed — 2026-03-01**

---

## 📌 Problem Statement

No registration or profile management existed. Farmers and buyers need to register (phone + OTP), set preferences (language, location), and manage their profiles.

---

## 🏗️ Implementation Spec

### Registration Flow
```
1. POST /auth/register → {phone, type: farmer|buyer}
2. System sends OTP (mock in dev, Twilio/MSG91 in prod)
3. POST /auth/verify-otp → {phone, otp} → Returns JWT token
4. POST /auth/profile → {name, district, language_pref, ...}
```

### Service Methods
```python
class RegistrationService:
    async def register(self, phone: str, user_type: str) -> dict:
        """Create user record + send OTP."""
    
    async def verify_otp(self, phone: str, otp: str) -> TokenResponse:
        """Verify OTP, return JWT token."""
    
    async def update_profile(self, user_id: str, user_type: str, data: dict) -> ProfileResponse:
        """Update name, district, GPS, language preference."""
    
    async def get_profile(self, user_id: str, user_type: str) -> ProfileResponse:
        """Fetch complete user profile with stats."""

    async def register_farmer(self, name, phone, district, village=None) -> dict:
        """Voice agent direct registration (skips OTP flow)."""
```

### JWT Token Structure
```python
{
    "sub": "user_uuid",
    "type": "farmer",  # or "buyer"
    "phone": "+91...",
    "district": "Bangalore Rural",
    "iat": timestamp,
    "exp": timestamp + 30d,
}
```

### Profile Enrichment
- Auto-detect district from GPS coordinates (Karnataka centroid-based, 10 districts)
- Language preference inferred from district (all 31 Karnataka districts → Kannada)
- Quality score initialized at 0.5 (neutral)
- First listing creation triggers onboarding flow

---

## ✅ Acceptance Criteria

| # | Criterion | Weight | Status |
|---|-----------|--------|--------|
| 1 | Phone-based registration with OTP (mock) | 25% | ✅ Done |
| 2 | JWT token generation and validation | 25% | ✅ Done |
| 3 | Profile CRUD with language preference | 20% | ✅ Done |
| 4 | Separate farmer/buyer profiles with type-specific fields | 15% | ✅ Done |
| 5 | Voice agent `register` intent creates profile | 15% | ✅ Done |

---

## 🏁 Completion Evidence — 2026-03-01

### Files Created / Modified

| Action | File | Description |
|--------|------|-------------|
| NEW | `src/api/services/registration_service.py` | Full `RegistrationService`: OTP generation + 10-min expiry, stdlib HS256 JWT (no external deps), phone normalisation (+91 prefix), district→language mapping (all 31 Karnataka districts), GPS→district centroid lookup, Aadhaar hash, `register_farmer()` for voice agent compat. Pydantic models: `RegisterRequest`, `VerifyOTPRequest`, `UpdateFarmerProfileRequest`, `UpdateBuyerProfileRequest`, `TokenResponse`, `ProfileResponse`, `OTPRecord`. Factory: `get_registration_service()` |
| NEW | `src/api/routers/auth.py` | 6 REST endpoints: `POST /auth/register`, `POST /auth/verify-otp`, `GET /auth/me`, `GET /auth/profile/{user_id}`, `PATCH /auth/profile/{user_id}`, `PATCH /auth/buyer-profile/{user_id}` |
| UPDATED | `src/db/postgres_client.py` | Added 5 new DB methods: `get_buyer()` (by UUID), `get_farmer_by_phone()`, `get_buyer_by_phone()`, `update_farmer()` (allowed-field whitelist), `update_buyer()` (allowed-field whitelist) |
| UPDATED | `src/api/main.py` | Registered auth router at `/api/v1` |
| NEW | `tests/unit/test_registration_service.py` | 64 unit tests across 7 test classes — full AC coverage |

### Test Results

```
tests/unit/test_registration_service.py — 64 passed in 0.20s
Full suite — 340 passed, 0 regressions (from 276 baseline)
```

### AC Validation

| # | Acceptance Criterion | Evidence |
|---|----------------------|----------|
| AC1 | Phone-based registration with OTP | `TestRegisterOTP` (13 tests) — OTP store, 10-min expiry, phone normalisation (10-digit → +91), wrong OTP raises ValueError, expired OTP clears store, verify cleanup. OTP stub logs to console via `[OTP-STUB]`. |
| AC2 | JWT generation and validation | `TestJWTTokens` (12 tests) — stdlib HS256 (no pyjwt dep), 3-part structure, `sub=user_id`, `type=user_type`, `phone`, `district` in payload, 30-day expiry, tampered sig raises ValueError, wrong secret raises, expired token raises. `TestFactory` (6 tests) — env var secret, custom secret, shared OTP store. |
| AC3 | Profile CRUD with language preference | `TestProfileCRUD` (10 tests) — language inferred from district, explicit language not overridden, Aadhaar SHA-256 hash replaces last4, update calls DB method, initial quality_score=0.5, no-DB graceful None return. |
| AC4 | Separate farmer/buyer profiles | `TestFarmerBuyerProfiles` (8 tests) — farmer creates farmer record, buyer creates buyer record, existing phone skips create, farmer profile has no buyer_type, buyer profile has buyer_type + credit_limit, buyer_type validator rejects invalid values. |
| AC5 | Voice agent `register` intent | `TestVoiceAgentIntegration` (6 tests) — `register_farmer()` returns farmer_id UUID, language_pref=kn, district passed through, DB create called with correct name+district, village forwarded, no-DB returns valid UUID. |

### REST API Endpoints

```
POST   /api/v1/auth/register              → {phone, user_type} → OTP sent
POST   /api/v1/auth/verify-otp           → {phone, otp} → TokenResponse (JWT)
GET    /api/v1/auth/me                   → Decode JWT → user_id, type, phone, district
GET    /api/v1/auth/profile/{user_id}    → ProfileResponse (farmer or buyer)
PATCH  /api/v1/auth/profile/{user_id}    → Update farmer fields
PATCH  /api/v1/auth/buyer-profile/{id}   → Update buyer-specific fields
```

### JWT Token Design

| Field | Value |
|-------|-------|
| Algorithm | HS256 (stdlib hmac + sha256 — no pyjwt required) |
| Expiry | 30 days |
| Payload | sub, type, phone, district, iat, exp |
| Secret | `JWT_SECRET` env var (falls back to dev constant) |
| Verification | `secrets.compare_digest()` prevents timing attacks |

### Karnataka District→Language Coverage

All 31 Karnataka districts map to `kn` (Kannada). Includes common alternate names:
- Bangalore / Bengaluru / Bangalore Urban / Bangalore Rural
- Mysuru / Mysore
- Tumkur / Tumakuru
- Bellary / Ballari
- Belgaum / Belagavi
- Bijapur / Vijayapura
- Gulbarga / Kalaburagi
- Shimoga / Shivamogga

### DB Methods Added (postgres_client.py)

| Method | Description |
|--------|-------------|
| `get_buyer(buyer_id)` | Fetch buyer by UUID (was missing in Task 6) |
| `get_farmer_by_phone(phone)` | Lookup existing farmer at registration |
| `get_buyer_by_phone(phone)` | Lookup existing buyer at registration |
| `update_farmer(farmer_id, updates)` | Profile update with allowed-field whitelist |
| `update_buyer(buyer_id, updates)` | Profile update with allowed-field whitelist |
