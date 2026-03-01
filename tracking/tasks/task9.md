# Task 9: Build Farmer/Buyer Registration & Profile Service

> **Priority:** 🟠 P1 | **Phase:** 2 | **Effort:** 2–3 days  
> **Files:** `src/api/services/registration_service.py` [NEW], `src/api/routers/auth.py` [NEW]  
> **Score Target:** 9/10 — OTP-based registration with profile management

---

## 📌 Problem Statement

No registration or profile management exists. Farmers and buyers need to register (phone + OTP), set preferences (language, location), and manage their profiles.

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
    
    async def verify_otp(self, phone: str, otp: str) -> dict:
        """Verify OTP, return JWT token."""
    
    async def update_profile(self, user_id: str, data: dict) -> dict:
        """Update name, district, GPS, language preference."""
    
    async def get_profile(self, user_id: str) -> dict:
        """Fetch complete user profile with stats."""
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
- Auto-detect district from GPS coordinates
- Language preference default from district (Karnataka → Kannada)
- Quality score initialized at 0.5 (neutral)
- First listing creation triggers onboarding flow

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Phone-based registration with OTP (mock) | 25% |
| 2 | JWT token generation and validation | 25% |
| 3 | Profile CRUD with language preference | 20% |
| 4 | Separate farmer/buyer profiles with type-specific fields | 15% |
| 5 | Voice agent `register` intent creates profile | 15% |
