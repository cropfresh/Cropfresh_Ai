"""
Authentication & Registration API Router
=========================================
REST endpoints for CropFresh farmer/buyer registration, OTP verification,
JWT token management, and profile CRUD.

Endpoints:
    POST   /api/v1/auth/register          — Initiate phone registration, send OTP
    POST   /api/v1/auth/verify-otp        — Verify OTP, return JWT token
    GET    /api/v1/auth/profile/{user_id} — Fetch farmer or buyer profile
    PATCH  /api/v1/auth/profile/{user_id} — Update profile fields
    GET    /api/v1/auth/me                — Decode JWT and return current user info
"""

# * AUTH ROUTER MODULE
# NOTE: RegistrationService resolved from app.state or instantiated per request
# NOTE: JWT verification is handled inline via RegistrationService.verify_token()

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from loguru import logger

from src.api.services.registration_service import (
    ProfileResponse,
    RegisterRequest,
    TokenResponse,
    UpdateBuyerProfileRequest,
    UpdateFarmerProfileRequest,
    VerifyOTPRequest,
    get_registration_service,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# * Dependency helpers
# ─────────────────────────────────────────────────────────────

def _service(request: Request):
    """Resolve RegistrationService from app.state or create a bare instance."""
    if hasattr(request.app.state, "registration_service"):
        return request.app.state.registration_service
    return get_registration_service(
        db=getattr(request.app.state, "db", None),
    )


def _extract_bearer_token(authorization: Optional[str]) -> str:
    """
    Extract token string from 'Bearer <token>' Authorization header.

    Raises:
        HTTPException 401 if header is missing or malformed.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Expected: Bearer <token>",
        )
    return parts[1]


# ─────────────────────────────────────────────────────────────
# * POST /auth/register — Initiate registration / send OTP
# ─────────────────────────────────────────────────────────────

@router.post(
    "/auth/register",
    status_code=200,
    summary="Register phone number and receive OTP",
    tags=["auth"],
    response_model=dict,
)
async def register(
    body: RegisterRequest,
    request: Request,
) -> dict:
    """
    Initiate phone-based registration for a farmer or buyer.

    - Accepts Indian mobile numbers (10-digit or +91 prefix).
    - Creates a minimal user record if the phone is new.
    - Sends a 6-digit OTP via SMS stub (Twilio/MSG91 in production).
    - OTP is valid for 10 minutes.

    **Dev note:** OTP is logged to server console — check terminal output.
    """
    svc = _service(request)
    try:
        result = await svc.register(body.phone, body.user_type)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Registration error for {body.phone}: {exc}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


# ─────────────────────────────────────────────────────────────
# * POST /auth/verify-otp — Verify OTP and return JWT
# ─────────────────────────────────────────────────────────────

@router.post(
    "/auth/verify-otp",
    response_model=TokenResponse,
    status_code=200,
    summary="Verify OTP and receive JWT access token",
    tags=["auth"],
)
async def verify_otp(
    body: VerifyOTPRequest,
    request: Request,
) -> TokenResponse:
    """
    Verify the OTP received via SMS and return a JWT access token.

    - Token is valid for **30 days**.
    - Token payload includes: `sub` (user_id), `type`, `phone`, `district`.
    - Use the token as `Authorization: Bearer <token>` for authenticated requests.
    """
    svc = _service(request)
    try:
        return await svc.verify_otp(body.phone, body.otp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"OTP verification error for {body.phone}: {exc}")
        raise HTTPException(status_code=500, detail="Verification failed. Please try again.")


# ─────────────────────────────────────────────────────────────
# * GET /auth/me — Decode JWT and return current user info
# ─────────────────────────────────────────────────────────────

@router.get(
    "/auth/me",
    status_code=200,
    summary="Get current user info from JWT",
    tags=["auth"],
    response_model=dict,
)
async def get_current_user(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
) -> dict:
    """
    Decode and validate the Bearer JWT and return the current user's info.

    Requires `Authorization: Bearer <token>` header.

    Returns the decoded JWT payload: user_id, user_type, phone, district.
    """
    token = _extract_bearer_token(authorization)
    svc = _service(request)
    try:
        payload = svc.verify_token(token)
        return {
            "user_id": payload.get("sub"),
            "user_type": payload.get("type"),
            "phone": payload.get("phone"),
            "district": payload.get("district"),
            "token_expires_at": payload.get("exp"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /auth/profile/{user_id} — Fetch profile
# ─────────────────────────────────────────────────────────────

@router.get(
    "/auth/profile/{user_id}",
    response_model=ProfileResponse,
    status_code=200,
    summary="Get farmer or buyer profile",
    tags=["auth"],
)
async def get_profile(
    user_id: str,
    request: Request,
    user_type: str = Query(default="farmer", description="'farmer' or 'buyer'"),
) -> ProfileResponse:
    """
    Fetch the complete profile for a farmer or buyer by UUID.

    - Pass `?user_type=buyer` to fetch a buyer profile.
    - Returns language preference, district, quality score, and type-specific fields.
    """
    svc = _service(request)
    try:
        profile = await svc.get_profile(user_id, user_type)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"{user_type.capitalize()} {user_id} not found",
            )
        return profile
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Profile fetch error for {user_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch profile.")


# ─────────────────────────────────────────────────────────────
# * PATCH /auth/profile/{user_id} — Update profile
# ─────────────────────────────────────────────────────────────

@router.patch(
    "/auth/profile/{user_id}",
    response_model=ProfileResponse,
    status_code=200,
    summary="Update farmer or buyer profile",
    tags=["auth"],
)
async def update_farmer_profile(
    user_id: str,
    body: UpdateFarmerProfileRequest,
    request: Request,
    user_type: str = Query(default="farmer", description="'farmer' or 'buyer'"),
) -> ProfileResponse:
    """
    Update profile fields for a farmer.

    - Auto-infers `language_pref` from `district` if not supplied.
    - GPS coordinates trigger automatic district detection.
    - Aadhaar last 4 digits are SHA-256 hashed before storage.

    Pass `?user_type=buyer` to update a buyer profile instead.
    """
    svc = _service(request)
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields provided to update.")
    try:
        profile = await svc.update_profile(user_id, user_type, updates)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"{user_type.capitalize()} {user_id} not found",
            )
        return profile
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Profile update error for {user_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to update profile.")


@router.patch(
    "/auth/buyer-profile/{user_id}",
    response_model=ProfileResponse,
    status_code=200,
    summary="Update buyer-specific profile fields",
    tags=["auth"],
)
async def update_buyer_profile(
    user_id: str,
    body: UpdateBuyerProfileRequest,
    request: Request,
) -> ProfileResponse:
    """
    Update profile fields specific to a buyer (type, credit_limit, subscription_tier).
    """
    svc = _service(request)
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields provided to update.")
    try:
        profile = await svc.update_profile(user_id, "buyer", updates)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Buyer {user_id} not found")
        return profile
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Buyer profile update error for {user_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to update buyer profile.")
