"""
Registration Service — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.api.services.registration_pkg` package.
! Import from `src.api.services.registration_pkg` directly in new code.
"""

from src.api.services.registration_pkg.models import (
    BUYER_TYPES,
    INITIAL_QUALITY_SCORE,
    JWT_EXPIRY_DAYS,
    KARNATAKA_DISTRICT_LANGUAGE,
    OTP_EXPIRY_MINUTES,
    OTP_LENGTH,
    OTPRecord,
    ProfileResponse,
    RegisterRequest,
    TokenResponse,
    USER_TYPES,
    UpdateBuyerProfileRequest,
    UpdateFarmerProfileRequest,
    VerifyOTPRequest,
)
from src.api.services.registration_pkg.service import (
    RegistrationService,
    get_registration_service,
)

__all__ = [
    "BUYER_TYPES",
    "INITIAL_QUALITY_SCORE",
    "JWT_EXPIRY_DAYS",
    "KARNATAKA_DISTRICT_LANGUAGE",
    "OTP_EXPIRY_MINUTES",
    "OTP_LENGTH",
    "OTPRecord",
    "ProfileResponse",
    "RegisterRequest",
    "RegistrationService",
    "TokenResponse",
    "USER_TYPES",
    "UpdateBuyerProfileRequest",
    "UpdateFarmerProfileRequest",
    "VerifyOTPRequest",
    "get_registration_service",
]
