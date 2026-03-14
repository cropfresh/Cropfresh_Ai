"""
Registration Package — Re-exports for backward compatibility.
"""

from src.api.services.registration_pkg.models import (
    BUYER_TYPES,
    INITIAL_QUALITY_SCORE,
    JWT_EXPIRY_DAYS,
    KARNATAKA_DISTRICT_LANGUAGE,
    OTP_EXPIRY_MINUTES,
    OTP_LENGTH,
    USER_TYPES,
    OTPRecord,
    ProfileResponse,
    RegisterRequest,
    TokenResponse,
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
