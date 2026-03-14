"""
Registration service data models, constants, and configuration.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ═══════════════════════════════════════════════════════════════
# * Constants
# ═══════════════════════════════════════════════════════════════

JWT_EXPIRY_DAYS: int = 30
OTP_LENGTH: int = 6
OTP_EXPIRY_MINUTES: int = 10

_DEFAULT_JWT_SECRET: str = "cropfresh-dev-jwt-secret-change-in-production"

# * Karnataka district → default language mapping
KARNATAKA_DISTRICT_LANGUAGE: dict[str, str] = {
    "bangalore": "kn", "bangalore urban": "kn", "bangalore rural": "kn",
    "bengaluru": "kn", "bengaluru urban": "kn", "bengaluru rural": "kn",
    "mysuru": "kn", "mysore": "kn", "tumkur": "kn", "tumakuru": "kn",
    "kolar": "kn", "chikkaballapur": "kn", "ramanagara": "kn",
    "mandya": "kn", "hassan": "kn", "chikkamagaluru": "kn",
    "shivamogga": "kn", "shimoga": "kn", "davanagere": "kn",
    "chitradurga": "kn", "bellary": "kn", "ballari": "kn",
    "raichur": "kn", "koppal": "kn", "gadag": "kn", "dharwad": "kn",
    "belagavi": "kn", "belgaum": "kn", "vijayapura": "kn",
    "bijapur": "kn", "bagalkot": "kn", "haveri": "kn",
    "uttara kannada": "kn", "udupi": "kn", "dakshina kannada": "kn",
    "kodagu": "kn", "chamarajanagar": "kn", "bidar": "kn",
    "kalaburagi": "kn", "gulbarga": "kn", "yadgir": "kn",
}

BUYER_TYPES: set[str] = {"retailer", "restaurant", "hotel", "exporter", "wholesaler", "other"}
USER_TYPES: set[str] = {"farmer", "buyer"}
INITIAL_QUALITY_SCORE: float = 0.5


# ═══════════════════════════════════════════════════════════════
# * Pydantic Models
# ═══════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    """Initial registration — phone number + user type only."""
    phone: str = Field(min_length=10, max_length=15)
    user_type: str = Field(description="'farmer' or 'buyer'")

    @field_validator("user_type")
    @classmethod
    def validate_user_type(cls, v: str) -> str:
        if v not in USER_TYPES:
            raise ValueError(f"user_type must be one of {USER_TYPES}")
        return v

    @field_validator("phone")
    @classmethod
    def normalise_phone(cls, v: str) -> str:
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) == 10:
            return f"+91{digits}"
        if len(digits) == 12 and digits.startswith("91"):
            return f"+{digits}"
        return v


class VerifyOTPRequest(BaseModel):
    """OTP verification — returns JWT on success."""
    phone: str
    otp: str = Field(min_length=4, max_length=8)

    @field_validator("phone")
    @classmethod
    def normalise_phone(cls, v: str) -> str:
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) == 10:
            return f"+91{digits}"
        if len(digits) == 12 and digits.startswith("91"):
            return f"+{digits}"
        return v


class UpdateFarmerProfileRequest(BaseModel):
    """Farmer-specific profile fields."""
    name: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    language_pref: Optional[str] = Field(default=None, description="kn | hi | en | ta | te")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    aadhaar_last4: Optional[str] = Field(default=None, min_length=4, max_length=4)


class UpdateBuyerProfileRequest(BaseModel):
    """Buyer-specific profile fields."""
    name: Optional[str] = None
    district: Optional[str] = None
    buyer_type: Optional[str] = None
    credit_limit: Optional[float] = Field(default=None, ge=0)
    subscription_tier: Optional[str] = Field(default=None)

    @field_validator("buyer_type")
    @classmethod
    def validate_buyer_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in BUYER_TYPES:
            raise ValueError(f"buyer_type must be one of {BUYER_TYPES}")
        return v


class TokenResponse(BaseModel):
    """JWT token returned after successful OTP verification."""
    access_token: str
    token_type: str = "bearer"
    expires_in_days: int = JWT_EXPIRY_DAYS
    user_id: str
    user_type: str
    phone: str
    district: Optional[str] = None
    language_pref: str


class ProfileResponse(BaseModel):
    """Complete user profile returned for farmer or buyer."""
    user_id: str
    user_type: str
    phone: str
    name: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    language_pref: str = "kn"
    quality_score: float = INITIAL_QUALITY_SCORE
    is_active: bool = True
    created_at: Optional[datetime] = None
    buyer_type: Optional[str] = None
    credit_limit: Optional[float] = None
    subscription_tier: Optional[str] = None


class OTPRecord(BaseModel):
    """In-memory OTP record with expiry."""
    otp: str
    user_type: str
    user_id: Optional[str] = None
    expires_at: datetime
