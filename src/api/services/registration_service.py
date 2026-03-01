"""
Farmer/Buyer Registration & Profile Service
============================================
OTP-based phone authentication, JWT token management, and profile CRUD
for the CropFresh marketplace.

Responsibilities:
- Phone-based registration with OTP (mock in dev, Twilio/MSG91 in prod)
- JWT token generation and validation (HS256 using stdlib hmac)
- Farmer/buyer profile creation with type-specific fields
- District auto-detection from GPS coordinates (Karnataka geofence)
- Language preference inferred from district (Karnataka → Kannada default)
- Quality score initialized at 0.5 (neutral) for new farmers
- Voice agent compatibility: register_farmer() method

JWT Structure:
    {"sub": "user_uuid", "type": "farmer|buyer", "phone": "+91...",
     "district": "Bangalore Rural", "iat": epoch, "exp": epoch + 30d}
"""

# * REGISTRATION SERVICE MODULE
# NOTE: OTP store uses in-memory dict for dev/test; swap for Redis in production
# NOTE: JWT uses stdlib hmac+hashlib to avoid pyjwt dependency

import base64
import hashlib
import hmac
import json
import random
import secrets
import string
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════
# * Constants
# ═══════════════════════════════════════════════════════════════

JWT_EXPIRY_DAYS: int = 30
OTP_LENGTH: int = 6
OTP_EXPIRY_MINUTES: int = 10

# * Default JWT secret — override via JWT_SECRET env variable in production
_DEFAULT_JWT_SECRET: str = "cropfresh-dev-jwt-secret-change-in-production"

# * Karnataka district → default language mapping
# NOTE: Covers all 31 Karnataka districts; fallback is Kannada
KARNATAKA_DISTRICT_LANGUAGE: dict[str, str] = {
    "bangalore": "kn",
    "bangalore urban": "kn",
    "bangalore rural": "kn",
    "bengaluru": "kn",
    "bengaluru urban": "kn",
    "bengaluru rural": "kn",
    "mysuru": "kn",
    "mysore": "kn",
    "tumkur": "kn",
    "tumakuru": "kn",
    "kolar": "kn",
    "chikkaballapur": "kn",
    "ramanagara": "kn",
    "mandya": "kn",
    "hassan": "kn",
    "chikkamagaluru": "kn",
    "shivamogga": "kn",
    "shimoga": "kn",
    "davanagere": "kn",
    "chitradurga": "kn",
    "bellary": "kn",
    "ballari": "kn",
    "raichur": "kn",
    "koppal": "kn",
    "gadag": "kn",
    "dharwad": "kn",
    "belagavi": "kn",
    "belgaum": "kn",
    "vijayapura": "kn",
    "bijapur": "kn",
    "bagalkot": "kn",
    "haveri": "kn",
    "uttara kannada": "kn",
    "udupi": "kn",
    "dakshina kannada": "kn",
    "kodagu": "kn",
    "chamarajanagar": "kn",
    "bidar": "kn",
    "kalaburagi": "kn",
    "gulbarga": "kn",
    "yadgir": "kn",
}

# * Buyer type options
BUYER_TYPES: set[str] = {"retailer", "restaurant", "hotel", "exporter", "wholesaler", "other"}

# * User types
USER_TYPES: set[str] = {"farmer", "buyer"}

# * Initial quality score for new users
INITIAL_QUALITY_SCORE: float = 0.5


# ═══════════════════════════════════════════════════════════════
# * Pydantic Models
# ═══════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    """Initial registration — phone number + user type only."""
    phone: str = Field(min_length=10, max_length=15, description="Mobile number (Indian format)")
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
    subscription_tier: Optional[str] = Field(default=None, description="free | standard | premium")

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
    # Farmer-specific
    buyer_type: Optional[str] = None
    credit_limit: Optional[float] = None
    subscription_tier: Optional[str] = None


class OTPRecord(BaseModel):
    """In-memory OTP record with expiry."""
    otp: str
    user_type: str
    user_id: Optional[str] = None
    expires_at: datetime


# ═══════════════════════════════════════════════════════════════
# * RegistrationService
# ═══════════════════════════════════════════════════════════════

class RegistrationService:
    """
    Full farmer/buyer registration lifecycle for CropFresh.

    Dependencies (all optional — service degrades gracefully):
        db: AuroraPostgresClient — for persistence
        jwt_secret: str — HMAC-SHA256 signing key (defaults to dev key)
        otp_store: dict — in-memory OTP cache (swap for Redis in prod)

    Usage:
        svc = RegistrationService(db=client, jwt_secret=os.environ["JWT_SECRET"])
        result = await svc.register("+919876543210", "farmer")
        token = await svc.verify_otp("+919876543210", "123456")
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        jwt_secret: str = _DEFAULT_JWT_SECRET,
        otp_store: Optional[dict[str, OTPRecord]] = None,
    ) -> None:
        self.db = db
        self._jwt_secret = jwt_secret
        # * In-memory OTP store — shared across instances in tests; swappable for Redis
        self._otp_store: dict[str, OTPRecord] = otp_store if otp_store is not None else {}

    # ─────────────────────────────────────────────────────────
    # * Registration Flow
    # ─────────────────────────────────────────────────────────

    async def register(self, phone: str, user_type: str) -> dict[str, Any]:
        """
        Step 1 of registration: create/lookup user record and send OTP.

        Args:
            phone: Indian mobile number — normalised to +91XXXXXXXXXX internally.
            user_type: 'farmer' or 'buyer'.

        Returns:
            Dict with phone, user_type, otp_sent=True, message.
        """
        if user_type not in USER_TYPES:
            raise ValueError(f"user_type must be one of {USER_TYPES}")

        phone = self._normalise_phone(phone)

        otp = self._generate_otp()
        self._store_otp(phone, otp, user_type)

        # ! NOTE: In production, call Twilio/MSG91 here
        # TODO: Replace mock OTP send with real SMS provider
        await self._send_otp_stub(phone, otp)

        logger.info(f"OTP sent to {phone} (type={user_type})")
        return {
            "phone": phone,
            "user_type": user_type,
            "otp_sent": True,
            "message": f"OTP sent to {phone}. Valid for {OTP_EXPIRY_MINUTES} minutes.",
        }

    async def verify_otp(self, phone: str, otp: str) -> TokenResponse:
        """
        Step 2 of registration: verify OTP and return JWT token.

        Creates the user record (farmer or buyer) in DB on first call.
        On subsequent calls (existing user), returns a fresh token.

        Args:
            phone: Mobile number (+91 prefix).
            otp: 6-digit OTP code.

        Returns:
            TokenResponse with JWT access_token.

        Raises:
            ValueError: If OTP is invalid or expired.
        """
        phone = self._normalise_phone(phone)
        record = self._validate_otp(phone, otp)
        user_type = record.user_type

        # * Look up or create user in DB
        user_id, district, language_pref = await self._get_or_create_user(
            phone, user_type
        )

        # * Clean up used OTP
        self._otp_store.pop(phone, None)

        # * Generate JWT
        token = self._generate_token(
            user_id=user_id,
            user_type=user_type,
            phone=phone,
            district=district or "",
        )

        logger.info(f"OTP verified for {phone} — user_id={user_id} type={user_type}")

        return TokenResponse(
            access_token=token,
            user_id=user_id,
            user_type=user_type,
            phone=phone,
            district=district,
            language_pref=language_pref,
        )

    # ─────────────────────────────────────────────────────────
    # * Profile Management
    # ─────────────────────────────────────────────────────────

    async def get_profile(self, user_id: str, user_type: str) -> Optional[ProfileResponse]:
        """
        Fetch complete user profile by UUID.

        Args:
            user_id: User UUID string.
            user_type: 'farmer' or 'buyer'.

        Returns:
            ProfileResponse, or None if not found.
        """
        row = await self._fetch_user(user_id, user_type)
        if row is None:
            return None
        return self._row_to_profile(row, user_type)

    async def update_profile(
        self,
        user_id: str,
        user_type: str,
        updates: dict[str, Any],
    ) -> Optional[ProfileResponse]:
        """
        Update farmer or buyer profile fields.

        Handles district → language inference and GPS → district lookup.

        Args:
            user_id: User UUID string.
            user_type: 'farmer' or 'buyer'.
            updates: Dict of fields to update.

        Returns:
            Updated ProfileResponse, or None if user not found.
        """
        # * Auto-infer language from district if not explicitly set
        if updates.get("district") and not updates.get("language_pref"):
            updates["language_pref"] = self._district_to_language(updates["district"])

        # * Auto-detect district from GPS if provided (farmer only)
        if user_type == "farmer" and updates.get("latitude") and updates.get("longitude"):
            detected = self._gps_to_district(updates["latitude"], updates["longitude"])
            if detected and not updates.get("district"):
                updates["district"] = detected
                updates.setdefault("language_pref", self._district_to_language(detected))

        # * Hash aadhaar last 4 digits if provided
        if updates.get("aadhaar_last4"):
            updates["aadhaar_hash"] = hashlib.sha256(
                updates.pop("aadhaar_last4").encode()
            ).hexdigest()

        await self._persist_update(user_id, user_type, updates)

        row = await self._fetch_user(user_id, user_type)
        if row is None:
            return None
        return self._row_to_profile(row, user_type)

    # ─────────────────────────────────────────────────────────
    # * Voice Agent Compatibility
    # ─────────────────────────────────────────────────────────

    async def register_farmer(
        self,
        name: str,
        phone: str,
        district: str,
        village: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Direct farmer registration from voice agent (skips OTP flow).

        Used by VoiceAgent._handle_register() to create a minimal farmer
        profile from voice-captured name, phone, and district.

        Args:
            name: Farmer's name (from voice).
            phone: Mobile number.
            district: Karnataka district.
            village: Optional village name.

        Returns:
            Dict with farmer_id, name, district, language_pref.
        """
        language_pref = self._district_to_language(district)
        farmer_data: dict[str, Any] = {
            "name": name,
            "phone": phone,
            "district": district,
            "language_pref": language_pref,
        }
        if village:
            farmer_data["village"] = village

        farmer_id: str = await self._persist_farmer(farmer_data)

        logger.info(f"Voice registration: farmer '{name}' ({phone}) → {farmer_id}")
        return {
            "farmer_id": farmer_id,
            "name": name,
            "phone": phone,
            "district": district,
            "language_pref": language_pref,
        }

    # ─────────────────────────────────────────────────────────
    # * JWT Helpers
    # ─────────────────────────────────────────────────────────

    def _generate_token(
        self,
        user_id: str,
        user_type: str,
        phone: str,
        district: str,
    ) -> str:
        """
        Generate a HS256 JWT using Python stdlib (no external deps).

        Header: {"alg": "HS256", "typ": "JWT"}
        Payload: {sub, type, phone, district, iat, exp}

        Returns:
            Signed JWT string.
        """
        now = int(datetime.now(UTC).timestamp())
        exp = now + (JWT_EXPIRY_DAYS * 86400)

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()

        payload = base64.urlsafe_b64encode(
            json.dumps({
                "sub": user_id,
                "type": user_type,
                "phone": phone,
                "district": district,
                "iat": now,
                "exp": exp,
            }, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()

        signing_input = f"{header}.{payload}"
        signature = base64.urlsafe_b64encode(
            hmac.new(
                self._jwt_secret.encode(),
                signing_input.encode(),
                hashlib.sha256,
            ).digest()
        ).rstrip(b"=").decode()

        return f"{signing_input}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any]:
        """
        Verify a JWT token and return its decoded payload.

        Args:
            token: JWT string to verify.

        Returns:
            Decoded payload dict.

        Raises:
            ValueError: If token is malformed, signature invalid, or expired.
        """
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token: expected 3 parts")

        header_b64, payload_b64, provided_sig = parts
        signing_input = f"{header_b64}.{payload_b64}"

        expected_sig = base64.urlsafe_b64encode(
            hmac.new(
                self._jwt_secret.encode(),
                signing_input.encode(),
                hashlib.sha256,
            ).digest()
        ).rstrip(b"=").decode()

        if not secrets.compare_digest(provided_sig, expected_sig):
            raise ValueError("Invalid token signature")

        # * Decode payload (pad to multiple of 4 for base64)
        padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())

        now = int(datetime.now(UTC).timestamp())
        if payload.get("exp", 0) < now:
            raise ValueError("Token has expired")

        return payload

    # ─────────────────────────────────────────────────────────
    # * OTP Helpers
    # ─────────────────────────────────────────────────────────

    def _generate_otp(self) -> str:
        """Generate a numeric OTP of length OTP_LENGTH."""
        return "".join(random.choices(string.digits, k=OTP_LENGTH))

    def _store_otp(self, phone: str, otp: str, user_type: str) -> None:
        """Store OTP in memory with expiry timestamp."""
        self._otp_store[phone] = OTPRecord(
            otp=otp,
            user_type=user_type,
            expires_at=datetime.now(UTC).replace(tzinfo=None)
            + timedelta(minutes=OTP_EXPIRY_MINUTES),
        )

    def _validate_otp(self, phone: str, otp: str) -> OTPRecord:
        """
        Validate OTP value and expiry.

        Returns:
            OTPRecord if valid.

        Raises:
            ValueError: If OTP not found, expired, or incorrect.
        """
        record = self._otp_store.get(phone)
        if record is None:
            raise ValueError(f"No OTP found for {phone}. Please request a new OTP.")

        now = datetime.now(UTC).replace(tzinfo=None)
        if now > record.expires_at:
            self._otp_store.pop(phone, None)
            raise ValueError("OTP has expired. Please request a new OTP.")

        if not secrets.compare_digest(record.otp, otp):
            raise ValueError("Incorrect OTP. Please try again.")

        return record

    async def _send_otp_stub(self, phone: str, otp: str) -> None:
        """
        Stub: log OTP in dev mode. Replace with Twilio/MSG91 in production.

        # TODO: Wire to Twilio or MSG91 SMS provider
        """
        # ! NOTE: OTP is logged for dev only — never log OTPs in production
        logger.info(f"[OTP-STUB] {phone} → {otp} (expires in {OTP_EXPIRY_MINUTES}m)")

    # ─────────────────────────────────────────────────────────
    # * Phone normalisation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_phone(phone: str) -> str:
        """Normalise Indian mobile number to +91XXXXXXXXXX format."""
        digits = "".join(ch for ch in phone if ch.isdigit())
        if len(digits) == 10:
            return f"+91{digits}"
        if len(digits) == 12 and digits.startswith("91"):
            return f"+{digits}"
        return phone

    # ─────────────────────────────────────────────────────────
    # * District / Language Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _district_to_language(district: str) -> str:
        """
        Map Karnataka district name to language preference.

        Returns 'kn' (Kannada) for all Karnataka districts.
        Falls back to 'kn' for unknown districts.
        """
        key = district.lower().strip()
        return KARNATAKA_DISTRICT_LANGUAGE.get(key, "kn")

    @staticmethod
    def _gps_to_district(latitude: float, longitude: float) -> Optional[str]:
        """
        Mock GPS → district lookup for Karnataka.

        TODO: Replace with PostGIS reverse geocoding:
            SELECT district FROM karnataka_boundaries
            WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint($lon,$lat), 4326))

        Returns:
            District name string, or None if outside Karnataka bounds.
        """
        # * Karnataka bounding box: 11.5°N–18.5°N, 74.0°E–78.5°E
        if not (11.5 <= latitude <= 18.5 and 74.0 <= longitude <= 78.5):
            return None

        # * Simplified centroid-based lookup (nearest major district)
        centroids: dict[str, tuple[float, float]] = {
            "Bangalore Urban": (12.97, 77.56),
            "Mysuru": (12.29, 76.64),
            "Tumakuru": (13.34, 77.10),
            "Kolar": (13.13, 78.13),
            "Mandya": (12.52, 76.90),
            "Hassan": (13.00, 76.10),
            "Davanagere": (14.46, 75.92),
            "Ballari": (15.14, 76.92),
            "Dharwad": (15.46, 75.01),
            "Belagavi": (15.85, 74.50),
        }

        nearest = min(
            centroids.items(),
            key=lambda kv: (kv[1][0] - latitude) ** 2 + (kv[1][1] - longitude) ** 2,
        )
        return nearest[0]

    # ─────────────────────────────────────────────────────────
    # * Private — DB helpers
    # ─────────────────────────────────────────────────────────

    async def _get_or_create_user(
        self, phone: str, user_type: str
    ) -> tuple[str, Optional[str], str]:
        """
        Look up existing user by phone or create a minimal record.

        Returns:
            (user_id, district, language_pref) tuple.
        """
        if self.db is None:
            # * No DB — return a deterministic UUID in test/dev mode
            user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_type}:{phone}"))
            return user_id, None, "kn"

        if user_type == "farmer":
            row = await self.db.get_farmer_by_phone(phone)
            if row:
                return (
                    str(row["id"]),
                    row.get("district"),
                    row.get("language_pref", "kn"),
                )
            farmer_id = await self.db.create_farmer(
                {"name": "", "phone": phone, "district": "Unknown"}
            )
            return farmer_id, None, "kn"

        # * Buyer path
        row = await self.db.get_buyer_by_phone(phone)
        if row:
            return (
                str(row["id"]),
                row.get("district"),
                "kn",
            )
        buyer_id = await self.db.create_buyer(
            {"name": "", "phone": phone, "type": "retailer", "district": "Unknown"}
        )
        return buyer_id, None, "kn"

    async def _fetch_user(
        self, user_id: str, user_type: str
    ) -> Optional[dict[str, Any]]:
        """Fetch farmer or buyer row by UUID."""
        if self.db is None:
            return None
        if user_type == "farmer":
            if hasattr(self.db, "get_farmer"):
                return await self.db.get_farmer(user_id)
            return None
        if hasattr(self.db, "get_buyer"):
            return await self.db.get_buyer(user_id)
        return None

    async def _persist_farmer(self, farmer_data: dict[str, Any]) -> str:
        """Persist a new farmer; return UUID."""
        if self.db and hasattr(self.db, "create_farmer"):
            return await self.db.create_farmer(farmer_data)
        return str(uuid.uuid4())

    async def _persist_update(
        self, user_id: str, user_type: str, updates: dict[str, Any]
    ) -> None:
        """Write profile updates to DB."""
        if self.db is None:
            return
        if user_type == "farmer" and hasattr(self.db, "update_farmer"):
            await self.db.update_farmer(user_id, updates)
        elif user_type == "buyer" and hasattr(self.db, "update_buyer"):
            await self.db.update_buyer(user_id, updates)

    # ─────────────────────────────────────────────────────────
    # * Private — row converters
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_profile(row: dict[str, Any], user_type: str) -> ProfileResponse:
        """Convert a DB row to a ProfileResponse."""
        return ProfileResponse(
            user_id=str(row.get("id", "")),
            user_type=user_type,
            phone=row.get("phone", ""),
            name=row.get("name") or None,
            district=row.get("district") or None,
            village=row.get("village") or None,
            language_pref=row.get("language_pref", "kn"),
            quality_score=float(row.get("quality_score", INITIAL_QUALITY_SCORE)),
            is_active=bool(row.get("is_active", True)),
            created_at=row.get("created_at"),
            buyer_type=row.get("type") or row.get("buyer_type") or None,
            credit_limit=float(row["credit_limit"]) if row.get("credit_limit") is not None else None,
            subscription_tier=row.get("subscription_tier") or None,
        )


# ═══════════════════════════════════════════════════════════════
# * Module-level factory
# ═══════════════════════════════════════════════════════════════

def get_registration_service(
    db: Optional[Any] = None,
    jwt_secret: Optional[str] = None,
    otp_store: Optional[dict] = None,
) -> RegistrationService:
    """
    Factory for creating a RegistrationService with injected dependencies.

    Args:
        db: AuroraPostgresClient instance.
        jwt_secret: HMAC signing key (reads JWT_SECRET env var if not supplied).
        otp_store: Optional shared in-memory OTP dict (useful in testing).

    Returns:
        Configured RegistrationService.
    """
    import os
    secret = jwt_secret or os.environ.get("JWT_SECRET", _DEFAULT_JWT_SECRET)
    return RegistrationService(db=db, jwt_secret=secret, otp_store=otp_store)
