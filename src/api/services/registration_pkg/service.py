"""
Registration Service — OTP auth, JWT tokens, and profile management.
"""

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

from src.api.services.registration_pkg.models import (
    _DEFAULT_JWT_SECRET,
    INITIAL_QUALITY_SCORE,
    JWT_EXPIRY_DAYS,
    KARNATAKA_DISTRICT_LANGUAGE,
    OTP_EXPIRY_MINUTES,
    OTP_LENGTH,
    USER_TYPES,
    OTPRecord,
    ProfileResponse,
    TokenResponse,
)


class RegistrationService:
    """Full farmer/buyer registration lifecycle for CropFresh."""

    def __init__(
        self,
        db: Optional[Any] = None,
        jwt_secret: str = _DEFAULT_JWT_SECRET,
        otp_store: Optional[dict[str, OTPRecord]] = None,
    ) -> None:
        self.db = db
        self._jwt_secret = jwt_secret
        self._otp_store: dict[str, OTPRecord] = otp_store if otp_store is not None else {}

    # ── Registration Flow ─────────────────────────────────────

    async def register(self, phone: str, user_type: str) -> dict[str, Any]:
        """Step 1: create/lookup user record and send OTP."""
        if user_type not in USER_TYPES:
            raise ValueError(f"user_type must be one of {USER_TYPES}")
        phone = self._normalise_phone(phone)
        otp = self._generate_otp()
        self._store_otp(phone, otp, user_type)
        await self._send_otp_stub(phone, otp)
        logger.info(f"OTP sent to {phone} (type={user_type})")
        return {
            "phone": phone, "user_type": user_type,
            "otp_sent": True,
            "message": f"OTP sent to {phone}. Valid for {OTP_EXPIRY_MINUTES} minutes.",
        }

    async def verify_otp(self, phone: str, otp: str) -> TokenResponse:
        """Step 2: verify OTP and return JWT token."""
        phone = self._normalise_phone(phone)
        record = self._validate_otp(phone, otp)
        user_type = record.user_type
        user_id, district, language_pref = await self._get_or_create_user(phone, user_type)
        self._otp_store.pop(phone, None)
        token = self._generate_token(
            user_id=user_id, user_type=user_type, phone=phone, district=district or "",
        )
        logger.info(f"OTP verified for {phone} — user_id={user_id} type={user_type}")
        return TokenResponse(
            access_token=token, user_id=user_id, user_type=user_type,
            phone=phone, district=district, language_pref=language_pref,
        )

    # ── Profile Management ────────────────────────────────────

    async def get_profile(self, user_id: str, user_type: str) -> Optional[ProfileResponse]:
        """Fetch complete user profile by UUID."""
        row = await self._fetch_user(user_id, user_type)
        return self._row_to_profile(row, user_type) if row else None

    async def update_profile(
        self, user_id: str, user_type: str, updates: dict[str, Any],
    ) -> Optional[ProfileResponse]:
        """Update farmer or buyer profile fields."""
        if updates.get("district") and not updates.get("language_pref"):
            updates["language_pref"] = self._district_to_language(updates["district"])
        if user_type == "farmer" and updates.get("latitude") and updates.get("longitude"):
            detected = self._gps_to_district(updates["latitude"], updates["longitude"])
            if detected and not updates.get("district"):
                updates["district"] = detected
                updates.setdefault("language_pref", self._district_to_language(detected))
        if updates.get("aadhaar_last4"):
            updates["aadhaar_hash"] = hashlib.sha256(
                updates.pop("aadhaar_last4").encode()
            ).hexdigest()
        await self._persist_update(user_id, user_type, updates)
        row = await self._fetch_user(user_id, user_type)
        return self._row_to_profile(row, user_type) if row else None

    # ── Voice Agent Compatibility ─────────────────────────────

    async def register_farmer(
        self, name: str, phone: str, district: str, village: Optional[str] = None,
    ) -> dict[str, Any]:
        """Direct farmer registration from voice agent (skips OTP)."""
        language_pref = self._district_to_language(district)
        farmer_data: dict[str, Any] = {
            "name": name, "phone": phone, "district": district,
            "language_pref": language_pref,
        }
        if village:
            farmer_data["village"] = village
        farmer_id = await self._persist_farmer(farmer_data)
        logger.info(f"Voice registration: farmer '{name}' ({phone}) → {farmer_id}")
        return {
            "farmer_id": farmer_id, "name": name, "phone": phone,
            "district": district, "language_pref": language_pref,
        }

    # ── JWT Helpers ───────────────────────────────────────────

    def _generate_token(self, user_id, user_type, phone, district) -> str:
        """Generate HS256 JWT using Python stdlib."""
        now = int(datetime.now(UTC).timestamp())
        exp = now + (JWT_EXPIRY_DAYS * 86400)
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps({
                "sub": user_id, "type": user_type, "phone": phone,
                "district": district, "iat": now, "exp": exp,
            }, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        signing_input = f"{header}.{payload}"
        signature = base64.urlsafe_b64encode(
            hmac.new(self._jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        return f"{signing_input}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token and return decoded payload."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token: expected 3 parts")
        header_b64, payload_b64, provided_sig = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(self._jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not secrets.compare_digest(provided_sig, expected_sig):
            raise ValueError("Invalid token signature")
        padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        now = int(datetime.now(UTC).timestamp())
        if payload.get("exp", 0) < now:
            raise ValueError("Token has expired")
        return payload

    # ── OTP Helpers ───────────────────────────────────────────

    def _generate_otp(self) -> str:
        return "".join(random.choices(string.digits, k=OTP_LENGTH))

    def _store_otp(self, phone, otp, user_type):
        self._otp_store[phone] = OTPRecord(
            otp=otp, user_type=user_type,
            expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=OTP_EXPIRY_MINUTES),
        )

    def _validate_otp(self, phone, otp) -> OTPRecord:
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

    async def _send_otp_stub(self, phone, otp):
        logger.info(f"[OTP-STUB] {phone} → {otp} (expires in {OTP_EXPIRY_MINUTES}m)")

    @staticmethod
    def _normalise_phone(phone: str) -> str:
        digits = "".join(ch for ch in phone if ch.isdigit())
        if len(digits) == 10:
            return f"+91{digits}"
        if len(digits) == 12 and digits.startswith("91"):
            return f"+{digits}"
        return phone

    @staticmethod
    def _district_to_language(district: str) -> str:
        return KARNATAKA_DISTRICT_LANGUAGE.get(district.lower().strip(), "kn")

    @staticmethod
    def _gps_to_district(latitude: float, longitude: float) -> Optional[str]:
        if not (11.5 <= latitude <= 18.5 and 74.0 <= longitude <= 78.5):
            return None
        centroids = {
            "Bangalore Urban": (12.97, 77.56), "Mysuru": (12.29, 76.64),
            "Tumakuru": (13.34, 77.10), "Kolar": (13.13, 78.13),
            "Mandya": (12.52, 76.90), "Hassan": (13.00, 76.10),
            "Davanagere": (14.46, 75.92), "Ballari": (15.14, 76.92),
            "Dharwad": (15.46, 75.01), "Belagavi": (15.85, 74.50),
        }
        nearest = min(
            centroids.items(),
            key=lambda kv: (kv[1][0] - latitude) ** 2 + (kv[1][1] - longitude) ** 2,
        )
        return nearest[0]

    # ── DB helpers ────────────────────────────────────────────

    async def _get_or_create_user(self, phone, user_type) -> tuple[str, Optional[str], str]:
        if self.db is None:
            user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_type}:{phone}"))
            return user_id, None, "kn"
        if user_type == "farmer":
            row = await self.db.get_farmer_by_phone(phone)
            if row:
                return str(row["id"]), row.get("district"), row.get("language_pref", "kn")
            farmer_id = await self.db.create_farmer({"name": "", "phone": phone, "district": "Unknown"})
            return farmer_id, None, "kn"
        row = await self.db.get_buyer_by_phone(phone)
        if row:
            return str(row["id"]), row.get("district"), "kn"
        buyer_id = await self.db.create_buyer(
            {"name": "", "phone": phone, "type": "retailer", "district": "Unknown"}
        )
        return buyer_id, None, "kn"

    async def _fetch_user(self, user_id, user_type) -> Optional[dict[str, Any]]:
        if self.db is None:
            return None
        if user_type == "farmer":
            return await self.db.get_farmer(user_id) if hasattr(self.db, "get_farmer") else None
        return await self.db.get_buyer(user_id) if hasattr(self.db, "get_buyer") else None

    async def _persist_farmer(self, farmer_data) -> str:
        if self.db and hasattr(self.db, "create_farmer"):
            return await self.db.create_farmer(farmer_data)
        return str(uuid.uuid4())

    async def _persist_update(self, user_id, user_type, updates):
        if self.db is None:
            return
        if user_type == "farmer" and hasattr(self.db, "update_farmer"):
            await self.db.update_farmer(user_id, updates)
        elif user_type == "buyer" and hasattr(self.db, "update_buyer"):
            await self.db.update_buyer(user_id, updates)

    @staticmethod
    def _row_to_profile(row: dict[str, Any], user_type: str) -> ProfileResponse:
        return ProfileResponse(
            user_id=str(row.get("id", "")), user_type=user_type,
            phone=row.get("phone", ""), name=row.get("name") or None,
            district=row.get("district") or None, village=row.get("village") or None,
            language_pref=row.get("language_pref", "kn"),
            quality_score=float(row.get("quality_score", INITIAL_QUALITY_SCORE)),
            is_active=bool(row.get("is_active", True)),
            created_at=row.get("created_at"),
            buyer_type=row.get("type") or row.get("buyer_type") or None,
            credit_limit=float(row["credit_limit"]) if row.get("credit_limit") is not None else None,
            subscription_tier=row.get("subscription_tier") or None,
        )


def get_registration_service(
    db=None, jwt_secret=None, otp_store=None,
) -> RegistrationService:
    """Factory for creating a RegistrationService."""
    import os
    secret = jwt_secret or os.environ.get("JWT_SECRET", _DEFAULT_JWT_SECRET)
    return RegistrationService(db=db, jwt_secret=secret, otp_store=otp_store)
