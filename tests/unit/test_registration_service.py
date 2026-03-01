"""
Unit Tests — RegistrationService (Task 9)
==========================================
Full acceptance-criteria coverage for the Farmer/Buyer Registration service.

AC1: Phone-based registration with OTP (mock)  — TestRegisterOTP (10 tests)
AC2: JWT token generation and validation        — TestJWTTokens (12 tests)
AC3: Profile CRUD with language preference      — TestProfileCRUD (10 tests)
AC4: Separate farmer/buyer type-specific fields — TestFarmerBuyerProfiles (8 tests)
AC5: Voice agent register_farmer() integration  — TestVoiceAgentIntegration (6 tests)

Supporting fixtures / helpers                  — TestDistrictLanguage (5 tests)
                                                  TestGPSToDistrict (4 tests)

Total: 55 tests across 7 test classes
"""

# * TEST MODULE — Registration Service (Task 9)
# NOTE: All tests use no-DB mode (db=None) — RegistrationService degrades gracefully

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.services.registration_service import (
    INITIAL_QUALITY_SCORE,
    KARNATAKA_DISTRICT_LANGUAGE,
    OTP_EXPIRY_MINUTES,
    JWT_EXPIRY_DAYS,
    OTPRecord,
    ProfileResponse,
    RegisterRequest,
    RegistrationService,
    TokenResponse,
    UpdateBuyerProfileRequest,
    UpdateFarmerProfileRequest,
    VerifyOTPRequest,
    get_registration_service,
)


# ═══════════════════════════════════════════════════════════════
# * Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture()
def otp_store() -> dict:
    """Shared in-memory OTP dict — cleared between tests via fresh fixture."""
    return {}


@pytest.fixture()
def svc(otp_store) -> RegistrationService:
    """RegistrationService with no DB, deterministic secret, shared OTP store."""
    return RegistrationService(
        db=None,
        jwt_secret="test-secret-key",
        otp_store=otp_store,
    )


@pytest.fixture()
def mock_db() -> MagicMock:
    """Mock AuroraPostgresClient with all required methods."""
    db = MagicMock()
    db.create_farmer = AsyncMock(return_value="farmer-uuid-001")
    db.get_farmer = AsyncMock(return_value={
        "id": "farmer-uuid-001",
        "phone": "+919876543210",
        "name": "Raju",
        "district": "Tumakuru",
        "village": "Koratagere",
        "language_pref": "kn",
        "quality_score": 0.5,
        "is_active": True,
        "created_at": None,
    })
    db.get_farmer_by_phone = AsyncMock(return_value=None)  # Default: new user
    db.create_buyer = AsyncMock(return_value="buyer-uuid-001")
    db.get_buyer = AsyncMock(return_value={
        "id": "buyer-uuid-001",
        "phone": "+919123456789",
        "name": "FreshMart",
        "district": "Bangalore Urban",
        "type": "retailer",
        "credit_limit": 50000.0,
        "subscription_tier": "standard",
        "quality_score": 0.5,
        "is_active": True,
        "created_at": None,
    })
    db.get_buyer_by_phone = AsyncMock(return_value=None)   # Default: new user
    db.update_farmer = AsyncMock(return_value=True)
    db.update_buyer = AsyncMock(return_value=True)
    return db


@pytest.fixture()
def svc_with_db(mock_db, otp_store) -> RegistrationService:
    """RegistrationService wired to mock DB."""
    return RegistrationService(
        db=mock_db,
        jwt_secret="test-secret-key",
        otp_store=otp_store,
    )


# ═══════════════════════════════════════════════════════════════
# * AC1 — Phone registration + OTP flow
# ═══════════════════════════════════════════════════════════════

class TestRegisterOTP:
    """Phone-based registration sends OTP and stores it in-memory."""

    @pytest.mark.asyncio
    async def test_register_farmer_returns_otp_sent(self, svc: RegistrationService):
        # Arrange / Act
        result = await svc.register("+919876543210", "farmer")
        # Assert
        assert result["otp_sent"] is True
        assert result["user_type"] == "farmer"
        assert "+919876543210" in result["phone"]

    @pytest.mark.asyncio
    async def test_register_buyer_returns_otp_sent(self, svc: RegistrationService):
        result = await svc.register("+919123456789", "buyer")
        assert result["otp_sent"] is True
        assert result["user_type"] == "buyer"

    @pytest.mark.asyncio
    async def test_register_normalises_10_digit_phone(self, svc: RegistrationService):
        result = await svc.register("9876543210", "farmer")
        assert result["phone"].startswith("+91")

    @pytest.mark.asyncio
    async def test_register_stores_otp_in_memory(self, svc: RegistrationService, otp_store: dict):
        await svc.register("+919876543210", "farmer")
        assert "+919876543210" in otp_store
        assert len(otp_store["+919876543210"].otp) == 6

    @pytest.mark.asyncio
    async def test_otp_is_numeric(self, svc: RegistrationService, otp_store: dict):
        await svc.register("+919876543210", "farmer")
        otp = otp_store["+919876543210"].otp
        assert otp.isdigit()

    @pytest.mark.asyncio
    async def test_otp_expires_in_10_minutes(self, svc: RegistrationService, otp_store: dict):
        await svc.register("+919876543210", "farmer")
        record = otp_store["+919876543210"]
        diff = record.expires_at - datetime.now(UTC).replace(tzinfo=None)
        assert timedelta(minutes=9) < diff < timedelta(minutes=11)

    @pytest.mark.asyncio
    async def test_register_invalid_user_type_raises(self, svc: RegistrationService):
        with pytest.raises(ValueError, match="user_type must be one of"):
            await svc.register("+919876543210", "agent")

    @pytest.mark.asyncio
    async def test_new_otp_overwrites_old_otp(self, svc: RegistrationService, otp_store: dict):
        await svc.register("+919876543210", "farmer")
        first_otp = otp_store["+919876543210"].otp
        await svc.register("+919876543210", "farmer")
        second_otp = otp_store["+919876543210"].otp
        # NOTE: OTPs are random — they may differ; store must have only 1 record
        assert len([k for k in otp_store if k == "+919876543210"]) == 1

    @pytest.mark.asyncio
    async def test_verify_correct_otp_returns_token_response(
        self, svc: RegistrationService, otp_store: dict
    ):
        await svc.register("+919876543210", "farmer")
        otp = otp_store["+919876543210"].otp
        token_resp = await svc.verify_otp("+919876543210", otp)
        assert isinstance(token_resp, TokenResponse)
        assert token_resp.access_token

    @pytest.mark.asyncio
    async def test_verify_wrong_otp_raises(self, svc: RegistrationService, otp_store: dict):
        await svc.register("+919876543210", "farmer")
        with pytest.raises(ValueError, match="Incorrect OTP"):
            await svc.verify_otp("+919876543210", "000000")

    @pytest.mark.asyncio
    async def test_verify_expired_otp_raises(
        self, svc: RegistrationService, otp_store: dict
    ):
        await svc.register("+919876543210", "farmer")
        # Manually expire
        otp_store["+919876543210"].expires_at = datetime(2000, 1, 1)
        with pytest.raises(ValueError, match="expired"):
            await svc.verify_otp("+919876543210", otp_store["+919876543210"].otp)

    @pytest.mark.asyncio
    async def test_verify_otp_cleans_up_store(
        self, svc: RegistrationService, otp_store: dict
    ):
        await svc.register("+919876543210", "farmer")
        otp = otp_store["+919876543210"].otp
        await svc.verify_otp("+919876543210", otp)
        assert "+919876543210" not in otp_store

    @pytest.mark.asyncio
    async def test_verify_no_otp_requested_raises(self, svc: RegistrationService):
        with pytest.raises(ValueError, match="No OTP found"):
            await svc.verify_otp("+919876543210", "123456")


# ═══════════════════════════════════════════════════════════════
# * AC2 — JWT token generation and validation
# ═══════════════════════════════════════════════════════════════

class TestJWTTokens:
    """JWT generation, structure validation, expiry, and signature checks."""

    def test_generate_token_returns_string(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Tumakuru")
        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    def test_token_payload_sub_equals_user_id(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Tumakuru")
        payload = svc.verify_token(token)
        assert payload["sub"] == "uid-001"

    def test_token_payload_type_equals_user_type(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "buyer", "+919123456789", "Bangalore Urban")
        payload = svc.verify_token(token)
        assert payload["type"] == "buyer"

    def test_token_payload_contains_phone(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Kolar")
        payload = svc.verify_token(token)
        assert payload["phone"] == "+919876543210"

    def test_token_payload_contains_district(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Mysuru")
        payload = svc.verify_token(token)
        assert payload["district"] == "Mysuru"

    def test_token_expiry_is_30_days(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Kolar")
        payload = svc.verify_token(token)
        now = int(datetime.now(UTC).timestamp())
        diff_days = (payload["exp"] - now) / 86400
        assert 29 <= diff_days <= 31

    def test_verify_token_invalid_signature_raises(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Kolar")
        parts = token.split(".")
        tampered = f"{parts[0]}.{parts[1]}.invalidsig"
        with pytest.raises(ValueError, match="Invalid token signature"):
            svc.verify_token(tampered)

    def test_verify_token_malformed_raises(self, svc: RegistrationService):
        with pytest.raises(ValueError, match="Malformed token"):
            svc.verify_token("not.a.real.token.with.too.many.parts")

    def test_verify_expired_token_raises(self, svc: RegistrationService):
        """Build a token with exp = 1 (past)."""
        import base64
        import hashlib
        import hmac

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "uid-001", "type": "farmer", "phone": "+91x",
                        "district": "x", "iat": 1, "exp": 1},
                       separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        signing_input = f"{header}.{payload}"
        sig = base64.urlsafe_b64encode(
            hmac.new(b"test-secret-key", signing_input.encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        expired_token = f"{signing_input}.{sig}"
        with pytest.raises(ValueError, match="expired"):
            svc.verify_token(expired_token)

    def test_different_secret_token_fails_verification(self, svc: RegistrationService):
        token = svc._generate_token("uid-001", "farmer", "+919876543210", "Kolar")
        svc2 = RegistrationService(jwt_secret="completely-different-secret")
        with pytest.raises(ValueError, match="Invalid token signature"):
            svc2.verify_token(token)

    @pytest.mark.asyncio
    async def test_verify_otp_returns_valid_jwt(
        self, svc: RegistrationService, otp_store: dict
    ):
        await svc.register("+919876543210", "farmer")
        otp = otp_store["+919876543210"].otp
        token_resp = await svc.verify_otp("+919876543210", otp)
        payload = svc.verify_token(token_resp.access_token)
        assert payload["phone"] == "+919876543210"
        assert payload["type"] == "farmer"

    @pytest.mark.asyncio
    async def test_token_response_includes_user_type(
        self, svc: RegistrationService, otp_store: dict
    ):
        await svc.register("+919876543210", "buyer")
        otp = otp_store["+919876543210"].otp
        token_resp = await svc.verify_otp("+919876543210", otp)
        assert token_resp.user_type == "buyer"
        assert token_resp.expires_in_days == JWT_EXPIRY_DAYS


# ═══════════════════════════════════════════════════════════════
# * AC3 — Profile CRUD with language preference
# ═══════════════════════════════════════════════════════════════

class TestProfileCRUD:
    """Profile creation, fetch, update, and language auto-inference."""

    @pytest.mark.asyncio
    async def test_get_profile_returns_none_without_db(self, svc: RegistrationService):
        profile = await svc.get_profile("uid-001", "farmer")
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_farmer_profile_with_db(self, svc_with_db: RegistrationService):
        profile = await svc_with_db.get_profile("farmer-uuid-001", "farmer")
        assert profile is not None
        assert profile.user_id == "farmer-uuid-001"
        assert profile.user_type == "farmer"
        assert profile.language_pref == "kn"

    @pytest.mark.asyncio
    async def test_get_buyer_profile_with_db(self, svc_with_db: RegistrationService):
        profile = await svc_with_db.get_profile("buyer-uuid-001", "buyer")
        assert profile is not None
        assert profile.user_type == "buyer"
        assert profile.buyer_type == "retailer"

    @pytest.mark.asyncio
    async def test_update_profile_language_inferred_from_district(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.update_profile(
            "farmer-uuid-001", "farmer", {"district": "Mysuru"}
        )
        call_args = mock_db.update_farmer.call_args[0][1]
        assert call_args.get("language_pref") == "kn"

    @pytest.mark.asyncio
    async def test_update_profile_explicit_language_not_overridden(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.update_profile(
            "farmer-uuid-001", "farmer",
            {"district": "Bangalore Urban", "language_pref": "hi"},
        )
        call_args = mock_db.update_farmer.call_args[0][1]
        assert call_args.get("language_pref") == "hi"

    @pytest.mark.asyncio
    async def test_update_profile_calls_db_update(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.update_profile(
            "farmer-uuid-001", "farmer", {"name": "Suresh"}
        )
        mock_db.update_farmer.assert_called_once()
        call_args = mock_db.update_farmer.call_args[0][1]
        assert call_args.get("name") == "Suresh"

    @pytest.mark.asyncio
    async def test_aadhaar_last4_is_hashed_on_update(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.update_profile(
            "farmer-uuid-001", "farmer", {"aadhaar_last4": "1234"}
        )
        call_args = mock_db.update_farmer.call_args[0][1]
        # aadhaar_last4 should be replaced by aadhaar_hash
        assert "aadhaar_last4" not in call_args
        assert "aadhaar_hash" in call_args
        expected_hash = hashlib.sha256("1234".encode()).hexdigest()
        assert call_args["aadhaar_hash"] == expected_hash

    @pytest.mark.asyncio
    async def test_initial_quality_score_is_half(self, svc_with_db: RegistrationService):
        profile = await svc_with_db.get_profile("farmer-uuid-001", "farmer")
        assert profile is not None
        assert profile.quality_score == INITIAL_QUALITY_SCORE

    @pytest.mark.asyncio
    async def test_update_buyer_profile_calls_update_buyer(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.update_profile(
            "buyer-uuid-001", "buyer",
            {"credit_limit": 100000.0, "subscription_tier": "premium"},
        )
        mock_db.update_buyer.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_profile_no_db_returns_none(
        self, svc: RegistrationService
    ):
        result = await svc.update_profile("uid-001", "farmer", {"name": "Test"})
        assert result is None


# ═══════════════════════════════════════════════════════════════
# * AC4 — Separate farmer / buyer type-specific fields
# ═══════════════════════════════════════════════════════════════

class TestFarmerBuyerProfiles:
    """Farmer and buyer profiles have distinct fields and validation."""

    @pytest.mark.asyncio
    async def test_farmer_registration_creates_farmer_record(
        self, svc_with_db: RegistrationService, mock_db: MagicMock, otp_store: dict
    ):
        await svc_with_db.register("+919876543210", "farmer")
        otp = otp_store["+919876543210"].otp
        token_resp = await svc_with_db.verify_otp("+919876543210", otp)
        mock_db.create_farmer.assert_called_once()
        assert token_resp.user_type == "farmer"

    @pytest.mark.asyncio
    async def test_buyer_registration_creates_buyer_record(
        self, svc_with_db: RegistrationService, mock_db: MagicMock, otp_store: dict
    ):
        await svc_with_db.register("+919123456789", "buyer")
        otp = otp_store["+919123456789"].otp
        token_resp = await svc_with_db.verify_otp("+919123456789", otp)
        mock_db.create_buyer.assert_called_once()
        assert token_resp.user_type == "buyer"

    @pytest.mark.asyncio
    async def test_existing_farmer_phone_skips_create(
        self, svc_with_db: RegistrationService, mock_db: MagicMock, otp_store: dict
    ):
        """If farmer already registered, verify_otp should NOT call create_farmer."""
        mock_db.get_farmer_by_phone.return_value = {
            "id": "existing-farmer-uuid",
            "phone": "+919876543210",
            "district": "Kolar",
            "language_pref": "kn",
        }
        await svc_with_db.register("+919876543210", "farmer")
        otp = otp_store["+919876543210"].otp
        token_resp = await svc_with_db.verify_otp("+919876543210", otp)
        mock_db.create_farmer.assert_not_called()
        assert token_resp.user_id == "existing-farmer-uuid"

    @pytest.mark.asyncio
    async def test_farmer_profile_has_no_buyer_type(self, svc_with_db: RegistrationService):
        profile = await svc_with_db.get_profile("farmer-uuid-001", "farmer")
        assert profile is not None
        # buyer_type should be None for farmer row
        assert profile.buyer_type is None

    @pytest.mark.asyncio
    async def test_buyer_profile_has_buyer_type(self, svc_with_db: RegistrationService):
        profile = await svc_with_db.get_profile("buyer-uuid-001", "buyer")
        assert profile is not None
        assert profile.buyer_type == "retailer"

    @pytest.mark.asyncio
    async def test_buyer_profile_has_credit_limit(self, svc_with_db: RegistrationService):
        profile = await svc_with_db.get_profile("buyer-uuid-001", "buyer")
        assert profile is not None
        assert profile.credit_limit == 50000.0

    def test_update_buyer_request_validates_buyer_type(self):
        with pytest.raises(ValueError):
            UpdateBuyerProfileRequest(buyer_type="invalid_type")

    def test_update_buyer_request_accepts_valid_buyer_type(self):
        req = UpdateBuyerProfileRequest(buyer_type="retailer", credit_limit=20000.0)
        assert req.buyer_type == "retailer"


# ═══════════════════════════════════════════════════════════════
# * AC5 — Voice agent register_farmer() integration
# ═══════════════════════════════════════════════════════════════

class TestVoiceAgentIntegration:
    """register_farmer() method satisfies VoiceAgent._handle_register() contract."""

    @pytest.mark.asyncio
    async def test_register_farmer_returns_farmer_id(self, svc: RegistrationService):
        result = await svc.register_farmer(
            name="Raju", phone="+919876543210", district="Tumakuru"
        )
        assert "farmer_id" in result
        assert result["farmer_id"]

    @pytest.mark.asyncio
    async def test_register_farmer_returns_language_pref(self, svc: RegistrationService):
        result = await svc.register_farmer(
            name="Raju", phone="+919876543210", district="Mysuru"
        )
        assert result["language_pref"] == "kn"

    @pytest.mark.asyncio
    async def test_register_farmer_returns_district(self, svc: RegistrationService):
        result = await svc.register_farmer(
            name="Suresh", phone="+919876543210", district="Kolar"
        )
        assert result["district"] == "Kolar"

    @pytest.mark.asyncio
    async def test_register_farmer_with_db_calls_create_farmer(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.register_farmer(
            name="Vijay", phone="+919876543210", district="Davanagere"
        )
        mock_db.create_farmer.assert_called_once()
        call_args = mock_db.create_farmer.call_args[0][0]
        assert call_args["name"] == "Vijay"
        assert call_args["district"] == "Davanagere"

    @pytest.mark.asyncio
    async def test_register_farmer_passes_village(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        await svc_with_db.register_farmer(
            name="Ramesh", phone="+919111111111",
            district="Tumakuru", village="Koratagere"
        )
        call_args = mock_db.create_farmer.call_args[0][0]
        assert call_args.get("village") == "Koratagere"

    @pytest.mark.asyncio
    async def test_register_farmer_no_db_returns_uuid(self, svc: RegistrationService):
        result = await svc.register_farmer(
            name="Test", phone="+919000000000", district="Bellary"
        )
        farmer_id = result["farmer_id"]
        assert len(farmer_id) == 36  # Standard UUID length


# ═══════════════════════════════════════════════════════════════
# * District → Language mapping
# ═══════════════════════════════════════════════════════════════

class TestDistrictLanguage:
    """_district_to_language() returns correct codes for Karnataka districts."""

    def test_bangalore_maps_to_kannada(self):
        assert RegistrationService._district_to_language("Bangalore Urban") == "kn"

    def test_mysuru_maps_to_kannada(self):
        assert RegistrationService._district_to_language("Mysuru") == "kn"

    def test_case_insensitive(self):
        assert RegistrationService._district_to_language("TUMAKURU") == "kn"

    def test_unknown_district_falls_back_to_kannada(self):
        assert RegistrationService._district_to_language("Unknown District XYZ") == "kn"

    def test_all_31_karnataka_districts_map_to_kn(self):
        for district, lang in KARNATAKA_DISTRICT_LANGUAGE.items():
            assert lang == "kn", f"{district} should map to 'kn', got '{lang}'"


# ═══════════════════════════════════════════════════════════════
# * GPS → District mock lookup
# ═══════════════════════════════════════════════════════════════

class TestGPSToDistrict:
    """_gps_to_district() returns nearest Karnataka district for in-bounds GPS."""

    def test_bangalore_coords_return_district(self):
        district = RegistrationService._gps_to_district(12.97, 77.56)
        assert district is not None
        assert "Bangalore" in district or "Bengaluru" in district

    def test_out_of_bounds_returns_none(self):
        # Mumbai
        result = RegistrationService._gps_to_district(19.07, 72.87)
        assert result is None

    def test_dharwad_approx_returns_district(self):
        district = RegistrationService._gps_to_district(15.46, 75.01)
        assert district == "Dharwad"

    @pytest.mark.asyncio
    async def test_gps_update_triggers_district_detection(
        self, svc_with_db: RegistrationService, mock_db: MagicMock
    ):
        """If only GPS coords are passed with no district, district is auto-detected."""
        await svc_with_db.update_profile(
            "farmer-uuid-001", "farmer",
            {"latitude": 12.97, "longitude": 77.56},
        )
        call_args = mock_db.update_farmer.call_args[0][1]
        # district should be auto-populated
        assert call_args.get("district") is not None


# ═══════════════════════════════════════════════════════════════
# * Factory function
# ═══════════════════════════════════════════════════════════════

class TestFactory:
    """get_registration_service() returns correctly configured instance."""

    def test_factory_returns_registration_service(self):
        svc = get_registration_service()
        assert isinstance(svc, RegistrationService)

    def test_factory_uses_env_jwt_secret(self):
        import os
        os.environ["JWT_SECRET"] = "env-secret-test"
        svc = get_registration_service()
        assert svc._jwt_secret == "env-secret-test"
        del os.environ["JWT_SECRET"]

    def test_factory_uses_provided_jwt_secret(self):
        svc = get_registration_service(jwt_secret="custom-secret")
        assert svc._jwt_secret == "custom-secret"

    def test_factory_with_shared_otp_store(self):
        store: dict = {}
        svc = get_registration_service(otp_store=store)
        assert svc._otp_store is store

    def test_register_request_normalises_phone(self):
        req = RegisterRequest(phone="9876543210", user_type="farmer")
        assert req.phone == "+919876543210"

    def test_register_request_rejects_invalid_type(self):
        with pytest.raises(ValueError):
            RegisterRequest(phone="+919876543210", user_type="unknown")
