"""
Unit tests for HITLNotificationService (FR9).

FR9 HITL trigger conditions:
  - confidence < 0.70
  - grade == "A+"
  - defect_count > 3
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.services.hitl_service import (
    HITL_CONFIDENCE_THRESHOLD,
    HITL_DEFECT_COUNT_THRESHOLD,
    HITL_GRADE_ALWAYS_REVIEW,
    HITLNotificationService,
    HITLTriggerResult,
    get_hitl_service,
)


# * ═══════════════════════════════════════════════════════════════
# * Fixtures
# * ═══════════════════════════════════════════════════════════════


@pytest.fixture
def svc() -> HITLNotificationService:
    """HITLNotificationService with no DB and no Redis (log-only mode)."""
    return HITLNotificationService(db=None, redis_client=None)


@pytest.fixture
def svc_with_redis() -> tuple[HITLNotificationService, AsyncMock]:
    """HITLNotificationService with a mock Redis client."""
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)
    return HITLNotificationService(db=None, redis_client=mock_redis), mock_redis


@pytest.fixture
def svc_with_db_and_redis() -> tuple[HITLNotificationService, MagicMock, AsyncMock]:
    """HITLNotificationService with mock DB and Redis."""
    mock_db = AsyncMock()
    mock_db.create_hitl_review = AsyncMock(return_value="review-uuid-001")
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)
    return HITLNotificationService(db=mock_db, redis_client=mock_redis), mock_db, mock_redis


# * ═══════════════════════════════════════════════════════════════
# * should_trigger() static helper
# * ═══════════════════════════════════════════════════════════════


class TestShouldTriggerHelper:

    def test_low_confidence_triggers(self):
        """Confidence below threshold must trigger."""
        assert HITLNotificationService.should_trigger(0.65, "A", 1) is True

    def test_grade_a_plus_triggers(self):
        """Grade A+ must always trigger regardless of confidence or defects."""
        assert HITLNotificationService.should_trigger(0.95, "A+", 0) is True

    def test_high_defect_count_triggers(self):
        """More than 3 defects must trigger."""
        assert HITLNotificationService.should_trigger(0.80, "B", 5) is True

    def test_all_conditions_clear_does_not_trigger(self):
        """Normal case: conf=0.82, grade=A, defects=2 → no trigger."""
        assert HITLNotificationService.should_trigger(0.82, "A", 2) is False

    def test_boundary_confidence_at_threshold_does_not_trigger(self):
        """Confidence exactly at threshold (0.70) must NOT trigger."""
        assert HITLNotificationService.should_trigger(0.70, "A", 1) is False

    def test_boundary_confidence_just_below_triggers(self):
        """Confidence at 0.6999 (just below) must trigger."""
        assert HITLNotificationService.should_trigger(0.6999, "B", 0) is True

    def test_boundary_defect_count_at_threshold_does_not_trigger(self):
        """Exactly 3 defects must NOT trigger (threshold is 'more than 3')."""
        assert HITLNotificationService.should_trigger(0.80, "B", 3) is False

    def test_threshold_constants(self):
        """Verify threshold constants match FR9 spec."""
        assert HITL_CONFIDENCE_THRESHOLD == 0.70
        assert HITL_DEFECT_COUNT_THRESHOLD == 3
        assert HITL_GRADE_ALWAYS_REVIEW == "A+"


# * ═══════════════════════════════════════════════════════════════
# * trigger_review() — no trigger cases
# * ═══════════════════════════════════════════════════════════════


class TestNoTrigger:

    @pytest.mark.asyncio
    async def test_normal_case_does_not_trigger(self, svc: HITLNotificationService):
        """Normal assessment should return triggered=False."""
        result = await svc.trigger_review(
            listing_id="lst-normal",
            confidence=0.82,
            grade="A",
            defect_count=1,
        )
        assert result.triggered is False
        assert result.reason == "no_trigger"
        assert result.listing_id == "lst-normal"
        assert result.review_id is None


# * ═══════════════════════════════════════════════════════════════
# * trigger_review() — trigger cases (log-only mode)
# * ═══════════════════════════════════════════════════════════════


class TestTriggerReasons:

    @pytest.mark.asyncio
    async def test_low_confidence_reason(self, svc: HITLNotificationService):
        result = await svc.trigger_review("lst-001", confidence=0.60, grade="B", defect_count=1)
        assert result.triggered is True
        assert result.reason == "confidence_low"

    @pytest.mark.asyncio
    async def test_grade_a_plus_reason(self, svc: HITLNotificationService):
        result = await svc.trigger_review("lst-002", confidence=0.92, grade="A+", defect_count=0)
        assert result.triggered is True
        assert result.reason == "grade_a_plus"

    @pytest.mark.asyncio
    async def test_high_defects_reason(self, svc: HITLNotificationService):
        result = await svc.trigger_review("lst-003", confidence=0.80, grade="B", defect_count=6)
        assert result.triggered is True
        assert result.reason == "defects_high"

    @pytest.mark.asyncio
    async def test_combined_reason_when_two_conditions_met(
        self, svc: HITLNotificationService
    ):
        """Low confidence + grade A+ → combined."""
        result = await svc.trigger_review("lst-004", confidence=0.55, grade="A+", defect_count=0)
        assert result.triggered is True
        assert result.reason == "combined"

    @pytest.mark.asyncio
    async def test_all_three_conditions_combined(self, svc: HITLNotificationService):
        result = await svc.trigger_review("lst-005", confidence=0.50, grade="A+", defect_count=5)
        assert result.triggered is True
        assert result.reason == "combined"


# * ═══════════════════════════════════════════════════════════════
# * trigger_review() — with Redis
# * ═══════════════════════════════════════════════════════════════


class TestRedisIntegration:

    @pytest.mark.asyncio
    async def test_redis_setex_called_on_trigger(
        self, svc_with_redis: tuple[HITLNotificationService, AsyncMock]
    ):
        """Trigger should push a Redis event with hitl:pending:{listing_id} key."""
        svc, mock_redis = svc_with_redis
        result = await svc.trigger_review("lst-redis-001", 0.60, "B", 1)

        assert result.triggered is True
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        redis_key = call_args[0][0]
        assert redis_key == "hitl:pending:lst-redis-001"

    @pytest.mark.asyncio
    async def test_redis_not_called_when_no_trigger(
        self, svc_with_redis: tuple[HITLNotificationService, AsyncMock]
    ):
        """No trigger → Redis must not be called."""
        svc, mock_redis = svc_with_redis
        await svc.trigger_review("lst-ok", 0.90, "A", 0)
        mock_redis.setex.assert_not_called()


# * ═══════════════════════════════════════════════════════════════
# * trigger_review() — with DB
# * ═══════════════════════════════════════════════════════════════


class TestDBIntegration:

    @pytest.mark.asyncio
    async def test_db_create_called_on_trigger(
        self, svc_with_db_and_redis: tuple
    ):
        """Trigger should persist a review record to DB."""
        svc, mock_db, _ = svc_with_db_and_redis
        result = await svc.trigger_review("lst-db-001", 0.55, "B", 5)

        assert result.triggered is True
        mock_db.create_hitl_review.assert_called_once()
        call_payload = mock_db.create_hitl_review.call_args[0][0]
        assert call_payload["listing_id"] == "lst-db-001"
        assert call_payload["status"] == "pending"

    @pytest.mark.asyncio
    async def test_review_id_returned_from_db(
        self, svc_with_db_and_redis: tuple
    ):
        """review_id from DB must be returned in HITLTriggerResult."""
        svc, _, _ = svc_with_db_and_redis
        result = await svc.trigger_review("lst-db-002", 0.60, "A+", 0)
        assert result.review_id == "review-uuid-001"

    @pytest.mark.asyncio
    async def test_db_failure_does_not_raise(
        self, svc_with_db_and_redis: tuple
    ):
        """DB failure should log a warning but not propagate (graceful degradation)."""
        svc, mock_db, _ = svc_with_db_and_redis
        mock_db.create_hitl_review.side_effect = Exception("DB connection refused")

        # Must not raise
        result = await svc.trigger_review("lst-db-fail", 0.50, "B", 6)
        assert result.triggered is True
        assert result.review_id is None   # None because DB failed, but trigger still happened


# * ═══════════════════════════════════════════════════════════════
# * Factory
# * ═══════════════════════════════════════════════════════════════


class TestFactory:

    def test_get_hitl_service_returns_instance(self):
        svc = get_hitl_service()
        assert isinstance(svc, HITLNotificationService)

    def test_get_hitl_service_with_deps(self):
        mock_db = MagicMock()
        mock_redis = MagicMock()
        svc = get_hitl_service(db=mock_db, redis_client=mock_redis)
        assert svc.db is mock_db
        assert svc._redis is mock_redis
