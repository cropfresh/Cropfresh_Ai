"""
HITL Notification Service
==========================
Dispatches Human-In-The-Loop review requests when the AI quality
pipeline cannot confidently determine crop grade.

FR9 Trigger Conditions:
  - confidence < 0.70  (vision model uncertain)
  - grade == "A+"      (export-quality claims always require manual sign-off)
  - defect_count > 3   (too many defects for autonomous decision)

Actions:
  1. Insert a record into hitl_review_queue table (when DB available)
  2. Push a JSON event to Redis key  hitl:pending:{listing_id}  (TTL=48h)
  3. Structured Loguru log event (Phase 5: replace with AWS SNS/webhook)

Design: Degrades gracefully — no DB and no Redis means log-only mode.
"""

# * HITL SERVICE MODULE
# NOTE: DB and Redis are both optional; service never raises in production.
# NOTE: Phase 5 TODO — replace Loguru structured log with SNS publish.

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Optional

from loguru import logger

# * ─── HITL trigger thresholds (must stay in sync with vision_models.py) ──────

HITL_CONFIDENCE_THRESHOLD: float = 0.70
HITL_DEFECT_COUNT_THRESHOLD: int = 3
HITL_GRADE_ALWAYS_REVIEW: str = "A+"

# Redis TTL for pending HITL events — field agent must review within 48h
HITL_REDIS_TTL_SECONDS: int = 48 * 3600


# * ─── result model ────────────────────────────────────────────────────────────


class HITLTriggerResult:
    """Lightweight result returned by trigger_review()."""

    def __init__(
        self,
        triggered: bool,
        reason: str,
        listing_id: str,
        review_id: Optional[str] = None,
    ) -> None:
        self.triggered = triggered
        self.reason = reason
        self.listing_id = listing_id
        self.review_id = review_id  # DB hitl_review_queue.id if persisted

    def __repr__(self) -> str:
        return (
            f"HITLTriggerResult(triggered={self.triggered}, "
            f"reason={self.reason!r}, listing_id={self.listing_id!r}, "
            f"review_id={self.review_id!r})"
        )


# * ─── service ────────────────────────────────────────────────────────────────


class HITLNotificationService:
    """
    Dispatches HITL review requests when the AI quality pipeline
    cannot confidently determine crop grade (FR9).

    Usage:
        svc = HITLNotificationService(db=db_client, redis_client=redis)
        result = await svc.trigger_review(
            listing_id="lst-abc123",
            confidence=0.62,
            grade="A",
            defect_count=5,
        )
        if result.triggered:
            logger.info("HITL review queued: {}", result.review_id)
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        # NOTE: db is AuroraPostgresClient — optional; log-only mode when absent
        self.db = db
        self._redis = redis_client

    # ── public API ──────────────────────────────────────────────────────────

    async def trigger_review(
        self,
        listing_id: str,
        confidence: float,
        grade: str,
        defect_count: int,
    ) -> HITLTriggerResult:
        """
        Evaluate trigger conditions and raise a HITL review if needed.

        Args:
            listing_id:   Listing UUID.
            confidence:   AI confidence score from quality assessment.
            grade:        Assigned grade from quality assessment.
            defect_count: Number of defects detected.

        Returns:
            HITLTriggerResult — always succeeds; errors are logged not raised.
        """
        reason = self._determine_reason(confidence, grade, defect_count)
        if not reason:
            return HITLTriggerResult(
                triggered=False,
                reason="no_trigger",
                listing_id=listing_id,
            )

        review_id: Optional[str] = None

        # * Step 1: persist to hitl_review_queue
        review_id = await self._persist_review(
            listing_id=listing_id,
            trigger_reason=reason,
            confidence=confidence,
            grade=grade,
            defect_count=defect_count,
        )

        # * Step 2: push to Redis for fast field-agent polling
        await self._push_redis_event(listing_id, reason, confidence, grade, defect_count)

        # * Step 3: structured log (Phase 5: → SNS publish)
        logger.bind(
            event="hitl_triggered",
            listing_id=listing_id,
            trigger_reason=reason,
            confidence=confidence,
            grade=grade,
            defect_count=defect_count,
            review_id=review_id,
        ).warning(
            "HITL review triggered for listing {} — reason={} confidence={:.2f} grade={} defects={}",
            listing_id, reason, confidence, grade, defect_count,
        )

        return HITLTriggerResult(
            triggered=True,
            reason=reason,
            listing_id=listing_id,
            review_id=review_id,
        )

    @staticmethod
    def should_trigger(confidence: float, grade: str, defect_count: int) -> bool:
        """
        Stateless helper — returns True if HITL should be triggered.
        Mirrors the condition in vision_models.CropVisionPipeline.
        Use this for unit testing without instantiating the service.
        """
        return (
            confidence < HITL_CONFIDENCE_THRESHOLD
            or grade == HITL_GRADE_ALWAYS_REVIEW
            or defect_count > HITL_DEFECT_COUNT_THRESHOLD
        )

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _determine_reason(
        confidence: float, grade: str, defect_count: int
    ) -> Optional[str]:
        """
        Map trigger conditions to a single reason label for DB storage.

        Priority: combined > confidence_low > grade_a_plus > defects_high.
        Returns None if no condition is met.
        """
        low_conf = confidence < HITL_CONFIDENCE_THRESHOLD
        a_plus   = grade == HITL_GRADE_ALWAYS_REVIEW
        high_def = defect_count > HITL_DEFECT_COUNT_THRESHOLD

        active = sum([low_conf, a_plus, high_def])
        if active == 0:
            return None
        if active > 1:
            return "combined"
        if low_conf:
            return "confidence_low"
        if a_plus:
            return "grade_a_plus"
        return "defects_high"

    async def _persist_review(
        self,
        listing_id: str,
        trigger_reason: str,
        confidence: float,
        grade: str,
        defect_count: int,
    ) -> Optional[str]:
        """Insert row into hitl_review_queue; returns new review UUID or None."""
        if not (self.db and hasattr(self.db, "create_hitl_review")):
            return None
        try:
            review_id = await self.db.create_hitl_review({
                "listing_id": listing_id,
                "trigger_reason": trigger_reason,
                "confidence": confidence,
                "grade": grade,
                "defect_count": defect_count,
                "status": "pending",
            })
            return str(review_id) if review_id else None
        except Exception as exc:
            logger.warning("Failed to persist HITL review for listing {}: {}", listing_id, exc)
            return None

    async def _push_redis_event(
        self,
        listing_id: str,
        trigger_reason: str,
        confidence: float,
        grade: str,
        defect_count: int,
    ) -> None:
        """Push a JSON event to Redis for field-agent polling (TTL=48h)."""
        if not self._redis:
            return
        try:
            key = f"hitl:pending:{listing_id}"
            payload = json.dumps({
                "listing_id": listing_id,
                "trigger_reason": trigger_reason,
                "confidence": round(confidence, 4),
                "grade": grade,
                "defect_count": defect_count,
                "queued_at": datetime.now(UTC).isoformat(),
            })
            await self._redis.setex(key, HITL_REDIS_TTL_SECONDS, payload)
        except Exception as exc:
            logger.warning("Failed to push HITL Redis event for listing {}: {}", listing_id, exc)


# * ─── factory ────────────────────────────────────────────────────────────────


def get_hitl_service(
    db: Optional[Any] = None,
    redis_client: Optional[Any] = None,
) -> HITLNotificationService:
    """
    Factory for creating a HITLNotificationService with optional dependencies.

    Args:
        db: AuroraPostgresClient instance (optional).
        redis_client: redis.asyncio client (optional).

    Returns:
        Configured HITLNotificationService.
    """
    return HITLNotificationService(db=db, redis_client=redis_client)
