"""Pydantic models for the Vision API surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VisionHealthResponse(BaseModel):
    """Health and model readiness for the shared quality-vision agent."""

    service_ready: bool
    vision_ready: bool
    assessment_mode: str
    model_dir: str


class GradeAttachPreview(BaseModel):
    """Reusable payload for the existing listing grade-attach route."""

    grade: str
    cv_confidence: float = Field(ge=0, le=1)
    defect_types: list[str] = Field(default_factory=list)


class VisionAssessRequest(BaseModel):
    """Request body for a one-shot vision or text quality assessment."""

    commodity: str
    listing_id: str | None = None
    description: str = ""
    image_b64: str | None = None
    require_upgrade_review: bool = False


class VisionAssessResponse(BaseModel):
    """Canonical response returned to the static Vision Lab."""

    listing_id: str
    commodity: str
    grade: str
    confidence: float
    defects: list[str] = Field(default_factory=list)
    defect_count: int
    hitl_required: bool
    shelf_life_days: int
    assessment_id: str
    digital_twin_linked: bool
    assessment_mode: str
    vision_ready: bool
    message: str
    grade_attach_preview: GradeAttachPreview
