"""
Quality Assessment Agent (CV-QG)
================================
AI-powered produce grading with HITL (Human-In-The-Loop) fallback.

Business Logic (from ARCHITECTURE.md):
  - Grade produce into A / B / C using image analysis
  - If AI confidence >= 95% → auto-grade
  - If AI confidence < 95%  → flag for HITL review by field agent
  - Create Digital Twin record for dispute resolution
  - Detect defects: bruise, worm_hole, colour_off, size_irregular

Phase 2 adds YOLOv8/ViT inference; this version uses LLM-based
analysis as the initial working implementation.

Author: CropFresh AI Team
Version: 2.0.0
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState
from src.orchestrator.llm_provider import BaseLLMProvider


# * ═══════════════════════════════════════════════════════════════
# * DATA MODELS
# * ═══════════════════════════════════════════════════════════════

VALID_GRADES = ("A", "B", "C", "Unverified")
DEFECT_TYPES = (
    "bruise", "worm_hole", "colour_off", "size_irregular",
    "over_ripe", "under_ripe", "fungal", "mechanical_damage",
)

HITL_CONFIDENCE_THRESHOLD = 0.95


class GradeAssessment(BaseModel):
    """Result of a single produce quality assessment."""
    listing_id: str
    commodity: str
    grade: str
    confidence: float
    hitl_required: bool
    defects_detected: list[str] = Field(default_factory=list)
    shelf_life_days: int = 0
    reasoning: str = ""
    assessed_at: datetime = Field(default_factory=datetime.now)


class QualityReport(BaseModel):
    """Full quality report that can be stored as a Digital Twin."""
    assessment: GradeAssessment
    image_count: int = 0
    method: str = "llm"  # "llm" | "yolo_vit" | "manual"


# * ═══════════════════════════════════════════════════════════════
# * GRADING RULES (rule-based fallback when no LLM / no images)
# * ═══════════════════════════════════════════════════════════════

COMMODITY_SHELF_LIFE = {
    "tomato": 7, "onion": 30, "potato": 45, "beans": 5,
    "cabbage": 14, "carrot": 21, "brinjal": 5, "chilli": 10,
    "capsicum": 7, "cucumber": 5, "peas": 3, "cauliflower": 5,
}


class QualityAssessmentAgent(BaseAgent):
    """
    CV-QG Agent — grades produce and flags items for HITL review.

    Current implementation uses LLM analysis of user-provided
    descriptions (and optionally base64 images). Phase 2 will add
    YOLOv8 + ViT vision inference for <500ms automated grading.

    Usage:
        agent = QualityAssessmentAgent(llm=provider)
        await agent.initialize()
        report = await agent.assess(listing_id="abc", commodity="Tomato",
                                     description="Red, firm, no spots")
    """

    def __init__(self, llm: Optional[BaseLLMProvider] = None, **kwargs: Any):
        config = AgentConfig(
            name="quality_assessment",
            description="AI produce grading with HITL fallback for dispute-proof quality verification",
            max_retries=1,
            temperature=0.2,
            max_tokens=400,
            kb_categories=["agronomy"],
        )
        super().__init__(config=config, llm=llm, **kwargs)

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return """You are CropFresh's Quality Assessment Agent.
Given a description (and optionally photos) of harvested produce,
determine the quality grade and any defects.

Respond ONLY with a JSON object:
{
    "grade": "A" | "B" | "C",
    "confidence": 0.0-1.0,
    "defects": ["defect_type", ...],
    "shelf_life_days": integer,
    "reasoning": "brief explanation"
}

Grade definitions:
  A — Premium: uniform size/colour, zero visible defects, firm texture
  B — Standard: minor cosmetic issues, 1-2 small defects, still fresh
  C — Economy: multiple defects, softening, colour variation, short shelf life

Valid defect types: bruise, worm_hole, colour_off, size_irregular,
over_ripe, under_ripe, fungal, mechanical_damage"""

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """Process a natural-language quality question via the supervisor."""
        if not self.llm:
            return AgentResponse(
                content=self._rule_based_response(query),
                agent_name=self.name,
                confidence=0.6,
                steps=["rule_based_grading"],
            )

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query},
        ]
        response_text = await self.generate_with_llm(messages)

        return AgentResponse(
            content=response_text,
            agent_name=self.name,
            confidence=0.8,
            steps=["llm_quality_analysis"],
        )

    async def assess(
        self,
        listing_id: str,
        commodity: str,
        description: str = "",
        image_b64: Optional[str] = None,
    ) -> QualityReport:
        """
        Perform a quality assessment on a produce listing.

        Args:
            listing_id: The listing to assess
            commodity: Commodity name (e.g., "Tomato")
            description: Text description of produce condition
            image_b64: Optional base64-encoded photo

        Returns:
            QualityReport with grade, confidence, and HITL flag
        """
        if self.llm and description:
            return await self._assess_with_llm(
                listing_id, commodity, description, image_b64,
            )

        return self._assess_rule_based(listing_id, commodity, description)

    async def _assess_with_llm(
        self,
        listing_id: str,
        commodity: str,
        description: str,
        image_b64: Optional[str],
    ) -> QualityReport:
        """Grade using LLM analysis."""
        import json as json_mod

        prompt = f"Commodity: {commodity}\nCondition: {description}"
        if image_b64:
            prompt += "\n(Photo provided — analyze based on description)"

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self.generate_with_llm(messages, temperature=0.1)

            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json_mod.loads(text)

            grade = data.get("grade", "C")
            if grade not in VALID_GRADES:
                grade = "C"

            confidence = float(data.get("confidence", 0.5))
            defects = [d for d in data.get("defects", []) if d in DEFECT_TYPES]
            shelf_life = int(data.get("shelf_life_days", self._default_shelf_life(commodity)))
            reasoning = data.get("reasoning", "")

            hitl = confidence < HITL_CONFIDENCE_THRESHOLD

            assessment = GradeAssessment(
                listing_id=listing_id,
                commodity=commodity,
                grade=grade,
                confidence=confidence,
                hitl_required=hitl,
                defects_detected=defects,
                shelf_life_days=shelf_life,
                reasoning=reasoning,
            )

            logger.info(
                "Quality assessed: {} grade={} conf={:.2f} hitl={}",
                commodity, grade, confidence, hitl,
            )

            return QualityReport(
                assessment=assessment,
                image_count=1 if image_b64 else 0,
                method="llm",
            )

        except Exception as exc:
            logger.warning("LLM quality assessment failed: {}", exc)
            return self._assess_rule_based(listing_id, commodity, description)

    def _assess_rule_based(
        self,
        listing_id: str,
        commodity: str,
        description: str,
    ) -> QualityReport:
        """Fallback rule-based grading from keywords in description."""
        desc_lower = description.lower()

        defects: list[str] = []
        for defect in DEFECT_TYPES:
            keyword = defect.replace("_", " ")
            if keyword in desc_lower:
                defects.append(defect)

        negative_kw = ["soft", "damaged", "rotten", "old", "spotted", "brown"]
        positive_kw = ["fresh", "firm", "red", "green", "clean", "uniform", "new"]

        neg_count = sum(1 for kw in negative_kw if kw in desc_lower)
        pos_count = sum(1 for kw in positive_kw if kw in desc_lower)

        if neg_count == 0 and (pos_count >= 2 or not description):
            grade, confidence = "B", 0.55
        elif pos_count > neg_count:
            grade, confidence = "B", 0.50
        elif neg_count > 0:
            grade, confidence = "C", 0.45
        else:
            grade, confidence = "B", 0.40

        if len(defects) >= 3:
            grade, confidence = "C", 0.50
        elif len(defects) == 0 and pos_count >= 3:
            grade, confidence = "A", 0.50

        assessment = GradeAssessment(
            listing_id=listing_id,
            commodity=commodity,
            grade=grade,
            confidence=confidence,
            hitl_required=True,  # rule-based always flags HITL
            defects_detected=defects,
            shelf_life_days=self._default_shelf_life(commodity),
            reasoning="Rule-based assessment — HITL review recommended",
        )

        return QualityReport(assessment=assessment, method="rule_based")

    def _rule_based_response(self, query: str) -> str:
        """Quick text response for supervisor routing."""
        return (
            "Based on the description, I'd recommend getting a field agent "
            "to verify the produce quality. You can submit photos for AI "
            "grading, or describe the condition and I'll give a preliminary grade."
        )

    @staticmethod
    def _default_shelf_life(commodity: str) -> int:
        return COMMODITY_SHELF_LIFE.get(commodity.lower(), 7)
