"""
ADCL Agent — Multi-Language Summary Generator
===============================================
Generates farmer-friendly weekly summaries in English, Hindi, and Kannada.

Two modes:
  - No LLM (llm=None) : Deterministic template-based output.
    Fully testable without any external calls.
  - With LLM          : Calls llm.generate() with a structured prompt.
    LLM is injected; no import at module level.
"""

# * ADCL SUMMARY MODULE
# NOTE: Template summaries are deterministic and test-friendly.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.adcl.models import ADCLCrop


class SummaryGenerator:
    """
    Produces 3-language weekly narrative for WeeklyReport.

    Usage (no LLM):
        gen = SummaryGenerator()
        summaries = gen.generate(crops)
        # {"en": "...", "hi": "...", "kn": "..."}

    Usage (with LLM):
        gen = SummaryGenerator(llm=my_llm)
        summaries = await gen.generate_async(crops)
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Sync template-based (no LLM)
    # ------------------------------------------------------------------

    def generate(self, crops: list[ADCLCrop]) -> dict[str, str]:
        """
        Generate summaries without LLM (deterministic templates).

        Returns:
            {"en": str, "hi": str, "kn": str}
        """
        green_crops = [c for c in crops if c.green_label]
        top_green = [c.commodity.title() for c in green_crops[:5]]
        count = len(green_crops)

        return {
            "en": self._template_en(count, top_green, crops),
            "hi": self._template_hi(count, top_green),
            "kn": self._template_kn(count, top_green),
        }

    # ------------------------------------------------------------------
    # Async LLM-based (optional)
    # ------------------------------------------------------------------

    async def generate_async(self, crops: list[ADCLCrop]) -> dict[str, str]:
        """
        Generate summaries using LLM if available, else fall back to templates.

        Returns:
            {"en": str, "hi": str, "kn": str}
        """
        if self._llm is None:
            return self.generate(crops)

        green_crops = [c for c in crops if c.green_label]
        top_green = [c.commodity.title() for c in green_crops[:5]]
        count = len(green_crops)

        #! Updated: references demand_trend and sow_season_fit for richer context
        crop_lines = "\n".join(
            f"- {c.commodity.title()}: demand={c.demand_score:.0%}, "
            f"demand_trend={c.demand_trend}, sow_fit={c.sow_season_fit}"
            for c in crops[:10]
        )

        results: dict[str, str] = {}
        for lang, lang_name in [("en", "English"), ("hi", "Hindi"), ("kn", "Kannada")]:
            prompt = (
                f"You are an agricultural advisor. Write a 2–3 sentence weekly crop "
                f"recommendation summary in {lang_name} for Indian farmers.\n\n"
                f"This week {count} crops are recommended for sowing (green-label): "
                f"{', '.join(top_green) or 'none'}.\n\n"
                f"Crop data:\n{crop_lines}\n\n"
                f"Emphasize that these crops are recommended for PLANTING NOW "
                f"because demand will be high when they are harvested in 2-4 months. "
                f"Keep the language simple and encouraging. Output only the summary text."
            )
            try:
                messages = [{"role": "user", "content": prompt}]
                results[lang] = await self._llm.generate(messages)
            except Exception:
                # Graceful fallback to template if LLM fails
                results[lang] = self.generate(crops)[lang]

        return results

    # ------------------------------------------------------------------
    # Template builders
    # ------------------------------------------------------------------

    def _template_en(self, count: int, top_green: list[str], crops: list[ADCLCrop]) -> str:
        if count == 0:
            return (
                "This week no crops have been given the green label. "
                "Market demand is mixed — consult your local agriculture officer before planting."
            )
        names = ", ".join(top_green)
        top = crops[0].commodity.title() if crops else ""
        return (
            f"This week {count} crop(s) are recommended for planting: {names}. "
            f"{top} shows the highest buyer demand this season. "
            "Plant these now for the best price and market access at harvest time."
        )

    def _template_hi(self, count: int, top_green: list[str]) -> str:
        if count == 0:
            return (
                "इस सप्ताह कोई भी फसल 'ग्रीन लेबल' नहीं मिला है। "
                "बाजार की मांग मिश्रित है — कृपया स्थानीय कृषि अधिकारी से सलाह लें।"
            )
        names = ", ".join(top_green)
        return (
            f"इस सप्ताह {count} फसल(ों) की बुवाई की सिफारिश की गई है: {names}। "
            "इन फसलों की बाजार में अच्छी मांग और बढ़ती कीमतें हैं। "
            "अभी बोएं — कटाई के समय सबसे अच्छा दाम मिलेगा।"
        )

    def _template_kn(self, count: int, top_green: list[str]) -> str:
        if count == 0:
            return (
                "ಈ ವಾರ ಯಾವುದೇ ಬೆಳೆಗೆ 'ಹಸಿರು ಲೇಬಲ್' ಸಿಕ್ಕಿಲ್ಲ. "
                "ಮಾರುಕಟ್ಟೆ ಬೇಡಿಕೆ ಮಿಶ್ರವಾಗಿದೆ — ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
            )
        names = ", ".join(top_green)
        return (
            f"ಈ ವಾರ {count} ಬೆಳೆ(ಗಳನ್ನು) ಬಿತ್ತನೆಗೆ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ: {names}. "
            "ಈ ಬೆಳೆಗಳಿಗೆ ಹೆಚ್ಚಿನ ಖರೀದಿದಾರರ ಬೇಡಿಕೆ ಮತ್ತು ಉತ್ತಮ ಬೆಲೆ ಇದೆ. "
            "ಈಗ ಬಿತ್ತಿ — ಕೊಯ್ಲಿನ ಸಮಯದಲ್ಲಿ ಅತ್ಯುತ್ತಮ ಬೆಲೆ ಸಿಗುತ್ತದೆ."
        )
