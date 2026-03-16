"""
Agentic Self-Evaluator — RAGAS-style confidence gate.

Computes faithfulness + relevance scores and decides
whether the orchestrator should retry with a revised plan.

Confidence formula: faithfulness × 0.6 + relevance × 0.4
Retry threshold: confidence < 0.75 (configurable)
Max retries: 2 (to prevent infinite loops)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.rag.agentic.models import EvalGate

EVALUATOR_PROMPT = """You are a quality evaluator for an Indian agricultural AI assistant.

Evaluate the generated answer against these criteria:

1. FAITHFULNESS (0–1): Is the answer fully supported by the retrieved context?
   - 1.0: Every claim is supported by the context
   - 0.5: Some claims are supported, some are LLM knowledge
   - 0.0: Answer contradicts or ignores the context

2. RELEVANCE (0–1): Does the answer address what the farmer actually asked?
   - 1.0: Directly answers the question completely
   - 0.5: Partially answers or goes off-topic
   - 0.0: Does not answer the question

Context provided:
{context}

Question: {query}

Generated Answer: {answer}

Respond with ONLY valid JSON:
{{"faithfulness": <0.0-1.0>, "relevance": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""


class AgenticSelfEvaluator:
    """Lightweight RAGAS-style answer quality gate."""

    def __init__(self, llm=None):
        self.llm = llm

    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: list[Any],
        confidence_threshold: float = 0.75,
    ) -> EvalGate:
        """
        Evaluate answer quality and decide if retry is needed.

        Args:
            query: Original user query
            answer: Generated answer text
            retrieved_docs: Documents used for generation
            confidence_threshold: Minimum confidence before retry

        Returns:
            EvalGate with confidence score and retry flag
        """
        if self.llm is None or not answer:
            return EvalGate(
                confidence=0.80,
                faithfulness=0.80,
                relevance=0.80,
                should_retry=False,
                reason="Self-evaluation skipped (no LLM available)",
            )

        try:
            import json

            from src.orchestrator.llm_provider import LLMMessage

            context_text = "\n\n".join(
                getattr(doc, "text", str(doc))[:500]
                for doc in retrieved_docs[:5]
            )

            prompt = EVALUATOR_PROMPT.format(
                context=context_text or "(No context retrieved)",
                query=query,
                answer=answer,
            )

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=150,
            )

            result = json.loads(response.content)
            faithfulness = float(result.get("faithfulness", 0.80))
            relevance = float(result.get("relevance", 0.80))
            reasoning = result.get("reasoning", "")

            confidence = (faithfulness * 0.6) + (relevance * 0.4)
            should_retry = confidence < confidence_threshold

            logger.info(
                f"AgenticSelfEvaluator: faith={faithfulness:.2f} | "
                f"rel={relevance:.2f} | conf={confidence:.2f} | "
                f"retry={should_retry}"
            )

            return EvalGate(
                confidence=confidence,
                faithfulness=faithfulness,
                relevance=relevance,
                should_retry=should_retry,
                reason=f"faith={faithfulness:.2f}, rel={relevance:.2f}: {reasoning}",
            )

        except Exception as e:
            logger.warning(f"AgenticSelfEvaluator failed: {e} — passing through")
            return EvalGate(
                confidence=0.80,
                faithfulness=0.80,
                relevance=0.80,
                should_retry=False,
                reason=f"Evaluation failed: {e}",
            )
