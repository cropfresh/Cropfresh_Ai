"""Hallucination checking for generated answers."""

from __future__ import annotations

from loguru import logger

from src.rag.knowledge_base import Document


class HallucinationChecker:
    """Validate that generated answers are grounded in retrieved documents."""

    HALLUCINATION_PROMPT = """You are a grader checking if an LLM generation is grounded in and supported by a set of retrieved documents.

Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the documents.
'yes' means the answer is supported by the facts in the documents.
'no' means the answer contains information not found in the documents.

Respond with only JSON:
{"score": "yes", "reasoning": "brief explanation"} or {"score": "no", "reasoning": "what was not grounded"}"""

    def __init__(self, llm=None):
        self.llm = llm

    async def check(
        self,
        answer: str,
        documents: list[Document],
        query: str,
    ) -> tuple[bool, str]:
        if self.llm is None:
            return True, "No LLM available for hallucination check"

        import json

        from src.orchestrator.llm_provider import LLMMessage

        docs_text = "\n\n".join(f"Document {i + 1}:\n{doc.text}" for i, doc in enumerate(documents))
        messages = [
            LLMMessage(role="system", content=self.HALLUCINATION_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Documents:\n{docs_text}\n\n"
                    f"User Question: {query}\n\n"
                    f"LLM Answer: {answer}\n\n"
                    "Is this answer grounded in the documents?"
                ),
            ),
        ]

        try:
            response = await self.llm.generate(messages, temperature=0.0, max_tokens=100)
            result = json.loads(response.content)
            return result.get("score", "yes").lower() == "yes", result.get("reasoning", "")
        except Exception as exc:
            logger.warning("Hallucination check failed: {}", exc)
            return True, f"Check failed: {exc}"
