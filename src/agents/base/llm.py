"""
Base Agent LLM Mixin
====================
LLM generation and memory integrations.
"""

import asyncio
from typing import Any, Optional

from loguru import logger


class LLMMixin:
    """Mixin providing LLM interaction and conversational memory capabilities."""

    def _build_memory_context(self, context: Optional[dict]) -> str:
        """
        Phase 4 (G4) — Build a concise memory block from the session context.
        """
        if not context:
            return ""

        lines: list[str] = []

        entities = context.get("entities", {})
        if entities:
            # Filter internal bookkeeping keys (prefixed with __)
            user_entities = {k: v for k, v in entities.items() if not k.startswith("__")}
            if user_entities:
                entity_parts = [
                    f"{k.replace('_', ' ')}: {v}"
                    for k, v in user_entities.items()
                ]
                lines.append("[Session Memory] " + " | ".join(entity_parts))

        summary = context.get("conversation_summary", "")
        if summary:
            lines.append(f"[Conversation so far] {summary}")

        prev_agent = (
            entities.get("__current_agent")
            or context.get("current_agent", "")
        )
        if prev_agent:
            lines.append(f"[Previous agent] {prev_agent}")

        return "\n".join(lines)

    async def generate_with_llm(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> str:
        """
        Generate response using LLM.
        """
        if not self.llm:
            raise ValueError("No LLM configured for agent")

        from src.orchestrator.llm_provider import LLMMessage

        # Phase 4: inject memory as a system message between domain prompt and user
        working_messages = list(messages)  # shallow copy
        memory_text = self._build_memory_context(context)
        if memory_text:
            # Insert after the last system message (don't push in front of domain prompt)
            last_system_idx = -1
            for i, m in enumerate(working_messages):
                if m.get("role") == "system":
                    last_system_idx = i
            insert_at = last_system_idx + 1
            working_messages.insert(
                insert_at,
                {"role": "system", "content": memory_text},
            )

        llm_messages = [
            LLMMessage(role=m["role"], content=m["content"])
            for m in working_messages
        ]

        response = await self.llm.generate(
            llm_messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        return response.content

    async def _retry_operation(
        self,
        operation,
        *args,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Retry an async operation with exponential backoff.
        """
        retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retry {attempt + 1}/{retries} after {wait}s: {e}")
                    await asyncio.sleep(wait)

        raise last_error
