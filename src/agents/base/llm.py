"""
Base Agent LLM Mixin
====================
LLM generation and memory integrations.
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from src.agents.kannada import get_kannada_context
from src.rag.language_support import build_generation_language_instruction
from src.shared.language import resolve_language


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
                entity_parts = [f"{k.replace('_', ' ')}: {v}" for k, v in user_entities.items()]
                lines.append("[Session Memory] " + " | ".join(entity_parts))

        summary = context.get("conversation_summary", "")
        if summary:
            lines.append(f"[Conversation so far] {summary}")

        prev_agent = entities.get("__current_agent") or context.get("current_agent", "")
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
            self._insert_system_message(working_messages, memory_text)

        kannada_context = self._build_kannada_context(working_messages, context)
        if kannada_context:
            self._insert_system_message(working_messages, kannada_context)

        language_instruction = self._build_language_instruction(working_messages, context)
        if language_instruction:
            self._insert_system_message(working_messages, language_instruction)

        llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in working_messages]

        response = await self.llm.generate(
            llm_messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        return response.content

    def _build_language_instruction(
        self,
        messages: list[dict],
        context: Optional[dict],
    ) -> str:
        """Build per-turn language guidance for domain agents."""
        if self.name == "supervisor":
            return ""

        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                user_text = str(message["content"])
                break

        language = resolve_language(query=user_text, context=context, default="en")
        return build_generation_language_instruction(
            user_text,
            language=language,
        )

    def _build_kannada_context(
        self,
        messages: list[dict],
        context: Optional[dict],
    ) -> str:
        """Inject centralized Kannada domain guidance for custom prompt agents."""
        if self.name == "supervisor":
            return ""
        if self._already_has_kannada_context(messages):
            return ""

        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                user_text = str(message["content"])
                break

        language = resolve_language(query=user_text, context=context, default="en")
        if language != "kn":
            return ""
        return get_kannada_context(self.name, context)

    @staticmethod
    def _insert_system_message(messages: list[dict], content: str) -> None:
        """Insert a system message after the last existing system block."""
        last_system_idx = -1
        for index, message in enumerate(messages):
            if message.get("role") == "system":
                last_system_idx = index

        messages.insert(
            last_system_idx + 1,
            {"role": "system", "content": content},
        )

    @staticmethod
    def _already_has_kannada_context(messages: list[dict]) -> bool:
        """Avoid duplicating the shared Kannada block when a prompt already has it."""
        return any(
            message.get("role") == "system"
            and "## Kannada Language Guidelines" in str(message.get("content", ""))
            for message in messages
        )

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
                    wait = 2**attempt  # Exponential backoff
                    logger.warning(f"Retry {attempt + 1}/{retries} after {wait}s: {e}")
                    await asyncio.sleep(wait)

        raise last_error
