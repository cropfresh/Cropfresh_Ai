"""
Agronomy Agent
==============
Specialized agent for agricultural knowledge with deep-reasoning,
strict RAG grounding, and full multilingual support.

See also:
- agronomy_prompt.py  — system prompt, role, weather keywords
- agronomy_helpers.py — parse_follow_ups, compute_confidence, avg_score

Author: CropFresh AI Team
Version: 3.0.0
"""

from typing import Any, Optional
import re

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.prompt_context import build_system_prompt
from src.agents.agronomy_prompt import (
    AGRONOMY_ROLE,
    AGRONOMY_SYSTEM_PROMPT,
    WEATHER_KEYWORDS,
)
from src.agents.agronomy_helpers import (
    avg_score,
    compute_confidence,
    parse_follow_ups,
)
from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry


class AgronomyAgent(BaseAgent):
    """
    Specialized agent for agricultural knowledge (v3.0).

    Features:
    - Chain-of-Thought reasoning in every response
    - Strict RAG grounding (no hallucinated dosages)
    - Structured output (Analysis → Actions → Organic → Cautions → Follow-ups)
    - Dynamic multilingual follow-ups parsed from LLM output
    - Confidence scoring based on document relevance

    Usage:
        agent = AgronomyAgent(llm=provider, knowledge_base=kb)
        await agent.initialize()
        response = await agent.process("ಟೊಮೆಟೊ ಬೆಳೆಯಲು ಹೇಗೆ?")
    """

    def __init__(
        self,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        config = AgentConfig(
            name="agronomy_agent",
            description="Expert in crop cultivation, pest management, soil health, and farming practices",
            max_retries=2,
            temperature=0.3,       # ? Lower for factual accuracy
            max_tokens=1200,       # ? Room for structured CoT output
            kb_categories=["agronomy", "general"],
            tool_categories=["weather", "calculator"],
        )
        super().__init__(  # type: ignore[call-arg]
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get agronomy system prompt with shared CropFresh context."""
        return build_system_prompt(
            role_description=AGRONOMY_ROLE,
            domain_prompt=AGRONOMY_SYSTEM_PROMPT,
            context=context,
            agent_domain="agronomy",
        )

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """Process an agronomy query with deep reasoning."""
        truncated: str = query[:80]  # type: ignore[index]
        logger.info(f"AgronomyAgent processing: '{truncated}...'")

        try:
            # * Step 1 — Retrieve relevant context
            if execution:
                self.state_manager.add_step(execution.execution_id, "retrieve_context")

            documents = await self.retrieve_context(
                query=query, top_k=5, categories=["agronomy", "general"],
            )
            if execution:
                execution.documents = documents

            # * Step 2 — Weather tool (multilingual keyword check)
            tool_results = await self._maybe_fetch_weather(query, context, execution)

            # * Step 3 — Build LLM messages
            if execution:
                self.state_manager.add_step(execution.execution_id, "generate_response")

            messages = self._build_messages(query, context, documents, tool_results)

            # * Step 4 — Generate
            if self.llm:
                answer = await self.generate_with_llm(messages, context=context)
            else:
                answer = self._generate_fallback(query, documents)

            # * Step 5 — Parse follow-ups & compute confidence
            follow_ups = parse_follow_ups(answer)
            confidence = compute_confidence(documents, bool(tool_results))

            # Strip out [LANG: xx] tag from answer to clean up UI output
            clean_answer = re.sub(r"^\[LANG:\s*[a-zA-Z]+\]\s*\n*", "", answer).strip()

            return AgentResponse(
                content=clean_answer,
                agent_name=self.name,
                confidence=confidence,
                sources=self._extract_sources(documents),
                reasoning=(
                    f"Retrieved {len(documents)} docs "
                    f"(avg relevance {avg_score(documents):.2f}), "
                    f"used agronomy expertise with CoT reasoning"
                ),
                tools_used=[r["tool"] for r in tool_results],
                steps=["retrieve_context", "generate_response"],
                suggested_actions=follow_ups,
            )

        except Exception as e:
            logger.error(f"AgronomyAgent error: {e}")
            import traceback
            traceback.print_exc()
            return AgentResponse(
                content="I apologize, but I encountered an error processing your "
                        "agricultural query. Please try rephrasing your question.",
                agent_name=self.name,
                confidence=0.0,
                error=str(e),
                steps=["error"],
            )

    # ── Private helpers ────────────────────────────────────────

    async def _maybe_fetch_weather(
        self, query: str, context: Optional[dict], execution: Optional[AgentExecutionState],
    ) -> list[dict]:
        """Invoke weather tool if the query mentions weather in any language."""
        if not self.tools:
            return []
        if not any(kw in query.lower() for kw in WEATHER_KEYWORDS):
            return []

        location = "Kolar"
        if context:
            location = (
                context.get("user_profile", {}).get("location")
                or context.get("entities", {}).get("location")
                or location
            )
        result = await self.use_tool("get_weather", execution=execution, location=location)
        if result.success:
            return [{"tool": "get_weather", "success": True, "result": result.result}]
        return []

    def _build_messages(
        self, query: str, context: Optional[dict],
        documents: list[dict], tool_results: list[dict],
    ) -> list[dict]:
        """Assemble the LLM message list with system, context, and user."""
        messages: list[dict] = [
            {"role": "system", "content": self._get_system_prompt(context)},
        ]

        context_text = ""
        if documents:
            context_text = (
                "\n\n**Retrieved Knowledge (use these as primary source):**\n"
                + self.format_context(documents)
            )
        if tool_results:
            context_text += (
                "\n\n**Real-time Tool Data:**\n"
                + self.format_tool_results(tool_results)
            )

        if context and context.get("conversation_summary"):
            messages.append({
                "role": "system",
                "content": f"Previous conversation:\n{context['conversation_summary']}",
            })

        user_message = f"{query}\n{context_text}" if context_text else query
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _generate_fallback(query: str, documents: list[dict[str, Any]]) -> str:
        """Generate response without LLM using document content."""
        if not documents:
            return (
                "I don't have specific information about that topic. "
                "Please contact your local KVK (Krishi Vigyan Kendra) or "
                "agricultural extension office for detailed guidance."
            )
        parts: list[str] = ["Based on available information:\n"]
        top_docs = documents[:3]  # type: ignore[index]
        for doc in top_docs:
            text: str = doc.get("text", "")
            parts.append(f"• {text[:300]}...")
        return "\n".join(parts)
