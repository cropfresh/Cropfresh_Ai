"""
LLM-powered Retrieval Planner for the Agentic RAG Orchestrator.

Generates a structured JSON tool execution plan for complex queries.
Uses Groq Llama-3.1-8B-Instant (~80ms, ~₹0.001/call) since planning
is a lightweight task.
"""

from __future__ import annotations

from loguru import logger

from ai.rag.agentic.models import RetrievalPlan, ToolCall


PLANNER_SYSTEM_PROMPT = """You are a retrieval orchestrator for CropFresh, an Indian agricultural AI assistant.

Available tools (use ONLY these exact names):
- "vector_search"  : Search the agricultural knowledge base (crop guides, schemes, agronomy)
- "graph_rag"      : Query Neo4j for farmer/buyer/crop relationships and supply chain
- "price_api"      : Fetch live mandi prices from eNAM for today
- "weather_api"    : Get IMD weather forecast for next 3-5 days
- "browser_scrape" : Scrape live government/news websites for scheme updates
- "direct_llm"     : Answer directly without retrieval (for simple sub-questions)

Rules:
- Use MINIMUM tools necessary to answer accurately
- Mark independent tools as can_parallelize: true for speed
- Set confidence_threshold higher (0.85) for safety-critical advice (pesticides, loans)
- Set confidence_threshold lower (0.70) for general agronomy knowledge

Respond with ONLY valid JSON (no markdown):
{
  "plan": [
    {"tool_name": "price_api", "params": {"commodity": "tomato", "location": "Hubli"}, "can_parallelize": true, "priority": 1},
    {"tool_name": "vector_search", "params": {"query": "tomato sell vs hold strategy"}, "can_parallelize": true, "priority": 1}
  ],
  "confidence_threshold": 0.75,
  "plan_reasoning": "User needs live price + agronomic decision advice"
}"""


class RetrievalPlanner:
    """
    LLM-powered retrieval planner.

    Generates a structured JSON tool execution plan for complex queries.
    """

    def __init__(self, llm=None):
        self.llm = llm

    async def plan(self, query: str, context: str = "") -> RetrievalPlan:
        """
        Generate a retrieval plan for a query.

        Args:
            query: User query text
            context: Optional session context (previous turns)

        Returns:
            RetrievalPlan with ordered tool calls
        """
        if self.llm is None:
            return self._fallback_plan(query)

        try:
            import json
            from src.orchestrator.llm_provider import LLMMessage

            user_msg = f"Query: {query}"
            if context:
                user_msg += f"\nSession context: {context[:300]}"

            messages = [
                LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_msg),
            ]

            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=400,
            )

            result = json.loads(response.content)
            plan_data = result.get("plan", [])

            tool_calls = [
                ToolCall(
                    tool_name=step.get("tool_name", "vector_search"),
                    params=step.get("params", {}),
                    can_parallelize=step.get("can_parallelize", True),
                    priority=step.get("priority", 1),
                )
                for step in plan_data
            ]

            logger.info(
                f"RetrievalPlanner: generated {len(tool_calls)}-step plan | "
                f"tools={[t.tool_name for t in tool_calls]}"
            )

            return RetrievalPlan(
                plan=tool_calls,
                confidence_threshold=float(result.get("confidence_threshold", 0.75)),
                query=query,
                plan_reasoning=result.get("plan_reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"RetrievalPlanner LLM failed: {e} — using fallback plan")
            return self._fallback_plan(query)

    def _fallback_plan(self, query: str) -> RetrievalPlan:
        """Simple fallback plan when LLM is unavailable: just vector search."""
        return RetrievalPlan(
            plan=[
                ToolCall(
                    tool_name="vector_search",
                    params={"query": query},
                    can_parallelize=False,
                )
            ],
            confidence_threshold=0.70,
            query=query,
            plan_reasoning="Fallback plan: LLM unavailable",
        )
