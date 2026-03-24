"""
LLM-powered Retrieval Planner for the Agentic RAG Orchestrator.

Generates a structured JSON tool execution plan for complex queries.
Uses Groq Llama-3.1-8B-Instant (~80ms, ~₹0.001/call) since planning
is a lightweight task.
"""

from __future__ import annotations

from loguru import logger

from src.rag.agentic.models import RetrievalPlan, ToolCall
from src.rag.routing.prefilter import extract_entities

PLANNER_SYSTEM_PROMPT = """You are a retrieval orchestrator for CropFresh, an Indian agricultural AI assistant.

Available tools (use ONLY these exact names):
- "vector_search"     : Search the agricultural knowledge base (crop guides, schemes, agronomy)
- "graph_rag"         : Query Neo4j for farmer/buyer/crop relationships and supply chain
- "multi_source_rates": Fetch Karnataka mandi, support price, fuel, or gold rates from multiple sources
- "price_api"         : Backward-compatible mandi-only alias of multi_source_rates
- "weather_api"       : Get IMD weather forecast for next 3-5 days
- "browser_scrape"    : Scrape live government/news websites for scheme updates
- "direct_llm"        : Answer directly without retrieval (for simple sub-questions)

Rules:
- Use MINIMUM tools necessary to answer accurately
- Mark independent tools as can_parallelize: true for speed
- Set confidence_threshold higher (0.85) for safety-critical advice (pesticides, loans)
- Set confidence_threshold lower (0.70) for general agronomy knowledge
- Use "multi_source_rates" for price, fuel, gold, mandi, MSP, or support-price queries
- Set "rate_kinds" to one or more of: mandi_wholesale, retail_produce, fuel, gold, support_price
- Use "market" for a specific mandi/APMC and "district" for district-wide filters

Respond with ONLY valid JSON (no markdown):
{
  "plan": [
    {"tool_name": "multi_source_rates", "params": {"rate_kinds": ["mandi_wholesale"], "commodity": "tomato", "market": "Hubli"}, "can_parallelize": true, "priority": 1},
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
        """Rule-based fallback plan when the planner LLM is unavailable."""
        query_lower = query.lower()
        entities = extract_entities(query)
        crop = entities["crops"][0] if entities["crops"] else None
        location = entities["locations"][0] if entities["locations"] else None

        if any(token in query_lower for token in ("fuel", "petrol", "diesel")):
            return RetrievalPlan(
                plan=[
                    ToolCall(
                        tool_name="multi_source_rates",
                        params={"rate_kinds": ["fuel"], "state": "Karnataka", "market": location},
                        can_parallelize=False,
                    )
                ],
                confidence_threshold=0.75,
                query=query,
                plan_reasoning="Fallback plan detected a fuel price request",
            )

        if "gold" in query_lower:
            return RetrievalPlan(
                plan=[
                    ToolCall(
                        tool_name="multi_source_rates",
                        params={"rate_kinds": ["gold"], "state": "Karnataka", "market": location},
                        can_parallelize=False,
                    )
                ],
                confidence_threshold=0.75,
                query=query,
                plan_reasoning="Fallback plan detected a gold price request",
            )

        if any(token in query_lower for token in ("msp", "support price", "floor price")) and crop:
            return RetrievalPlan(
                plan=[
                    ToolCall(
                        tool_name="multi_source_rates",
                        params={"rate_kinds": ["support_price"], "commodity": crop, "state": "Karnataka"},
                        can_parallelize=False,
                    )
                ],
                confidence_threshold=0.75,
                query=query,
                plan_reasoning="Fallback plan detected a support-price request",
            )

        if any(token in query_lower for token in ("price", "rate", "mandi", "market", "apmc")) and crop:
            return RetrievalPlan(
                plan=[
                    ToolCall(
                        tool_name="multi_source_rates",
                        params={
                            "rate_kinds": ["mandi_wholesale"],
                            "commodity": crop,
                            "state": "Karnataka",
                            "market": location,
                        },
                        can_parallelize=False,
                    )
                ],
                confidence_threshold=0.75,
                query=query,
                plan_reasoning="Fallback plan detected a mandi-price request",
            )

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
