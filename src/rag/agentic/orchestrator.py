"""Agentic retrieval orchestrator (ADR-007)."""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger

from src.rag.agentic.evaluator import AgenticSelfEvaluator
from src.rag.agentic.executor import execute_retrieval_plan
from src.rag.agentic.models import OrchestratorResult, RetrievalPlan, ToolCall
from src.rag.agentic.planner import RetrievalPlanner
from src.rag.agentic.speculative import SpeculativeDraftEngine
from src.rag.agentic.tool_handlers import AgenticToolHandlers


class AgenticOrchestrator:
    """Advanced agentic RAG orchestrator."""

    MAX_RETRIES = 2

    def __init__(
        self,
        planner_llm=None,
        drafter_llm=None,
        verifier_llm=None,
        evaluator_llm=None,
        knowledge_base=None,
        graph_retriever=None,
        price_client=None,
        weather_client=None,
        browser_rag=None,
    ):
        del price_client  # price_api now routes through the shared rate hub
        self.planner = RetrievalPlanner(llm=planner_llm)
        self.speculative_engine = SpeculativeDraftEngine(
            drafter_llm=drafter_llm,
            verifier_llm=verifier_llm,
        )
        self.evaluator = AgenticSelfEvaluator(llm=evaluator_llm)
        self.handlers = AgenticToolHandlers(
            knowledge_base=knowledge_base,
            graph_retriever=graph_retriever,
            weather_client=weather_client,
            browser_rag=browser_rag,
        )
        self._tools = {
            "vector_search": self.handlers.vector_search,
            "graph_rag": self.handlers.graph_rag,
            "multi_source_rates": self.handlers.multi_source_rates,
            "price_api": self.handlers.price_api,
            "weather_api": self.handlers.weather_api,
            "browser_scrape": self.handlers.browser_scrape,
            "direct_llm": self.handlers.direct_llm,
        }
        logger.info("AgenticOrchestrator initialized | tools={}", list(self._tools))

    async def orchestrate(self, query: str, has_image: bool = False, context: str = "") -> OrchestratorResult:
        """Full agentic orchestration: plan → execute → draft → evaluate."""
        del has_image
        start = time.perf_counter()
        retry_count = 0
        plan: Optional[RetrievalPlan] = None
        plan_feedback = ""

        while retry_count <= self.MAX_RETRIES:
            planned_query = query if not plan_feedback else f"{query}\n\nPrevious attempt feedback: {plan_feedback}"
            plan = await self.planner.plan(planned_query, context=context)
            logger.info("AgenticOrchestrator[retry={}] plan={}", retry_count, [tool.tool_name for tool in plan.plan])
            documents = await execute_retrieval_plan(plan, self._call_tool)
            answer, _ = await self.speculative_engine.generate_and_select(documents=documents, query=query)
            gate = await self.evaluator.evaluate(
                query=query,
                answer=answer,
                retrieved_docs=documents,
                confidence_threshold=plan.confidence_threshold,
            )
            if not gate.should_retry or retry_count >= self.MAX_RETRIES:
                total_ms = (time.perf_counter() - start) * 1000
                return OrchestratorResult(
                    answer=answer,
                    retrieved_documents=documents,
                    plan_used=plan,
                    eval_gate=gate,
                    retry_count=retry_count,
                    total_latency_ms=total_ms,
                    estimated_cost_inr=0.03 * (retry_count + 1) * max(len(plan.plan), 1),
                    tools_called=[tool.tool_name for tool in plan.plan],
                )
            logger.info(
                "AgenticOrchestrator retrying after low confidence {:.2f} < {:.2f}",
                gate.confidence,
                plan.confidence_threshold,
            )
            plan_feedback = gate.reason
            retry_count += 1

        total_ms = (time.perf_counter() - start) * 1000
        return OrchestratorResult(
            answer="Unable to generate a high-confidence answer. Please try rephrasing.",
            retry_count=retry_count,
            total_latency_ms=total_ms,
        )

    async def _call_tool(self, tool_call: ToolCall) -> list[Any]:
        """Route a ToolCall to its implementation."""
        tool_fn = self._tools.get(tool_call.tool_name)
        if tool_fn is None:
            logger.warning("Unknown tool: {}", tool_call.tool_name)
            return []
        return await tool_fn(tool_call.params)
