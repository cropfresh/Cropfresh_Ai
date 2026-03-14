"""
AgenticOrchestrator — Main orchestration class (ADR-007).

Autonomously plans, executes, and validates retrieval for complex queries.

Pipeline:
1. Plan: RetrievalPlanner generates JSON tool execution plan
2. Execute: ToolExecutor runs parallel/sequential tool calls
3. Draft: SpeculativeDraftEngine generates 3 parallel drafts → verifier picks best
4. Evaluate: AgenticSelfEvaluator gates answer quality (retry if < threshold)

Only engaged for FULL_AGENTIC and BROWSER_SCRAPE routes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from ai.rag.agentic.evaluator import AgenticSelfEvaluator
from ai.rag.agentic.models import OrchestratorResult, RetrievalPlan, ToolCall
from ai.rag.agentic.planner import RetrievalPlanner
from ai.rag.agentic.speculative import SpeculativeDraftEngine


class AgenticOrchestrator:
    """Advanced Agentic RAG Orchestrator (ADR-007)."""

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
        self.planner = RetrievalPlanner(llm=planner_llm)
        self.speculative_engine = SpeculativeDraftEngine(
            drafter_llm=drafter_llm,
            verifier_llm=verifier_llm,
        )
        self.evaluator = AgenticSelfEvaluator(llm=evaluator_llm)

        self._tools = {
            "vector_search": self._tool_vector_search,
            "graph_rag": self._tool_graph_rag,
            "price_api": self._tool_price_api,
            "weather_api": self._tool_weather_api,
            "browser_scrape": self._tool_browser_scrape,
            "direct_llm": self._tool_direct_llm,
        }

        self._kb = knowledge_base
        self._graph = graph_retriever
        self._price = price_client
        self._weather = weather_client
        self._browser = browser_rag

        logger.info(
            f"AgenticOrchestrator initialized | tools={list(self._tools.keys())}"
        )

    async def orchestrate(
        self,
        query: str,
        has_image: bool = False,
        context: str = "",
    ) -> OrchestratorResult:
        """Full agentic orchestration: plan → execute → speculate → evaluate."""
        start = time.perf_counter()
        retry_count = 0
        plan: Optional[RetrievalPlan] = None
        plan_feedback = ""

        while retry_count <= self.MAX_RETRIES:
            plan_query = query if not plan_feedback else f"{query}\n\nPrevious attempt feedback: {plan_feedback}"
            plan = await self.planner.plan(plan_query, context=context)

            logger.info(
                f"AgenticOrchestrator[retry={retry_count}]: "
                f"plan={[t.tool_name for t in plan.plan]}"
            )

            documents = await self._execute_plan(plan)
            tools_called = [t.tool_name for t in plan.plan]

            answer, _ = await self.speculative_engine.generate_and_select(
                documents=documents, query=query,
            )

            gate = await self.evaluator.evaluate(
                query=query, answer=answer,
                retrieved_docs=documents,
                confidence_threshold=plan.confidence_threshold,
            )

            if not gate.should_retry or retry_count >= self.MAX_RETRIES:
                total_ms = (time.perf_counter() - start) * 1000
                if gate.should_retry:
                    logger.warning(
                        f"AgenticOrchestrator: max retries reached | "
                        f"final_confidence={gate.confidence:.2f}"
                    )
                est_cost = 0.03 * (retry_count + 1) * len(plan.plan)
                return OrchestratorResult(
                    answer=answer,
                    retrieved_documents=documents,
                    plan_used=plan,
                    eval_gate=gate,
                    retry_count=retry_count,
                    total_latency_ms=total_ms,
                    estimated_cost_inr=est_cost,
                    tools_called=tools_called,
                )

            logger.info(
                f"AgenticOrchestrator: retrying (confidence={gate.confidence:.2f} < "
                f"{plan.confidence_threshold}) | reason={gate.reason}"
            )
            plan_feedback = gate.reason
            retry_count += 1

        total_ms = (time.perf_counter() - start) * 1000
        return OrchestratorResult(
            answer="Unable to generate a high-confidence answer. Please try rephrasing.",
            retry_count=retry_count,
            total_latency_ms=total_ms,
        )

    async def _execute_plan(self, plan: RetrievalPlan) -> list[Any]:
        """Execute a retrieval plan, respecting parallel execution hints."""
        all_docs: list[Any] = []

        parallel_calls = [t for t in plan.plan if t.can_parallelize]
        sequential_calls = sorted(
            [t for t in plan.plan if not t.can_parallelize],
            key=lambda t: t.priority,
        )

        if parallel_calls:
            results = await asyncio.gather(
                *[self._call_tool(tc) for tc in parallel_calls],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, list):
                    all_docs.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Parallel tool failed: {result}")

        for tc in sequential_calls:
            try:
                result = await self._call_tool(tc)
                if isinstance(result, list):
                    all_docs.extend(result)
            except Exception as e:
                logger.warning(f"Sequential tool {tc.tool_name} failed: {e}")

        logger.info(f"AgenticOrchestrator: collected {len(all_docs)} documents")
        return all_docs

    async def _call_tool(self, tool_call: ToolCall) -> list[Any]:
        """Route a ToolCall to its implementation."""
        tool_fn = self._tools.get(tool_call.tool_name)
        if tool_fn is None:
            logger.warning(f"Unknown tool: {tool_call.tool_name}")
            return []
        return await tool_fn(tool_call.params)

    # ── Tool Implementations ──────────────────────────────────────────

    async def _tool_vector_search(self, params: dict) -> list[Any]:
        """Execute vector knowledge base search."""
        if self._kb is None:
            return []
        try:
            query = params.get("query", "")
            top_k = params.get("top_k", 5)
            result = await self._kb.search(query, top_k=top_k)
            return result.documents if hasattr(result, "documents") else []
        except Exception as e:
            logger.warning(f"vector_search tool failed: {e}")
            return []

    async def _tool_graph_rag(self, params: dict) -> list[Any]:
        """Execute graph RAG retrieval."""
        if self._graph is None:
            return []
        try:
            from types import SimpleNamespace
            query = params.get("query", params.get("entity", ""))
            graph_ctx = await self._graph.retrieve(query)
            if graph_ctx and graph_ctx.context_text:
                doc = SimpleNamespace(
                    text=graph_ctx.context_text, id="graph_context",
                    score=1.0, metadata={"source": "neo4j"},
                )
                return [doc]
            return []
        except Exception as e:
            logger.warning(f"graph_rag tool failed: {e}")
            return []

    async def _tool_price_api(self, params: dict) -> list[Any]:
        """Fetch live mandi prices from eNAM."""
        if self._price is None:
            return []
        try:
            from types import SimpleNamespace
            commodity = params.get("commodity", "")
            location = params.get("location", "")
            price_data = await self._price.get_prices(commodity=commodity, mandi=location)
            if price_data:
                text = (
                    f"Live mandi price for {commodity} in {location}: "
                    f"₹{price_data.get('modal_price', 'N/A')}/quintal "
                    f"(Min: ₹{price_data.get('min_price', 'N/A')}, "
                    f"Max: ₹{price_data.get('max_price', 'N/A')}) "
                    f"as of {price_data.get('date', 'today')}."
                )
                doc = SimpleNamespace(
                    text=text, id="price_api", score=1.0,
                    metadata={"source": "enam", **price_data},
                )
                return [doc]
            return []
        except Exception as e:
            logger.warning(f"price_api tool failed: {e}")
            return []

    async def _tool_weather_api(self, params: dict) -> list[Any]:
        """Fetch IMD weather forecast."""
        if self._weather is None:
            return []
        try:
            from types import SimpleNamespace
            location = params.get("location", params.get("district", ""))
            days = params.get("days", 5)
            weather_data = await self._weather.get_forecast(location=location, days=days)
            if weather_data:
                doc = SimpleNamespace(
                    text=f"Weather forecast for {location}: {weather_data}",
                    id="weather_api", score=1.0, metadata={"source": "imd"},
                )
                return [doc]
            return []
        except Exception as e:
            logger.warning(f"weather_api tool failed: {e}")
            return []

    async def _tool_browser_scrape(self, params: dict) -> list[Any]:
        """Scrape live web sources for scheme updates / disease alerts."""
        if self._browser is None:
            return []
        try:
            return await self._browser.retrieve_live(query=params.get("query", ""))
        except Exception as e:
            logger.warning(f"browser_scrape tool failed: {e}")
            return []

    async def _tool_direct_llm(self, params: dict) -> list[Any]:
        """Direct LLM answer — returns empty docs."""
        logger.debug("direct_llm tool: no retrieval — generation from LLM knowledge")
        return []
