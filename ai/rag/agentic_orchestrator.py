"""
Agentic RAG Orchestrator
========================
Autonomous multi-tool retrieval planner with speculative generation (ADR-007).

Replaces the fixed 4-node LangGraph pipeline for complex queries with an
adaptive, self-evaluating orchestrator:

1. RetrievalPlanner  — Groq 8B plans which tools to call (and how)
2. ToolExecutor      — Parallel execution via asyncio.gather()
3. SpeculativeDraftEngine — 3 parallel drafts → verifier picks best (–51% latency)
4. AgenticSelfEvaluator — RAGAS-style confidence gate; retries on < 0.75

Only engaged when AdaptiveQueryRouter routes to FULL_AGENTIC or BROWSER_SCRAPE.
All other routes bypass the orchestrator for lower latency.

Architecture: docs/architecture/agentic_rag_system.md
ADR: docs/decisions/ADR-007-agentic-rag-orchestrator.md
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """A single tool call in a retrieval plan."""
    tool_name: str = Field(description="Name of the tool to call")
    params: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    can_parallelize: bool = Field(
        default=True,
        description="True if this call can run in parallel with other parallelizable calls"
    )
    priority: int = Field(default=1, description="Execution priority (lower = run first)")


class RetrievalPlan(BaseModel):
    """
    A structured retrieval plan from the RetrievalPlanner.

    Contains an ordered list of tool calls to execute,
    with parallel execution hints.
    """
    plan: list[ToolCall] = Field(description="Ordered list of tool calls to execute")
    confidence_threshold: float = Field(
        default=0.75,
        description="Minimum answer confidence before returning (0–1)"
    )
    query: str = Field(description="Original query this plan was generated for")
    plan_reasoning: str = Field(default="", description="Planner's reasoning for LangSmith")


class EvalGate(BaseModel):
    """Result of the AgenticSelfEvaluator confidence check."""
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence 0–1")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Faithfulness to retrieved docs")
    relevance: float = Field(ge=0.0, le=1.0, description="Answer relevancy to query")
    should_retry: bool = Field(description="True if confidence < threshold")
    reason: str = Field(description="Evaluation explanation for logging")


class Draft(BaseModel):
    """A single speculative draft from a drafter LLM."""
    content: str = Field(description="Generated draft text")
    source_doc_indices: list[int] = Field(
        default_factory=list,
        description="Which document subset indices this draft used"
    )
    drafter_model: str = Field(default="groq/llama-3.1-8b-instant")
    generation_ms: float = Field(default=0.0)


class OrchestratorResult(BaseModel):
    """Final result from the AgenticOrchestrator."""
    answer: str
    retrieved_documents: list[Any] = Field(default_factory=list)
    plan_used: Optional[RetrievalPlan] = None
    eval_gate: Optional[EvalGate] = None
    retry_count: int = 0
    total_latency_ms: float = 0.0
    estimated_cost_inr: float = 0.0
    tools_called: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Planner (LLM-powered)
# ─────────────────────────────────────────────────────────────────────────────

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
    LLM-powered retrieval planner using Groq Llama-3.1-8B-Instant.

    Generates a structured JSON tool execution plan for complex queries.
    Fast model choice (~80ms, ~₹0.001/call) since planning is a lightweight task.
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


# ─────────────────────────────────────────────────────────────────────────────
# Speculative Draft Engine (–51% voice latency)
# ─────────────────────────────────────────────────────────────────────────────

class SpeculativeDraftEngine:
    """
    Parallel speculative generation engine.

    Splits retrieved documents into N subsets and generates N drafts in parallel.
    A verifier LLM selects the best draft.

    Latency model:
    - Sequential (70B): 3.8s
    - Speculative (3× 8B + verifier): 1.6s  →  –58% latency

    Based on: "Speculative RAG: Enhancing Retrieval Augmented Generation" (Google, 2024)
    """

    VERIFIER_PROMPT = """You are a quality controller for an Indian agricultural AI assistant.

You will receive {n} draft answers to the same question. Select the BEST draft based on:
1. Factual accuracy and groundedness in the provided context
2. Completeness — does it answer all parts of the question?
3. Clarity and actionability for an Indian farmer

Respond with ONLY valid JSON:
{{"best_draft_index": <0-based index of best draft>, "reason": "<one sentence>"}}"""

    DRAFTER_PROMPT = """You are a helpful agricultural AI assistant for Indian farmers.
Answer the following question based ONLY on the provided context documents.
Be concise, practical, and use simple language suitable for farmers.

Context:
{context}

Question: {query}

Answer:"""

    def __init__(self, drafter_llm=None, verifier_llm=None, n_subsets: int = 3):
        """
        Initialize the speculative draft engine.

        Args:
            drafter_llm: Fast LLM for generating drafts (Groq Llama-3.1-8B)
            verifier_llm: Higher quality LLM for selecting best draft (Gemini Flash / Groq 70B)
            n_subsets: Number of parallel drafts to generate (default: 3)
        """
        self.drafter_llm = drafter_llm
        self.verifier_llm = verifier_llm
        self.n_subsets = n_subsets

    async def generate_and_select(
        self,
        documents: list[Any],
        query: str,
    ) -> tuple[str, int]:
        """
        Generate N parallel drafts and select the best one.

        Args:
            documents: Retrieved documents (will be split into subsets)
            query: User query

        Returns:
            (best_draft_content, best_draft_index)
        """
        if not documents:
            return "No relevant information found.", 0

        # No LLM available — single draft from all docs
        if self.drafter_llm is None:
            return f"Based on {len(documents)} retrieved documents: [LLM unavailable]", 0

        # Split documents into N subsets
        subsets = self._split_documents(documents, self.n_subsets)
        actual_n = len(subsets)

        logger.info(
            f"SpeculativeDraftEngine: generating {actual_n} parallel drafts | "
            f"docs={len(documents)} | query={query[:60]}..."
        )

        # Generate all drafts in parallel
        start = time.perf_counter()
        drafts = await asyncio.gather(
            *[self._generate_draft(subset, query, idx) for idx, subset in enumerate(subsets)],
            return_exceptions=True,
        )
        gen_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"SpeculativeDraftEngine: {actual_n} drafts in {gen_ms:.0f}ms")

        # Filter out exceptions
        valid_drafts: list[Draft] = [
            d for d in drafts if isinstance(d, Draft) and d.content
        ]

        if not valid_drafts:
            return "Unable to generate an answer from retrieved documents.", 0

        if len(valid_drafts) == 1:
            return valid_drafts[0].content, 0

        # Select best draft using verifier LLM
        best_idx = await self._select_best_draft(valid_drafts, query)
        best_content = valid_drafts[best_idx].content

        logger.info(f"SpeculativeDraftEngine: selected draft {best_idx}/{len(valid_drafts)}")
        return best_content, best_idx

    async def _generate_draft(
        self,
        doc_subset: list[Any],
        query: str,
        subset_idx: int,
    ) -> Draft:
        """Generate a single draft from a document subset."""
        start = time.perf_counter()

        try:
            from src.orchestrator.llm_provider import LLMMessage

            # Build context from subset
            context_text = "\n\n".join(
                getattr(doc, "text", str(doc)) for doc in doc_subset
            )

            prompt = self.DRAFTER_PROMPT.format(
                context=context_text[:3000],  # Cap to avoid token limits
                query=query,
            )

            messages = [LLMMessage(role="user", content=prompt)]

            response = await self.drafter_llm.generate(
                messages,
                temperature=0.2,
                max_tokens=500,
            )

            gen_ms = (time.perf_counter() - start) * 1000
            return Draft(
                content=response.content,
                source_doc_indices=[subset_idx],
                generation_ms=gen_ms,
            )

        except Exception as e:
            logger.warning(f"SpeculativeDraftEngine: draft {subset_idx} failed: {e}")
            gen_ms = (time.perf_counter() - start) * 1000
            return Draft(
                content="",
                source_doc_indices=[subset_idx],
                generation_ms=gen_ms,
            )

    async def _select_best_draft(
        self,
        drafts: list[Draft],
        query: str,
    ) -> int:
        """Use verifier LLM to select the best draft."""
        if self.verifier_llm is None:
            # No verifier — return longest draft as heuristic
            return max(range(len(drafts)), key=lambda i: len(drafts[i].content))

        try:
            import json
            from src.orchestrator.llm_provider import LLMMessage

            # Format all drafts for comparison
            drafts_text = "\n\n---\n\n".join(
                f"Draft {i}:\n{d.content}" for i, d in enumerate(drafts)
            )

            system_prompt = self.VERIFIER_PROMPT.format(n=len(drafts))
            user_content = f"Question: {query}\n\n{drafts_text}"

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_content),
            ]

            response = await self.verifier_llm.generate(
                messages,
                temperature=0.0,
                max_tokens=100,
            )

            result = json.loads(response.content)
            best_idx = int(result.get("best_draft_index", 0))

            # Clamp to valid range
            return max(0, min(best_idx, len(drafts) - 1))

        except Exception as e:
            logger.warning(f"SpeculativeDraftEngine: verifier failed: {e} — using draft 0")
            return 0

    def _split_documents(
        self,
        documents: list[Any],
        n: int,
    ) -> list[list[Any]]:
        """Split documents into n equal subsets for parallel drafting."""
        if not documents:
            return []

        # Don't create more subsets than documents
        actual_n = min(n, len(documents))
        chunk_size = max(1, len(documents) // actual_n)

        subsets = []
        for i in range(actual_n):
            start = i * chunk_size
            end = start + chunk_size if i < actual_n - 1 else len(documents)
            subset = documents[start:end]
            if subset:
                subsets.append(subset)

        return subsets


# ─────────────────────────────────────────────────────────────────────────────
# Self-Evaluator (Confidence Gate)
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Lightweight RAGAS-style answer quality gate.

    Computes faithfulness + relevance scores and decides
    whether the orchestrator should retry with a revised plan.

    Confidence formula: faithfulness × 0.6 + relevance × 0.4
    Retry threshold: confidence < 0.75 (configurable)
    Max retries: 2 (to prevent infinite loops)
    """

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
            retrieved_docs: Documents used for generation (for faithfulness check)
            confidence_threshold: Minimum confidence before retry (default: 0.75)

        Returns:
            EvalGate with confidence score and retry flag
        """
        if self.llm is None or not answer:
            # No evaluator LLM — pass through with default confidence
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

            # Build context text from retrieved docs
            context_text = "\n\n".join(
                getattr(doc, "text", str(doc))[:500]  # Cap per-doc length
                for doc in retrieved_docs[:5]  # Cap to 5 docs
            )

            prompt = EVALUATOR_PROMPT.format(
                context=context_text or "(No context retrieved)",
                query=query,
                answer=answer,
            )

            messages = [LLMMessage(role="user", content=prompt)]

            response = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=150,
            )

            result = json.loads(response.content)
            faithfulness = float(result.get("faithfulness", 0.80))
            relevance = float(result.get("relevance", 0.80))
            reasoning = result.get("reasoning", "")

            # Weighted confidence: faithfulness matters more for grounding
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


# ─────────────────────────────────────────────────────────────────────────────
# AgenticOrchestrator — Main Class
# ─────────────────────────────────────────────────────────────────────────────

class AgenticOrchestrator:
    """
    Advanced Agentic RAG Orchestrator (ADR-007).

    Autonomously plans, executes, and validates retrieval for complex queries.

    Pipeline:
    1. Plan: RetrievalPlanner generates JSON tool execution plan
    2. Execute: ToolExecutor runs parallel/sequential tool calls
    3. Draft: SpeculativeDraftEngine generates 3 parallel drafts → verifier selects best
    4. Evaluate: AgenticSelfEvaluator gates answer quality (retry if < threshold)

    Only engaged for FULL_AGENTIC and BROWSER_SCRAPE routes.

    Usage:
        orchestrator = AgenticOrchestrator(
            planner_llm=groq_8b,
            drafter_llm=groq_8b,
            verifier_llm=gemini_flash,
            evaluator_llm=groq_8b,
        )
        result = await orchestrator.orchestrate(
            query="Should I sell my tomatoes now or wait 2 weeks?",
            has_image=False,
        )
    """

    MAX_RETRIES = 2  # Safety limit on self-correction loop

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
        """
        Initialize the orchestrator with tool and LLM dependencies.

        All dependencies are optional — tools unavailable at runtime
        are skipped gracefully (logged as warnings).

        Args:
            planner_llm: LLM for plan generation (Groq 8B recommended)
            drafter_llm: LLM for draft generation (Groq 8B for speed)
            verifier_llm: LLM for draft verification (Gemini Flash / 70B for quality)
            evaluator_llm: LLM for self-evaluation (Groq 8B)
            knowledge_base: Qdrant vector knowledge base instance
            graph_retriever: Neo4j graph retriever instance
            price_client: eNAM price API client
            weather_client: IMD weather client
            browser_rag: BrowserRAGIntegration instance
        """
        self.planner = RetrievalPlanner(llm=planner_llm)
        self.speculative_engine = SpeculativeDraftEngine(
            drafter_llm=drafter_llm,
            verifier_llm=verifier_llm,
        )
        self.evaluator = AgenticSelfEvaluator(llm=evaluator_llm)

        # Tool registry
        self._tools = {
            "vector_search": self._tool_vector_search,
            "graph_rag": self._tool_graph_rag,
            "price_api": self._tool_price_api,
            "weather_api": self._tool_weather_api,
            "browser_scrape": self._tool_browser_scrape,
            "direct_llm": self._tool_direct_llm,
        }

        # Tool dependencies
        self._kb = knowledge_base
        self._graph = graph_retriever
        self._price = price_client
        self._weather = weather_client
        self._browser = browser_rag

        logger.info(
            f"AgenticOrchestrator initialized | "
            f"tools={list(self._tools.keys())}"
        )

    async def orchestrate(
        self,
        query: str,
        has_image: bool = False,
        context: str = "",
    ) -> OrchestratorResult:
        """
        Full agentic orchestration: plan → execute → speculate → evaluate.

        Args:
            query: User query text
            has_image: True if an image was attached
            context: Optional session context from recent turns

        Returns:
            OrchestratorResult with answer, sources, and metadata
        """
        start = time.perf_counter()
        retry_count = 0
        plan: Optional[RetrievalPlan] = None
        plan_feedback = ""

        while retry_count <= self.MAX_RETRIES:
            # ── Step 1: Plan ─────────────────────────────────────────────────
            plan_query = query if not plan_feedback else f"{query}\n\nPrevious attempt feedback: {plan_feedback}"
            plan = await self.planner.plan(plan_query, context=context)

            logger.info(
                f"AgenticOrchestrator[retry={retry_count}]: "
                f"plan={[t.tool_name for t in plan.plan]}"
            )

            # ── Step 2: Execute Tool Plan ─────────────────────────────────────
            documents = await self._execute_plan(plan)
            tools_called = [t.tool_name for t in plan.plan]

            # ── Step 3: Speculative Draft + Verify ───────────────────────────
            answer, _ = await self.speculative_engine.generate_and_select(
                documents=documents,
                query=query,
            )

            # ── Step 4: Self-Evaluate (Confidence Gate) ───────────────────────
            gate = await self.evaluator.evaluate(
                query=query,
                answer=answer,
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

                # Estimate cost: planner + drafts + verifier + evaluator calls
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

            # Retry with feedback from evaluator
            logger.info(
                f"AgenticOrchestrator: retrying (confidence={gate.confidence:.2f} < "
                f"{plan.confidence_threshold}) | reason={gate.reason}"
            )
            plan_feedback = gate.reason
            retry_count += 1

        # Should never reach here but just in case
        total_ms = (time.perf_counter() - start) * 1000
        return OrchestratorResult(
            answer="Unable to generate a high-confidence answer. Please try rephrasing.",
            retry_count=retry_count,
            total_latency_ms=total_ms,
        )

    async def _execute_plan(self, plan: RetrievalPlan) -> list[Any]:
        """
        Execute a retrieval plan, respecting parallel execution hints.

        Parallelizable calls run together via asyncio.gather().
        Sequential calls run in priority order.
        """
        all_docs: list[Any] = []

        # Separate parallel and sequential tool calls
        parallel_calls = [t for t in plan.plan if t.can_parallelize]
        sequential_calls = sorted(
            [t for t in plan.plan if not t.can_parallelize],
            key=lambda t: t.priority,
        )

        # Run parallelizable calls simultaneously
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

        # Run sequential calls in order
        for tc in sequential_calls:
            try:
                result = await self._call_tool(tc)
                if isinstance(result, list):
                    all_docs.extend(result)
            except Exception as e:
                logger.warning(f"Sequential tool {tc.tool_name} failed: {e}")

        logger.info(f"AgenticOrchestrator: collected {len(all_docs)} documents from plan")
        return all_docs

    async def _call_tool(self, tool_call: ToolCall) -> list[Any]:
        """Route a ToolCall to its implementation."""
        tool_fn = self._tools.get(tool_call.tool_name)
        if tool_fn is None:
            logger.warning(f"Unknown tool: {tool_call.tool_name}")
            return []
        return await tool_fn(tool_call.params)

    # ── Tool Implementations ──────────────────────────────────────────────────

    async def _tool_vector_search(self, params: dict) -> list[Any]:
        """Execute vector knowledge base search."""
        if self._kb is None:
            logger.warning("vector_search: No knowledge base available")
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
            logger.warning("graph_rag: No graph retriever available")
            return []
        try:
            query = params.get("query", params.get("entity", ""))
            graph_ctx = await self._graph.retrieve(query)
            if graph_ctx and graph_ctx.context_text:
                # Wrap graph context as a pseudo-document
                from types import SimpleNamespace
                doc = SimpleNamespace(
                    text=graph_ctx.context_text,
                    id="graph_context",
                    score=1.0,
                    metadata={"source": "neo4j"},
                )
                return [doc]
            return []
        except Exception as e:
            logger.warning(f"graph_rag tool failed: {e}")
            return []

    async def _tool_price_api(self, params: dict) -> list[Any]:
        """Fetch live mandi prices from eNAM."""
        if self._price is None:
            logger.warning("price_api: No price client available")
            return []
        try:
            commodity = params.get("commodity", "")
            location = params.get("location", "")
            # Attempt to fetch live price
            price_data = await self._price.get_prices(commodity=commodity, mandi=location)

            if price_data:
                from types import SimpleNamespace
                text = (
                    f"Live mandi price for {commodity} in {location}: "
                    f"₹{price_data.get('modal_price', 'N/A')}/quintal "
                    f"(Min: ₹{price_data.get('min_price', 'N/A')}, "
                    f"Max: ₹{price_data.get('max_price', 'N/A')}) "
                    f"as of {price_data.get('date', 'today')}."
                )
                doc = SimpleNamespace(
                    text=text,
                    id="price_api",
                    score=1.0,
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
            logger.warning("weather_api: No weather client available")
            return []
        try:
            location = params.get("location", params.get("district", ""))
            days = params.get("days", 5)
            weather_data = await self._weather.get_forecast(location=location, days=days)

            if weather_data:
                from types import SimpleNamespace
                text = f"Weather forecast for {location}: {weather_data}"
                doc = SimpleNamespace(
                    text=text,
                    id="weather_api",
                    score=1.0,
                    metadata={"source": "imd"},
                )
                return [doc]
            return []
        except Exception as e:
            logger.warning(f"weather_api tool failed: {e}")
            return []

    async def _tool_browser_scrape(self, params: dict) -> list[Any]:
        """Scrape live web sources for scheme updates / disease alerts."""
        if self._browser is None:
            logger.warning("browser_scrape: No browser RAG integration available")
            return []
        try:
            query = params.get("query", "")
            return await self._browser.retrieve_live(query=query)
        except Exception as e:
            logger.warning(f"browser_scrape tool failed: {e}")
            return []

    async def _tool_direct_llm(self, params: dict) -> list[Any]:
        """
        Direct LLM answer — returns empty docs (answer comes from LLM knowledge).

        The orchestrator will still run generation, but without retrieved context.
        """
        logger.debug("direct_llm tool: no retrieval — generation from LLM knowledge")
        return []
