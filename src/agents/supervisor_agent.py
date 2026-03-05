"""
Supervisor Agent
================
Central orchestrator for multi-agent RAG system.

Responsibilities:
- Query analysis and intent classification
- Routing to specialized domain agents
- Multi-step reasoning coordination
- Response aggregation from multiple agents
- Fallback handling when no agent matches

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.prompt_context import get_identity_preamble
from src.memory.state_manager import AgentExecutionState, AgentStateManager, Message
from src.tools.registry import ToolRegistry


class RoutingDecision(BaseModel):
    """Decision about which agent to route to."""
    
    agent_name: str
    confidence: float
    reasoning: str
    requires_multiple: bool = False
    secondary_agents: list[str] = Field(default_factory=list)


# Routing prompt for the supervisor
ROUTING_PROMPT = f"""You are the Supervisor Agent for CropFresh AI.

{get_identity_preamble()}

Your job is to analyze user queries and route them to the most appropriate specialized agent.

Available agents:
1. **agronomy_agent**: Expert in crops, farming practices, pest management, soil health, irrigation, organic farming
   - Keywords: grow, plant, cultivate, harvest, pest, disease, fertilizer, soil, seed, irrigation, variety

2. **commerce_agent**: Expert in market prices, trading, sell/hold recommendations, AISP calculations
   - Keywords: price, sell, buy, mandi, market, rate, cost, profit, AISP, logistics

3. **platform_agent**: Expert in CropFresh app features, registration, account support, and platform usage
   - Keywords: register, login, app, feature, account, logistics, order, payment

4. **web_scraping_agent**: Expert in fetching LIVE data from websites - current mandi prices, weather, news
   - Keywords: live, current, today, real-time, fetch, scrape, website, portal, eNAM, Agmarknet, weather
   - Use for: "What's the current tomato price?", "Get today's weather advisory", "Latest agri news"

5. **browser_agent**: Expert in interactive web tasks requiring login, form submission, navigation
   - Keywords: login, submit, navigate, download, portal, form, interactive, authenticated
   - Use for: "Check my eNAM dashboard", "Submit an application", "Download price report"

6. **research_agent**: Expert in deep research with multiple sources, citations, and comprehensive reports
   - Keywords: research, investigate, comprehensive, detailed, compare, analysis, report, study
   - Use for: "Research best tomato varieties", "Compare farming methods", "Detailed analysis of..."

7. **general_agent**: Fallback for greetings, general questions, or unclear intents
   - Keywords: hello, hi, thanks, help, who are you

8. **buyer_matching_agent**: Expert in matching farmers and buyers using grade, price, and distance
   - Keywords: find buyer, match buyer, who will buy, find farmer, supplier match, buyer matching, sell my produce

9. **quality_assessment_agent**: Expert in produce grading (A+/A/B/C), defect detection, shelf life, and HITL verification
    - Keywords: quality check, grade produce, defects, bruise, fungal, shelf life, quality assessment, inspect crop

10. **adcl_agent**: Expert in crop recommendations, what to sow now, weekly demand analysis
    - Keywords: recommend, sow, what to grow, demand, crop suggestion, weekly report, which crop

11. **price_prediction_agent**: Expert in price forecasting, trend analysis, sell/hold timing
    - Keywords: predict, forecast, trend, future price, will price go up, hold or sell, price tomorrow

12. **crop_listing_agent**: Expert in creating/managing produce listings for sale
    - Keywords: list my crop, sell my produce, create listing, my listings, cancel listing, update listing

13. **logistics_agent**: Expert in delivery routing, transport cost, vehicle assignment
    - Keywords: delivery, transport, route, vehicle, logistics cost, shipping, pickup

14. **knowledge_agent**: Deep knowledge retrieval from agricultural knowledge base
    - Keywords: explain, tell me about, information, knowledge, learn, what is, how does

Analyze the user query and respond with a JSON object:
{{
    "agent_name": "name_of_agent",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "requires_multiple": true/false,
    "secondary_agents": ["other_agent"] // only if requires_multiple is true
}}

Only output the JSON, nothing else."""


class SupervisorAgent(BaseAgent):
    """
    Central Supervisor Agent for CropFresh AI.
    
    Orchestrates the multi-agent system:
    - Analyzes incoming queries
    - Routes to appropriate specialized agent
    - Coordinates multi-agent responses
    - Manages conversation context
    
    Usage:
        supervisor = SupervisorAgent(llm=provider)
        await supervisor.initialize()
        supervisor.register_agent("agronomy", agronomy_agent)
        response = await supervisor.process("How to grow tomatoes?")
    """
    
    def __init__(
        self,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        """
        Initialize Supervisor Agent.
        
        Args:
            llm: LLM provider
            tool_registry: Global tool registry
            state_manager: State manager for sessions
            knowledge_base: Knowledge base for retrieval
        """
        config = AgentConfig(
            name="supervisor",
            description="Central orchestrator that routes queries to specialized agents",
            max_retries=2,
            temperature=0.3,  # Lower temperature for routing decisions
            max_tokens=500,
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )
        
        # Registered specialized agents
        self._agents: dict[str, BaseAgent] = {}
        
        # General/fallback agent
        self._fallback_agent: Optional[BaseAgent] = None
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register a specialized agent.
        
        Args:
            name: Agent identifier
            agent: Agent instance
        """
        self._agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    def set_fallback_agent(self, agent: BaseAgent) -> None:
        """Set the fallback agent for unroutable queries."""
        self._fallback_agent = agent
        logger.info(f"Set fallback agent: {agent.name}")
    
    async def initialize(self) -> bool:
        """Initialize supervisor and all registered agents."""
        # Initialize all registered agents
        for name, agent in self._agents.items():
            try:
                await agent.initialize()
                logger.debug(f"Initialized agent: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {name}: {e}")
        
        self._initialized = True
        return True
    
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Generate supervisor system prompt."""
        return ROUTING_PROMPT
    
    async def route_query(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Determine which agent should handle the query.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            RoutingDecision with selected agent
        """
        import json
        
        # If no LLM, use rule-based routing
        if not self.llm:
            return self._route_rule_based(query)
        
        try:
            # Build messages for LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": query},
            ]
            
            # Add conversation context if available
            if context and context.get("conversation_summary"):
                messages.insert(1, {
                    "role": "system",
                    "content": f"Previous conversation:\n{context['conversation_summary']}",
                })
            
            # Get routing decision
            response = await self.generate_with_llm(
                messages,
                temperature=0.1,  # Very low for consistent routing
                max_tokens=200,
            )
            
            # Parse JSON response
            # Handle potential markdown code blocks
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            decision = json.loads(response_text)
            
            routing = RoutingDecision(
                agent_name=decision.get("agent_name", "general_agent"),
                confidence=decision.get("confidence", 0.5),
                reasoning=decision.get("reasoning", ""),
                requires_multiple=decision.get("requires_multiple", False),
                secondary_agents=decision.get("secondary_agents", []),
            )
            
            logger.info(f"Routing decision: {routing.agent_name} (confidence: {routing.confidence})")
            return routing
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse routing response: {e}")
            return self._route_rule_based(query)
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return self._route_rule_based(query)
    
    def _route_rule_based(self, query: str) -> RoutingDecision:
        """
        Simple rule-based routing as fallback.
        
        Uses keyword matching when LLM is unavailable.
        """
        query_lower = query.lower()
        
        # Agronomy keywords
        agronomy_kw = [
            "grow", "plant", "cultivat", "harvest", "pest", "disease",
            "fertilizer", "soil", "seed", "irrigation", "organic", "variety",
            "crop", "farming", "agriculture"
        ]
        
        # Commerce keywords
        commerce_kw = [
            "price", "sell", "buy", "mandi", "market", "rate", "cost",
            "profit", "aisp", "logistics", "₹", "rupee", "quintal"
        ]
        
        # Platform keywords
        platform_kw = [
            "register", "login", "app", "feature", "account", "order",
            "payment", "cropfresh", "digital twin"
        ]

        # Buyer matching keywords
        matching_kw = [
            "find buyer", "match buyer", "buyer matching", "who wants to buy",
            "find farmer", "find supplier", "supplier match", "sell my produce",
            "need tomatoes", "need onion", "procurement"
        ]

        # Quality assessment keywords
        quality_kw = [
            "quality check", "quality assessment", "grade produce", "produce grade",
            "defect", "bruise", "worm hole", "fungal", "shelf life", "inspect quality",
            "a+ grade", "quality grading"
        ]

        # Prefer explicit matching intents before weighted keyword scoring.
        if any(keyword in query_lower for keyword in matching_kw):
            return RoutingDecision(
                agent_name="buyer_matching_agent",
                confidence=0.85,
                reasoning="Rule-based routing",
            )

        # Prefer explicit quality assessment intents before weighted scoring.
        if any(keyword in query_lower for keyword in quality_kw):
            return RoutingDecision(
                agent_name="quality_assessment_agent",
                confidence=0.84,
                reasoning="Rule-based routing",
            )
        
        # Web scraping keywords (live data)
        scraping_kw = [
            "live", "current", "today", "real-time", "realtime", "fetch",
            "scrape", "website", "portal", "enam", "agmarknet", "weather",
            "latest", "now", "today's"
        ]
        
        # Browser agent keywords (interactive)
        browser_kw = [
            "login to", "submit", "navigate", "download", "form",
            "interactive", "authenticated", "dashboard", "check my"
        ]
        
        # Research agent keywords (deep research)
        research_kw = [
            "research", "investigate", "comprehensive", "detailed",
            "compare", "analysis", "report", "study", "in-depth"
        ]

        # * NEW: ADCL crop recommendation keywords
        adcl_kw = [
            "recommend", "sow", "what to grow", "demand", "crop suggestion",
            "weekly report", "which crop", "what should i grow",
        ]

        # * NEW: Price prediction keywords
        prediction_kw = [
            "predict", "forecast", "trend", "future price", "will price",
            "hold or sell", "price tomorrow", "price next week",
        ]

        # * NEW: Crop listing keywords
        listing_kw = [
            "list my crop", "sell my produce", "create listing",
            "my listings", "cancel listing", "update listing",
        ]

        # * NEW: Logistics keywords
        logistics_kw = [
            "delivery", "transport", "route", "vehicle", "logistics cost",
            "shipping", "pickup", "truck", "tempo",
        ]

        # * NEW: Knowledge agent keywords
        knowledge_kw = [
            "explain", "tell me about", "information", "knowledge",
            "learn", "what is", "how does",
        ]

        # Direct/general keywords
        general_kw = [
            "hello", "hi", "thanks", "thank you", "bye", "help",
            "who are you", "what are you"
        ]

        # * Check explicit phrase matches first (before weighted scoring)
        for kw in adcl_kw:
            if kw in query_lower:
                return RoutingDecision(
                    agent_name="adcl_agent", confidence=0.83, reasoning="Rule-based: crop recommendation",
                )
        for kw in listing_kw:
            if kw in query_lower:
                return RoutingDecision(
                    agent_name="crop_listing_agent", confidence=0.83, reasoning="Rule-based: listing",
                )
        for kw in logistics_kw:
            if kw in query_lower:
                return RoutingDecision(
                    agent_name="logistics_agent", confidence=0.82, reasoning="Rule-based: logistics",
                )

        # Score each category
        scores = {
            "agronomy_agent": sum(1 for kw in agronomy_kw if kw in query_lower),
            "commerce_agent": sum(1 for kw in commerce_kw if kw in query_lower),
            "platform_agent": sum(1 for kw in platform_kw if kw in query_lower),
            "buyer_matching_agent": sum(1 for kw in matching_kw if kw in query_lower),
            "quality_assessment_agent": sum(1 for kw in quality_kw if kw in query_lower),
            "web_scraping_agent": sum(1 for kw in scraping_kw if kw in query_lower),
            "browser_agent": sum(1 for kw in browser_kw if kw in query_lower),
            "research_agent": sum(1 for kw in research_kw if kw in query_lower),
            "price_prediction_agent": sum(1 for kw in prediction_kw if kw in query_lower),
            "knowledge_agent": sum(1 for kw in knowledge_kw if kw in query_lower),
            "general_agent": sum(1 for kw in general_kw if kw in query_lower),
        }
        
        # Find best match
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]
        
        # Default to general if no keywords match
        if best_score == 0:
            best_agent = "general_agent"
        
        confidence = min(best_score * 0.2 + 0.3, 0.9)  # Scale to 0.3-0.9
        
        return RoutingDecision(
            agent_name=best_agent,
            confidence=confidence,
            reasoning="Rule-based routing",
        )
    
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process a user query through the multi-agent system.
        
        Args:
            query: User query
            context: Optional context (conversation history, user profile)
            execution: Optional execution state for tracking
            
        Returns:
            AgentResponse from the appropriate agent
        """
        logger.info(f"Supervisor processing: '{query[:50]}...'")
        
        # Create execution state if not provided
        if execution is None and self.state_manager:
            session = await self.state_manager.create_session()
            execution = self.state_manager.create_execution(
                session.session_id,
                query,
            )
        
        try:
            # * Step 0: Photo/image detection — auto-route to QA agent
            if context and context.get("image_b64"):
                routing = RoutingDecision(
                    agent_name="quality_assessment_agent",
                    confidence=0.95,
                    reasoning="Photo input detected — routing to quality assessment",
                )
                logger.info("Photo detected — routing to quality_assessment_agent")
            else:
                # Step 1: Route query
                if execution:
                    self.state_manager.add_step(execution.execution_id, "route_query")
                routing = await self.route_query(query, context)
            
            if execution:
                execution.selected_agent = routing.agent_name
                execution.routing_confidence = routing.confidence
                execution.routing_reasoning = routing.reasoning
            
            # Step 2: Get target agent
            target_agent = self._agents.get(routing.agent_name)
            
            if not target_agent:
                # Try fallback
                target_agent = self._fallback_agent
                logger.warning(f"Agent '{routing.agent_name}' not found, using fallback")
            
            if not target_agent:
                # No agent available
                return AgentResponse(
                    content="I apologize, but I'm unable to process your request at this time. Please try again later.",
                    agent_name="supervisor",
                    confidence=0.0,
                    error="No suitable agent available",
                    steps=["route_query", "error_no_agent"],
                )
            
            # Step 3: Process with target agent
            if execution:
                self.state_manager.add_step(execution.execution_id, f"agent:{routing.agent_name}")
            
            response = await target_agent.process(query, context, execution)
            
            # Step 4: Handle multi-agent if needed
            if routing.requires_multiple and routing.secondary_agents:
                for secondary_name in routing.secondary_agents:
                    secondary_agent = self._agents.get(secondary_name)
                    if secondary_agent:
                        if execution:
                            self.state_manager.add_step(execution.execution_id, f"agent:{secondary_name}")
                        
                        secondary_response = await secondary_agent.process(query, context, execution)
                        
                        # Merge responses
                        response = self._merge_responses(response, secondary_response)
            
            # Step 5: Finalize
            if execution:
                self.state_manager.complete_execution(execution.execution_id, response.content)
            
            # Add routing info to response
            response.steps = ["route_query"] + response.steps
            
            return response
            
        except Exception as e:
            logger.error(f"Supervisor processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            return AgentResponse(
                content=f"I encountered an error processing your request. Please try again.",
                agent_name="supervisor",
                confidence=0.0,
                error=str(e),
                steps=["route_query", "error"],
            )
    
    async def process_with_session(
        self,
        query: str,
        session_id: str,
    ) -> AgentResponse:
        """
        Process query with session context.
        
        Maintains conversation history across multiple queries.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            AgentResponse
        """
        if not self.state_manager:
            return await self.process(query)
        
        # Get session context
        session = await self.state_manager.get_context(session_id)
        if not session:
            # Create new session
            session = await self.state_manager.create_session()
        
        # Add user message to history
        await self.state_manager.add_message(
            session.session_id,
            Message(role="user", content=query),
        )
        
        # Build context from session
        context = {
            "user_profile": session.user_profile,
            "entities": session.entities,
            "conversation_summary": self.state_manager.get_conversation_summary(session),
        }
        
        # Create execution state
        execution = self.state_manager.create_execution(session.session_id, query)
        
        # Process
        response = await self.process(query, context, execution)
        
        # Add assistant response to history
        await self.state_manager.add_message(
            session.session_id,
            Message(role="assistant", content=response.content),
        )
        
        return response
    
    def _merge_responses(
        self,
        primary: AgentResponse,
        secondary: AgentResponse,
    ) -> AgentResponse:
        """
        Merge responses from multiple agents.
        
        Args:
            primary: Primary agent response
            secondary: Secondary agent response
            
        Returns:
            Merged AgentResponse
        """
        # Combine content
        merged_content = f"{primary.content}\n\n**Additional Information ({secondary.agent_name}):**\n{secondary.content}"
        
        # Combine sources and steps
        all_sources = list(set(primary.sources + secondary.sources))
        all_steps = primary.steps + [f"merge:{secondary.agent_name}"] + secondary.steps
        all_tools = primary.tools_used + secondary.tools_used
        
        return AgentResponse(
            content=merged_content,
            agent_name=f"{primary.agent_name}+{secondary.agent_name}",
            confidence=(primary.confidence + secondary.confidence) / 2,
            sources=all_sources,
            reasoning=f"{primary.reasoning}; {secondary.reasoning}",
            tools_used=all_tools,
            steps=all_steps,
        )
    
    def get_available_agents(self) -> list[dict]:
        """
        Get list of registered agents with their info.
        
        Returns:
            List of agent info dicts
        """
        agents = []
        for name, agent in self._agents.items():
            # * Some agents (e.g. PricingAgent) don't inherit BaseAgent
            agents.append({
                "name": name,
                "description": getattr(agent, "description", str(type(agent).__name__)),
                "categories": getattr(getattr(agent, "config", None), "kb_categories", []),
            })
        return agents
