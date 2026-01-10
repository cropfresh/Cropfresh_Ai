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
ROUTING_PROMPT = """You are the Supervisor Agent for CropFresh AI, an agricultural marketplace platform.

Your job is to analyze user queries and route them to the most appropriate specialized agent.

Available agents:
1. **agronomy_agent**: Expert in crops, farming practices, pest management, soil health, irrigation, organic farming
   - Keywords: grow, plant, cultivate, harvest, pest, disease, fertilizer, soil, seed, irrigation, variety
   
2. **commerce_agent**: Expert in market prices, trading, sell/hold recommendations, AISP calculations
   - Keywords: price, sell, buy, mandi, market, rate, cost, profit, AISP, logistics
   
3. **platform_agent**: Expert in CropFresh app features, registration, quality grades, how to use the platform
   - Keywords: register, login, app, feature, quality, grade, logistics, order, payment
   
4. **general_agent**: Fallback for greetings, general questions, or unclear intents
   - Keywords: hello, hi, thanks, help, who are you

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
            "payment", "cropfresh", "quality grade", "digital twin"
        ]
        
        # Direct/general keywords
        general_kw = [
            "hello", "hi", "thanks", "thank you", "bye", "help",
            "who are you", "what are you"
        ]
        
        # Score each category
        scores = {
            "agronomy_agent": sum(1 for kw in agronomy_kw if kw in query_lower),
            "commerce_agent": sum(1 for kw in commerce_kw if kw in query_lower),
            "platform_agent": sum(1 for kw in platform_kw if kw in query_lower),
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
            agents.append({
                "name": name,
                "description": agent.description,
                "categories": agent.config.kb_categories,
            })
        return agents
