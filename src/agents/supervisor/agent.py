"""
Supervisor Agent class definition.
"""

from typing import Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry

from src.agents.supervisor.models import RoutingDecision
from src.agents.supervisor.prompts import get_system_prompt
from src.agents.supervisor.router import route_query
from src.agents.supervisor.session import process_with_session
from src.agents.supervisor.utils import merge_responses

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
        """Get the system prompt using the dedicated prompt function."""
        return get_system_prompt(context)
    
    async def route_query(self, query: str, context: Optional[dict] = None) -> RoutingDecision:
        """Route query using the dedicated router function."""
        return await route_query(self, query, context)
        
    def _route_rule_based(self, query: str) -> RoutingDecision:
        """Route rule-based fallback."""
        from src.agents.supervisor.rules import route_rule_based
        return route_rule_based(query)

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process a user query through the multi-agent system.
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
            
    async def process_with_session(self, query: str, session_id: str) -> AgentResponse:
        """Process query with session context using dedicated session function."""
        return await process_with_session(self, query, session_id)
        
    def _merge_responses(self, primary: AgentResponse, secondary: AgentResponse) -> AgentResponse:
        """Merge responses using dedicated utils function."""
        return merge_responses(primary, secondary)
        
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
