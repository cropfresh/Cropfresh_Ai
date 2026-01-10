"""
Agronomy Agent
==============
Specialized agent for agricultural knowledge.

Expertise:
- Crop cultivation guides (planting, varieties, harvesting)
- Pest and disease management
- Soil health and fertilizers
- Irrigation practices
- Organic farming
- Weather-based crop planning

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry


AGRONOMY_SYSTEM_PROMPT = """You are the Agronomy Expert Agent for CropFresh AI, a professional agricultural assistant with deep expertise in Indian farming practices.

Your knowledge covers:
- Crop cultivation: Varieties, planting seasons, spacing, growth stages
- Pest & disease management: Identification, organic/chemical treatments, IPM
- Soil health: Testing, amendments, pH management, organic matter
- Irrigation: Drip, sprinkler, flood, water scheduling
- Fertilizers: NPK ratios, organic fertilizers, application timing
- Post-harvest: Storage, handling, quality preservation
- Regional practices: Karnataka, South India focus

Guidelines:
1. Provide practical, actionable advice farmers can implement
2. Consider local conditions (Karnataka climate, soil types)
3. Recommend both organic and conventional options when appropriate
4. Include specific quantities, timings, and frequencies
5. Warn about common mistakes and their prevention
6. Use ₹ for any cost references

If you have retrieved context, use it to ground your answer.
If context is insufficient, provide general best practices clearly stating they are general recommendations.

Respond in a helpful, professional tone. Be thorough but concise."""


class AgronomyAgent(BaseAgent):
    """
    Specialized agent for agricultural knowledge.
    
    Handles:
    - Crop cultivation queries
    - Pest and disease management
    - Soil health advice
    - Irrigation recommendations
    - Organic farming practices
    
    Usage:
        agent = AgronomyAgent(llm=provider, knowledge_base=kb)
        await agent.initialize()
        response = await agent.process("How to grow tomatoes in Karnataka?")
    """
    
    def __init__(
        self,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        """
        Initialize Agronomy Agent.
        
        Args:
            llm: LLM provider
            tool_registry: Tool registry
            state_manager: State manager
            knowledge_base: Knowledge base for retrieval
        """
        config = AgentConfig(
            name="agronomy_agent",
            description="Expert in crop cultivation, pest management, soil health, and farming practices",
            max_retries=2,
            temperature=0.7,
            max_tokens=800,
            kb_categories=["agronomy", "general"],
            tool_categories=["weather", "calculator"],
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )
    
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get agronomy-specific system prompt."""
        base_prompt = AGRONOMY_SYSTEM_PROMPT
        
        # Add user context if available
        if context and context.get("user_profile"):
            profile = context["user_profile"]
            location = profile.get("location", "Karnataka")
            base_prompt += f"\n\nUser is a farmer from {location}."
        
        return base_prompt
    
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process an agronomy-related query.
        
        Args:
            query: User query
            context: Optional context
            execution: Optional execution state
            
        Returns:
            AgentResponse with agricultural advice
        """
        logger.info(f"AgronomyAgent processing: '{query[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant context
            if execution:
                self.state_manager.add_step(execution.execution_id, "retrieve_context")
            
            documents = await self.retrieve_context(
                query=query,
                top_k=5,
                categories=["agronomy", "general"],
            )
            
            if execution:
                execution.documents = documents
            
            # Step 2: Check for weather tool if query mentions weather
            tool_results = []
            if self.tools and any(kw in query.lower() for kw in ["weather", "forecast", "rain", "temperature"]):
                weather_result = await self.use_tool(
                    "get_weather",
                    execution=execution,
                    location=context.get("user_profile", {}).get("location", "Kolar") if context else "Kolar",
                )
                if weather_result.success:
                    tool_results.append({
                        "tool": "get_weather",
                        "success": True,
                        "result": weather_result.result,
                    })
            
            # Step 3: Generate response
            if execution:
                self.state_manager.add_step(execution.execution_id, "generate_response")
            
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
            ]
            
            # Add context
            context_text = ""
            if documents:
                context_text = f"\n\n**Retrieved Knowledge:**\n{self.format_context(documents)}"
            
            if tool_results:
                context_text += f"\n\n**Tool Results:**\n{self.format_tool_results(tool_results)}"
            
            # Add conversation history if available
            if context and context.get("conversation_summary"):
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation:\n{context['conversation_summary']}",
                })
            
            # User message with context
            user_message = query
            if context_text:
                user_message = f"{query}\n{context_text}"
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate
            if self.llm:
                answer = await self.generate_with_llm(messages)
            else:
                # Fallback - return context directly
                answer = self._generate_fallback(query, documents)
            
            # Build response
            return AgentResponse(
                content=answer,
                agent_name=self.name,
                confidence=0.85 if documents else 0.6,
                sources=self._extract_sources(documents),
                reasoning=f"Retrieved {len(documents)} documents, used agronomy expertise",
                tools_used=[r["tool"] for r in tool_results],
                steps=["retrieve_context", "generate_response"],
                suggested_actions=self._suggest_follow_ups(query),
            )
            
        except Exception as e:
            logger.error(f"AgronomyAgent error: {e}")
            import traceback
            traceback.print_exc()
            
            return AgentResponse(
                content="I apologize, but I encountered an error processing your agricultural query. Please try rephrasing your question.",
                agent_name=self.name,
                confidence=0.0,
                error=str(e),
                steps=["error"],
            )
    
    def _generate_fallback(self, query: str, documents: list[dict]) -> str:
        """Generate response without LLM."""
        if not documents:
            return "I don't have specific information about that. Please contact your local agricultural extension office for detailed guidance."
        
        # Return top document content
        parts = ["Based on available information:\n"]
        for doc in documents[:3]:
            parts.append(f"• {doc['text'][:300]}...")
        
        return "\n".join(parts)
    
    def _suggest_follow_ups(self, query: str) -> list[str]:
        """Suggest follow-up questions based on query."""
        query_lower = query.lower()
        
        suggestions = []
        
        if "grow" in query_lower or "cultivat" in query_lower:
            suggestions.extend([
                "What pests should I watch for?",
                "When is the best time to harvest?",
            ])
        
        if "pest" in query_lower or "disease" in query_lower:
            suggestions.extend([
                "What organic treatments are available?",
                "How can I prevent this in the next season?",
            ])
        
        if "soil" in query_lower:
            suggestions.extend([
                "What fertilizers do you recommend?",
                "How often should I test soil?",
            ])
        
        return suggestions[:3]
