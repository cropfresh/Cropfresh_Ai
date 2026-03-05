"""
General Agent
=============
Fallback agent for greetings, general questions, and unclear intents.

Handles:
- Greetings and pleasantries
- General questions about CropFresh
- Unclear or ambiguous queries
- Redirecting to appropriate specialized agents

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.prompt_context import build_system_prompt
from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry


GENERAL_SYSTEM_PROMPT = """**What you can help with:**
- 🌱 **Farming advice**: Crop cultivation, pest management, soil health
- 💰 **Market prices**: Current mandi rates, sell/hold recommendations
- 📱 **App support**: Registration, orders, payments, features
- 📊 **Crop recommendations**: What to sow, demand analysis
- 🚚 **Logistics**: Delivery costs, route planning
- 🔍 **Quality grading**: Photo-based produce inspection
- ❓ **General questions**: About CropFresh, agriculture in India

For specialized queries, you might say:
- "I can help with that farming question!" (then answer)
- "Let me check the current market prices for you."
- "I can explain how to use that feature."

Remember: You're here to make farmers' lives easier!"""


GENERAL_ROLE = "You are Prashna Krishi (प्रश्न कृषि), the friendly AI assistant for CropFresh."


class GeneralAgent(BaseAgent):
    """
    Fallback agent for general queries and greetings.
    
    Handles:
    - Greetings and pleasantries
    - General questions
    - Unclear intents
    - Redirecting to specialized agents
    
    Usage:
        agent = GeneralAgent(llm=provider)
        await agent.initialize()
        response = await agent.process("Hello, how are you?")
    """
    
    def __init__(
        self,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        """
        Initialize General Agent.
        """
        config = AgentConfig(
            name="general_agent",
            description="Friendly assistant for greetings and general questions",
            max_retries=1,
            temperature=0.8,  # Slightly higher for more conversational tone
            max_tokens=400,
            kb_categories=["general", "platform"],
            tool_categories=[],
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )
    
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get general agent prompt with shared CropFresh context."""
        return build_system_prompt(
            role_description=GENERAL_ROLE,
            domain_prompt=GENERAL_SYSTEM_PROMPT,
            context=context,
            include_platform=True,
            agent_domain="general",
        )
    
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process a general query or greeting.
        
        Args:
            query: User query
            context: Optional context
            execution: Optional execution state
            
        Returns:
            AgentResponse with friendly reply
        """
        logger.info(f"GeneralAgent processing: '{query[:50]}...'")
        
        try:
            # Check for simple greetings first
            simple_response = self._handle_simple_greeting(query)
            if simple_response:
                return AgentResponse(
                    content=simple_response,
                    agent_name=self.name,
                    confidence=1.0,
                    steps=["greeting"],
                    suggested_actions=[
                        "How do I grow tomatoes?",
                        "What's the current onion price?",
                        "Help me register on CropFresh",
                    ],
                )
            
            # For other queries, use LLM
            if execution:
                self.state_manager.add_step(execution.execution_id, "generate_response")
            
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
                {"role": "user", "content": query},
            ]
            
            if self.llm:
                answer = await self.generate_with_llm(messages)
            else:
                answer = self._generate_fallback(query)
            
            return AgentResponse(
                content=answer,
                agent_name=self.name,
                confidence=0.7,
                steps=["generate_response"],
                suggested_actions=self._get_suggested_topics(),
            )
            
        except Exception as e:
            logger.error(f"GeneralAgent error: {e}")
            
            return AgentResponse(
                content="Hello! 👋 I'm Prashna Krishi, your CropFresh assistant. How can I help you today?",
                agent_name=self.name,
                confidence=0.5,
                error=str(e),
                steps=["error_fallback"],
            )
    
    def _handle_simple_greeting(self, query: str) -> Optional[str]:
        """Handle simple greetings without LLM."""
        query_lower = query.lower().strip()
        
        greetings = {
            "hello": "Hello! 👋 I'm Prashna Krishi, your CropFresh AI assistant. How can I help you today?",
            "hi": "Hi there! 👋 Welcome to CropFresh. How can I assist you?",
            "hey": "Hey! 👋 I'm here to help with farming advice, market prices, or app questions. What would you like to know?",
            "namaste": "नमस्ते! 🙏 I'm Prashna Krishi. How can I help you today?",
            "namaskara": "ನಮಸ್ಕಾರ! 🙏 ನಾನು ಪ್ರಶ್ನ ಕೃಷಿ, ನಿಮ್ಮ ಕ್ರಾಪ್‌ಫ್ರೆಶ್ ಎಐ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ?",
            "ನಮಸ್ಕಾರ": "ನಮಸ್ಕಾರ! 🙏 ನಾನು ಪ್ರಶ್ನ ಕೃಷಿ, ನಿಮ್ಮ ಕ್ರಾಪ್‌ಫ್ರೆಶ್ ಎಐ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ?",
            "thanks": "You're welcome! 😊 Is there anything else I can help you with?",
            "thank you": "You're welcome! 😊 Happy to help. Need anything else?",
            "dhanyavada": "ಧನ್ಯವಾದಗಳು! 😊 ನಿಮಗೆ ಬೇರೆ ಏನಾದರೂ ಸಹಾಯ ಬೇಕೆ?",
            "ಧನ್ಯವಾದಗಳು": "ಧನ್ಯವಾದಗಳು! 😊 ನಿಮಗೆ ಬೇರೆ ಏನಾದರೂ ಸಹಾಯ ಬೇಕೆ?",
            "bye": "Goodbye! 👋 Good luck with your farming. Come back anytime!",
            "help": "I can help you with:\n\n🌱 **Farming**: Crop cultivation, pests, soil health\n💰 **Prices**: Current market rates, sell/hold advice\n📱 **App**: Registration, orders, payments\n\nWhat would you like to know?",
            "ಸಹಾಯ": "ನಾನು ನಿಮಗೆ ಈ ಕೆಳಗಿನ ವಿಷಯಗಳಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:\n\n🌱 **ಕೃಷಿ**: ಬೆಳೆ ಸಾಗುವಳಿ, ಕೀಟಗಳು, ಮಣ್ಣಿನ ಆರೋಗ್ಯ\n💰 **ಬೆಲೆಗಳು**: ಪ್ರಸ್ತುತ ಮಾರುಕಟ್ಟೆ ದರಗಳು, ಮಾರ್ಗದರ್ಶನ\n📱 **ಆಪ್**: ನೋಂದಣಿ, ವಹಿವಾಟು, ಪಾವತಿಗಳು\n\nನೀವು ಏನನ್ನು ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಿ?",
        }
        
        # Check for exact or partial matches
        for key, response in greetings.items():
            if query_lower == key or query_lower.startswith(key + " ") or query_lower.startswith(key + ","):
                return response
        
        return None
    
    def _generate_fallback(self, query: str) -> str:
        """Generate response without LLM."""
        return f"""I understand you're asking about: "{query}"

I can help you with:
- 🌱 **Farming advice**: Growing crops, managing pests, soil health
- 💰 **Market prices**: Current rates, when to sell
- 📱 **CropFresh app**: Registration, orders, payments

Could you tell me more about what you need help with?"""
    
    def _get_suggested_topics(self) -> list[str]:
        """Get suggested follow-up topics."""
        return [
            "How do I grow tomatoes in Karnataka?",
            "What is the current onion price?",
            "How do I register on CropFresh?",
        ]
