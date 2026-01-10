"""
Platform Agent
==============
Specialized agent for CropFresh platform features and FAQs.

Expertise:
- CropFresh app features and navigation
- Registration and onboarding flows
- Quality grades (A/B/C, Digital Twin)
- Order management and logistics
- Payment and settlements
- Troubleshooting common issues

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry


PLATFORM_SYSTEM_PROMPT = """You are the Platform Expert Agent for CropFresh AI, specializing in our agricultural marketplace platform.

Your knowledge covers:

**For Farmers:**
- Registration: KYC, Aadhaar verification, bank details
- Listing produce: Photos, quantity, expected price, harvest date
- Quality grades: A (Premium), B (Standard), C (Economy), Digital Twin QR
- Order fulfillment: Accepting bids, logistics scheduling, packing
- Payments: Settlement timeline (T+2), bank transfers, transaction history

**For Buyers:**
- Registration: Business verification, GST, PAN
- Browsing: Categories, filters, quality grades, farm traceability
- Ordering: Cart, bidding, AISP breakdown, delivery scheduling
- Quality assurance: Inspection, rejections, refunds

**Platform Features:**
- Digital Twin: QR-based produce traceability
- Prashna Krishi: AI chatbot (that's you!)
- Voice assistant: Hindi/Kannada support
- Price alerts: Market price notifications
- Cold chain: Temperature-controlled logistics

**Troubleshooting:**
- Login issues: Password reset, OTP problems
- Payment delays: Settlement SLA, bank issues
- Order issues: Cancellation, disputes, refunds

Guidelines:
1. Be helpful and patient with first-time users
2. Provide step-by-step instructions when needed
3. Reference specific app screens/buttons when helpful
4. For complex issues, suggest contacting support
5. Celebrate user milestones (first sale, etc.)

Respond in a friendly, supportive tone. Make users feel welcome to the CropFresh community."""


class PlatformAgent(BaseAgent):
    """
    Specialized agent for CropFresh platform support.
    
    Handles:
    - Feature explanations
    - How-to guides
    - Troubleshooting
    - FAQs
    
    Usage:
        agent = PlatformAgent(llm=provider, knowledge_base=kb)
        await agent.initialize()
        response = await agent.process("How do I register as a farmer?")
    """
    
    def __init__(
        self,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        """
        Initialize Platform Agent.
        """
        config = AgentConfig(
            name="platform_agent",
            description="Expert in CropFresh app features, registration, and support",
            max_retries=2,
            temperature=0.7,
            max_tokens=600,
            kb_categories=["platform", "general"],
            tool_categories=["database"],
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )
    
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get platform-specific system prompt."""
        base_prompt = PLATFORM_SYSTEM_PROMPT
        
        if context and context.get("user_profile"):
            profile = context["user_profile"]
            user_type = profile.get("type", "user")
            base_prompt += f"\n\nUser type: {user_type}"
        
        return base_prompt
    
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process a platform-related query.
        
        Args:
            query: User query
            context: Optional context
            execution: Optional execution state
            
        Returns:
            AgentResponse with platform guidance
        """
        logger.info(f"PlatformAgent processing: '{query[:50]}...'")
        
        try:
            # Step 1: Retrieve platform knowledge
            if execution:
                self.state_manager.add_step(execution.execution_id, "retrieve_context")
            
            documents = await self.retrieve_context(
                query=query,
                top_k=5,
                categories=["platform", "general"],
            )
            
            # Step 2: Generate response
            if execution:
                self.state_manager.add_step(execution.execution_id, "generate_response")
            
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
            ]
            
            # Add context
            user_message = query
            if documents:
                context_text = f"\n\n**Platform Knowledge:**\n{self.format_context(documents)}"
                user_message = f"{query}{context_text}"
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate
            if self.llm:
                answer = await self.generate_with_llm(messages)
            else:
                answer = self._generate_fallback(query, documents)
            
            return AgentResponse(
                content=answer,
                agent_name=self.name,
                confidence=0.85 if documents else 0.7,
                sources=self._extract_sources(documents),
                reasoning=f"Retrieved {len(documents)} platform docs",
                steps=["retrieve_context", "generate_response"],
                suggested_actions=self._suggest_follow_ups(query),
            )
            
        except Exception as e:
            logger.error(f"PlatformAgent error: {e}")
            
            return AgentResponse(
                content="I apologize, but I couldn't process your request. For immediate help, contact support@cropfresh.ai or call 1800-XXX-XXXX.",
                agent_name=self.name,
                confidence=0.0,
                error=str(e),
                steps=["error"],
            )
    
    def _generate_fallback(self, query: str, documents: list) -> str:
        """Generate response without LLM."""
        query_lower = query.lower()
        
        # Predefined answers for common questions
        faqs = {
            "register": """
**How to Register on CropFresh:**

1. Download the CropFresh app from Play Store/App Store
2. Click "Register" and select Farmer or Buyer
3. Enter your mobile number and verify OTP
4. Complete KYC: Aadhaar, PAN (for buyers), Bank details
5. Wait for verification (usually 24-48 hours)

Need help? Contact support@cropfresh.ai
""",
            "quality": """
**CropFresh Quality Grades:**

- **Grade A (Premium)**: Top quality, minimal defects, commands best prices
- **Grade B (Standard)**: Good quality, some minor defects
- **Grade C (Economy)**: Acceptable quality, suitable for processing

Each produce gets a Digital Twin QR code for traceability!
""",
            "payment": """
**Payment & Settlements:**

- Farmer payments are settled within T+2 (2 business days)
- Payments are deposited directly to your registered bank account
- View transaction history in Profile > Transactions

Payment issues? Contact support@cropfresh.ai
""",
        }
        
        for keyword, answer in faqs.items():
            if keyword in query_lower:
                return answer
        
        if documents:
            return f"Based on CropFresh documentation:\n{documents[0].get('text', '')}"
        
        return "I don't have specific information about that. Please contact support@cropfresh.ai for help."
    
    def _suggest_follow_ups(self, query: str) -> list[str]:
        """Suggest follow-up questions."""
        query_lower = query.lower()
        
        if "register" in query_lower:
            return ["What documents do I need?", "How long does verification take?"]
        elif "payment" in query_lower:
            return ["How do I update my bank details?", "Why is my payment delayed?"]
        elif "order" in query_lower:
            return ["How do I track my order?", "How do I cancel an order?"]
        
        return ["How do I list my produce?", "How do quality grades work?"]
