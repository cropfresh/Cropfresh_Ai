"""whatsapp_bot Agent — CropFresh AI"""
from typing import Optional

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent


class WhatsappBotAgent(BaseAgent):
    """whatsapp_bot agent implementation."""

    def __init__(self):
        config = AgentConfig(
            name="whatsapp_bot",
            description="WhatsApp bot agent for CropFresh AI",
        )
        super().__init__(config=config)

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution=None,
    ) -> AgentResponse:
        """Process the agent's main task."""
        raise NotImplementedError("whatsapp_bot agent not yet implemented")

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get domain-specific system prompt."""
        return "You are a WhatsApp bot for CropFresh AI."

