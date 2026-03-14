import pytest
from unittest.mock import AsyncMock, patch
import json

from src.agents.supervisor import SupervisorAgent

@pytest.fixture
def supervisor():
    return SupervisorAgent()

@pytest.mark.asyncio
async def test_supervisor_rule_based_routing():
    """Test standard keyword-based fallback routing."""
    supervisor = SupervisorAgent() # No LLM provided
    
    # Agronomy routing
    decision = await supervisor.route_query("my tomato plants have a disease")
    assert decision.agent_name == "agronomy_agent"
    
    # Commerce routing 
    decision = await supervisor.route_query("what is the current price of onions in the mandi")
    assert decision.agent_name == "commerce_agent"
    
    # Buyer matching routing (exact phrase rule)
    decision = await supervisor.route_query("find buyer for my cabbage")
    assert decision.agent_name == "buyer_matching_agent"
    
    # Unknown queries hit general_agent
    decision = await supervisor.route_query("the sky is blue")
    assert decision.agent_name == "general_agent"

@pytest.mark.asyncio
async def test_supervisor_llm_routing_parsing():
    """Verify that SupervisorAgent correctly parses the JSON schema returned by the LLM."""
    mock_llm = AsyncMock()
    # Provide a mock JSON string typical of an LLM response (wrapped in markdown code blocks)
    mock_llm.return_value = '''```json
{
    "agent_name": "adcl_agent",
    "confidence": 0.95,
    "reasoning": "User is asking for crop recommendations.",
    "requires_multiple": false,
    "secondary_agents": []
}
```'''
    
    # Patch `generate_with_llm` to return our mock string
    supervisor = SupervisorAgent()
    with patch.object(supervisor, 'generate_with_llm', new=mock_llm):
        supervisor.llm = mock_llm  # Flag that we have an LLM
        decision = await supervisor.route_query("What crop should I grow next week?")
        
        assert decision.agent_name == "adcl_agent"
        assert decision.confidence == 0.95
        assert decision.requires_multiple is False
