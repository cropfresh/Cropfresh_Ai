"""
Routing logic for the Supervisor Agent.
"""

import json
from typing import Optional
from loguru import logger

from src.agents.supervisor.models import RoutingDecision
from src.agents.supervisor.prompts import get_system_prompt
from src.agents.supervisor.rules import route_rule_based

async def route_query(
    agent_instance,
    query: str,
    context: Optional[dict] = None,
) -> RoutingDecision:
    """
    Determine which agent should handle the query using LLM or fallback to rules.
    
    Args:
        agent_instance: The SupervisorAgent instance
        query: User query
        context: Optional conversation context
        
    Returns:
        RoutingDecision with selected agent
    """
    # If no LLM, use rule-based routing
    if not agent_instance.llm:
        return route_rule_based(query)
    
    try:
        # Build messages for LLM
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": query},
        ]
        
        # Add conversation context if available
        if context and context.get("conversation_summary"):
            messages.insert(1, {
                "role": "system",
                "content": f"Previous conversation:\n{context['conversation_summary']}",
            })
            
        # Get routing decision
        response = await agent_instance.generate_with_llm(
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
        return route_rule_based(query)
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        return route_rule_based(query)
