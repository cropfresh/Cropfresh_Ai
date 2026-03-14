"""
Utility functions for the Supervisor Agent.
"""

from src.agents.base_agent import AgentResponse


def merge_responses(primary: AgentResponse, secondary: AgentResponse) -> AgentResponse:
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
