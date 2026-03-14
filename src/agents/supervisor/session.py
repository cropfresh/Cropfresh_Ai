"""
Session handling logic for the Supervisor Agent.
"""

from typing import Any

from src.agents.base_agent import AgentResponse
from src.memory.state_manager import Message


async def process_with_session(
    agent_instance: Any,
    query: str,
    session_id: str,
) -> AgentResponse:
    """
    Process query with session context.

    Maintains conversation history across multiple queries.
    On every turn:
      1. Loads full ConversationContext from Redis/memory
      2. Extracts entities from user query (commodity, quantity, district)
      3. Builds rich context dict (history, entities, user_profile, current_agent)
      4. Delegates to target agent via process()
      5. Saves assistant reply + updates entities + current_agent back to session

    Args:
        agent_instance: The SupervisorAgent instance
        query: User query
        session_id: Session identifier

    Returns:
        AgentResponse
    """
    if not agent_instance.state_manager:
        return await agent_instance.process(query)

    # 1. Get session context
    session = await agent_instance.state_manager.get_context(session_id)
    if not session:
        # Create new session if it has expired or is unknown
        session = await agent_instance.state_manager.create_session()
        session_id = session.session_id

    # 2. Add user message to history
    await agent_instance.state_manager.add_message(
        session.session_id,
        Message(role="user", content=query),
    )

    # 3. Entity extraction on user query (Phase 3 / G3 fix)
    await agent_instance.state_manager.extract_and_merge_entities(session.session_id, query)

    # Reload to get freshly merged entities
    session = await agent_instance.state_manager.get_context(session.session_id) or session

    # 4. Build rich context dict (Phase 4 / G4 fix)
    context = {
        "user_profile": session.user_profile,
        "entities": session.entities,                          # structured facts
        "current_agent": session.current_agent,                # previous agent name
        "conversation_summary": agent_instance.state_manager.get_conversation_summary(session),
    }

    # 5. Create execution state
    execution = agent_instance.state_manager.create_execution(session.session_id, query)

    # 6. Process
    response = await agent_instance.process(query, context, execution)

    # 7. Save assistant reply
    await agent_instance.state_manager.add_message(
        session.session_id,
        Message(role="assistant", content=response.content),
    )

    # 8. Extract entities from assistant response too (catches commodity/price in answers)
    await agent_instance.state_manager.extract_and_merge_entities(
        session.session_id, response.content
    )

    # 9. Write current_agent back to session (Phase 5 / G5 fix)
    if response.agent_name:
        await agent_instance.state_manager.update_entities(
            session.session_id,
            {"__current_agent": response.agent_name},
        )

    return response
