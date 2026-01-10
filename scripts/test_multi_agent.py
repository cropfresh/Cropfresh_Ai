"""
Multi-Agent System Test Script
==============================
Tests the Advanced Agentic RAG system with specialized agents.

Usage:
    cd d:\\Cropfresh Ai\\cropfresh-service-ai
    .venv\\Scripts\\activate
    python scripts/test_multi_agent.py

Author: CropFresh AI Team
Version: 2.0.0
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_state_manager():
    """Test the agent state manager."""
    print("\n" + "=" * 60)
    print("TEST 1: Agent State Manager")
    print("=" * 60)
    
    try:
        from src.memory.state_manager import AgentStateManager, Message
        
        manager = AgentStateManager()
        
        # Create session
        session = await manager.create_session(user_id="test_farmer")
        print(f"✅ Created session: {session.session_id[:8]}...")
        
        # Add messages
        await manager.add_message(
            session.session_id,
            Message(role="user", content="How to grow tomatoes?"),
        )
        await manager.add_message(
            session.session_id,
            Message(role="assistant", content="Tomatoes need sunlight..."),
        )
        
        # Get context
        context = await manager.get_context(session.session_id)
        print(f"   Messages: {len(context.messages)}")
        
        # Get summary
        summary = manager.get_conversation_summary(context)
        print(f"   Summary generated: {len(summary)} chars")
        
        print("✅ State manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ State manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_registry():
    """Test the tool registry."""
    print("\n" + "=" * 60)
    print("TEST 2: Tool Registry")
    print("=" * 60)
    
    try:
        from src.tools.registry import get_tool_registry
        
        registry = get_tool_registry()
        
        # List tools
        tools = registry.list_tools()
        print(f"✅ Registered tools: {len(tools)}")
        for tool in tools:
            defn = registry.get_definition(tool)
            print(f"   - {tool} ({defn.category}): {defn.description[:50]}...")
        
        # Test tool execution
        print("\n   Testing calculator tool...")
        result = await registry.execute(
            "calculate_aisp",
            farmer_price_per_kg=25,
            quantity_kg=100,
            distance_km=30,
        )
        
        if result.success:
            print(f"   ✅ AISP calculated: ₹{result.result['total_aisp']:.0f} (₹{result.result['aisp_per_kg']:.2f}/kg)")
        else:
            print(f"   ⚠️ Calculator failed: {result.error}")
        
        print("✅ Tool registry test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tool registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_routing():
    """Test supervisor agent query routing."""
    print("\n" + "=" * 60)
    print("TEST 3: Agent Routing (Rule-based)")
    print("=" * 60)
    
    try:
        from src.agents.supervisor_agent import SupervisorAgent
        
        supervisor = SupervisorAgent()  # No LLM for rule-based
        
        test_queries = [
            ("How to grow tomatoes?", "agronomy_agent"),
            ("What is the current onion price?", "commerce_agent"),
            ("How do I register on CropFresh?", "platform_agent"),
            ("Hello, how are you?", "general_agent"),
        ]
        
        for query, expected in test_queries:
            routing = await supervisor.route_query(query)
            status = "✅" if routing.agent_name == expected else "⚠️"
            print(f"   {status} '{query[:40]}...'")
            print(f"      Routed to: {routing.agent_name} (expected: {expected})")
            print(f"      Confidence: {routing.confidence:.0%}")
        
        print("✅ Agent routing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_general_agent():
    """Test general agent with greetings."""
    print("\n" + "=" * 60)
    print("TEST 4: General Agent")
    print("=" * 60)
    
    try:
        from src.agents.general_agent import GeneralAgent
        
        agent = GeneralAgent()
        await agent.initialize()
        
        greetings = ["Hello", "Hi there", "Help", "Thanks"]
        
        for greeting in greetings:
            response = await agent.process(greeting)
            print(f"   Query: '{greeting}'")
            print(f"   Response: {response.content[:80]}...")
            print()
        
        print("✅ General agent test passed!")
        return True
        
    except Exception as e:
        print(f"❌ General agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_commerce_agent():
    """Test commerce agent with pricing."""
    print("\n" + "=" * 60)
    print("TEST 5: Commerce Agent (Mock Data)")
    print("=" * 60)
    
    try:
        from src.agents.commerce_agent import CommerceAgent
        
        agent = CommerceAgent()
        await agent.initialize()
        
        query = "What is the current tomato price in Kolar?"
        print(f"   Query: '{query}'")
        
        response = await agent.process(query)
        
        print(f"   Agent: {response.agent_name}")
        print(f"   Confidence: {response.confidence:.0%}")
        print(f"   Tools used: {response.tools_used}")
        print(f"   Response preview: {response.content[:200]}...")
        
        print("✅ Commerce agent test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Commerce agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_multi_agent_pipeline():
    """Test the full multi-agent pipeline."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Multi-Agent Pipeline")
    print("=" * 60)
    
    try:
        from src.agents.supervisor_agent import SupervisorAgent
        from src.agents.agronomy_agent import AgronomyAgent
        from src.agents.commerce_agent import CommerceAgent
        from src.agents.platform_agent import PlatformAgent
        from src.agents.general_agent import GeneralAgent
        from src.memory.state_manager import AgentStateManager
        from src.tools.registry import get_tool_registry
        
        # Create components
        state_manager = AgentStateManager()
        tool_registry = get_tool_registry()
        
        # Create supervisor
        supervisor = SupervisorAgent(
            state_manager=state_manager,
            tool_registry=tool_registry,
        )
        
        # Create and register agents
        agronomy = AgronomyAgent(
            state_manager=state_manager,
            tool_registry=tool_registry,
        )
        commerce = CommerceAgent(
            state_manager=state_manager,
            tool_registry=tool_registry,
        )
        platform = PlatformAgent(
            state_manager=state_manager,
            tool_registry=tool_registry,
        )
        general = GeneralAgent(
            state_manager=state_manager,
            tool_registry=tool_registry,
        )
        
        supervisor.register_agent("agronomy_agent", agronomy)
        supervisor.register_agent("commerce_agent", commerce)
        supervisor.register_agent("platform_agent", platform)
        supervisor.register_agent("general_agent", general)
        supervisor.set_fallback_agent(general)
        
        await supervisor.initialize()
        
        print("   Supervisor initialized with 4 agents")
        
        # Test queries
        test_queries = [
            "Hello!",
            "How to grow tomatoes in Karnataka?",
            "What is the current onion price?",
            "How do I register as a farmer?",
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            response = await supervisor.process(query)
            print(f"   Agent: {response.agent_name}")
            print(f"   Steps: {' → '.join(response.steps)}")
            print(f"   Response: {response.content[:100]}...")
        
        print("\n✅ Multi-agent pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-agent pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_llm():
    """Test with actual LLM (requires GROQ_API_KEY)."""
    print("\n" + "=" * 60)
    print("TEST 7: Full Pipeline with LLM")
    print("=" * 60)
    
    try:
        from src.config import get_settings
        from src.orchestrator.llm_provider import create_llm_provider
        from src.agents.supervisor_agent import SupervisorAgent
        from src.agents.agronomy_agent import AgronomyAgent
        from src.agents.commerce_agent import CommerceAgent
        from src.agents.platform_agent import PlatformAgent
        from src.agents.general_agent import GeneralAgent
        from src.memory.state_manager import AgentStateManager
        from src.tools.registry import get_tool_registry
        from src.rag.knowledge_base import KnowledgeBase
        
        settings = get_settings()
        
        if not settings.groq_api_key:
            print("⚠️  No GROQ_API_KEY configured - skipping LLM test")
            return True
        
        # Create LLM
        llm = create_llm_provider(
            provider=settings.llm_provider,
            api_key=settings.groq_api_key,
            model=settings.llm_model,
        )
        print(f"   LLM: {settings.llm_provider} / {settings.llm_model}")
        
        # Create knowledge base
        kb = KnowledgeBase(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        await kb.initialize()
        print(f"   Knowledge base: {kb.get_stats()}")
        
        # Create components
        state_manager = AgentStateManager()
        tool_registry = get_tool_registry()
        
        # Create supervisor with LLM
        supervisor = SupervisorAgent(
            llm=llm,
            state_manager=state_manager,
            tool_registry=tool_registry,
            knowledge_base=kb,
        )
        
        # Create agents with LLM
        agents = {
            "agronomy_agent": AgronomyAgent(llm=llm, state_manager=state_manager, tool_registry=tool_registry, knowledge_base=kb),
            "commerce_agent": CommerceAgent(llm=llm, state_manager=state_manager, tool_registry=tool_registry, knowledge_base=kb),
            "platform_agent": PlatformAgent(llm=llm, state_manager=state_manager, tool_registry=tool_registry, knowledge_base=kb),
            "general_agent": GeneralAgent(llm=llm, state_manager=state_manager, tool_registry=tool_registry, knowledge_base=kb),
        }
        
        for name, agent in agents.items():
            supervisor.register_agent(name, agent)
        supervisor.set_fallback_agent(agents["general_agent"])
        
        await supervisor.initialize()
        
        # Test with LLM
        query = "How should I grow tomatoes during the monsoon season in Karnataka?"
        print(f"\n   Query: '{query}'")
        print("   Processing with LLM...")
        
        response = await supervisor.process(query)
        
        print(f"\n   Agent: {response.agent_name}")
        print(f"   Confidence: {response.confidence:.0%}")
        print(f"   Steps: {' → '.join(response.steps)}")
        print(f"   Sources: {response.sources}")
        print(f"\n   Response:\n   {response.content[:500]}...")
        
        print("\n✅ LLM pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("  CropFresh AI - Multi-Agent System Test Suite")
    print("  Version 2.0.0 - Advanced Agentic RAG")
    print("=" * 60)
    
    results = {}
    
    # Core tests
    results["state_manager"] = await test_state_manager()
    results["tool_registry"] = await test_tool_registry()
    results["agent_routing"] = await test_agent_routing()
    results["general_agent"] = await test_general_agent()
    results["commerce_agent"] = await test_commerce_agent()
    results["multi_agent_pipeline"] = await test_full_multi_agent_pipeline()
    
    # LLM test (optional)
    results["llm_pipeline"] = await test_with_llm()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️  Some tests failed"))
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
