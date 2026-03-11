# CropFresh AI — Agent Design Guide

> How to create, register, and test a new agent in the CropFresh AI multi-agent system.

---

## Step 1: Understand the Agent Architecture

Every agent in CropFresh follows this pattern:

```
User Query → SupervisorAgent routes → YourAgent.process() → AgentResponse
```

Agents come in two flavors:
1. **BaseAgent subclass** — Inherits `BaseAgent` for full tool/memory/LLM integration
2. **Standalone agent** — Does NOT inherit `BaseAgent`, attached via `state_manager` attribute

---

## Step 2: Create Your Agent File

### Option A: BaseAgent Subclass (Recommended)

Create `src/agents/your_agent.py`:

```python
"""
Your Agent — Brief description.
"""

from typing import Optional
from loguru import logger
from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState


class YourAgent(BaseAgent):
    """Your agent description."""

    def __init__(self, llm=None, tool_registry=None,
                 state_manager=None, knowledge_base=None):
        config = AgentConfig(
            name="your_agent",
            description="What this agent does",
            max_retries=2,
            temperature=0.7,
            max_tokens=1000,
            tool_categories=["your_category"],  # Filter tools
            kb_categories=["your_kb_category"],  # Filter KB search
        )
        super().__init__(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )

    def _get_system_prompt(self, context=None) -> str:
        """Domain-specific system prompt."""
        return """You are a CropFresh AI assistant specializing in...
        Respond in the user's language. Be concise and helpful."""

    async def process(self, query, context=None,
                      execution=None) -> AgentResponse:
        """Main agent logic."""
        # 1. Retrieve relevant documents
        docs = await self.retrieve_context(query)

        # 2. Use tools if needed
        tool_result = await self.use_tool("your_tool", param="value")

        # 3. Build prompt with context
        messages = [
            {"role": "system", "content": self._get_system_prompt(context)},
            {"role": "user", "content": f"Context:\n{self.format_context(docs)}\n\nQuery: {query}"},
        ]

        # 4. Generate response (auto-injects memory)
        content = await self.generate_with_llm(messages, context=context)

        return AgentResponse(
            content=content,
            agent_name=self.name,
            confidence=0.85,
            sources=self._extract_sources(docs),
            tools_used=["your_tool"] if tool_result.success else [],
        )
```

### Option B: Standalone Agent

If your agent doesn't fit the BaseAgent pattern, create it independently and attach `state_manager` as an attribute in the registry.

---

## Step 3: Register Your Agent

Edit `src/agents/agent_registry.py`:

1. **Add a creation function:**

```python
def _create_your_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    agent_kwargs = {k: v for k, v in kwargs.items() if k != "settings"}
    try:
        from src.agents.your_agent import YourAgent
        agents["your_agent"] = YourAgent(**agent_kwargs)
    except Exception as exc:
        logger.warning("YourAgent creation failed: {}", exc)
    return agents
```

2. **Call it in `create_agent_system()`:**

```python
all_agents.update(_create_your_agents(kwargs))
```

---

## Step 4: Add Routing Keywords

Edit `src/agents/supervisor_agent.py`:

1. **Add to the `ROUTING_PROMPT`:**

```python
15. **your_agent**: Expert in [your domain]
    - Keywords: keyword1, keyword2, keyword3
    - Use for: "Example query 1", "Example query 2"
```

2. **Add rule-based keywords in `_route_rule_based()`:**

```python
your_kw = ["keyword1", "keyword2", "keyword3"]
# Add to scores dict:
"your_agent": sum(1 for kw in your_kw if kw in query_lower),
```

---

## Step 5: Write Tests

Create `tests/unit/test_your_agent.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.your_agent import YourAgent

@pytest.fixture
def agent():
    return YourAgent(
        llm=AsyncMock(),
        tool_registry=MagicMock(),
        state_manager=AsyncMock(),
    )

@pytest.mark.asyncio
async def test_process_happy_path(agent):
    agent.llm.generate = AsyncMock(
        return_value=MagicMock(content="Test response")
    )
    response = await agent.process("test query")
    assert response.content == "Test response"
    assert response.agent_name == "your_agent"

@pytest.mark.asyncio
async def test_process_no_llm():
    agent = YourAgent()
    response = await agent.process("test query")
    # Should handle gracefully
    assert response.agent_name == "your_agent"
```

---

## Step 6: Update Documentation

1. Add entry to `docs/agents/REGISTRY.md`
2. Update `docs/agents-index.yaml` with module paths
3. Log the change in `WORKFLOW_STATUS.md`

---

## AgentConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | Agent identifier used in routing |
| `description` | str | (required) | Human-readable description |
| `max_retries` | int | 2 | Retry count for failed operations |
| `temperature` | float | 0.7 | LLM temperature |
| `max_tokens` | int | 1000 | Max response tokens |
| `tool_categories` | list[str] | [] | Tool registry category filter |
| `kb_categories` | list[str] | [] | Knowledge base category filter |
| `system_prompt` | str | "" | Static system prompt (usually via `_get_system_prompt()`) |

## AgentResponse Reference

| Field | Type | Description |
|-------|------|-------------|
| `content` | str | Main response text |
| `agent_name` | str | Name of the responding agent |
| `confidence` | float | Confidence score (0.0–1.0) |
| `sources` | list[str] | Source references |
| `reasoning` | str | Explanation of how answer was derived |
| `tools_used` | list[str] | Tools executed during processing |
| `steps` | list[str] | Execution steps taken |
| `suggested_actions` | list[str] | Follow-up action suggestions |
| `error` | str or None | Error message if failed |
