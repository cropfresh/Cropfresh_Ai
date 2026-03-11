---
description: How to create or update agent design documentation for CropFresh AI
---

# Agent Design Doc Workflow

Follow these steps when creating or documenting an agent.

## Steps

1. **Read the agent source file:**
   - Check `src/agents/<agent_name>.py`
   - Note: does it inherit `BaseAgent`? What tools/KB does it use?

2. **Read the routing configuration:**
   - Check `src/agents/supervisor_agent.py` → `ROUTING_PROMPT`
   - Note keywords and confidence thresholds

3. **Read the registry wiring:**
   - Check `src/agents/agent_registry.py`
   - Note which creation group it belongs to

4. **Create/update the agent entry in `docs/agents/REGISTRY.md`:**
   - Agent name, module path, inheritance status
   - Keywords that trigger routing
   - Tools and KB categories it accesses
   - Status (stable, partial, todo)

5. **If the agent is complex, create a standalone doc:**
   - Create `docs/agents/<agent_name>.md`
   - Include Mermaid diagram of the agent's processing flow
   - Document multi-turn flows, intents, entity requirements

6. **Update `docs/agents-index.yaml`:**
   ```yaml
   - name: your_agent
     module: src/agents/your_agent.py
     inherits_base: true
     tools: [tool1, tool2]
   ```

7. **Log the change in `WORKFLOW_STATUS.md`**
