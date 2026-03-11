---
description: How to implement a new feature with accompanying documentation and tests
---

# Implement Feature With Docs

Follow these steps when implementing any new feature in CropFresh AI.

## Steps

1. **Read context files:**
   - `PLAN.md` — Understand the product vision
   - `tracking/PROJECT_STATUS.md` — Current state
   - Active sprint file in `tracking/sprints/`

2. **Plan the implementation:**
   - Identify which files to create/modify
   - Note which agents, tools, or APIs are affected
   - Check `docs/agents/REGISTRY.md` for existing agent patterns

3. **Write the code:**
   - Follow patterns in `src/agents/base_agent.py` for new agents
   - Use `loguru.logger` (never `print()`)
   - Use structured types (Pydantic models)

4. **Write tests:**
   - Create `tests/unit/test_<feature>.py`
   - Cover: happy path, edge cases, errors
   - Mock external dependencies (Qdrant, Groq, Redis)

5. **Update documentation:**
   - If new agent → update `docs/agents/REGISTRY.md`
   - If new endpoint → update `docs/api/endpoints-reference.md`
   - If new feature → create `docs/features/<feature>.md` with Mermaid diagram
   - Always update `WORKFLOW_STATUS.md` file changes log

6. **Run verification:**
   ```bash
   uv run pytest -v
   uv run mypy src/
   uv run ruff check src/
   ```

7. **Commit with sprint tag:**
   ```bash
   git commit -m "Sprint-XX: <concise task description>"
   ```
