---
description: Artifact-first documentation rules for CropFresh AI development
---

# CropFresh Docs & Artifacts Rule

This rule enforces consistent documentation practices across all AI agent work.

## Core Rules

### 1. No Markdown Overwrite
- **Never** overwrite existing `.md` files without explicit user approval
- Append to existing docs; use Git for version history
- Sprint outcomes, daily logs, and retrospectives are append-only

### 2. Required Mermaid Diagrams
Every new architecture or feature document **must** include at least one Mermaid diagram:
```markdown
```mermaid
graph TD
    A --> B
```
```

### 3. Documentation With Every Feature
When implementing a new feature or agent:
1. Update `docs/agents/REGISTRY.md` if adding an agent
2. Update `docs/api/endpoints-reference.md` if adding an endpoint
3. Log the change in `WORKFLOW_STATUS.md` file changes table
4. Create or update the relevant `docs/features/` document

### 4. File Naming Convention
| Type | Pattern | Example |
|------|---------|---------|
| Architecture docs | `docs/architecture/kebab-case.md` | `system-architecture.md` |
| Feature docs | `docs/features/kebab-case.md` | `voice-pipeline.md` |
| ADRs | `docs/decisions/ADR-NNN-kebab-case.md` | `ADR-007-agentic-rag.md` |
| Sprint files | `tracking/sprints/sprint-NN-theme.md` | `sprint-05-advanced-rag.md` |
| Daily logs | `tracking/daily/YYYY-MM-DD.md` | `2026-03-11.md` |

### 5. Cross-References
All docs should link to related documents using relative paths:
```markdown
See [Agent Registry](../agents/REGISTRY.md) for details.
```

### 6. Protected Paths
These files require extra care — read before modifying:
- `PLAN.md` — Master product plan
- `AGENTS.md` — AI agent instructions
- `ai/rag/` — RAG pipeline (check eval results before modifying)
- `src/agents/base_agent.py` — Base class (changes affect all agents)
