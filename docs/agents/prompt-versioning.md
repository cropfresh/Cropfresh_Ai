# Prompt Versioning System

## Structure
Each agent has a prompts/ directory:
``nsrc/agents/{agent_name}/prompts/
  v1_system.md
  v2_system.md
  registry.yaml  # points to active version
``n
## Rules
1. Never edit an existing version — create a new one
2. Update registry.yaml to point to active version
3. Run evals before activating a new version
4. Document changes in agent changelog
