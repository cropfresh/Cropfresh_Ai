# Agent Design Principles

## 1. Single Responsibility
Each agent handles one domain. The supervisor routes.

## 2. Tool-Based Architecture
Agents use explicit tools (functions) — no unstructured reasoning.

## 3. Constrained Output
All agents return structured Pydantic responses.

## 4. Evaluation-Driven
Every agent must have evals before production deployment.

## 5. Prompt Versioning
All prompts are versioned. Active version tracked in registry.yaml.

## 6. Graceful Degradation
If an agent fails, return helpful fallback rather than error.
